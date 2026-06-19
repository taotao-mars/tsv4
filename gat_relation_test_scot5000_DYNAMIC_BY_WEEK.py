# ============================================================
# Dynamic ASIN relation test for 5000 + SCOT joint ASINs
# ------------------------------------------------------------
# Goal:
#   Test whether same-category ASIN-ASIN relations can be learned dynamically
#   at each order_week / forecast origin.
#
# Key design:
#   - node = ASIN
#   - graph/order-week origin = each rolling forecast origin
#   - node profile and edge features use ONLY history before origin_week
#   - weak relation label uses a future label window after origin_week
#   - relation classes: positive / competitive / neutral
#
# Core features emphasized:
#   total_dph, buy_box_dph, in_stock_dph, fbi_demand, hbt, ind_top10_brand
#
# Usage is at the bottom. Designed for Jupyter:
#   %run -i gat_relation_test_scot5000_DYNAMIC_BY_WEEK.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------
# Utilities
# -----------------------------

def _safe_numeric(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def _safe_col(df, col, default=0.0):
    if col in df.columns:
        return _safe_numeric(df[col], default)
    return pd.Series(default, index=df.index)


def _safe_str_col(df, col, default="MISSING"):
    if col in df.columns:
        return df[col].astype(str).fillna(default).replace({"nan": default, "None": default, "": default})
    return pd.Series(default, index=df.index)


def _corr_aligned(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)


def _gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.sum(x) <= 0:
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def _mode_str(x, default="MISSING"):
    x = pd.Series(x).astype(str).replace({"nan": default, "None": default, "": default})
    if len(x) == 0:
        return default
    return str(x.mode().iloc[0]) if len(x.mode()) else default


def _prepare_joint_data(data_raw1, scot_df=None, n_asins=5000, seed=42):
    df = data_raw1.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])

    rng = np.random.default_rng(seed)
    unique_asins = df["asin"].dropna().unique()
    sampled = rng.choice(unique_asins, size=min(n_asins, len(unique_asins)), replace=False)
    sampled = set(map(str, sampled))

    if scot_df is not None and "asin" in scot_df.columns:
        scot_asins = set(scot_df["asin"].astype(str).dropna().unique())
        joint = sampled.intersection(scot_asins)
        out = df[df["asin"].isin(joint)].copy()
        print(f"Sampled ASINs: {len(sampled)} | SCOT ASINs: {len(scot_asins)} | Joint ASINs: {len(joint)} | Rows: {len(out)}")
    else:
        out = df[df["asin"].isin(sampled)].copy()
        print(f"Sampled ASINs: {len(sampled)} | Rows: {len(out)} | scot_df not used")

    # required numeric fields
    for c in ["fbi_demand", "total_dph", "buy_box_dph", "in_stock_dph"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = _safe_numeric(out[c]).clip(lower=0.0)

    # important static/dynamic fields
    out["hbt"] = _safe_str_col(out, "hbt")
    if "ind_top10_brand" not in out.columns:
        out["ind_top10_brand"] = 0.0
    out["ind_top10_brand"] = _safe_numeric(out["ind_top10_brand"]).clip(0, 1)

    if "category_code" not in out.columns:
        out["category_code"] = "UNKNOWN"
    out["category_code"] = _safe_str_col(out, "category_code", default="UNKNOWN")

    if "gl_product_group" not in out.columns:
        out["gl_product_group"] = "UNKNOWN"
    out["gl_product_group"] = _safe_str_col(out, "gl_product_group", default="UNKNOWN")

    if "customer_active_review_count" in out.columns:
        out["log_review_count"] = np.log1p(_safe_numeric(out["customer_active_review_count"]).clip(lower=0.0))
    else:
        out["log_review_count"] = 0.0

    if "our_price" in out.columns:
        out["log_price"] = np.log1p(_safe_numeric(out["our_price"]).clip(lower=0.0))
    else:
        out["log_price"] = 0.0

    if "ind_promotion" in out.columns:
        out["ind_promotion"] = _safe_numeric(out["ind_promotion"]).clip(0, 1)
    else:
        out["ind_promotion"] = 0.0

    if "ind_prime_week" in out.columns:
        out["ind_prime_week"] = _safe_numeric(out["ind_prime_week"]).clip(0, 1)
    else:
        out["ind_prime_week"] = 0.0

    return out.sort_values(["asin", "order_week"]).reset_index(drop=True)


# -----------------------------
# Dynamic profile builder
# -----------------------------

SIGNAL_COLS = ["total_dph", "buy_box_dph", "in_stock_dph", "fbi_demand"]
SIGNAL_PREFIX = {
    "total_dph": "total",
    "buy_box_dph": "buybox",
    "in_stock_dph": "instock",
    "fbi_demand": "demand",
}


def _series_summary_features(x, prefix):
    x = np.asarray(x, dtype=float)
    x = np.clip(x[np.isfinite(x)], 0, None)
    if len(x) == 0:
        x = np.array([0.0])
    active = (x > 0).astype(float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    q75 = float(np.quantile(x, 0.75))
    q90 = float(np.quantile(x, 0.90))
    q95 = float(np.quantile(x, 0.95))
    mx = float(np.max(x))
    s = float(np.sum(x))
    active_vals = x[x > 0]
    active_mean = float(np.mean(active_vals)) if len(active_vals) else 0.0
    active_q90 = float(np.quantile(active_vals, 0.90)) if len(active_vals) else 0.0
    recent = x[-13:] if len(x) >= 13 else x
    recent_mean = float(np.mean(recent)) if len(recent) else 0.0
    return {
        f"{prefix}_log_sum": np.log1p(s),
        f"{prefix}_log_mean": np.log1p(mean),
        f"{prefix}_log_q75": np.log1p(q75),
        f"{prefix}_log_q90": np.log1p(q90),
        f"{prefix}_log_q95": np.log1p(q95),
        f"{prefix}_log_max": np.log1p(mx),
        f"{prefix}_log_recent13_mean": np.log1p(recent_mean),
        f"{prefix}_active_rate": float(np.mean(active)),
        f"{prefix}_zero_rate": float(1.0 - np.mean(active)),
        f"{prefix}_cv": float(std / (mean + 1e-6)),
        f"{prefix}_gini": _gini(x),
        f"{prefix}_active_log_mean": np.log1p(active_mean),
        f"{prefix}_active_log_q90": np.log1p(active_q90),
    }


def build_dynamic_node_profile(df, origin_week, min_hist_weeks=13):
    """Build ASIN node features using only weeks < origin_week."""
    origin_week = pd.to_datetime(origin_week)
    hist = df[df["order_week"] < origin_week].copy()
    if hist.empty:
        return pd.DataFrame()

    rows = []
    for asin, g in hist.groupby("asin"):
        g = g.sort_values("order_week")
        if len(g) < min_hist_weeks:
            continue
        row = {"asin": asin, "origin_week": origin_week}
        row["category_code"] = _mode_str(g["category_code"], "UNKNOWN")
        row["gl_product_group"] = _mode_str(g["gl_product_group"], "UNKNOWN")
        row["hbt"] = _mode_str(g["hbt"], "MISSING")
        hbt = row["hbt"].lower()
        row["hbt_is_head"] = float(hbt == "head")
        row["hbt_is_body"] = float(hbt == "body")
        row["hbt_is_tail"] = float(hbt == "tail")
        row["hbt_is_unknown"] = float(hbt not in ["head", "body", "tail"])
        row["ind_top10_brand"] = float(np.nanmax(_safe_numeric(g["ind_top10_brand"]).values))
        row["log_review_last"] = float(g["log_review_count"].iloc[-1])
        row["log_review_mean"] = float(g["log_review_count"].mean())
        row["log_price_mean"] = float(g["log_price"].mean())
        row["promo_rate"] = float(g["ind_promotion"].mean())
        row["prime_rate"] = float(g["ind_prime_week"].mean())
        row["hist_len"] = float(len(g))

        for c in SIGNAL_COLS:
            row.update(_series_summary_features(g[c].values, SIGNAL_PREFIX[c]))

        # compound strength used for edge construction
        row["funnel_strength"] = (
            row["total_log_q90"] + row["buybox_log_q90"] + row["instock_log_q90"] + row["demand_log_q90"]
        ) / 4.0
        row["active_strength"] = (
            row["total_active_rate"] + row["buybox_active_rate"] + row["instock_active_rate"] + row["demand_active_rate"]
        ) / 4.0
        rows.append(row)

    prof = pd.DataFrame(rows)
    return prof.reset_index(drop=True)


def _get_recent_sequences(df, origin_week, asins, lookback_weeks=52):
    """Return dict asin -> recent aligned sequences by signal, weeks < origin."""
    origin_week = pd.to_datetime(origin_week)
    hist = df[(df["order_week"] < origin_week) & (df["asin"].isin(asins))].copy()
    if hist.empty:
        return {}
    # choose last lookback calendar weeks globally before origin
    weeks = sorted(hist["order_week"].dropna().unique())[-lookback_weeks:]
    hist = hist[hist["order_week"].isin(weeks)]
    out = {}
    for asin, g in hist.groupby("asin"):
        g = g.set_index("order_week").sort_index()
        item = {}
        for c in SIGNAL_COLS:
            item[c] = g[c].reindex(weeks).fillna(0.0).values.astype(float)
        out[asin] = item
    return out


# -----------------------------
# Pair construction and dynamic labels
# -----------------------------

EDGE_BASE_COLS = []


def _edge_features(row_i, row_j, seq_i=None, seq_j=None):
    feat = {}

    # hbt/top-brand relation
    feat["same_hbt"] = float(str(row_i["hbt"]) == str(row_j["hbt"]))
    feat["hbt_diff"] = 1.0 - feat["same_hbt"]
    feat["same_top10_brand"] = float(row_i["ind_top10_brand"] == row_j["ind_top10_brand"])
    feat["top10_brand_diff"] = 1.0 - feat["same_top10_brand"]
    feat["j_is_top10_brand"] = float(row_j["ind_top10_brand"] > 0.5)
    feat["i_is_top10_brand"] = float(row_i["ind_top10_brand"] > 0.5)
    feat["j_top10_i_not"] = float((row_j["ind_top10_brand"] > 0.5) and (row_i["ind_top10_brand"] <= 0.5))

    # direct DPH/demand gap features
    gap_cols = []
    for prefix in ["total", "buybox", "instock", "demand"]:
        for stat in ["log_mean", "log_q75", "log_q90", "log_q95", "log_max", "log_recent13_mean", "active_rate", "zero_rate", "active_log_mean", "active_log_q90"]:
            c = f"{prefix}_{stat}"
            if c in row_i.index and c in row_j.index:
                diff = float(row_j[c] - row_i[c])
                feat[f"{c}_abs_gap"] = abs(diff)
                feat[f"{c}_signed_gap_j_minus_i"] = diff
                gap_cols.append(abs(diff))

    # stronger indicators: neighbor j stronger than current i
    for prefix in ["total", "buybox", "instock", "demand"]:
        c = f"{prefix}_log_q90"
        if c in row_i.index and c in row_j.index:
            feat[f"j_stronger_{prefix}_q90"] = float(row_j[c] > row_i[c] + 0.25)
    feat["j_stronger_funnel"] = float(row_j["funnel_strength"] > row_i["funnel_strength"] + 0.25)
    feat["funnel_strength_signed_gap"] = float(row_j["funnel_strength"] - row_i["funnel_strength"])
    feat["funnel_strength_abs_gap"] = abs(feat["funnel_strength_signed_gap"])
    feat["active_strength_signed_gap"] = float(row_j["active_strength"] - row_i["active_strength"])
    feat["active_strength_abs_gap"] = abs(feat["active_strength_signed_gap"])

    # interactions important for competitive pressure
    feat["stronger_top10_competitor"] = float(feat["j_top10_i_not"] * feat["j_stronger_funnel"])
    feat["hbt_diff_and_j_stronger"] = float(feat["hbt_diff"] * feat["j_stronger_funnel"])
    feat["brand_diff_and_j_stronger"] = float(feat["top10_brand_diff"] * feat["j_stronger_funnel"])

    # historical aligned correlations, if sequences available
    if seq_i is not None and seq_j is not None:
        for c in SIGNAL_COLS:
            p = SIGNAL_PREFIX[c]
            corr = _corr_aligned(seq_i.get(c, []), seq_j.get(c, []))
            feat[f"hist_corr_{p}"] = corr
            feat[f"hist_neg_corr_{p}"] = max(0.0, -corr)
            feat[f"hist_pos_corr_{p}"] = max(0.0, corr)
    else:
        for p in ["total", "buybox", "instock", "demand"]:
            feat[f"hist_corr_{p}"] = 0.0
            feat[f"hist_neg_corr_{p}"] = 0.0
            feat[f"hist_pos_corr_{p}"] = 0.0

    # compact totals
    feat["mean_abs_gap_all"] = float(np.mean(gap_cols)) if gap_cols else 0.0
    return feat


def _future_profile(df, origin_week, asins, label_horizon=4):
    origin_week = pd.to_datetime(origin_week)
    weeks = sorted(df[df["order_week"] >= origin_week]["order_week"].dropna().unique())[:label_horizon]
    fut = df[(df["order_week"].isin(weeks)) & (df["asin"].isin(asins))].copy()
    out = {}
    for asin, g in fut.groupby("asin"):
        g = g.set_index("order_week").sort_index()
        item = {}
        for c in SIGNAL_COLS:
            seq = g[c].reindex(weeks).fillna(0.0).values.astype(float)
            item[c] = seq
            p = SIGNAL_PREFIX[c]
            item[f"{p}_sum"] = float(np.sum(seq))
            item[f"{p}_mean"] = float(np.mean(seq)) if len(seq) else 0.0
            item[f"{p}_active_rate"] = float(np.mean(seq > 0)) if len(seq) else 0.0
        out[asin] = item
    return out


def _weak_relation_label(edge_feat, fut_i=None, fut_j=None):
    """
    Return 2=positive, 1=competitive, 0=neutral.
    Uses future label window only as training label. This is not used at forecast time.
    """
    if fut_i is None or fut_j is None:
        # fallback based on historical edge features only
        pos_like = (
            edge_feat.get("mean_abs_gap_all", 9) < 0.45 and
            edge_feat.get("same_hbt", 0) > 0.5 and
            edge_feat.get("same_top10_brand", 0) > 0.5
        )
        comp_like = (
            edge_feat.get("j_stronger_funnel", 0) > 0.5 and
            (edge_feat.get("hbt_diff", 0) > 0.5 or edge_feat.get("top10_brand_diff", 0) > 0.5) and
            edge_feat.get("funnel_strength_abs_gap", 0) > 0.6
        )
        return 2 if pos_like else (1 if comp_like else 0)

    # future correlations and strength gaps
    fut_corrs = []
    for c in SIGNAL_COLS:
        fut_corrs.append(_corr_aligned(fut_i[c], fut_j[c]))
    mean_pos_corr = float(np.mean([max(0.0, c) for c in fut_corrs]))
    mean_neg_corr = float(np.mean([max(0.0, -c) for c in fut_corrs]))

    # future strength: use DPH funnel + demand
    strength_i = np.log1p(fut_i["total_sum"] + fut_i["buybox_sum"] + fut_i["instock_sum"] + fut_i["demand_sum"])
    strength_j = np.log1p(fut_j["total_sum"] + fut_j["buybox_sum"] + fut_j["instock_sum"] + fut_j["demand_sum"])
    strength_gap = float(strength_j - strength_i)
    abs_strength_gap = abs(strength_gap)
    both_active = float((fut_i["instock_active_rate"] > 0) and (fut_j["instock_active_rate"] > 0))

    # positive: similar future movement + similar scale + same hbt/top10 usually
    positive = (
        mean_pos_corr >= 0.35 and
        abs_strength_gap <= 0.75 and
        both_active > 0 and
        edge_feat.get("mean_abs_gap_all", 9) <= 0.75
    ) or (
        edge_feat.get("mean_abs_gap_all", 9) <= 0.35 and
        edge_feat.get("same_hbt", 0) > 0.5 and
        edge_feat.get("same_top10_brand", 0) > 0.5
    )

    # competitive: same category pair with clear strong/weak contrast, often hbt/top10 diff
    competitive = (
        abs_strength_gap >= 1.00 and
        (edge_feat.get("hbt_diff", 0) > 0.5 or edge_feat.get("top10_brand_diff", 0) > 0.5 or edge_feat.get("stronger_top10_competitor", 0) > 0.5) and
        (mean_neg_corr >= 0.10 or edge_feat.get("funnel_strength_abs_gap", 0) >= 0.75)
    )

    if competitive and not positive:
        return 1
    if positive and not competitive:
        return 2
    if competitive and positive:
        # if relation is ambiguous, use strength gap to separate
        return 1 if abs_strength_gap > 1.25 else 2
    return 0


def build_dynamic_pair_dataset(
    df,
    origin_weeks=None,
    history_min_weeks=13,
    label_horizon=4,
    lookback_weeks=52,
    max_pairs_per_category=1500,
    max_origins=12,
    seed=42,
):
    rng = np.random.default_rng(seed)
    all_weeks = sorted(pd.to_datetime(df["order_week"].dropna().unique()))
    if origin_weeks is None:
        # choose rolling origins with enough history and enough future labels
        valid = []
        for w in all_weeks:
            n_hist = (df["order_week"] < w).sum()
            future_weeks = [x for x in all_weeks if x >= w]
            if len(future_weeks) >= label_horizon and n_hist > 0:
                valid.append(w)
        # use recent but multiple origins
        if len(valid) > max_origins:
            idx = np.linspace(0, len(valid) - 1, max_origins).round().astype(int)
            origin_weeks = [valid[i] for i in idx]
        else:
            origin_weeks = valid
    else:
        origin_weeks = [pd.to_datetime(w) for w in origin_weeks]

    pair_rows = []
    profile_rows = []
    print(f"Dynamic origins: {len(origin_weeks)}")

    for oi, origin in enumerate(origin_weeks, 1):
        prof = build_dynamic_node_profile(df, origin, min_hist_weeks=history_min_weeks)
        if prof.empty:
            continue
        prof["origin_week"] = origin
        profile_rows.append(prof)
        asin_set = set(prof["asin"].astype(str))
        seqs = _get_recent_sequences(df, origin, asin_set, lookback_weeks=lookback_weeks)
        fut = _future_profile(df, origin, asin_set, label_horizon=label_horizon)

        prof_idx = prof.set_index("asin")
        cat_counts = prof["category_code"].value_counts()
        usable_cats = cat_counts[cat_counts >= 2].index.tolist()
        origin_pairs = 0

        for cat in usable_cats:
            asins = prof.loc[prof["category_code"] == cat, "asin"].astype(str).values
            n = len(asins)
            # all pairs if small, random sample if large
            possible = n * (n - 1)
            cap = min(max_pairs_per_category, possible)
            if possible <= max_pairs_per_category:
                pairs = [(a, b) for a in asins for b in asins if a != b]
            else:
                pairs = []
                seen = set()
                attempts = 0
                while len(pairs) < cap and attempts < cap * 10:
                    a, b = rng.choice(asins, size=2, replace=False)
                    key = (a, b)
                    if key not in seen:
                        pairs.append(key)
                        seen.add(key)
                    attempts += 1

            for a, b in pairs:
                ri = prof_idx.loc[a]
                rj = prof_idx.loc[b]
                ef = _edge_features(ri, rj, seqs.get(a), seqs.get(b))
                y = _weak_relation_label(ef, fut.get(a), fut.get(b))
                row = {
                    "origin_week": origin,
                    "asin_i": a,
                    "asin_j": b,
                    "category_code": cat,
                    "label": int(y),
                    "label_name": {0: "neutral", 1: "competitive", 2: "positive"}[int(y)],
                    "hbt_i": ri["hbt"],
                    "hbt_j": rj["hbt"],
                    "top10_i": ri["ind_top10_brand"],
                    "top10_j": rj["ind_top10_brand"],
                    "funnel_strength_i": ri["funnel_strength"],
                    "funnel_strength_j": rj["funnel_strength"],
                }
                row.update(ef)
                pair_rows.append(row)
                origin_pairs += 1

        print(f"[{oi}/{len(origin_weeks)}] origin={str(origin)[:10]} | ASINs={len(prof)} | categories={len(usable_cats)} | pairs={origin_pairs}")

    pair_df = pd.DataFrame(pair_rows)
    profile_df = pd.concat(profile_rows, ignore_index=True) if profile_rows else pd.DataFrame()
    return pair_df, profile_df


# -----------------------------
# Relation classifier
# -----------------------------

class PairRelationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EdgeAwareRelationClassifier(nn.Module):
    """
    Lightweight edge-aware relation classifier.
    This is intentionally separated from the exposure model.
    It tests whether dynamic edge features can distinguish:
      neutral / competitive / positive.
    """
    def __init__(self, input_dim, hidden=128, dropout=0.20, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def _balanced_sample(pair_df, max_per_class=60000, seed=42):
    rng = np.random.default_rng(seed)
    parts = []
    for lab, g in pair_df.groupby("label"):
        if len(g) > max_per_class:
            parts.append(g.sample(n=max_per_class, random_state=seed + int(lab)))
        else:
            parts.append(g)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def train_relation_classifier(pair_df, epochs=30, batch_size=512, lr=1e-3, seed=42, verbose=True):
    if pair_df.empty:
        raise ValueError("pair_df is empty")

    # feature columns: numeric only, excluding IDs/labels
    exclude = {"origin_week", "asin_i", "asin_j", "category_code", "label", "label_name", "hbt_i", "hbt_j"}
    feat_cols = [c for c in pair_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(pair_df[c])]
    df = pair_df.copy()
    for c in feat_cols:
        df[c] = _safe_numeric(df[c]).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # temporal split by origin weeks
    origins = sorted(pd.to_datetime(df["origin_week"].unique()))
    if len(origins) >= 3:
        split = int(len(origins) * 0.75)
        train_origins = set(origins[:split])
        test_origins = set(origins[split:])
        tr = df[pd.to_datetime(df["origin_week"]).isin(train_origins)].copy()
        te = df[pd.to_datetime(df["origin_week"]).isin(test_origins)].copy()
    else:
        # fallback random split
        tr = df.sample(frac=0.75, random_state=seed)
        te = df.drop(tr.index)

    tr = _balanced_sample(tr, max_per_class=60000, seed=seed)
    te = _balanced_sample(te, max_per_class=60000, seed=seed + 1)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[feat_cols].values.astype(float))
    Xte = scaler.transform(te[feat_cols].values.astype(float))
    ytr = tr["label"].values.astype(int)
    yte = te["label"].values.astype(int)

    tr_ds = PairRelationDataset(Xtr, ytr)
    te_ds = PairRelationDataset(Xte, yte)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model = EdgeAwareRelationClassifier(input_dim=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # class weights
    counts = np.bincount(ytr, minlength=3).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weight = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    best_sd, best_f1 = None, -1
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in tr_ld:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        preds, probs, ys = [], [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(DEVICE)
                logits = model(xb)
                pr = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(pr)
                preds.append(np.argmax(pr, axis=1))
                ys.append(yb.numpy())
        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        f1 = f1_score(ys, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            acc = accuracy_score(ys, preds)
            print(f"Epoch {ep+1:03d} | train_loss={np.mean(losses):.4f} | test_acc={acc:.4f} | macro_f1={f1:.4f}")

    if best_sd is not None:
        model.load_state_dict(best_sd)

    # final prediction on full pair_df for diagnostics
    Xall = scaler.transform(df[feat_cols].values.astype(float))
    all_ld = DataLoader(PairRelationDataset(Xall, df["label"].values.astype(int)), batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in all_ld:
            all_probs.append(torch.softmax(model(xb.to(DEVICE)), dim=-1).cpu().numpy())
    all_probs = np.concatenate(all_probs)

    scored = df.copy()
    scored["score_neutral"] = all_probs[:, 0]
    scored["score_competitive"] = all_probs[:, 1]
    scored["score_positive"] = all_probs[:, 2]
    scored["pred_label"] = np.argmax(all_probs, axis=1)
    scored["pred_label_name"] = scored["pred_label"].map({0: "neutral", 1: "competitive", 2: "positive"})

    # heldout report
    Xte2 = scaler.transform(te[feat_cols].values.astype(float))
    with torch.no_grad():
        pte = torch.softmax(model(torch.tensor(Xte2, dtype=torch.float32, device=DEVICE)), dim=-1).cpu().numpy()
    pred_te = np.argmax(pte, axis=1)
    report = classification_report(yte, pred_te, target_names=["neutral", "competitive", "positive"], output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred_te, labels=[0,1,2])

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feat_cols,
        "scored_pair_df": scored,
        "heldout_report": report,
        "confusion_matrix": cm,
        "best_macro_f1": best_f1,
        "train_df": tr,
        "test_df": te,
    }


# -----------------------------
# Diagnostics
# -----------------------------

def diagnose_relation_scores(scored_pair_df, top_n=30, verbose=True):
    df = scored_pair_df.copy()
    top_pos = df.sort_values("score_positive", ascending=False).head(top_n)
    top_comp = df.sort_values("score_competitive", ascending=False).head(top_n)

    summary = (
        df.groupby("pred_label_name")
        .agg(
            n_pairs=("asin_i", "size"),
            mean_score_pos=("score_positive", "mean"),
            mean_score_comp=("score_competitive", "mean"),
            mean_total_gap=("total_log_q90_abs_gap", "mean") if "total_log_q90_abs_gap" in df.columns else ("score_positive", "mean"),
            mean_buybox_gap=("buybox_log_q90_abs_gap", "mean") if "buybox_log_q90_abs_gap" in df.columns else ("score_positive", "mean"),
            mean_instock_gap=("instock_log_q90_abs_gap", "mean") if "instock_log_q90_abs_gap" in df.columns else ("score_positive", "mean"),
            mean_demand_gap=("demand_log_q90_abs_gap", "mean") if "demand_log_q90_abs_gap" in df.columns else ("score_positive", "mean"),
            same_hbt_rate=("same_hbt", "mean"),
            same_top10_rate=("same_top10_brand", "mean"),
            stronger_top10_rate=("stronger_top10_competitor", "mean"),
        )
        .reset_index()
        .sort_values("n_pairs", ascending=False)
    )

    by_origin = (
        df.groupby("origin_week")
        .agg(
            n_pairs=("asin_i", "size"),
            avg_pos_score=("score_positive", "mean"),
            avg_comp_score=("score_competitive", "mean"),
            pred_positive_rate=("pred_label", lambda x: float(np.mean(np.asarray(x)==2))),
            pred_competitive_rate=("pred_label", lambda x: float(np.mean(np.asarray(x)==1))),
        )
        .reset_index()
    )

    by_category = (
        df.groupby("category_code")
        .agg(
            n_pairs=("asin_i", "size"),
            avg_pos_score=("score_positive", "mean"),
            avg_comp_score=("score_competitive", "mean"),
            pred_positive_rate=("pred_label", lambda x: float(np.mean(np.asarray(x)==2))),
            pred_competitive_rate=("pred_label", lambda x: float(np.mean(np.asarray(x)==1))),
        )
        .reset_index()
        .sort_values("n_pairs", ascending=False)
    )

    corr_cols = [c for c in [
        "total_log_q90_abs_gap", "buybox_log_q90_abs_gap", "instock_log_q90_abs_gap", "demand_log_q90_abs_gap",
        "same_hbt", "same_top10_brand", "top10_brand_diff", "stronger_top10_competitor",
        "hist_corr_total", "hist_corr_buybox", "hist_corr_instock", "hist_corr_demand",
        "funnel_strength_abs_gap", "active_strength_abs_gap",
    ] if c in df.columns]
    corr_rows = []
    for c in corr_cols:
        for s in ["score_positive", "score_competitive"]:
            val = _corr_aligned(df[c].values, df[s].values)
            corr_rows.append({"feature": c, "score": s, "corr": val})
    score_feature_corr = pd.DataFrame(corr_rows).sort_values(["score", "corr"], ascending=[True, False])

    if verbose:
        print("\n=== Relation prediction summary ===")
        print(summary.to_string(index=False))
        print("\n=== By origin week ===")
        print(by_origin.tail(10).to_string(index=False))
        print("\n=== Top feature correlations with scores ===")
        print(score_feature_corr.head(30).to_string(index=False))
        print("\n=== Top positive pairs preview ===")
        cols = [c for c in ["origin_week", "asin_i", "asin_j", "category_code", "score_positive", "score_competitive", "hbt_i", "hbt_j", "top10_i", "top10_j", "same_hbt", "same_top10_brand", "total_log_q90_abs_gap", "buybox_log_q90_abs_gap", "instock_log_q90_abs_gap", "demand_log_q90_abs_gap"] if c in top_pos.columns]
        print(top_pos[cols].head(10).to_string(index=False))
        print("\n=== Top competitive pairs preview ===")
        print(top_comp[cols].head(10).to_string(index=False))

    return {
        "summary": summary,
        "by_origin": by_origin,
        "by_category": by_category,
        "top_positive_pairs": top_pos,
        "top_competitive_pairs": top_comp,
        "score_feature_corr": score_feature_corr,
    }


# -----------------------------
# Main run function
# -----------------------------

def run_dynamic_gat_relation_test_scot5000(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
    history_min_weeks=13,
    label_horizon=4,
    lookback_weeks=52,
    max_origins=12,
    max_pairs_per_category=1500,
    epochs=30,
    batch_size=512,
):
    df = _prepare_joint_data(data_raw1, scot_df=scot_df, n_asins=n_asins, seed=seed)
    pair_df, profile_df = build_dynamic_pair_dataset(
        df,
        history_min_weeks=history_min_weeks,
        label_horizon=label_horizon,
        lookback_weeks=lookback_weeks,
        max_pairs_per_category=max_pairs_per_category,
        max_origins=max_origins,
        seed=seed,
    )
    if pair_df.empty:
        raise ValueError("No pair data constructed. Check category sizes / history length / data columns.")

    print("\nPair label distribution:")
    print(pair_df["label_name"].value_counts().to_string())

    train_out = train_relation_classifier(
        pair_df,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        verbose=True,
    )
    diag = diagnose_relation_scores(train_out["scored_pair_df"], verbose=True)

    print("\nHeldout report:")
    print(pd.DataFrame(train_out["heldout_report"]).T.to_string())
    print("\nConfusion matrix [neutral, competitive, positive]:")
    print(train_out["confusion_matrix"])

    return {
        "joint_df": df,
        "profile_df": profile_df,
        "pair_df": pair_df,
        "model": train_out["model"],
        "scaler": train_out["scaler"],
        "feature_cols": train_out["feature_cols"],
        "scored_pair_df": train_out["scored_pair_df"],
        "heldout_report": train_out["heldout_report"],
        "confusion_matrix": train_out["confusion_matrix"],
        "diagnostics": diag,
        "top_positive_pairs": diag["top_positive_pairs"],
        "top_competitive_pairs": diag["top_competitive_pairs"],
        "category_relation_summary": diag["by_category"],
        "origin_relation_summary": diag["by_origin"],
        "score_feature_corr": diag["score_feature_corr"],
    }


# ============================================================
# FINAL USAGE ONLY
# ============================================================
# In Jupyter, run this file after data_raw1 and scot_df exist:
# %run -i gat_relation_test_scot5000_DYNAMIC_BY_WEEK.py
#
# Then run:
#
# dynamic_gat_relation_result = run_dynamic_gat_relation_test_scot5000(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     history_min_weeks=13,
#     label_horizon=4,
#     lookback_weeks=52,
#     max_origins=12,
#     max_pairs_per_category=1200,
#     epochs=30,
#     batch_size=512,
# )
#
# scored_pair_df = dynamic_gat_relation_result["scored_pair_df"]
# top_positive_pairs = dynamic_gat_relation_result["top_positive_pairs"]
# top_competitive_pairs = dynamic_gat_relation_result["top_competitive_pairs"]
# category_relation_summary = dynamic_gat_relation_result["category_relation_summary"]
# origin_relation_summary = dynamic_gat_relation_result["origin_relation_summary"]
