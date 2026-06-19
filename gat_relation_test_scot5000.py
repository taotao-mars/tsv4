# ============================================================
# Standalone GAT-style ASIN Relation Test for 5000 + SCOT joint ASINs
# Purpose:
#   Test whether same-category ASIN relations can be learned separately
#   from the exposure forecasting model.
#
# What this file does:
#   1) sample 5000 ASINs from data_raw1 and intersect with scot_df ASINs
#   2) build ASIN-level historical profiles using only history before final holdout
#   3) build same-category ASIN-ASIN relation pairs
#      - positive: similar exposure/demand/hbt/top-brand profile
#      - competitive: same category but stronger/contrastive competitor
#      - neutral: same category but weak/no clear relation
#   4) train an edge-aware GAT-style pair relation classifier
#   5) output diagnostics: metrics, top positive/competitive pairs, category summaries
#
# Important:
#   - This is NOT the exposure forecasting model.
#   - It is a standalone graph/relation diagnostic.
#   - Final holdout weeks are NOT used for training labels/features.
#   - Final holdout weeks are used only for optional out-of-sample validation diagnostics.
# ============================================================

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass


def _safe_numeric(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def _first_existing(df: pd.DataFrame, candidates: List[str], default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a = a[-n:]
    b = b[-n:]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cv(x):
    x = np.asarray(x, dtype=float)
    m = np.mean(x) if len(x) else 0.0
    s = np.std(x) if len(x) else 0.0
    return float(s / (m + 1e-6))


def _gini_nonnegative(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0, None)
    if len(x) == 0 or np.sum(x) <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * np.sum(x)) - (n + 1) / n)


def _hbt_to_code(v):
    s = str(v).strip().lower()
    if s == "head":
        return 2.0
    if s == "body":
        return 1.0
    if s == "tail":
        return 0.0
    return -1.0


def _norm_str(v):
    s = str(v).strip()
    if s.lower() in ["nan", "none", "", "missing"]:
        return "MISSING"
    return s


# ============================================================
# 1. Data prep: 5000 + SCOT joint ASINs
# ============================================================

def prepare_scot5000_joint_relation_data(
    data_raw1: pd.DataFrame,
    scot_df: Optional[pd.DataFrame] = None,
    n_asins: int = 5000,
    seed: int = 42,
    asin_col: str = "asin",
    week_col: str = "order_week",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Samples n_asins from data_raw1 and intersects with scot_df ASINs if scot_df is provided.

    Returns
    -------
    joint_df : rows in data_raw1 for sampled ∩ SCOT ASINs
    sample_asin_df : sampled ASIN table
    joint_asin_df : final joint ASIN table
    """
    df = data_raw1.copy()
    df[asin_col] = df[asin_col].astype(str)
    df[week_col] = pd.to_datetime(df[week_col])

    rng = np.random.default_rng(seed)
    unique_asins = df[asin_col].dropna().astype(str).unique()
    sample_asins = rng.choice(unique_asins, size=min(n_asins, len(unique_asins)), replace=False)
    sample_asin_df = pd.DataFrame({asin_col: sample_asins})

    if scot_df is not None and asin_col in scot_df.columns:
        scot_asins = set(scot_df[asin_col].astype(str).dropna().unique())
        joint_asins = sorted(set(sample_asins).intersection(scot_asins))
    else:
        joint_asins = sorted(set(sample_asins))

    joint_asin_df = pd.DataFrame({asin_col: joint_asins})
    joint_df = df[df[asin_col].isin(set(joint_asins))].copy()
    print(f"Sampled ASINs: {len(sample_asins)} | Joint ASINs: {len(joint_asins)} | Rows: {len(joint_df)}")
    return joint_df, sample_asin_df, joint_asin_df


# ============================================================
# 2. ASIN historical profile construction
# ============================================================

@dataclass
class RelationProfileOutput:
    profile_df: pd.DataFrame
    node_feature_cols: List[str]
    hist_series: Dict[str, Dict[str, np.ndarray]]
    future_series: Dict[str, Dict[str, np.ndarray]]
    meta_cols: List[str]


def _series_summary(x, prefix: str):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    if len(x) == 0:
        x = np.array([0.0])
    active = x > 0
    return {
        f"{prefix}_mean_log": float(np.log1p(np.mean(x))),
        f"{prefix}_median_log": float(np.log1p(np.median(x))),
        f"{prefix}_q75_log": float(np.log1p(np.quantile(x, 0.75))),
        f"{prefix}_q90_log": float(np.log1p(np.quantile(x, 0.90))),
        f"{prefix}_q95_log": float(np.log1p(np.quantile(x, 0.95))),
        f"{prefix}_max_log": float(np.log1p(np.max(x))),
        f"{prefix}_sum_log": float(np.log1p(np.sum(x))),
        f"{prefix}_active_rate": float(np.mean(active)),
        f"{prefix}_zero_rate": float(np.mean(~active)),
        f"{prefix}_cv": _cv(x),
        f"{prefix}_gini": _gini_nonnegative(x),
        f"{prefix}_mean13_log": float(np.log1p(np.mean(x[-13:]))) if len(x) else 0.0,
        f"{prefix}_mean4_log": float(np.log1p(np.mean(x[-4:]))) if len(x) else 0.0,
        f"{prefix}_last_log": float(np.log1p(x[-1])) if len(x) else 0.0,
    }


def _transition_features(x, prefix="instock"):
    x = np.asarray(x, dtype=float)
    active = (x > 0).astype(int)
    if len(active) < 2:
        return {
            f"{prefix}_zero_to_active_rate": 0.0,
            f"{prefix}_active_to_zero_rate": 0.0,
            f"{prefix}_last_active_streak_log": 0.0,
            f"{prefix}_last_zero_streak_log": 0.0,
            f"{prefix}_weeks_since_last_positive_log": float(np.log1p(99.0)),
        }
    z2a = np.mean((active[:-1] == 0) & (active[1:] == 1))
    a2z = np.mean((active[:-1] == 1) & (active[1:] == 0))
    # streaks at end
    last_val = active[-1]
    streak = 1
    for v in active[-2::-1]:
        if v == last_val:
            streak += 1
        else:
            break
    last_active_streak = streak if last_val == 1 else 0
    last_zero_streak = streak if last_val == 0 else 0
    pos_idx = np.where(active == 1)[0]
    wslp = len(active) if len(pos_idx) == 0 else len(active) - 1 - pos_idx[-1]
    return {
        f"{prefix}_zero_to_active_rate": float(z2a),
        f"{prefix}_active_to_zero_rate": float(a2z),
        f"{prefix}_last_active_streak_log": float(np.log1p(last_active_streak)),
        f"{prefix}_last_zero_streak_log": float(np.log1p(last_zero_streak)),
        f"{prefix}_weeks_since_last_positive_log": float(np.log1p(wslp)),
    }


def build_asin_relation_profiles(
    joint_df: pd.DataFrame,
    holdout_horizon: int = 20,
    min_hist_weeks: int = 13,
    asin_col: str = "asin",
    week_col: str = "order_week",
    category_col: str = "category_code",
) -> RelationProfileOutput:
    """
    Builds ASIN-level historical profiles for relation learning.
    Uses only data before the final holdout_horizon weeks per ASIN.
    """
    df = joint_df.copy()
    df[asin_col] = df[asin_col].astype(str)
    df[week_col] = pd.to_datetime(df[week_col])
    df = df.sort_values([asin_col, week_col])

    # Required / fallback columns
    required = ["fbi_demand", "total_dph", "buy_box_dph", "in_stock_dph"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = _safe_numeric(df[c]).clip(lower=0.0)

    if category_col not in df.columns:
        print(f"Warning: {category_col} not found. Using gl_product_group if available, else ALL.")
        category_col = "gl_product_group" if "gl_product_group" in df.columns else "__all_category__"
        if category_col == "__all_category__":
            df[category_col] = "ALL"

    rows = []
    hist_series = {}
    future_series = {}

    for asin, g in df.groupby(asin_col):
        g = g.sort_values(week_col).reset_index(drop=True)
        if len(g) < min_hist_weeks + holdout_horizon:
            continue
        hist = g.iloc[:-holdout_horizon].copy()
        fut = g.iloc[-holdout_horizon:].copy()
        if len(hist) < min_hist_weeks:
            continue

        total = hist["total_dph"].values.astype(float)
        buy = hist["buy_box_dph"].values.astype(float)
        instock = hist["in_stock_dph"].values.astype(float)
        demand = hist["fbi_demand"].values.astype(float)

        row = {asin_col: asin}
        row["category_code_raw"] = _norm_str(hist[category_col].iloc[-1])
        row["gl_product_group_raw"] = _norm_str(hist["gl_product_group"].iloc[-1]) if "gl_product_group" in hist.columns else "MISSING"
        row["hbt_raw"] = _norm_str(hist["hbt"].iloc[-1]) if "hbt" in hist.columns else "MISSING"
        row["hbt_code"] = _hbt_to_code(row["hbt_raw"])
        row["hbt_is_head"] = 1.0 if str(row["hbt_raw"]).lower() == "head" else 0.0
        row["hbt_is_body"] = 1.0 if str(row["hbt_raw"]).lower() == "body" else 0.0
        row["hbt_is_tail"] = 1.0 if str(row["hbt_raw"]).lower() == "tail" else 0.0

        if "ind_top10_brand" in hist.columns:
            row["ind_top10_brand"] = float(_safe_numeric(hist["ind_top10_brand"].iloc[-1:]).iloc[0])
        else:
            row["ind_top10_brand"] = 0.0
        row["ind_top10_brand"] = float(np.clip(row["ind_top10_brand"], 0, 1))

        if "customer_active_review_count" in hist.columns:
            row["review_last_log"] = float(np.log1p(_safe_numeric(hist["customer_active_review_count"]).iloc[-1]))
            row["review_mean_log"] = float(np.log1p(_safe_numeric(hist["customer_active_review_count"]).mean()))
        else:
            row["review_last_log"] = 0.0
            row["review_mean_log"] = 0.0

        if "our_price" in hist.columns:
            row["price_mean_log"] = float(np.log1p(_safe_numeric(hist["our_price"]).clip(lower=0).mean()))
        else:
            row["price_mean_log"] = 0.0

        row["promo_rate"] = float(_safe_numeric(hist["ind_promotion"]).clip(0, 1).mean()) if "ind_promotion" in hist.columns else 0.0
        row["prime_rate"] = float(_safe_numeric(hist["ind_prime_week"]).clip(0, 1).mean()) if "ind_prime_week" in hist.columns else 0.0
        row["oos_rate"] = float(_safe_numeric(hist["scot_oos"]).clip(0, 1).mean()) if "scot_oos" in hist.columns else 0.0
        row["oos_rate13"] = float(_safe_numeric(hist["scot_oos"].tail(13)).clip(0, 1).mean()) if "scot_oos" in hist.columns else 0.0

        # Core time-series summaries explicitly include all DPH funnel channels and demand.
        for prefix, arr in [
            ("total", total),
            ("buybox", buy),
            ("instock", instock),
            ("demand", demand),
        ]:
            row.update(_series_summary(arr, prefix))

        row.update(_transition_features(instock, "instock"))
        row.update(_transition_features(buy, "buybox"))

        # Strength scores used for relation labels/diagnostics.
        row["exposure_strength"] = (
            0.25 * row["total_q90_log"]
            + 0.30 * row["buybox_q90_log"]
            + 0.35 * row["instock_q90_log"]
            + 0.10 * row["demand_q90_log"]
        )
        row["funnel_active_strength"] = (
            0.30 * row["total_active_rate"]
            + 0.30 * row["buybox_active_rate"]
            + 0.30 * row["instock_active_rate"]
            + 0.10 * row["demand_active_rate"]
        )
        rows.append(row)

        hist_series[asin] = {
            "total": total,
            "buybox": buy,
            "instock": instock,
            "demand": demand,
        }
        future_series[asin] = {
            "total": fut["total_dph"].values.astype(float),
            "buybox": fut["buy_box_dph"].values.astype(float),
            "instock": fut["in_stock_dph"].values.astype(float),
            "demand": fut["fbi_demand"].values.astype(float),
        }

    profile_df = pd.DataFrame(rows)
    if profile_df.empty:
        raise ValueError("No ASIN profiles built. Check history length and holdout_horizon.")

    # Frequency encodings for category / GL / HBT.
    for raw_col in ["category_code_raw", "gl_product_group_raw", "hbt_raw"]:
        freq = profile_df[raw_col].value_counts(normalize=True)
        profile_df[raw_col.replace("_raw", "_freq")] = profile_df[raw_col].map(freq).fillna(0.0).astype(float)
        codes, uniques = pd.factorize(profile_df[raw_col].astype(str))
        denom = max(len(uniques) - 1, 1)
        profile_df[raw_col.replace("_raw", "_code_norm")] = codes.astype(float) / denom

    meta_cols = [asin_col, "category_code_raw", "gl_product_group_raw", "hbt_raw"]
    non_feature = set(meta_cols)
    node_feature_cols = [
        c for c in profile_df.columns
        if c not in non_feature and pd.api.types.is_numeric_dtype(profile_df[c])
    ]
    profile_df[node_feature_cols] = profile_df[node_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"Built profiles: ASINs={len(profile_df)} | node_feature_dim={len(node_feature_cols)}")
    print(f"Categories: {profile_df['category_code_raw'].nunique()} | GLs: {profile_df['gl_product_group_raw'].nunique()}")

    return RelationProfileOutput(profile_df, node_feature_cols, hist_series, future_series, meta_cols)


# ============================================================
# 3. Edge features and weak labels
# ============================================================

CORE_SIM_COLS = [
    "total_mean_log", "total_q75_log", "total_q90_log", "total_q95_log", "total_max_log", "total_active_rate", "total_zero_rate",
    "buybox_mean_log", "buybox_q75_log", "buybox_q90_log", "buybox_q95_log", "buybox_max_log", "buybox_active_rate", "buybox_zero_rate",
    "instock_mean_log", "instock_q75_log", "instock_q90_log", "instock_q95_log", "instock_max_log", "instock_active_rate", "instock_zero_rate",
    "demand_mean_log", "demand_q75_log", "demand_q90_log", "demand_q95_log", "demand_max_log", "demand_active_rate", "demand_zero_rate",
    "hbt_code", "ind_top10_brand", "review_last_log", "price_mean_log",
]


def _edge_features(row_i: pd.Series, row_j: pd.Series, hist_series_i: Dict[str, np.ndarray], hist_series_j: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {}

    # Explicit DPH/demand gaps: these are the core features the user identified.
    for base in ["total", "buybox", "instock", "demand"]:
        for stat in ["mean_log", "q75_log", "q90_log", "q95_log", "max_log", "active_rate", "zero_rate", "mean13_log"]:
            ci = f"{base}_{stat}"
            out[f"gap_{ci}"] = abs(float(row_i.get(ci, 0.0)) - float(row_j.get(ci, 0.0)))
        out[f"j_minus_i_{base}_q90_log"] = float(row_j.get(f"{base}_q90_log", 0.0)) - float(row_i.get(f"{base}_q90_log", 0.0))
        out[f"j_stronger_{base}"] = 1.0 if out[f"j_minus_i_{base}_q90_log"] > 0.25 else 0.0
        out[f"corr_{base}"] = _safe_corr(hist_series_i[base], hist_series_j[base])

    # Funnel relation features.
    out["same_hbt"] = 1.0 if str(row_i.get("hbt_raw")) == str(row_j.get("hbt_raw")) else 0.0
    out["hbt_diff"] = 1.0 - out["same_hbt"]
    out["hbt_code_gap"] = abs(float(row_i.get("hbt_code", -1.0)) - float(row_j.get("hbt_code", -1.0)))

    top_i = float(row_i.get("ind_top10_brand", 0.0))
    top_j = float(row_j.get("ind_top10_brand", 0.0))
    out["same_top10_brand"] = 1.0 if int(round(top_i)) == int(round(top_j)) else 0.0
    out["brand_diff"] = 1.0 - out["same_top10_brand"]
    out["j_is_top10_brand"] = top_j
    out["i_is_top10_brand"] = top_i
    out["j_top10_i_not"] = 1.0 if (top_j >= 0.5 and top_i < 0.5) else 0.0

    out["same_gl"] = 1.0 if str(row_i.get("gl_product_group_raw")) == str(row_j.get("gl_product_group_raw")) else 0.0
    out["same_category"] = 1.0 if str(row_i.get("category_code_raw")) == str(row_j.get("category_code_raw")) else 0.0
    out["review_gap"] = abs(float(row_i.get("review_last_log", 0.0)) - float(row_j.get("review_last_log", 0.0)))
    out["price_gap"] = abs(float(row_i.get("price_mean_log", 0.0)) - float(row_j.get("price_mean_log", 0.0)))

    # Aggregate similarity / competition scores for diagnostics and labels.
    dph_gap = np.mean([
        out["gap_total_q90_log"], out["gap_buybox_q90_log"], out["gap_instock_q90_log"], out["gap_demand_q90_log"],
        out["gap_total_active_rate"], out["gap_buybox_active_rate"], out["gap_instock_active_rate"], out["gap_demand_active_rate"],
    ])
    corr_mean = np.mean([out["corr_total"], out["corr_buybox"], out["corr_instock"], out["corr_demand"]])
    stronger_count = out["j_stronger_total"] + out["j_stronger_buybox"] + out["j_stronger_instock"] + out["j_stronger_demand"]
    strength_gap = abs(float(row_j.get("exposure_strength", 0.0)) - float(row_i.get("exposure_strength", 0.0)))

    out["dph_gap_mean"] = float(dph_gap)
    out["corr_mean"] = float(corr_mean)
    out["j_stronger_count"] = float(stronger_count)
    out["strength_gap"] = float(strength_gap)
    out["positive_proxy_score"] = float(corr_mean - 0.35 * dph_gap + 0.10 * out["same_hbt"] + 0.08 * out["same_top10_brand"])
    out["competitive_proxy_score"] = float(
        0.45 * strength_gap
        + 0.20 * out["hbt_diff"]
        + 0.18 * out["brand_diff"]
        + 0.20 * out["j_top10_i_not"]
        + 0.12 * stronger_count
        - 0.10 * max(corr_mean, 0.0)
    )
    return out


def build_relation_pair_dataset(
    profiles: RelationProfileOutput,
    max_pairs_per_category: int = 3000,
    pos_k: int = 8,
    comp_k: int = 8,
    neutral_per_category: int = 800,
    seed: int = 42,
    min_category_size: int = 8,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds same-category ASIN-ASIN pair dataset with weak labels:
      label 0 = neutral
      label 1 = positive / attraction / similar co-movement
      label 2 = competitive / contrast / possible repulsion
    """
    rng = np.random.default_rng(seed)
    p = profiles.profile_df.reset_index(drop=True).copy()
    asin_col = "asin"
    p["node_idx"] = np.arange(len(p))

    # Feature space for nearest positive neighbors.
    core_cols = [c for c in CORE_SIM_COLS if c in p.columns]
    X_core = p[core_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
    X_core = StandardScaler().fit_transform(X_core)
    for k, col in enumerate(core_cols):
        p[f"__core_scaled_{k}"] = X_core[:, k]
    scaled_cols = [f"__core_scaled_{k}" for k in range(len(core_cols))]

    pair_rows = []

    for cat, idxs in p.groupby("category_code_raw").groups.items():
        idxs = list(idxs)
        if len(idxs) < min_category_size:
            continue
        cat_df = p.loc[idxs].copy().reset_index(drop=True)
        cat_indices = cat_df["node_idx"].values.astype(int)
        cat_X = cat_df[scaled_cols].values.astype(float)
        n = len(cat_df)

        # Positive candidates via KNN in explicit DPH/demand/hbt/top-brand feature space.
        k_nn = min(pos_k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_nn, metric="euclidean")
        nbrs.fit(cat_X)
        distances, neighbors = nbrs.kneighbors(cat_X)

        used_pairs = set()

        def _add_pair(i_global, j_global, label, rel_source):
            if i_global == j_global:
                return
            key = (int(i_global), int(j_global), int(label))
            if key in used_pairs:
                return
            used_pairs.add(key)
            ri = p.loc[int(i_global)]
            rj = p.loc[int(j_global)]
            asin_i = str(ri[asin_col])
            asin_j = str(rj[asin_col])
            efeat = _edge_features(ri, rj, profiles.hist_series[asin_i], profiles.hist_series[asin_j])
            pair_rows.append({
                "asin_i": asin_i,
                "asin_j": asin_j,
                "i_idx": int(i_global),
                "j_idx": int(j_global),
                "category_code": cat,
                "gl_i": ri.get("gl_product_group_raw", "MISSING"),
                "gl_j": rj.get("gl_product_group_raw", "MISSING"),
                "hbt_i": ri.get("hbt_raw", "MISSING"),
                "hbt_j": rj.get("hbt_raw", "MISSING"),
                "top10_i": float(ri.get("ind_top10_brand", 0.0)),
                "top10_j": float(rj.get("ind_top10_brand", 0.0)),
                "label": int(label),
                "relation": {0: "neutral", 1: "positive", 2: "competitive"}[int(label)],
                "rel_source": rel_source,
                **efeat,
            })

        # Add positive pairs: nearest neighbors, filtered by positive proxy reasonably.
        for local_i in range(n):
            i_global = cat_indices[local_i]
            count = 0
            for local_j in neighbors[local_i][1:]:
                j_global = cat_indices[int(local_j)]
                # add both directions because influence can be asymmetric later
                _add_pair(i_global, j_global, 1, "knn_positive")
                count += 1
                if count >= pos_k:
                    break

        # Competitive pairs: same category but contrastive / stronger neighbors.
        # For each i, score all j by a simple fast proxy first, then edge_features refines diagnostics.
        strength = cat_df["exposure_strength"].values.astype(float)
        active = cat_df["funnel_active_strength"].values.astype(float)
        hbt = cat_df["hbt_code"].values.astype(float)
        top = cat_df["ind_top10_brand"].values.astype(float)
        demand_s = cat_df["demand_q90_log"].values.astype(float)
        for local_i in range(n):
            si, ai, hi, ti, di = strength[local_i], active[local_i], hbt[local_i], top[local_i], demand_s[local_i]
            score = (
                0.45 * np.abs(strength - si)
                + 0.15 * np.abs(active - ai)
                + 0.18 * (hbt != hi).astype(float)
                + 0.18 * (top != ti).astype(float)
                + 0.15 * np.abs(demand_s - di)
                + 0.25 * ((top >= 0.5) & (ti < 0.5)).astype(float)
                + 0.20 * (strength > si).astype(float)
            )
            score[local_i] = -np.inf
            top_js = np.argsort(-score)[:min(comp_k, n - 1)]
            i_global = cat_indices[local_i]
            for local_j in top_js:
                j_global = cat_indices[int(local_j)]
                _add_pair(i_global, j_global, 2, "contrast_competitive")

        # Neutral pairs: same category random pairs with not-too-extreme proxy scores.
        max_neutral = min(neutral_per_category, max_pairs_per_category)
        tries, added = 0, 0
        while added < max_neutral and tries < max_neutral * 20:
            tries += 1
            a, b = rng.choice(n, size=2, replace=False)
            i_global = cat_indices[int(a)]
            j_global = cat_indices[int(b)]
            # avoid turning nearest/contrast pairs into neutral too often
            if abs(strength[a] - strength[b]) > np.quantile(np.abs(strength[:, None] - strength[None, :]).ravel(), 0.75):
                continue
            _add_pair(i_global, j_global, 0, "random_same_category_neutral")
            added += 1

        # Cap per category if very large.
        if len(pair_rows) > 0 and len(pair_rows) > max_pairs_per_category * max(1, p["category_code_raw"].nunique()):
            pass

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        raise ValueError("No relation pairs built. Check category sizes.")

    edge_feature_cols = [
        c for c in pair_df.columns
        if c not in ["asin_i", "asin_j", "category_code", "gl_i", "gl_j", "hbt_i", "hbt_j", "relation", "rel_source"]
        and c not in ["i_idx", "j_idx", "label"]
        and pd.api.types.is_numeric_dtype(pair_df[c])
    ]
    pair_df[edge_feature_cols] = pair_df[edge_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print("Pair dataset:")
    print(pair_df["relation"].value_counts())
    print(f"Pairs={len(pair_df)} | edge_feature_dim={len(edge_feature_cols)} | categories={pair_df['category_code'].nunique()}")
    return pair_df, edge_feature_cols


# ============================================================
# 4. Dataset and edge-aware GAT-style relation classifier
# ============================================================

class PairRelationDataset(Dataset):
    def __init__(self, pair_df, node_features_scaled, edge_features_scaled, pair_indices):
        self.pair_df = pair_df.iloc[pair_indices].reset_index(drop=True)
        self.node_features = node_features_scaled.astype(np.float32)
        self.edge_features = edge_features_scaled[pair_indices].astype(np.float32)
        self.i_idx = self.pair_df["i_idx"].values.astype(np.int64)
        self.j_idx = self.pair_df["j_idx"].values.astype(np.int64)
        self.y = self.pair_df["label"].values.astype(np.int64)

    def __len__(self):
        return len(self.pair_df)

    def __getitem__(self, idx):
        i = self.i_idx[idx]
        j = self.j_idx[idx]
        return {
            "x_i": torch.tensor(self.node_features[i], dtype=torch.float32),
            "x_j": torch.tensor(self.node_features[j], dtype=torch.float32),
            "edge": torch.tensor(self.edge_features[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
        }


class EdgeAwarePairGAT(nn.Module):
    """
    Standalone edge-aware GAT-style relation learner.

    This is intentionally pair-level, not the full exposure model.
    It learns whether ASIN_i -> ASIN_j is positive, competitive, or neutral using:
      - node profile of i
      - node profile of j
      - explicit edge features: DPH/demand/hbt/top-brand gaps and correlations

    The attention-like part is an edge scorer over [h_i, h_j, edge_ij].
    """
    def __init__(self, node_dim, edge_dim, hidden=128, graph_dim=64, dropout=0.15, n_classes=3):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, graph_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, graph_dim),
            nn.ReLU(),
        )
        # GAT-style relation attention gate: how much edge context should modify pair interaction.
        self.edge_gate = nn.Sequential(
            nn.Linear(graph_dim * 5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, graph_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(graph_dim * 5, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x_i, x_j, edge):
        h_i = self.node_encoder(x_i)
        h_j = self.node_encoder(x_j)
        e = self.edge_encoder(edge)
        diff = torch.abs(h_i - h_j)
        prod = h_i * h_j
        gate_in = torch.cat([h_i, h_j, diff, prod, e], dim=-1)
        edge_gate = self.edge_gate(gate_in)
        e_gated = edge_gate * e
        z = torch.cat([h_i, h_j, diff, prod, e_gated], dim=-1)
        logits = self.classifier(z)
        return logits, {"edge_gate_mean": edge_gate.mean(dim=-1), "edge_gate": edge_gate}


def _batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_pair_relation_model(
    profiles: RelationProfileOutput,
    pair_df: pd.DataFrame,
    edge_feature_cols: List[str],
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 6,
    hidden: int = 128,
    graph_dim: int = 64,
    dropout: float = 0.15,
    seed: int = 42,
    device=None,
):
    device = torch.device(device) if device is not None else DEVICE
    np.random.seed(seed)
    torch.manual_seed(seed)

    profile_df = profiles.profile_df.reset_index(drop=True).copy()
    node_X = profile_df[profiles.node_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
    node_scaler = StandardScaler()
    node_X_scaled = node_scaler.fit_transform(node_X)

    edge_X = pair_df[edge_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
    edge_scaler = StandardScaler()
    edge_X_scaled = edge_scaler.fit_transform(edge_X)

    labels = pair_df["label"].values.astype(int)
    indices = np.arange(len(pair_df))
    tr_idx, te_idx = train_test_split(indices, test_size=0.20, random_state=seed, stratify=labels)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=seed, stratify=labels[tr_idx])

    tr_ds = PairRelationDataset(pair_df, node_X_scaled, edge_X_scaled, tr_idx)
    va_ds = PairRelationDataset(pair_df, node_X_scaled, edge_X_scaled, va_idx)
    te_ds = PairRelationDataset(pair_df, node_X_scaled, edge_X_scaled, te_idx)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = EdgeAwarePairGAT(
        node_dim=node_X_scaled.shape[1],
        edge_dim=edge_X_scaled.shape[1],
        hidden=hidden,
        graph_dim=graph_dim,
        dropout=dropout,
        n_classes=3,
    ).to(device)

    # Class weights to handle imbalance.
    counts = np.bincount(labels, minlength=3).astype(float)
    weights = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_sd, wait = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for b in tr_ld:
            b = _batch_to_device(b, device)
            logits, aux = model(b["x_i"], b["x_j"], b["edge"])
            loss = F.cross_entropy(logits, b["y"], weight=class_weight)
            # Mild gate regularization to avoid all edge gate dimensions opening.
            loss = loss + 0.001 * aux["edge_gate_mean"].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * b["y"].shape[0]
            tr_n += b["y"].shape[0]

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for b in va_ld:
                b = _batch_to_device(b, device)
                logits, aux = model(b["x_i"], b["x_j"], b["edge"])
                loss = F.cross_entropy(logits, b["y"], weight=class_weight)
                loss = loss + 0.001 * aux["edge_gate_mean"].mean()
                va_loss += loss.item() * b["y"].shape[0]
                va_n += b["y"].shape[0]
        tr_loss /= max(tr_n, 1)
        va_loss /= max(va_n, 1)

        if ep == 1 or ep % 5 == 0:
            print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")
        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep} | best_val={best_val:.4f}")
                break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    def _predict(loader):
        ys, probs, gates = [], [], []
        model.eval()
        with torch.no_grad():
            for b in loader:
                b = _batch_to_device(b, device)
                logits, aux = model(b["x_i"], b["x_j"], b["edge"])
                p = torch.softmax(logits, dim=-1)
                ys.append(b["y"].detach().cpu().numpy())
                probs.append(p.detach().cpu().numpy())
                gates.append(aux["edge_gate_mean"].detach().cpu().numpy())
        return np.concatenate(ys), np.concatenate(probs), np.concatenate(gates)

    y_test, p_test, gate_test = _predict(te_ld)
    pred_test = p_test.argmax(axis=1)
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_macro_f1": float(f1_score(y_test, pred_test, average="macro")),
        "test_weighted_f1": float(f1_score(y_test, pred_test, average="weighted")),
        "edge_gate_mean_test": float(np.mean(gate_test)),
    }
    for cls, name in [(0, "neutral"), (1, "positive"), (2, "competitive")]:
        try:
            metrics[f"auc_{name}"] = float(roc_auc_score((y_test == cls).astype(int), p_test[:, cls]))
        except Exception:
            metrics[f"auc_{name}"] = np.nan

    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("\nClassification report:")
    print(classification_report(y_test, pred_test, target_names=["neutral", "positive", "competitive"]))
    print("Confusion matrix [neutral, positive, competitive]:")
    print(confusion_matrix(y_test, pred_test))

    # Score all pairs.
    all_ds = PairRelationDataset(pair_df, node_X_scaled, edge_X_scaled, np.arange(len(pair_df)))
    all_ld = DataLoader(all_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_all, p_all, gate_all = _predict(all_ld)
    scored_pair_df = pair_df.copy().reset_index(drop=True)
    scored_pair_df["p_neutral"] = p_all[:, 0]
    scored_pair_df["p_positive"] = p_all[:, 1]
    scored_pair_df["p_competitive"] = p_all[:, 2]
    scored_pair_df["pred_relation"] = np.array(["neutral", "positive", "competitive"])[p_all.argmax(axis=1)]
    scored_pair_df["edge_gate_mean"] = gate_all

    result = {
        "model": model,
        "metrics": metrics,
        "pair_df": scored_pair_df,
        "profile_df": profile_df,
        "node_feature_cols": profiles.node_feature_cols,
        "edge_feature_cols": edge_feature_cols,
        "node_scaler": node_scaler,
        "edge_scaler": edge_scaler,
        "test_indices": te_idx,
    }
    return result


# ============================================================
# 5. Diagnostics
# ============================================================

def add_future_validation_features(pair_df: pd.DataFrame, future_series: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    out = pair_df.copy()
    vals = []
    for _, r in out.iterrows():
        ai, aj = str(r["asin_i"]), str(r["asin_j"])
        f_i = future_series.get(ai, {})
        f_j = future_series.get(aj, {})
        d = {}
        for base in ["total", "buybox", "instock", "demand"]:
            xi = np.asarray(f_i.get(base, np.zeros(0)), dtype=float)
            xj = np.asarray(f_j.get(base, np.zeros(0)), dtype=float)
            d[f"future_{base}_corr"] = _safe_corr(xi, xj)
            d[f"future_{base}_sum_i_log"] = float(np.log1p(np.sum(np.clip(xi, 0, None)))) if len(xi) else 0.0
            d[f"future_{base}_sum_j_log"] = float(np.log1p(np.sum(np.clip(xj, 0, None)))) if len(xj) else 0.0
            d[f"future_{base}_j_minus_i_sum_log"] = d[f"future_{base}_sum_j_log"] - d[f"future_{base}_sum_i_log"]
        vals.append(d)
    fut_df = pd.DataFrame(vals)
    return pd.concat([out.reset_index(drop=True), fut_df], axis=1)


def diagnose_relation_learning(result: Dict, profiles: RelationProfileOutput, top_n: int = 20, verbose: bool = True):
    pair_df = result["pair_df"].copy()
    pair_df = add_future_validation_features(pair_df, profiles.future_series)

    top_positive = pair_df.sort_values("p_positive", ascending=False).head(top_n).copy()
    top_competitive = pair_df.sort_values("p_competitive", ascending=False).head(top_n).copy()

    # Category summary.
    cat_summary = pair_df.groupby("category_code").agg(
        n_pairs=("asin_i", "count"),
        mean_p_positive=("p_positive", "mean"),
        mean_p_competitive=("p_competitive", "mean"),
        mean_edge_gate=("edge_gate_mean", "mean"),
        mean_corr_instock=("corr_instock", "mean"),
        mean_future_corr_instock=("future_instock_corr", "mean"),
        mean_comp_strength_gap=("strength_gap", "mean"),
    ).reset_index().sort_values("n_pairs", ascending=False)

    # Relation quality summary.
    relation_summary = pair_df.groupby("pred_relation").agg(
        n_pairs=("asin_i", "count"),
        mean_p_positive=("p_positive", "mean"),
        mean_p_competitive=("p_competitive", "mean"),
        mean_corr_instock=("corr_instock", "mean"),
        mean_corr_buybox=("corr_buybox", "mean"),
        mean_corr_demand=("corr_demand", "mean"),
        mean_dph_gap=("dph_gap_mean", "mean"),
        mean_strength_gap=("strength_gap", "mean"),
        mean_same_hbt=("same_hbt", "mean"),
        mean_same_top10=("same_top10_brand", "mean"),
        mean_j_top10_i_not=("j_top10_i_not", "mean"),
        mean_future_instock_corr=("future_instock_corr", "mean"),
        mean_future_instock_j_minus_i=("future_instock_j_minus_i_sum_log", "mean"),
    ).reset_index()

    # Correlations between learned relation probs and interpretable pair features.
    corr_rows = []
    for score in ["p_positive", "p_competitive"]:
        for feat in [
            "corr_total", "corr_buybox", "corr_instock", "corr_demand", "dph_gap_mean",
            "strength_gap", "same_hbt", "same_top10_brand", "j_top10_i_not",
            "future_instock_corr", "future_instock_j_minus_i_sum_log",
        ]:
            if feat in pair_df.columns:
                corr_rows.append({
                    "score": score,
                    "feature": feat,
                    "corr": _safe_corr(pair_df[score].values, pair_df[feat].values),
                })
    score_feature_corr = pd.DataFrame(corr_rows).sort_values(["score", "corr"], ascending=[True, False])

    if verbose:
        print("\n=== Relation summary by predicted relation ===")
        print(relation_summary.round(4).to_string(index=False))
        print("\n=== Top predicted positive pairs ===")
        display_cols = [
            "asin_i", "asin_j", "category_code", "hbt_i", "hbt_j", "top10_i", "top10_j",
            "p_positive", "p_competitive", "corr_instock", "corr_buybox", "corr_demand",
            "dph_gap_mean", "same_hbt", "same_top10_brand", "future_instock_corr",
        ]
        print(top_positive[[c for c in display_cols if c in top_positive.columns]].round(4).to_string(index=False))
        print("\n=== Top predicted competitive pairs ===")
        display_cols2 = [
            "asin_i", "asin_j", "category_code", "hbt_i", "hbt_j", "top10_i", "top10_j",
            "p_positive", "p_competitive", "strength_gap", "j_stronger_count", "j_top10_i_not",
            "hbt_diff", "brand_diff", "future_instock_j_minus_i_sum_log",
        ]
        print(top_competitive[[c for c in display_cols2 if c in top_competitive.columns]].round(4).to_string(index=False))
        print("\n=== Relation score vs feature correlations ===")
        print(score_feature_corr.round(4).to_string(index=False))

    diagnostics = {
        "pair_df_with_future": pair_df,
        "top_positive_pairs": top_positive,
        "top_competitive_pairs": top_competitive,
        "category_relation_summary": cat_summary,
        "relation_summary": relation_summary,
        "score_feature_corr": score_feature_corr,
    }
    return diagnostics


# ============================================================
# 6. Main runner
# ============================================================

def run_gat_relation_test_scot5000(
    data_raw1: pd.DataFrame,
    scot_df: Optional[pd.DataFrame] = None,
    n_asins: int = 5000,
    seed: int = 42,
    holdout_horizon: int = 20,
    min_hist_weeks: int = 13,
    category_col: str = "category_code",
    max_pairs_per_category: int = 3000,
    pos_k: int = 8,
    comp_k: int = 8,
    neutral_per_category: int = 800,
    epochs: int = 30,
    patience: int = 6,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden: int = 128,
    graph_dim: int = 64,
    dropout: float = 0.15,
    device=None,
):
    joint_df, sample_asin_df, joint_asin_df = prepare_scot5000_joint_relation_data(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
    )

    profiles = build_asin_relation_profiles(
        joint_df=joint_df,
        holdout_horizon=holdout_horizon,
        min_hist_weeks=min_hist_weeks,
        category_col=category_col,
    )

    pair_df, edge_feature_cols = build_relation_pair_dataset(
        profiles=profiles,
        max_pairs_per_category=max_pairs_per_category,
        pos_k=pos_k,
        comp_k=comp_k,
        neutral_per_category=neutral_per_category,
        seed=seed,
    )

    train_result = train_pair_relation_model(
        profiles=profiles,
        pair_df=pair_df,
        edge_feature_cols=edge_feature_cols,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        hidden=hidden,
        graph_dim=graph_dim,
        dropout=dropout,
        seed=seed,
        device=device,
    )

    diagnostics = diagnose_relation_learning(train_result, profiles, top_n=20, verbose=True)

    out = {
        **train_result,
        **diagnostics,
        "profiles": profiles,
        "joint_df": joint_df,
        "sample_asin_df": sample_asin_df,
        "joint_asin_df": joint_asin_df,
    }
    return out


# ============================================================
# Single usage cell for Jupyter
# ============================================================

if "data_raw1" in globals():
    gat_relation_result = run_gat_relation_test_scot5000(
        data_raw1=data_raw1,
        scot_df=scot_df if "scot_df" in globals() else None,
        n_asins=5000,
        seed=42,
        holdout_horizon=20,
        min_hist_weeks=13,
        category_col="category_code",
        max_pairs_per_category=3000,
        pos_k=8,
        comp_k=8,
        neutral_per_category=800,
        epochs=30,
        patience=6,
        batch_size=512,
        lr=1e-3,
        hidden=128,
        graph_dim=64,
        dropout=0.15,
    )

    pair_df = gat_relation_result["pair_df"]
    top_positive_pairs = gat_relation_result["top_positive_pairs"]
    top_competitive_pairs = gat_relation_result["top_competitive_pairs"]
    relation_summary = gat_relation_result["relation_summary"]
    category_relation_summary = gat_relation_result["category_relation_summary"]
    score_feature_corr = gat_relation_result["score_feature_corr"]
else:
    print("Define data_raw1 (and optionally scot_df) in your notebook, then run this file/cell.")
