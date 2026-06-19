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
#   total_dph, buy_box_dph, in_stock_dph, fbi_demand, hbt, ind_top10_brand, ind_promotion, our_price, holiday/event indicators
#   dynamic category rank / magnitude hierarchy features by origin_week
#
# Usage is at the bottom. Designed for Jupyter:
#   %run -i gat_relation_test_scot5000_DYNAMIC_RANK_MAG_v4_OURPRICE_EVENT_SPARSE.py
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

    # price features: use our_price as the primary dynamic price signal.
    # list_price is kept only as optional diagnostic/fallback, because the user wants
    # rank/relation to react to the actual observed/current price.
    if "our_price" in out.columns:
        out["log_our_price"] = np.log1p(_safe_numeric(out["our_price"]).clip(lower=0.0))
    elif "list_price" in out.columns:
        out["log_our_price"] = np.log1p(_safe_numeric(out["list_price"]).clip(lower=0.0))
    else:
        out["log_our_price"] = 0.0

    if "list_price" in out.columns:
        out["log_list_price_raw"] = np.log1p(_safe_numeric(out["list_price"]).clip(lower=0.0))
    else:
        out["log_list_price_raw"] = out["log_our_price"]

    # backward-compatible aliases used by older feature names; they now point to our_price.
    out["log_list_price"] = out["log_our_price"]
    out["log_price"] = out["log_our_price"]

    # Holiday / event known-at-origin features.
    holiday_cols = [c for c in out.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in out.columns if c.startswith("distance_")]
    for c in holiday_cols:
        out[c] = _safe_numeric(out[c]).clip(0, 1)
    prox_cols = []
    for c in distance_cols:
        # Distance columns can be signed; transform to a [0,1] proximity score.
        out[c] = _safe_numeric(out[c]).clip(-12, 12)
        pc = f"event_proximity__{c}"
        out[pc] = (1.0 - out[c].abs() / 12.0).clip(0, 1)
        prox_cols.append(pc)
    if holiday_cols:
        out["holiday_event_index"] = out[holiday_cols].max(axis=1).astype(float)
    else:
        out["holiday_event_index"] = 0.0
    if prox_cols:
        out["distance_event_proximity"] = out[prox_cols].max(axis=1).astype(float)
    else:
        out["distance_event_proximity"] = 0.0

    if "ind_promotion" in out.columns:
        out["ind_promotion"] = _safe_numeric(out["ind_promotion"]).clip(0, 1)
    else:
        out["ind_promotion"] = 0.0

    if "ind_prime_week" in out.columns:
        out["ind_prime_week"] = _safe_numeric(out["ind_prime_week"]).clip(0, 1)
    else:
        out["ind_prime_week"] = 0.0

    # One compact event index used by dynamic rank. It is known from calendar / current-week metadata.
    out["event_index"] = np.maximum.reduce([
        out["holiday_event_index"].astype(float).values,
        out["distance_event_proximity"].astype(float).values,
        out["ind_prime_week"].astype(float).values,
    ])
    out["promo_event_index"] = np.maximum(out["ind_promotion"].astype(float), out["event_index"].astype(float))

    print("Dynamic rank v4 uses our_price as primary price signal.")
    print(f"Holiday indicator cols: {len([c for c in out.columns if c.startswith('holiday_indicator_')])} | distance cols: {len([c for c in out.columns if c.startswith('distance_')])}")

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


def _tail_mean(x, k):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    return float(np.mean(x[-min(k, len(x)):]))


def _tail_zero_rate(x, k):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 1.0
    t = x[-min(k, len(x)):]
    return float(np.mean(t <= 0))


def _series_summary_features(x, prefix):
    """
    Dynamic signal summary for one ASIN before one origin_week.

    In addition to long-term level/active/zero stats, this explicitly builds
    recent momentum and zero/stock penalties that should drive dynamic category rank:
      - promotion can move an ASIN up temporarily
      - long recent zero / low in-stock / low buybox should move it down
    """
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
    ssum = float(np.sum(x))
    active_vals = x[x > 0]
    active_mean = float(np.mean(active_vals)) if len(active_vals) else 0.0
    active_q90 = float(np.quantile(active_vals, 0.90)) if len(active_vals) else 0.0

    recent4 = _tail_mean(x, 4)
    recent13 = _tail_mean(x, 13)
    recent26 = _tail_mean(x, 26)
    long52 = _tail_mean(x, 52)

    def log_ratio(num, den):
        return float(np.log1p(max(num, 0.0)) - np.log1p(max(den, 0.0)))

    return {
        f"{prefix}_log_sum": np.log1p(ssum),
        f"{prefix}_log_mean": np.log1p(mean),
        f"{prefix}_log_q75": np.log1p(q75),
        f"{prefix}_log_q90": np.log1p(q90),
        f"{prefix}_log_q95": np.log1p(q95),
        f"{prefix}_log_max": np.log1p(mx),
        f"{prefix}_log_recent4_mean": np.log1p(recent4),
        f"{prefix}_log_recent13_mean": np.log1p(recent13),
        f"{prefix}_log_recent26_mean": np.log1p(recent26),
        f"{prefix}_log_long52_mean": np.log1p(long52),
        f"{prefix}_recent13_vs_long52_logratio": log_ratio(recent13, long52),
        f"{prefix}_recent4_vs_recent13_logratio": log_ratio(recent4, recent13),
        f"{prefix}_recent13_zero_rate": _tail_zero_rate(x, 13),
        f"{prefix}_recent26_zero_rate": _tail_zero_rate(x, 26),
        f"{prefix}_active_rate": float(np.mean(active)),
        f"{prefix}_zero_rate": float(1.0 - np.mean(active)),
        f"{prefix}_cv": float(std / (mean + 1e-6)),
        f"{prefix}_gini": _gini(x),
        f"{prefix}_active_log_mean": np.log1p(active_mean),
        f"{prefix}_active_log_q90": np.log1p(active_q90),
    }


def _safe_percentile_rank(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True, method="average").fillna(0.5).clip(0.0, 1.0)


def _add_category_dynamic_ranks(prof):
    """
    Add origin-specific, within-category dynamic magnitude rank priors.

    These are the standalone GAT diagnostics that match the user's intended use:
      - promotion / promo-response can push current rank upward
      - recent stock/demand zeros push rank downward
      - recent13 vs long52 momentum pushes rank up/down
      - HBT/top10 brand and price tier provide business hierarchy context
    """
    if prof is None or len(prof) == 0:
        return prof
    prof = prof.copy()

    # Current raw strength before within-category rank.
    # DPH funnel is weighted more heavily than demand because this is an exposure relation test.
    prof["dyn_level_strength_raw"] = (
        0.30 * prof.get("total_log_recent13_mean", 0.0) +
        0.25 * prof.get("buybox_log_recent13_mean", 0.0) +
        0.25 * prof.get("instock_log_recent13_mean", 0.0) +
        0.20 * prof.get("demand_log_recent13_mean", 0.0)
    )
    prof["dyn_long_strength_raw"] = (
        0.30 * prof.get("total_log_long52_mean", prof.get("total_log_mean", 0.0)) +
        0.25 * prof.get("buybox_log_long52_mean", prof.get("buybox_log_mean", 0.0)) +
        0.25 * prof.get("instock_log_long52_mean", prof.get("instock_log_mean", 0.0)) +
        0.20 * prof.get("demand_log_long52_mean", prof.get("demand_log_mean", 0.0))
    )
    prof["dyn_momentum_raw"] = (
        0.30 * prof.get("total_recent13_vs_long52_logratio", 0.0) +
        0.25 * prof.get("buybox_recent13_vs_long52_logratio", 0.0) +
        0.25 * prof.get("instock_recent13_vs_long52_logratio", 0.0) +
        0.20 * prof.get("demand_recent13_vs_long52_logratio", 0.0)
    )
    prof["dyn_zero_penalty_raw"] = (
        0.20 * prof.get("total_recent13_zero_rate", 1.0) +
        0.30 * prof.get("buybox_recent13_zero_rate", 1.0) +
        0.30 * prof.get("instock_recent13_zero_rate", 1.0) +
        0.20 * prof.get("demand_recent13_zero_rate", 1.0)
    )
    prof["dyn_event_boost_raw"] = (
        0.40 * prof.get("event_index_current", 0.0) +
        0.25 * prof.get("holiday_event_index_current", 0.0) +
        0.20 * prof.get("distance_event_proximity_current", 0.0) +
        0.15 * prof.get("event_recent13_rate", 0.0)
    )
    prof["dyn_promo_boost_raw"] = (
        0.40 * prof.get("ind_promotion_current", 0.0) +
        0.20 * prof.get("promo_recent13_rate", 0.0) +
        0.20 * prof.get("promo_response_strength", 0.0) +
        0.20 * prof["dyn_event_boost_raw"]
    )
    prof["dyn_active_eligibility_raw"] = prof.get("active_eligibility_raw", 0.0)
    prof["dyn_business_prior_raw"] = (
        0.20 * prof.get("hbt_is_head", 0.0) -
        0.10 * prof.get("hbt_is_tail", 0.0) +
        0.15 * prof.get("ind_top10_brand", 0.0) +
        0.05 * prof.get("log_review_last", 0.0)
    )

    # Dynamic composite strength.  This is NOT a label; it is a history/current-known prior.
    prof["dynamic_strength_raw"] = (
        0.45 * prof["dyn_level_strength_raw"] +
        0.20 * prof["dyn_long_strength_raw"] +
        0.15 * prof["dyn_momentum_raw"] +
        0.12 * prof["dyn_promo_boost_raw"] +
        0.06 * prof["dyn_event_boost_raw"] +
        0.08 * prof["dyn_business_prior_raw"] +
        0.07 * prof["dyn_active_eligibility_raw"] -
        0.38 * prof["dyn_zero_penalty_raw"]
    )

    # Category percentiles by origin/category.
    group_keys = ["origin_week", "category_code"] if "origin_week" in prof.columns else ["category_code"]
    rank_specs = {
        "cat_rank_recent13_total": "total_log_recent13_mean",
        "cat_rank_recent13_buybox": "buybox_log_recent13_mean",
        "cat_rank_recent13_instock": "instock_log_recent13_mean",
        "cat_rank_recent13_demand": "demand_log_recent13_mean",
        "cat_rank_long52_total": "total_log_long52_mean",
        "cat_rank_momentum": "dyn_momentum_raw",
        "cat_rank_promo_boost": "dyn_promo_boost_raw",
        "cat_rank_event_boost": "dyn_event_boost_raw",
        "cat_rank_active_eligibility": "dyn_active_eligibility_raw",
        "cat_rank_zero_good": "dyn_zero_penalty_raw",  # inverted below
        "cat_rank_dynamic_strength": "dynamic_strength_raw",
    }
    for out_col, src_col in rank_specs.items():
        if src_col not in prof.columns:
            prof[out_col] = 0.5
            continue
        if out_col == "cat_rank_zero_good":
            prof[out_col] = prof.groupby(group_keys)[src_col].transform(lambda x: 1.0 - _safe_percentile_rank(x))
        else:
            prof[out_col] = prof.groupby(group_keys)[src_col].transform(_safe_percentile_rank)

    # Final compact rank prior used by pair features.
    prof["cat_rank_composite_dynamic"] = (
        0.25 * prof["cat_rank_recent13_total"] +
        0.20 * prof["cat_rank_recent13_buybox"] +
        0.20 * prof["cat_rank_recent13_instock"] +
        0.15 * prof["cat_rank_recent13_demand"] +
        0.09 * prof["cat_rank_momentum"] +
        0.04 * prof["cat_rank_promo_boost"] +
        0.04 * prof["cat_rank_event_boost"] +
        0.05 * prof["cat_rank_active_eligibility"] +
        0.03 * prof["cat_rank_zero_good"]
    ).clip(0.0, 1.0)

    return prof

def build_dynamic_node_profile(df, origin_week, min_hist_weeks=13):
    """Build ASIN node features using only weeks < origin_week."""
    origin_week = pd.to_datetime(origin_week)
    hist = df[df["order_week"] < origin_week].copy()
    current = df[df["order_week"] == origin_week].copy()
    current_idx = current.sort_values("order_week").groupby("asin").tail(1).set_index("asin") if len(current) else pd.DataFrame()
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
        # our_price is the main dynamic price signal.
        row["log_our_price_mean"] = float(g["log_our_price"].mean())
        row["log_our_price_last"] = float(g["log_our_price"].iloc[-1])
        row["log_list_price_mean"] = row["log_our_price_mean"]       # backward-compatible alias
        row["log_list_price_last"] = row["log_our_price_last"]       # backward-compatible alias
        row["log_price_mean"] = row["log_our_price_mean"]

        # Current-origin known features. These are not future realized DPH/demand targets.
        if asin in current_idx.index:
            cur = current_idx.loc[asin]
            row["ind_promotion_current"] = float(pd.to_numeric(cur.get("ind_promotion", 0.0), errors="coerce"))
            row["ind_prime_week_current"] = float(pd.to_numeric(cur.get("ind_prime_week", 0.0), errors="coerce"))
            row["log_our_price_current"] = float(pd.to_numeric(cur.get("log_our_price", row["log_our_price_last"]), errors="coerce"))
            row["log_list_price_current"] = row["log_our_price_current"]
            row["event_index_current"] = float(pd.to_numeric(cur.get("event_index", 0.0), errors="coerce"))
            row["holiday_event_index_current"] = float(pd.to_numeric(cur.get("holiday_event_index", 0.0), errors="coerce"))
            row["distance_event_proximity_current"] = float(pd.to_numeric(cur.get("distance_event_proximity", 0.0), errors="coerce"))
            row["promo_event_index_current"] = float(pd.to_numeric(cur.get("promo_event_index", max(row["ind_promotion_current"], row["event_index_current"])), errors="coerce"))
        else:
            row["ind_promotion_current"] = 0.0
            row["ind_prime_week_current"] = 0.0
            row["log_our_price_current"] = row["log_our_price_last"]
            row["log_list_price_current"] = row["log_our_price_current"]
            row["event_index_current"] = 0.0
            row["holiday_event_index_current"] = 0.0
            row["distance_event_proximity_current"] = 0.0
            row["promo_event_index_current"] = row["ind_promotion_current"]
        row["log_our_price_current_gap_vs_mean"] = row["log_our_price_current"] - row["log_our_price_mean"]
        row["log_list_price_current_gap_vs_mean"] = row["log_our_price_current_gap_vs_mean"]

        promo_arr = _safe_numeric(g["ind_promotion"]).values.astype(float)
        row["promo_rate"] = float(np.mean(promo_arr))
        row["promo_recent13_rate"] = float(np.mean(promo_arr[-13:])) if len(promo_arr) else 0.0
        row["promo_ever"] = float(np.max(promo_arr) > 0) if len(promo_arr) else 0.0
        row["prime_rate"] = float(g["ind_prime_week"].mean())
        row["event_rate"] = float(g["event_index"].mean()) if "event_index" in g.columns else 0.0
        row["event_recent13_rate"] = float(g["event_index"].tail(13).mean()) if "event_index" in g.columns and len(g) else 0.0
        row["promo_event_rate"] = float(g["promo_event_index"].mean()) if "promo_event_index" in g.columns else row["promo_rate"]
        row["promo_event_recent13_rate"] = float(g["promo_event_index"].tail(13).mean()) if "promo_event_index" in g.columns and len(g) else row["promo_recent13_rate"]
        row["hist_len"] = float(len(g))

        for c in SIGNAL_COLS:
            row.update(_series_summary_features(g[c].values, SIGNAL_PREFIX[c]))

        # promotion response: how much demand / exposure lifts during promotion weeks
        promo_mask = promo_arr > 0.5
        nonpromo_mask = ~promo_mask
        for c in SIGNAL_COLS:
            pfx = SIGNAL_PREFIX[c]
            vals = np.asarray(g[c].values, dtype=float)
            promo_mean = float(np.mean(vals[promo_mask])) if np.any(promo_mask) else 0.0
            nonpromo_mean = float(np.mean(vals[nonpromo_mask])) if np.any(nonpromo_mask) else 0.0
            row[f"{pfx}_promo_log_mean"] = np.log1p(promo_mean)
            row[f"{pfx}_nonpromo_log_mean"] = np.log1p(nonpromo_mean)
            row[f"{pfx}_promo_lift"] = np.log1p(promo_mean) - np.log1p(nonpromo_mean)

        # compound strength used for edge construction
        row["funnel_strength"] = (
            row["total_log_q90"] + row["buybox_log_q90"] + row["instock_log_q90"] + row["demand_log_q90"]
        ) / 4.0
        row["active_strength"] = (
            row["total_active_rate"] + row["buybox_active_rate"] + row["instock_active_rate"] + row["demand_active_rate"]
        ) / 4.0
        row["promo_response_strength"] = (
            row["total_promo_lift"] + row["buybox_promo_lift"] + row["instock_promo_lift"] + row["demand_promo_lift"]
        ) / 4.0
        # Sparse-aware eligibility: if recent signals are all zero and there is no promo/event,
        # this ASIN should often be tied/close rather than forced into a higher/lower rank.
        row["active_eligibility_raw"] = (
            0.25 * (1.0 - row.get("total_recent13_zero_rate", 1.0)) +
            0.25 * (1.0 - row.get("buybox_recent13_zero_rate", 1.0)) +
            0.25 * (1.0 - row.get("instock_recent13_zero_rate", 1.0)) +
            0.15 * (1.0 - row.get("demand_recent13_zero_rate", 1.0)) +
            0.05 * row.get("ind_promotion_current", 0.0) +
            0.05 * row.get("event_index_current", 0.0)
        )
        row["both_zero_like_score"] = 1.0 - row["active_eligibility_raw"]
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
        item["ind_promotion"] = g["ind_promotion"].reindex(weeks).fillna(0.0).values.astype(float)
        item["event_index"] = g["event_index"].reindex(weeks).fillna(0.0).values.astype(float) if "event_index" in g.columns else np.zeros(len(weeks), dtype=float)
        item["promo_event_index"] = g["promo_event_index"].reindex(weeks).fillna(0.0).values.astype(float) if "promo_event_index" in g.columns else item["ind_promotion"]
        item["log_our_price"] = g["log_our_price"].reindex(weeks).ffill().bfill().fillna(0.0).values.astype(float)
        item["log_list_price"] = item["log_our_price"]
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

    # our_price / price-tier relation. Positive pairs usually share current price tier;
    # competitive pairs can have price gaps only when combined with strength / brand gaps.
    li = float(row_i.get("log_our_price_mean", row_i.get("log_price_mean", 0.0)))
    lj = float(row_j.get("log_our_price_mean", row_j.get("log_price_mean", 0.0)))
    feat["log_our_price_abs_gap"] = abs(lj - li)
    feat["log_our_price_signed_gap_j_minus_i"] = lj - li
    # backward-compatible aliases
    feat["log_list_price_abs_gap"] = feat["log_our_price_abs_gap"]
    feat["log_list_price_signed_gap_j_minus_i"] = feat["log_our_price_signed_gap_j_minus_i"]
    feat["same_price_tier_soft"] = float(abs(lj - li) <= 0.25)
    feat["price_tier_gap_large"] = float(abs(lj - li) >= 0.75)

    # promotion-level relation from ASIN profile
    pi = float(row_i.get("promo_rate", 0.0)); pj = float(row_j.get("promo_rate", 0.0))
    pri = float(row_i.get("promo_recent13_rate", 0.0)); prj = float(row_j.get("promo_recent13_rate", 0.0))
    feat["promo_rate_abs_gap"] = abs(pj - pi)
    feat["promo_recent13_abs_gap"] = abs(prj - pri)
    feat["same_promo_regime_soft"] = float(abs(pj - pi) <= 0.15 and abs(prj - pri) <= 0.20)
    feat["j_more_promoted"] = float(pj > pi + 0.20)

    for prefix in ["total", "buybox", "instock", "demand"]:
        lift_i = float(row_i.get(f"{prefix}_promo_lift", 0.0))
        lift_j = float(row_j.get(f"{prefix}_promo_lift", 0.0))
        feat[f"{prefix}_promo_lift_abs_gap"] = abs(lift_j - lift_i)
        feat[f"{prefix}_promo_lift_signed_gap_j_minus_i"] = lift_j - lift_i
    feat["promo_response_abs_gap"] = abs(float(row_j.get("promo_response_strength", 0.0)) - float(row_i.get("promo_response_strength", 0.0)))
    feat["promo_response_signed_gap_j_minus_i"] = float(row_j.get("promo_response_strength", 0.0)) - float(row_i.get("promo_response_strength", 0.0))

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

    # Dynamic magnitude hierarchy features: category-relative rank at origin_week.
    rank_cols = [
        "cat_rank_recent13_total", "cat_rank_recent13_buybox", "cat_rank_recent13_instock", "cat_rank_recent13_demand",
        "cat_rank_long52_total", "cat_rank_momentum", "cat_rank_promo_boost", "cat_rank_event_boost", "cat_rank_active_eligibility", "cat_rank_zero_good", "cat_rank_dynamic_strength",
        "cat_rank_composite_dynamic",
    ]
    for rc in rank_cols:
        ri_val = float(row_i.get(rc, 0.5))
        rj_val = float(row_j.get(rc, 0.5))
        short = rc.replace("cat_rank_", "rank_")
        feat[f"{short}_i"] = ri_val
        feat[f"{short}_j"] = rj_val
        feat[f"{short}_signed_gap_j_minus_i"] = rj_val - ri_val
        feat[f"{short}_abs_gap"] = abs(rj_val - ri_val)
        feat[f"j_higher_{short}"] = float(rj_val > ri_val + 0.10)
    feat["j_higher_dynamic_rank"] = float(feat.get("rank_composite_dynamic_signed_gap_j_minus_i", 0.0) > 0.10)
    feat["i_higher_dynamic_rank"] = float(feat.get("rank_composite_dynamic_signed_gap_j_minus_i", 0.0) < -0.10)
    feat["dynamic_rank_abs_gap"] = abs(feat.get("rank_composite_dynamic_signed_gap_j_minus_i", 0.0))

    # Current-origin promo interaction: promotion can temporarily lift rank.
    feat["i_promo_current"] = float(row_i.get("ind_promotion_current", 0.0))
    feat["j_promo_current"] = float(row_j.get("ind_promotion_current", 0.0))
    feat["j_promo_i_not_current"] = float(feat["j_promo_current"] > 0.5 and feat["i_promo_current"] <= 0.5)
    feat["both_promo_current"] = float(feat["j_promo_current"] > 0.5 and feat["i_promo_current"] > 0.5)
    feat["i_event_current"] = float(row_i.get("event_index_current", 0.0))
    feat["j_event_current"] = float(row_j.get("event_index_current", 0.0))
    feat["both_event_current"] = float(feat["i_event_current"] > 0.5 and feat["j_event_current"] > 0.5)
    feat["j_event_i_not_current"] = float(feat["j_event_current"] > 0.5 and feat["i_event_current"] <= 0.5)
    feat["event_current_j_minus_i"] = feat["j_event_current"] - feat["i_event_current"]
    feat["i_active_eligibility"] = float(row_i.get("active_eligibility_raw", 0.0))
    feat["j_active_eligibility"] = float(row_j.get("active_eligibility_raw", 0.0))
    feat["active_eligibility_abs_gap"] = abs(feat["j_active_eligibility"] - feat["i_active_eligibility"])
    feat["both_low_eligibility"] = float(feat["i_active_eligibility"] < 0.15 and feat["j_active_eligibility"] < 0.15)

    # interactions important for competitive pressure
    feat["stronger_top10_competitor"] = float(feat["j_top10_i_not"] * feat["j_stronger_funnel"])
    feat["hbt_diff_and_j_stronger"] = float(feat["hbt_diff"] * feat["j_stronger_funnel"])
    feat["brand_diff_and_j_stronger"] = float(feat["top10_brand_diff"] * feat["j_stronger_funnel"])
    feat["stronger_top10_price_gap"] = float(feat["stronger_top10_competitor"] * feat["log_our_price_abs_gap"])
    feat["j_promo_stronger_funnel"] = float(feat["j_more_promoted"] * feat["j_stronger_funnel"])

    # historical aligned correlations / active-overlap / promotion co-movement, if sequences available
    if seq_i is not None and seq_j is not None:
        promo_i = np.asarray(seq_i.get("ind_promotion", []), dtype=float) > 0.5
        promo_j = np.asarray(seq_j.get("ind_promotion", []), dtype=float) > 0.5
        if len(promo_i) == len(promo_j) and len(promo_i) > 0:
            feat["promo_overlap_rate"] = float(np.mean(promo_i & promo_j))
            feat["promo_i_only_rate"] = float(np.mean(promo_i & (~promo_j)))
            feat["promo_j_only_rate"] = float(np.mean((~promo_i) & promo_j))
            feat["promo_any_overlap_jaccard"] = float(np.sum(promo_i & promo_j) / (np.sum(promo_i | promo_j) + 1e-6))
        else:
            feat["promo_overlap_rate"] = 0.0
            feat["promo_i_only_rate"] = 0.0
            feat["promo_j_only_rate"] = 0.0
            feat["promo_any_overlap_jaccard"] = 0.0

        for c in SIGNAL_COLS:
            p = SIGNAL_PREFIX[c]
            xi = np.asarray(seq_i.get(c, []), dtype=float)
            xj = np.asarray(seq_j.get(c, []), dtype=float)
            corr = _corr_aligned(xi, xj)
            feat[f"hist_corr_{p}"] = corr
            feat[f"hist_neg_corr_{p}"] = max(0.0, -corr)
            feat[f"hist_pos_corr_{p}"] = max(0.0, corr)
            if len(xi) == len(xj) and len(xi) > 0:
                ai = xi > 0; aj = xj > 0
                feat[f"active_overlap_{p}"] = float(np.mean(ai & aj))
                feat[f"zero_overlap_{p}"] = float(np.mean((~ai) & (~aj)))
                feat[f"active_jaccard_{p}"] = float(np.sum(ai & aj) / (np.sum(ai | aj) + 1e-6))
                # promotion stealing proxy: j is promoted and strong while i is weak
                if len(promo_j) == len(xi):
                    j_promo = promo_j
                    j_high = xj > (np.quantile(xj, 0.75) + 1e-9)
                    i_low = xi <= (np.quantile(xi, 0.50) + 1e-9)
                    feat[f"promo_steal_j_to_i_{p}"] = float(np.mean(j_promo & j_high & i_low))
                else:
                    feat[f"promo_steal_j_to_i_{p}"] = 0.0
            else:
                feat[f"active_overlap_{p}"] = 0.0
                feat[f"zero_overlap_{p}"] = 0.0
                feat[f"active_jaccard_{p}"] = 0.0
                feat[f"promo_steal_j_to_i_{p}"] = 0.0
    else:
        for p in ["total", "buybox", "instock", "demand"]:
            feat[f"hist_corr_{p}"] = 0.0
            feat[f"hist_neg_corr_{p}"] = 0.0
            feat[f"hist_pos_corr_{p}"] = 0.0
            feat[f"active_overlap_{p}"] = 0.0
            feat[f"zero_overlap_{p}"] = 0.0
            feat[f"active_jaccard_{p}"] = 0.0
            feat[f"promo_steal_j_to_i_{p}"] = 0.0
        feat["promo_overlap_rate"] = 0.0
        feat["promo_i_only_rate"] = 0.0
        feat["promo_j_only_rate"] = 0.0
        feat["promo_any_overlap_jaccard"] = 0.0

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
            edge_feat.get("mean_abs_gap_all", 9) < 0.55 and
            edge_feat.get("same_hbt", 0) > 0.5 and
            edge_feat.get("same_top10_brand", 0) > 0.5 and
            edge_feat.get("same_price_tier_soft", 0) > 0.5 and
            edge_feat.get("same_promo_regime_soft", 0) > 0.5
        )
        comp_like = (
            edge_feat.get("j_stronger_funnel", 0) > 0.5 and
            (edge_feat.get("hbt_diff", 0) > 0.5 or edge_feat.get("top10_brand_diff", 0) > 0.5 or edge_feat.get("stronger_top10_competitor", 0) > 0.5) and
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

    # Positive: not all same-category pairs are positive.  A positive pair should look
    # similar historically in demand + exposure funnel, and usually share hbt/top10/price/promo regime.
    hist_level_close = edge_feat.get("mean_abs_gap_all", 9) <= 0.65
    product_regime_close = (
        edge_feat.get("same_hbt", 0) > 0.5 and
        edge_feat.get("same_top10_brand", 0) > 0.5 and
        edge_feat.get("same_price_tier_soft", 0) > 0.5 and
        edge_feat.get("same_promo_regime_soft", 0) > 0.5
    )
    active_overlap_good = np.mean([
        edge_feat.get("active_jaccard_total", 0),
        edge_feat.get("active_jaccard_buybox", 0),
        edge_feat.get("active_jaccard_instock", 0),
        edge_feat.get("active_jaccard_demand", 0),
    ]) >= 0.20
    promo_similar = edge_feat.get("promo_any_overlap_jaccard", 0) >= 0.20 or edge_feat.get("same_promo_regime_soft", 0) > 0.5

    positive = (
        hist_level_close and
        product_regime_close and
        both_active > 0 and
        abs_strength_gap <= 0.85 and
        (mean_pos_corr >= 0.20 or active_overlap_good or promo_similar)
    )

    # Competitive: same category, clear strong/weak contrast, often top-brand/head/HBT/price/promo response gap.
    promo_steal = np.mean([
        edge_feat.get("promo_steal_j_to_i_total", 0),
        edge_feat.get("promo_steal_j_to_i_buybox", 0),
        edge_feat.get("promo_steal_j_to_i_instock", 0),
        edge_feat.get("promo_steal_j_to_i_demand", 0),
    ])
    dominant_neighbor = (
        edge_feat.get("j_stronger_funnel", 0) > 0.5 or
        edge_feat.get("stronger_top10_competitor", 0) > 0.5 or
        edge_feat.get("j_promo_stronger_funnel", 0) > 0.5
    )
    regime_contrast = (
        edge_feat.get("hbt_diff", 0) > 0.5 or
        edge_feat.get("top10_brand_diff", 0) > 0.5 or
        edge_feat.get("price_tier_gap_large", 0) > 0.5 or
        edge_feat.get("promo_response_abs_gap", 0) > 0.50
    )
    competitive = (
        dominant_neighbor and
        regime_contrast and
        (abs_strength_gap >= 1.00 or edge_feat.get("funnel_strength_abs_gap", 0) >= 0.75 or promo_steal >= 0.10 or mean_neg_corr >= 0.10)
    )

    if competitive and not positive:
        return 1
    if positive and not competitive:
        return 2
    if competitive and positive:
        # if relation is ambiguous, use strength gap to separate
        return 1 if abs_strength_gap > 1.25 else 2
    return 0



def _future_rank_map(profile_df, future_dict):
    """Compute future label-window within-category rank for diagnostics/training target."""
    if profile_df is None or profile_df.empty:
        return {}
    tmp = profile_df[["asin", "category_code"]].copy()
    vals = []
    for asin in tmp["asin"].astype(str).values:
        fut = future_dict.get(asin, None)
        if fut is None:
            vals.append(0.0)
        else:
            vals.append(float(np.log1p(
                fut.get("total_sum", 0.0) +
                fut.get("buybox_sum", 0.0) +
                fut.get("instock_sum", 0.0) +
                fut.get("demand_sum", 0.0)
            )))
    tmp["future_strength"] = vals
    tmp["future_cat_rank"] = tmp.groupby("category_code")["future_strength"].transform(_safe_percentile_rank)
    return tmp.set_index("asin")[["future_strength", "future_cat_rank"]].to_dict("index")

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
        # Add dynamic category rank / magnitude hierarchy priors for this origin.
        prof = _add_category_dynamic_ranks(prof)
        profile_rows.append(prof)
        asin_set = set(prof["asin"].astype(str))
        seqs = _get_recent_sequences(df, origin, asin_set, lookback_weeks=lookback_weeks)
        fut = _future_profile(df, origin, asin_set, label_horizon=label_horizon)
        fut_rank = _future_rank_map(prof, fut)

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
                fri = fut_rank.get(a, {"future_strength": 0.0, "future_cat_rank": 0.5})
                frj = fut_rank.get(b, {"future_strength": 0.0, "future_cat_rank": 0.5})
                future_rank_gap = float(frj.get("future_cat_rank", 0.5) - fri.get("future_cat_rank", 0.5))
                # Sparse-aware + event-aware rank label:
                # - if both ASINs are effectively inactive and no current promo/event signal, keep them close.
                # - normal weeks use a wider close band to avoid fake ordering among many zeros.
                # - promo/event weeks allow rank to separate more easily.
                elig_i = float(ri.get("active_eligibility_raw", 0.0))
                elig_j = float(rj.get("active_eligibility_raw", 0.0))
                event_pair = max(float(ri.get("event_index_current", 0.0)), float(rj.get("event_index_current", 0.0)), float(ri.get("ind_promotion_current", 0.0)), float(rj.get("ind_promotion_current", 0.0)))
                both_inactive = (elig_i < 0.15 and elig_j < 0.15)
                close_margin = 0.10 if event_pair > 0.5 else 0.20
                if both_inactive:
                    rank_label, rank_label_name = 0, "close_rank"
                elif future_rank_gap > close_margin:
                    rank_label, rank_label_name = 1, "j_higher"
                elif future_rank_gap < -close_margin:
                    rank_label, rank_label_name = 2, "i_higher"
                else:
                    rank_label, rank_label_name = 0, "close_rank"

                row = {
                    "origin_week": origin,
                    "asin_i": a,
                    "asin_j": b,
                    "category_code": cat,
                    "label": int(y),
                    "label_name": {0: "neutral", 1: "competitive", 2: "positive"}[int(y)],
                    "rank_label": int(rank_label),
                    "rank_label_name": rank_label_name,
                    "future_strength_i": float(fri.get("future_strength", 0.0)),
                    "future_strength_j": float(frj.get("future_strength", 0.0)),
                    "future_cat_rank_i": float(fri.get("future_cat_rank", 0.5)),
                    "future_cat_rank_j": float(frj.get("future_cat_rank", 0.5)),
                    "future_rank_gap_j_minus_i": future_rank_gap,
                    "hbt_i": ri["hbt"],
                    "hbt_j": rj["hbt"],
                    "top10_i": ri["ind_top10_brand"],
                    "top10_j": rj["ind_top10_brand"],
                    "funnel_strength_i": ri["funnel_strength"],
                    "funnel_strength_j": rj["funnel_strength"],
                    "dynamic_rank_i": ri.get("cat_rank_composite_dynamic", 0.5),
                    "dynamic_rank_j": rj.get("cat_rank_composite_dynamic", 0.5),
                    "dynamic_strength_raw_i": ri.get("dynamic_strength_raw", 0.0),
                    "dynamic_strength_raw_j": rj.get("dynamic_strength_raw", 0.0),
                    "ind_promotion_current_i": ri.get("ind_promotion_current", 0.0),
                    "ind_promotion_current_j": rj.get("ind_promotion_current", 0.0),
                    "event_index_current_i": ri.get("event_index_current", 0.0),
                    "event_index_current_j": rj.get("event_index_current", 0.0),
                    "active_eligibility_i": ri.get("active_eligibility_raw", 0.0),
                    "active_eligibility_j": rj.get("active_eligibility_raw", 0.0),
                    "log_our_price_i": ri.get("log_our_price_current", ri.get("log_our_price_mean", ri.get("log_price_mean", 0.0))),
                    "log_our_price_j": rj.get("log_our_price_current", rj.get("log_our_price_mean", rj.get("log_price_mean", 0.0))),
                    "log_list_price_i": ri.get("log_our_price_current", ri.get("log_our_price_mean", ri.get("log_price_mean", 0.0))),
                    "log_list_price_j": rj.get("log_our_price_current", rj.get("log_our_price_mean", rj.get("log_price_mean", 0.0))),
                    "promo_rate_i": ri.get("promo_rate", 0.0),
                    "promo_rate_j": rj.get("promo_rate", 0.0),
                    "promo_response_i": ri.get("promo_response_strength", 0.0),
                    "promo_response_j": rj.get("promo_response_strength", 0.0),
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
    exclude = {
        "origin_week", "asin_i", "asin_j", "category_code", "label", "label_name", "hbt_i", "hbt_j",
        "rank_label", "rank_label_name",
        "future_strength_i", "future_strength_j", "future_cat_rank_i", "future_cat_rank_j", "future_rank_gap_j_minus_i",
    }
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




def train_pairwise_rank_classifier(pair_df, epochs=30, batch_size=512, lr=1e-3, seed=42, verbose=True):
    """
    Separate pairwise magnitude hierarchy test.

    Target:
      0 = close_rank
      1 = asin_j should rank higher than asin_i in the future label window
      2 = asin_i should rank higher than asin_j in the future label window

    Input uses only history/current-known dynamic rank features, not future rank columns.
    """
    if pair_df.empty:
        raise ValueError("pair_df is empty")
    if "rank_label" not in pair_df.columns:
        raise ValueError("pair_df must contain rank_label")

    exclude = {
        "origin_week", "asin_i", "asin_j", "category_code", "label", "label_name", "hbt_i", "hbt_j",
        "rank_label", "rank_label_name",
        "future_strength_i", "future_strength_j", "future_cat_rank_i", "future_cat_rank_j", "future_rank_gap_j_minus_i",
    }
    feat_cols = [c for c in pair_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(pair_df[c])]
    df = pair_df.copy()
    for c in feat_cols:
        df[c] = _safe_numeric(df[c]).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    origins = sorted(pd.to_datetime(df["origin_week"].unique()))
    if len(origins) >= 3:
        split = int(len(origins) * 0.75)
        train_origins = set(origins[:split])
        test_origins = set(origins[split:])
        tr = df[pd.to_datetime(df["origin_week"]).isin(train_origins)].copy()
        te = df[pd.to_datetime(df["origin_week"]).isin(test_origins)].copy()
    else:
        tr = df.sample(frac=0.75, random_state=seed)
        te = df.drop(tr.index)

    def _balanced_by_rank(d, max_per_class=60000, seed0=42):
        parts = []
        for lab, g in d.groupby("rank_label"):
            if len(g) > max_per_class:
                parts.append(g.sample(n=max_per_class, random_state=seed0 + int(lab)))
            else:
                parts.append(g)
        return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed0).reset_index(drop=True)

    tr = _balanced_by_rank(tr, max_per_class=60000, seed0=seed)
    te = _balanced_by_rank(te, max_per_class=60000, seed0=seed + 1)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[feat_cols].values.astype(float))
    Xte = scaler.transform(te[feat_cols].values.astype(float))
    ytr = tr["rank_label"].values.astype(int)
    yte = te["rank_label"].values.astype(int)

    tr_ds = PairRelationDataset(Xtr, ytr)
    te_ds = PairRelationDataset(Xte, yte)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model = EdgeAwareRelationClassifier(input_dim=Xtr.shape[1], n_classes=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                pr = torch.softmax(model(xb.to(DEVICE)), dim=-1).cpu().numpy()
                preds.append(np.argmax(pr, axis=1))
                ys.append(yb.numpy())
        preds = np.concatenate(preds)
        ys = np.concatenate(ys)
        f1 = f1_score(ys, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            acc = accuracy_score(ys, preds)
            print(f"[rank] Epoch {ep+1:03d} | train_loss={np.mean(losses):.4f} | test_acc={acc:.4f} | macro_f1={f1:.4f}")

    if best_sd is not None:
        model.load_state_dict(best_sd)

    Xall = scaler.transform(df[feat_cols].values.astype(float))
    all_ld = DataLoader(PairRelationDataset(Xall, df["rank_label"].values.astype(int)), batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in all_ld:
            all_probs.append(torch.softmax(model(xb.to(DEVICE)), dim=-1).cpu().numpy())
    all_probs = np.concatenate(all_probs)

    scored = df.copy()
    scored["score_close_rank"] = all_probs[:, 0]
    scored["score_j_higher"] = all_probs[:, 1]
    scored["score_i_higher"] = all_probs[:, 2]
    scored["pred_rank_label"] = np.argmax(all_probs, axis=1)
    scored["pred_rank_label_name"] = scored["pred_rank_label"].map({0: "close_rank", 1: "j_higher", 2: "i_higher"})

    Xte2 = scaler.transform(te[feat_cols].values.astype(float))
    with torch.no_grad():
        pte = torch.softmax(model(torch.tensor(Xte2, dtype=torch.float32, device=DEVICE)), dim=-1).cpu().numpy()
    pred_te = np.argmax(pte, axis=1)
    report = classification_report(yte, pred_te, target_names=["close_rank", "j_higher", "i_higher"], output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred_te, labels=[0, 1, 2])

    return {
        "rank_model": model,
        "rank_scaler": scaler,
        "rank_feature_cols": feat_cols,
        "rank_scored_pair_df": scored,
        "rank_heldout_report": report,
        "rank_confusion_matrix": cm,
        "rank_best_macro_f1": best_f1,
    }

# -----------------------------
# Diagnostics
# -----------------------------


def diagnose_dynamic_rank_scores(rank_scored_pair_df, profile_df=None, top_n=30, verbose=True):
    df = rank_scored_pair_df.copy()

    rank_summary = (
        df.groupby("pred_rank_label_name")
        .agg(
            n_pairs=("asin_i", "size"),
            mean_score_j_higher=("score_j_higher", "mean"),
            mean_score_i_higher=("score_i_higher", "mean"),
            mean_future_rank_gap=("future_rank_gap_j_minus_i", "mean"),
            mean_dynamic_rank_gap=("rank_composite_dynamic_signed_gap_j_minus_i", "mean") if "rank_composite_dynamic_signed_gap_j_minus_i" in df.columns else ("score_j_higher", "mean"),
            mean_dynamic_rank_abs_gap=("dynamic_rank_abs_gap", "mean") if "dynamic_rank_abs_gap" in df.columns else ("score_j_higher", "mean"),
            mean_j_promo_current=("j_promo_current", "mean") if "j_promo_current" in df.columns else ("score_j_higher", "mean"),
            mean_i_promo_current=("i_promo_current", "mean") if "i_promo_current" in df.columns else ("score_i_higher", "mean"),
        )
        .reset_index()
        .sort_values("n_pairs", ascending=False)
    )

    by_origin = (
        df.groupby("origin_week")
        .agg(
            n_pairs=("asin_i", "size"),
            j_higher_rate=("pred_rank_label", lambda x: float(np.mean(np.asarray(x)==1))),
            i_higher_rate=("pred_rank_label", lambda x: float(np.mean(np.asarray(x)==2))),
            close_rank_rate=("pred_rank_label", lambda x: float(np.mean(np.asarray(x)==0))),
            avg_score_j_higher=("score_j_higher", "mean"),
            avg_score_i_higher=("score_i_higher", "mean"),
        )
        .reset_index()
    )

    top_j = df.sort_values("score_j_higher", ascending=False).head(top_n)
    top_i = df.sort_values("score_i_higher", ascending=False).head(top_n)

    corr_cols = [c for c in [
        "rank_composite_dynamic_signed_gap_j_minus_i", "dynamic_rank_abs_gap",
        "rank_recent13_total_signed_gap_j_minus_i", "rank_recent13_buybox_signed_gap_j_minus_i",
        "rank_recent13_instock_signed_gap_j_minus_i", "rank_recent13_demand_signed_gap_j_minus_i",
        "rank_momentum_signed_gap_j_minus_i", "rank_promo_boost_signed_gap_j_minus_i", "rank_zero_good_signed_gap_j_minus_i",
        "j_promo_current", "i_promo_current", "j_promo_i_not_current", "both_promo_current",
        "funnel_strength_signed_gap", "active_strength_signed_gap",
        "promo_response_signed_gap_j_minus_i", "log_our_price_signed_gap_j_minus_i", "log_list_price_signed_gap_j_minus_i", "event_index_current_j", "active_eligibility_abs_gap",
    ] if c in df.columns]
    corr_rows = []
    for c in corr_cols:
        for score in ["score_j_higher", "score_i_higher"]:
            corr_rows.append({"feature": c, "score": score, "corr": _corr_aligned(df[c].values, df[score].values)})
    rank_score_feature_corr = pd.DataFrame(corr_rows).sort_values(["score", "corr"], ascending=[True, False])

    top_ranked_asins = pd.DataFrame()
    if profile_df is not None and not profile_df.empty and "cat_rank_composite_dynamic" in profile_df.columns:
        cols = [c for c in [
            "origin_week", "asin", "category_code", "cat_rank_composite_dynamic",
            "cat_rank_recent13_total", "cat_rank_recent13_buybox", "cat_rank_recent13_instock", "cat_rank_recent13_demand",
            "cat_rank_momentum", "cat_rank_promo_boost", "cat_rank_zero_good",
            "dynamic_strength_raw", "ind_promotion_current", "promo_response_strength", "hbt", "ind_top10_brand",
            "total_log_recent13_mean", "buybox_log_recent13_mean", "instock_log_recent13_mean", "demand_log_recent13_mean",
            "instock_recent13_zero_rate", "buybox_recent13_zero_rate", "demand_recent13_zero_rate",
        ] if c in profile_df.columns]
        top_ranked_asins = profile_df.sort_values(["origin_week", "category_code", "cat_rank_composite_dynamic"], ascending=[True, True, False])[cols].groupby(["origin_week", "category_code"]).head(5).reset_index(drop=True)

    if verbose:
        print("\n=== Dynamic magnitude-rank prediction summary ===")
        print(rank_summary.to_string(index=False))
        print("\n=== Rank by origin week ===")
        print(by_origin.tail(10).to_string(index=False))
        print("\n=== Rank score feature correlations ===")
        print(rank_score_feature_corr.head(30).to_string(index=False))
        preview_cols = [c for c in [
            "origin_week", "asin_i", "asin_j", "category_code", "score_j_higher", "score_i_higher", "score_close_rank",
            "dynamic_rank_i", "dynamic_rank_j", "rank_composite_dynamic_signed_gap_j_minus_i",
            "future_cat_rank_i", "future_cat_rank_j", "future_rank_gap_j_minus_i",
            "i_promo_current", "j_promo_current", "funnel_strength_i", "funnel_strength_j",
            "hbt_i", "hbt_j", "top10_i", "top10_j"
        ] if c in df.columns]
        print("\n=== Top pairs where ASIN_j should rank higher ===")
        print(top_j[preview_cols].head(10).to_string(index=False))
        print("\n=== Top pairs where ASIN_i should rank higher ===")
        print(top_i[preview_cols].head(10).to_string(index=False))
        if not top_ranked_asins.empty:
            print("\n=== Top dynamic-ranked ASINs per category/origin preview ===")
            print(top_ranked_asins.head(20).to_string(index=False))

    return {
        "rank_summary": rank_summary,
        "rank_by_origin": by_origin,
        "top_j_higher_pairs": top_j,
        "top_i_higher_pairs": top_i,
        "rank_score_feature_corr": rank_score_feature_corr,
        "top_ranked_asins": top_ranked_asins,
    }


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
        "log_our_price_abs_gap", "log_list_price_abs_gap", "same_price_tier_soft", "price_tier_gap_large", "both_low_eligibility", "j_event_i_not_current",
        "promo_rate_abs_gap", "promo_recent13_abs_gap", "same_promo_regime_soft",
        "promo_overlap_rate", "promo_any_overlap_jaccard", "promo_response_abs_gap",
        "promo_steal_j_to_i_total", "promo_steal_j_to_i_buybox", "promo_steal_j_to_i_instock", "promo_steal_j_to_i_demand",
        "active_jaccard_total", "active_jaccard_buybox", "active_jaccard_instock", "active_jaccard_demand",
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
        cols = [c for c in ["origin_week", "asin_i", "asin_j", "category_code", "score_positive", "score_competitive", "hbt_i", "hbt_j", "top10_i", "top10_j", "same_hbt", "same_top10_brand", "log_our_price_abs_gap", "same_price_tier_soft", "promo_rate_abs_gap", "event_index_current_i", "event_index_current_j", "active_eligibility_i", "active_eligibility_j", "promo_overlap_rate", "promo_response_abs_gap", "total_log_q90_abs_gap", "buybox_log_q90_abs_gap", "instock_log_q90_abs_gap", "demand_log_q90_abs_gap"] if c in top_pos.columns]
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

    print("\nPair relation label distribution:")
    print(pair_df["label_name"].value_counts().to_string())
    print("\nPair dynamic-rank label distribution:")
    print(pair_df["rank_label_name"].value_counts().to_string())

    print("\n" + "=" * 80)
    print("TRAIN RELATION CLASSIFIER: neutral / competitive / positive")
    print("=" * 80)
    train_out = train_relation_classifier(
        pair_df,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        verbose=True,
    )
    diag = diagnose_relation_scores(train_out["scored_pair_df"], verbose=True)

    print("\nRelation heldout report:")
    print(pd.DataFrame(train_out["heldout_report"]).T.to_string())
    print("\nRelation confusion matrix [neutral, competitive, positive]:")
    print(train_out["confusion_matrix"])

    print("\n" + "=" * 80)
    print("TRAIN PAIRWISE DYNAMIC MAGNITUDE-RANK CLASSIFIER")
    print("Classes: close_rank / asin_j higher / asin_i higher")
    print("=" * 80)
    rank_out = train_pairwise_rank_classifier(
        pair_df,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        verbose=True,
    )
    rank_diag = diagnose_dynamic_rank_scores(rank_out["rank_scored_pair_df"], profile_df=profile_df, verbose=True)

    print("\nRank heldout report:")
    print(pd.DataFrame(rank_out["rank_heldout_report"]).T.to_string())
    print("\nRank confusion matrix [close_rank, j_higher, i_higher]:")
    print(rank_out["rank_confusion_matrix"])

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

        "rank_model": rank_out["rank_model"],
        "rank_scaler": rank_out["rank_scaler"],
        "rank_feature_cols": rank_out["rank_feature_cols"],
        "rank_scored_pair_df": rank_out["rank_scored_pair_df"],
        "rank_heldout_report": rank_out["rank_heldout_report"],
        "rank_confusion_matrix": rank_out["rank_confusion_matrix"],
        "rank_diagnostics": rank_diag,
        "top_j_higher_pairs": rank_diag["top_j_higher_pairs"],
        "top_i_higher_pairs": rank_diag["top_i_higher_pairs"],
        "top_ranked_asins": rank_diag["top_ranked_asins"],
        "rank_origin_summary": rank_diag["rank_by_origin"],
        "rank_score_feature_corr": rank_diag["rank_score_feature_corr"],
    }


# ============================================================
# FINAL USAGE ONLY
# ============================================================
# In Jupyter, run this file after data_raw1 and scot_df exist:
# %run -i gat_relation_test_scot5000_DYNAMIC_RANK_MAG_v4_OURPRICE_EVENT_SPARSE.py
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
# rank_scored_pair_df = dynamic_gat_relation_result["rank_scored_pair_df"]
# top_ranked_asins = dynamic_gat_relation_result["top_ranked_asins"]
# top_j_higher_pairs = dynamic_gat_relation_result["top_j_higher_pairs"]
# top_i_higher_pairs = dynamic_gat_relation_result["top_i_higher_pairs"]
# top_positive_pairs = dynamic_gat_relation_result["top_positive_pairs"]
# top_competitive_pairs = dynamic_gat_relation_result["top_competitive_pairs"]
# category_relation_summary = dynamic_gat_relation_result["category_relation_summary"]
# origin_relation_summary = dynamic_gat_relation_result["origin_relation_summary"]
