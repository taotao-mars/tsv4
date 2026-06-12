"""
Joint demand-centered model with internal exposure-covariate decoder + dual-relation graph.

Goal
----
Demand is the main task. Exposure is an auxiliary covariate generator, not the final target.

Architecture
------------
Shared history encoder
  -> h_t
Dual-relation ASIN graph
  -> graph_embedding
Exposure covariate branch
  -> pred_total_dph, pred_buy_box_dph, pred_instock_dph, p_active_instock, exposure_hidden
Demand ENN head
  -> consumes h_t + future_context + graph_embedding + exposure branch outputs
  -> predicts demand NB mu/alpha and p50/p70

Leakage rule
------------
- Future true DPH/OOS are never used as demand inputs.
- Future true DPH is used only as auxiliary exposure target during training/evaluation.
- Graph features are built from historical rows before the final horizon cutoff by default.

Typical run
-----------
%run -i joint_demand_exposure_dualgraph_model.py

result = run_joint_demand_exposure_dualgraph(
    data_raw1=data_raw1,
    scot_df=scot_df,
    n_asins=5000,
    history=52,
    horizon=20,
    epochs=60,
    patience=8,
    batch_size=64,
    use_graph=True,
    graph_dim=16,
    neighbor_k=10,
    detach_exposure_for_demand=True,
    lambda_exp=0.10,
    lambda_active=0.10,
)

forecast_df = result["forecast_df"]
"""

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

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)


# =====================================================
# Utilities
# =====================================================

def _get_1d_col(df: pd.DataFrame, col: str) -> pd.Series:
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x


def _safe_numeric_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(_get_1d_col(df, col), errors="coerce").fillna(default)


def _standardize(s: pd.Series, clip: float = 5.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    sd = float(s.std())
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    out = (s - float(s.mean())) / sd
    return out.clip(-clip, clip)


def _factorize_norm(s: pd.Series) -> Tuple[pd.Series, pd.Series, Dict[str, int]]:
    ss = s.astype(str).fillna("MISSING")
    codes, uniques = pd.factorize(ss)
    denom = max(len(uniques) - 1, 1)
    code_norm = pd.Series(codes.astype(float) / denom, index=s.index)
    freq = ss.map(ss.value_counts(normalize=True)).fillna(0.0)
    mapping = {str(u): int(i) for i, u in enumerate(uniques)}
    return code_norm, freq, mapping


def _rolling_mean(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values


def _rolling_std(arr, window):
    return pd.Series(arr).rolling(window, min_periods=2).std().fillna(0).values


def _rolling_max_lag(arr, window):
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]
        out[i] = vals.max() if len(vals) > 0 else 0.0
    return out


def _rolling_positive_mean(arr, window):
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]
        vals = vals[vals > 0]
        out[i] = vals.mean() if len(vals) > 0 else 0.0
    return out


def _rolling_positive_quantile(arr, window, q):
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]
        vals = vals[vals > 0]
        out[i] = np.quantile(vals, q) if len(vals) > 0 else 0.0
    return out


def _zero_streak(active):
    out = np.zeros(len(active), dtype=np.float32)
    cur = 0
    for i, a in enumerate(active):
        if a > 0:
            cur = 0
        else:
            cur += 1
        out[i] = cur
    return out


def _oos_streak(oos):
    out = np.zeros(len(oos), dtype=np.float32)
    cur = 0
    for i, a in enumerate(oos):
        if a > 0:
            cur += 1
        else:
            cur = 0
        out[i] = cur
    return out


def _weeks_since_last_oos(oos):
    out = np.zeros(len(oos), dtype=np.float32)
    last = -1
    for i, a in enumerate(oos):
        if a > 0:
            last = i
        out[i] = (i - last) if last >= 0 else 52.0
    return out


def prepare_sample_intersection(data_raw1, scot_df=None, n_asins=5000, seed=42):
    df = data_raw1.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    rng = np.random.default_rng(seed)
    asins = df["asin"].dropna().unique()
    sample_asins = rng.choice(asins, size=min(n_asins, len(asins)), replace=False)
    sample_set = set(sample_asins)

    if scot_df is not None and "asin" in scot_df.columns:
        scot = scot_df.copy()
        scot["asin"] = scot["asin"].astype(str)
        sample_set = sample_set & set(scot["asin"].dropna().unique())

    out = df[df["asin"].isin(sample_set)].copy()
    print("Sample ASINs after intersection:", out["asin"].nunique())
    print("Sample rows:", len(out))
    return out


# =====================================================
# Data creation
# =====================================================

def _select_static_context_cols(df: pd.DataFrame) -> List[str]:
    candidates = [
        "gl_product_group", "category_code", "hbt", "ind_amxl_hb",
        "ind_top10_brand", "ind_top10_review_brand", "ind_new_asin",
        "customer_review_count", "customer_active_review_count",
        "cust_avg_active_review_rating", "customer_average_review_rating",
        "glance_view_band_cat", "hb_rank", "hb_score",
        "list_price", "our_price", "price_bands",
        "ind_promotion", "promotion_amount", "promotion_ratio",
        "promotion_pricing_amount", "promotion_type", "pricing_type",
        "pkg_height", "pkg_length", "pkg_width", "pkg_weight",
        "ind_prime_week",
    ]
    return [c for c in candidates if c in df.columns]


def _encode_context_features(df: pd.DataFrame, raw_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    out_cols = []
    for c in raw_cols:
        if c not in df.columns:
            continue
        c_low = c.lower()
        if pd.api.types.is_numeric_dtype(df[c]):
            val = pd.to_numeric(_get_1d_col(df, c), errors="coerce").fillna(0.0)
            if any(k in c_low for k in ["count", "price", "amount", "rank", "score", "height", "length", "width", "weight"]):
                val = np.log1p(val.clip(lower=0))
            new_c = f"ctx__{c}"
            df[new_c] = _standardize(val)
            out_cols.append(new_c)
        else:
            code, freq, _ = _factorize_norm(_get_1d_col(df, c))
            c_code = f"ctx__{c}__code"
            c_freq = f"ctx__{c}__freq"
            df[c_code] = code.astype(float)
            df[c_freq] = freq.astype(float)
            out_cols += [c_code, c_freq]
    return df, out_cols


def load_joint_real_data(data_raw: pd.DataFrame, dph_cap_q: float = 0.995):
    df = data_raw.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"], errors="coerce")

    for c in ["fbi_demand", "scot_oos", "total_dph", "buy_box_dph", "in_stock_dph", "our_price"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["fbi_demand", "total_dph", "buy_box_dph", "in_stock_dph", "our_price"]:
        df[c] = df[c].clip(lower=0)
    df["scot_oos"] = df["scot_oos"].fillna(0).clip(lower=0, upper=1)

    # Conservative DPH cap for heavy tails.
    pos_total = df["total_dph"].clip(lower=0)
    cap = float(pos_total.quantile(dph_cap_q)) if pos_total.sum() > 0 else np.inf
    if np.isfinite(cap) and cap > 0:
        for c in ["total_dph", "buy_box_dph", "in_stock_dph"]:
            df[c] = df[c].clip(upper=cap)

    # Calendar context.
    df["order_month"] = df["order_week"].dt.month.fillna(1).astype(float)
    df["week_index"] = ((df["order_week"] - df["order_week"].min()).dt.days // 7).fillna(0).astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["week_index"] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df["week_index"] / 52.0)
    df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12.0)
    df["season_winter"] = df["order_month"].isin([12, 1, 2]).astype(float)
    df["season_summer"] = df["order_month"].isin([6, 7, 8]).astype(float)

    # Holiday/promo context.
    holiday_cols = [c for c in df.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in df.columns if c.startswith("distance_")]
    for c in holiday_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(0, 1)
    for c in distance_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(-12, 12) / 12.0

    if "ind_promotion" in df.columns:
        df["promo_t"] = pd.to_numeric(df["ind_promotion"], errors="coerce").fillna(0.0).clip(0, 1)
    elif holiday_cols:
        df["promo_t"] = df[holiday_cols].max(axis=1)
    else:
        df["promo_t"] = 0.0

    static_cols = _select_static_context_cols(df)
    df, encoded_static_cols = _encode_context_features(df, static_cols)

    # Future context excludes true DPH/OOS. OOS is historical only to avoid leakage.
    context_cols = [
        "our_price", "week_sin", "week_cos", "month_sin", "month_cos",
        "season_winter", "season_summer", "promo_t",
    ] + holiday_cols + distance_cols + encoded_static_cols
    context_cols = list(dict.fromkeys([c for c in context_cols if c in df.columns]))

    # Normalize/log price context.
    df["our_price"] = np.log1p(df["our_price"].clip(lower=0))
    df["our_price"] = _standardize(df["our_price"])

    # Add forecast-origin-safe historical DPH proxy columns, filled in Dataset.
    dph_proxy_cols = [
        "hist_total_dph_last_log", "hist_total_dph_mean4_log", "hist_total_dph_mean13_log",
        "hist_buy_box_dph_last_log", "hist_buy_box_dph_mean4_log", "hist_buy_box_dph_mean13_log",
        "hist_instock_dph_last_log", "hist_instock_dph_mean4_log", "hist_instock_dph_mean13_log",
        "hist_demand_last_log", "hist_demand_mean4_log", "hist_demand_mean13_log",
        "hist_demand_active_rate", "hist_oos_rate13", "hist_oos_last",
    ]
    for c in dph_proxy_cols:
        df[c] = 0.0
    context_cols = context_cols + dph_proxy_cols

    df = df.sort_values(["asin", "order_week"]).reset_index(drop=True)

    data = {}
    for asin, g in df.groupby("asin", sort=False):
        g = g.reset_index(drop=True)
        demand = g["fbi_demand"].values.astype(float)
        active = (demand > 0).astype(float)
        total = g["total_dph"].values.astype(float)
        buy = g["buy_box_dph"].values.astype(float)
        instock = g["in_stock_dph"].values.astype(float)
        oos = g["scot_oos"].values.astype(float)
        t = g["week_index"].values.astype(int)

        dist_last = np.zeros(len(demand), dtype=np.float32)
        last = -1
        for i, a in enumerate(active):
            if a > 0:
                last = i
            dist_last[i] = (i - last) / 52.0 if last >= 0 else 1.0

        hist_nonzero_mean_52 = _rolling_positive_mean(demand, 52)
        hist_nonzero_p75_52 = _rolling_positive_quantile(demand, 52, 0.75)
        recent_peak_13 = _rolling_max_lag(demand, 13)
        active_rate_4 = _rolling_mean(active, 4)
        active_rate_13 = _rolling_mean(active, 13)
        oos_rate_4 = _rolling_mean(oos, 4)
        oos_rate_13 = _rolling_mean(oos, 13)
        instock_mean_4 = _rolling_mean(instock, 4)
        instock_mean_13 = _rolling_mean(instock, 13)
        total_mean_4 = _rolling_mean(total, 4)
        total_mean_13 = _rolling_mean(total, 13)
        buy_mean_4 = _rolling_mean(buy, 4)
        buy_mean_13 = _rolling_mean(buy, 13)
        zero_streak = _zero_streak(active) / 52.0
        pos_mean_4 = _rolling_positive_mean(demand, 4)
        pos_mean_13 = _rolling_positive_mean(demand, 13)
        pos_max_13 = _rolling_max_lag(demand, 13)
        pos_std_13 = _rolling_std(np.log1p(demand), 13)
        buy_box_rate = np.clip(buy / (total + 1.0), 0.0, 10.0)
        instock_rate = np.clip(instock / (total + 1.0), 0.0, 10.0)
        instock_given_buy = np.clip(instock / (buy + 1.0), 0.0, 10.0)

        features = np.stack([
            np.log1p(demand), active, dist_last,
            np.sin(2 * np.pi * t / 52.0), np.cos(2 * np.pi * t / 52.0),
            g["promo_t"].values.astype(float),
            np.sin(2 * np.pi * t / 13.0), np.cos(2 * np.pi * t / 13.0),
            np.log1p(hist_nonzero_mean_52), np.log1p(hist_nonzero_p75_52), np.log1p(recent_peak_13),
            np.log1p(instock), oos, active_rate_4, active_rate_13, oos_rate_4, oos_rate_13,
            np.log1p(instock_mean_4), np.log1p(instock_mean_13), zero_streak,
            g["our_price"].values.astype(float),
            np.log1p(pos_mean_4), np.log1p(pos_mean_13), np.log1p(pos_max_13), pos_std_13,
            np.log1p(total), np.log1p(buy), np.log1p(total_mean_4), np.log1p(total_mean_13),
            np.log1p(buy_mean_4), np.log1p(buy_mean_13),
            buy_box_rate, instock_rate, instock_given_buy,
        ], axis=1).astype(np.float32)

        data[asin] = {
            "features": features,
            "future_context": g[context_cols].values.astype(np.float32),
            "demand": demand.astype(np.float32),
            "week": g["order_week"].values,
            "oos": oos.astype(np.float32),
            "total_dph": total.astype(np.float32),
            "buy_box_dph": buy.astype(np.float32),
            "in_stock_dph": instock.astype(np.float32),
            "price_raw": np.expm1(g["our_price"].values.astype(float)).astype(np.float32),
            "dph_proxy_context_idx": {c: context_cols.index(c) for c in dph_proxy_cols if c in context_cols},
        }

    print("History encoder dim: 34")
    print("Context dim:", len(context_cols))
    print("Context includes category/GL/HBT/top10/review if present; excludes future true OOS/DPH.")
    return data, len(context_cols), context_cols, df


# =====================================================
# Dataset
# =====================================================

class JointDemandExposureDataset(Dataset):
    def __init__(self, data: Dict, asin_to_idx: Dict[str, int], history=52, horizon=20, mode="train", val_weeks=20):
        self.samples = []
        self.history = history
        self.horizon = horizon
        for asin, d in data.items():
            T = len(d["demand"])
            if mode == "train":
                max_start = max(0, T - val_weeks - horizon - history + 1)
                starts = range(max_start)
            else:
                s = T - history - horizon
                starts = [s] if s >= 0 else []

            for start in starts:
                y_slice = slice(start + history, start + history + horizon)
                self.samples.append({
                    "x": torch.tensor(d["features"][start:start+history], dtype=torch.float32),
                    "future_context": torch.tensor(self._make_future_context(d, start, history, horizon), dtype=torch.float32),
                    "y": torch.tensor(d["demand"][y_slice], dtype=torch.float32),
                    "y_exp": torch.tensor(np.stack([
                        d["total_dph"][y_slice],
                        d["buy_box_dph"][y_slice],
                        d["in_stock_dph"][y_slice],
                    ], axis=-1), dtype=torch.float32),
                    "asin": asin,
                    "asin_idx": torch.tensor(asin_to_idx[asin], dtype=torch.long),
                    "target_week": [str(w)[:10] for w in d["week"][y_slice]],
                    "oos": torch.tensor(d["oos"][y_slice], dtype=torch.float32),
                    "future_total_dph": torch.tensor(d["total_dph"][y_slice], dtype=torch.float32),
                    "future_buy_box_dph": torch.tensor(d["buy_box_dph"][y_slice], dtype=torch.float32),
                    "future_instock": torch.tensor(d["in_stock_dph"][y_slice], dtype=torch.float32),
                })

    def _hist_mean(self, arr, start, history, window):
        hist = arr[start:start+history]
        if len(hist) == 0:
            return 0.0
        hist = hist[-min(window, len(hist)):]
        return float(np.mean(hist))

    def _make_future_context(self, d, start, history, horizon):
        fc = d["future_context"][start+history:start+history+horizon].copy()
        idx = d.get("dph_proxy_context_idx", {})
        def fill_log(col, val):
            if col in idx:
                fc[:, idx[col]] = np.log1p(max(float(val), 0.0))
        def fill_raw(col, val):
            if col in idx:
                fc[:, idx[col]] = float(val)

        total, buy, instock, demand, oos = d["total_dph"], d["buy_box_dph"], d["in_stock_dph"], d["demand"], d["oos"]
        origin = start + history - 1
        if origin >= 0:
            fill_log("hist_total_dph_last_log", total[origin])
            fill_log("hist_buy_box_dph_last_log", buy[origin])
            fill_log("hist_instock_dph_last_log", instock[origin])
            fill_log("hist_demand_last_log", demand[origin])
            fill_raw("hist_oos_last", oos[origin])
        fill_log("hist_total_dph_mean4_log", self._hist_mean(total, start, history, 4))
        fill_log("hist_total_dph_mean13_log", self._hist_mean(total, start, history, 13))
        fill_log("hist_buy_box_dph_mean4_log", self._hist_mean(buy, start, history, 4))
        fill_log("hist_buy_box_dph_mean13_log", self._hist_mean(buy, start, history, 13))
        fill_log("hist_instock_dph_mean4_log", self._hist_mean(instock, start, history, 4))
        fill_log("hist_instock_dph_mean13_log", self._hist_mean(instock, start, history, 13))
        fill_log("hist_demand_mean4_log", self._hist_mean(demand, start, history, 4))
        fill_log("hist_demand_mean13_log", self._hist_mean(demand, start, history, 13))
        fill_raw("hist_demand_active_rate", np.mean(demand[start:start+history] > 0) if history > 0 else 0.0)
        fill_raw("hist_oos_rate13", self._hist_mean(oos, max(start, start+history-13), min(history, 13), 13))
        return fc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# =====================================================
# Graph construction
# =====================================================

def _mode_or_missing(s):
    ss = s.dropna().astype(str)
    if len(ss) == 0:
        return "MISSING"
    return ss.mode().iloc[0]


def build_dual_graph_inputs(raw_df_for_graph: pd.DataFrame, all_asins: List[str], neighbor_k=10):
    df = raw_df_for_graph.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    for c in ["fbi_demand", "total_dph", "buy_box_dph", "in_stock_dph", "scot_oos", "customer_review_count", "customer_active_review_count", "ind_top10_brand"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["gl_product_group", "category_code", "hbt"]:
        if c not in df.columns:
            df[c] = "MISSING"

    rows = []
    for asin in all_asins:
        g = df[df["asin"] == asin].sort_values("order_week")
        if len(g) == 0:
            rows.append({"asin": asin})
            continue
        inst = g["in_stock_dph"].clip(lower=0).values
        buy = g["buy_box_dph"].clip(lower=0).values
        tot = g["total_dph"].clip(lower=0).values
        dem = g["fbi_demand"].clip(lower=0).values
        oos = g["scot_oos"].clip(0, 1).values
        active_inst = inst > 0
        active_dem = dem > 0
        pos_inst = inst[active_inst]
        def q(arr, qq):
            return float(np.quantile(arr, qq)) if len(arr) > 0 else 0.0
        rows.append({
            "asin": asin,
            "gl": _mode_or_missing(g["gl_product_group"]),
            "category": _mode_or_missing(g["category_code"]),
            "hbt": _mode_or_missing(g["hbt"]),
            "ind_top10_brand": float(g["ind_top10_brand"].iloc[-1]) if len(g) else 0.0,
            "review_log": float(np.log1p(max(g["customer_active_review_count"].iloc[-1], g["customer_review_count"].iloc[-1], 0.0))),
            "inst_zero_rate": float(np.mean(inst <= 0)),
            "buy_zero_rate": float(np.mean(buy <= 0)),
            "total_zero_rate": float(np.mean(tot <= 0)),
            "demand_active_rate": float(np.mean(active_dem)),
            "oos_rate": float(np.mean(oos)),
            "last_oos": float(oos[-1]) if len(oos) else 0.0,
            "oos_streak": float(_oos_streak(oos)[-1]) if len(oos) else 0.0,
            "weeks_since_last_oos": float(_weeks_since_last_oos(oos)[-1]) if len(oos) else 52.0,
            "inst_mean_log": float(np.log1p(np.mean(inst))),
            "inst_q75_log": float(np.log1p(q(inst, 0.75))),
            "inst_q90_log": float(np.log1p(q(inst, 0.90))),
            "inst_q95_log": float(np.log1p(q(inst, 0.95))),
            "inst_max_log": float(np.log1p(np.max(inst) if len(inst) else 0.0)),
            "active_only_mean_log": float(np.log1p(np.mean(pos_inst))) if len(pos_inst) else 0.0,
            "active_only_q90_log": float(np.log1p(q(pos_inst, 0.90))) if len(pos_inst) else 0.0,
            "active_only_q95_log": float(np.log1p(q(pos_inst, 0.95))) if len(pos_inst) else 0.0,
            "buy_mean_log": float(np.log1p(np.mean(buy))),
            "total_mean_log": float(np.log1p(np.mean(tot))),
            "cv_instock": float(np.std(inst) / (np.mean(inst) + 1.0)),
            "top20_share": float(np.sort(inst)[-max(1, int(len(inst) * 0.2)):].sum() / (inst.sum() + 1.0)) if len(inst) else 0.0,
            "active_to_zero_rate": float(np.mean((inst[:-1] > 0) & (inst[1:] <= 0))) if len(inst) > 1 else 0.0,
            "zero_to_active_rate": float(np.mean((inst[:-1] <= 0) & (inst[1:] > 0))) if len(inst) > 1 else 0.0,
        })
    node_df = pd.DataFrame(rows).fillna(0.0)

    # Encode categorical graph columns.
    for c in ["gl", "category", "hbt"]:
        code, freq, _ = _factorize_norm(node_df[c].astype(str))
        node_df[f"{c}_code"] = code
        node_df[f"{c}_freq"] = freq
    hbt_str = node_df["hbt"].astype(str).str.lower()
    node_df["hbt_is_head"] = hbt_str.str.contains("head|h").astype(float)
    node_df["hbt_is_body"] = hbt_str.str.contains("body|b").astype(float)
    node_df["hbt_is_tail"] = hbt_str.str.contains("tail|t").astype(float)

    feature_cols = [
        "inst_zero_rate", "buy_zero_rate", "total_zero_rate", "demand_active_rate",
        "oos_rate", "last_oos", "oos_streak", "weeks_since_last_oos",
        "inst_mean_log", "inst_q75_log", "inst_q90_log", "inst_q95_log", "inst_max_log",
        "active_only_mean_log", "active_only_q90_log", "active_only_q95_log",
        "buy_mean_log", "total_mean_log", "cv_instock", "top20_share",
        "active_to_zero_rate", "zero_to_active_rate",
        "gl_code", "gl_freq", "category_code", "category_freq",
        "hbt_code", "hbt_freq", "hbt_is_head", "hbt_is_body", "hbt_is_tail",
        "ind_top10_brand", "review_log",
    ]
    X = node_df[feature_cols].astype(float).copy()
    for c in X.columns:
        X[c] = _standardize(X[c])
    node_feat = X.values.astype(np.float32)
    N = len(node_df)
    K = min(neighbor_k, max(1, N - 1))

    # Positive neighbors: KNN over balanced node feature representation.
    if N <= 1:
        pos_idx = np.zeros((N, neighbor_k), dtype=np.int64)
        comp_idx = np.zeros((N, neighbor_k), dtype=np.int64)
    else:
        nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
        nn.fit(node_feat)
        neigh = nn.kneighbors(node_feat, return_distance=False)[:, 1:K+1]
        if K < neighbor_k:
            pad = np.repeat(neigh[:, -1:], neighbor_k - K, axis=1)
            neigh = np.concatenate([neigh, pad], axis=1)
        pos_idx = neigh.astype(np.int64)

        # Competitive neighbors: same category preferred, stronger/high-review/head/top-brand items.
        strength = (
            1.2 * node_df["active_only_q95_log"].astype(float).values
            + 0.8 * node_df["review_log"].astype(float).values
            + 0.7 * node_df["ind_top10_brand"].astype(float).values
            + 0.7 * node_df["hbt_is_head"].astype(float).values
            + 0.4 * node_df["inst_q95_log"].astype(float).values
        )
        comp = np.zeros((N, neighbor_k), dtype=np.int64)
        cats = node_df["category"].astype(str).values
        gls = node_df["gl"].astype(str).values
        hbts = node_df["hbt"].astype(str).values
        for i in range(N):
            same_cat = np.where((cats == cats[i]) & (np.arange(N) != i))[0]
            if len(same_cat) < K:
                same_gl = np.where((gls == gls[i]) & (np.arange(N) != i))[0]
                cand = np.unique(np.concatenate([same_cat, same_gl]))
            else:
                cand = same_cat
            if len(cand) == 0:
                cand = np.where(np.arange(N) != i)[0]
            diff_hbt = (hbts[cand] != hbts[i]).astype(float)
            stronger = strength[cand] - strength[i]
            score = stronger + 0.5 * diff_hbt
            order = cand[np.argsort(-score)]
            take = order[:K]
            if len(take) < neighbor_k:
                take = np.concatenate([take, np.repeat(take[-1] if len(take) else i, neighbor_k-len(take))])
            comp[i] = take[:neighbor_k]
        comp_idx = comp

    asin_to_idx = {asin: i for i, asin in enumerate(node_df["asin"].astype(str).tolist())}
    print("Graph nodes:", N, "| node_feat_dim:", node_feat.shape[1], "| K:", neighbor_k)
    print("Graph feature includes category_code, GL, hbt, top10 brand, review, OOS summaries, exposure summaries.")
    return node_feat, pos_idx, comp_idx, asin_to_idx, node_df


# =====================================================
# Model components
# =====================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class SparsePeakAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4, beta_peak=1.0, soft_mask_scale=3.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.beta_peak = beta_peak
        self.soft_mask_scale = soft_mask_scale
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, active, peak_score):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        sparse_mask = (active == 0) & ~(active == 0).all(dim=1, keepdim=True)
        scores = scores - self.soft_mask_scale * sparse_mask.float()[:, None, None, :]
        peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)
        scores = scores + self.beta_peak * peak_norm[:, None, None, :]
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.norm(x + self.out_proj(out))


class SharedHistoryEncoder(nn.Module):
    def __init__(self, input_dim=34, d_model=32, horizon=20):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, d_model)
        dilations = [1, 2, 4, 8, 13, 26, 52]
        self.convs = nn.ModuleList([CausalConv1d(d_model, d_model, 2, d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])
        self.sparse_attn = SparsePeakAttention(d_model, n_heads=4, beta_peak=1.0)
        self.final_norm = nn.LayerNorm(d_model)
        self.base_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))
        self.alpha_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))
    def forward(self, x):
        active = x[:, :, 1]
        peak_score = torch.sqrt(torch.expm1(x[:, :, 0]).clamp(min=0) + 1e-6)
        h = self.input_proj(x).permute(0, 2, 1)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h) + h
            h = h.permute(0, 2, 1)
            h = F.gelu(norm(h))
            h = h.permute(0, 2, 1)
        enc_seq = self.sparse_attn(h.permute(0, 2, 1), active, peak_score)
        h_t = self.final_norm(enc_seq[:, -1, :])
        mu_base = F.softplus(self.base_head(h_t))
        alpha_base = F.softplus(self.alpha_head(h_t)) + 1e-4
        return mu_base, alpha_base, h_t, enc_seq


class DualGraphEncoder(nn.Module):
    def __init__(self, node_feat, pos_idx, comp_idx, graph_dim=16, dropout=0.05, message_scale=0.10):
        super().__init__()
        self.register_buffer("node_feat", torch.tensor(node_feat, dtype=torch.float32))
        self.register_buffer("pos_idx", torch.tensor(pos_idx, dtype=torch.long))
        self.register_buffer("comp_idx", torch.tensor(comp_idx, dtype=torch.long))
        in_dim = node_feat.shape[1]
        self.self_proj = nn.Linear(in_dim, graph_dim)
        self.pos_proj = nn.Linear(in_dim, graph_dim)
        self.comp_proj = nn.Linear(in_dim, graph_dim)
        self.out = nn.Sequential(
            nn.Linear(graph_dim * 3, graph_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, graph_dim),
            nn.LayerNorm(graph_dim),
        )
        self.message_scale = message_scale
    def forward_all(self):
        x = self.node_feat
        pos_mean = x[self.pos_idx].mean(dim=1)
        comp_mean = x[self.comp_idx].mean(dim=1)
        h_self = self.self_proj(x)
        h_pos = self.pos_proj(pos_mean) * self.message_scale
        h_comp = self.comp_proj(comp_mean) * self.message_scale
        return self.out(torch.cat([h_self, h_pos, h_comp], dim=-1))
    def forward(self, asin_idx):
        all_g = self.forward_all()
        return all_g[asin_idx]


class ExposureCovariateDecoder(nn.Module):
    def __init__(self, d_model, context_dim, graph_dim=16, hidden_dim=32, horizon=20):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        inp = d_model + context_dim + graph_dim
        self.net = nn.Sequential(
            nn.Linear(inp, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 96), nn.ReLU(),
        )
        self.hidden_proj = nn.Linear(96, hidden_dim)
        self.exp_head = nn.Linear(hidden_dim, 3)
        self.active_head = nn.Linear(hidden_dim, 3)
    def forward(self, h_t, future_context, graph_emb):
        B, H, _ = future_context.shape
        h_rep = h_t[:, None, :].expand(B, H, -1)
        g_rep = graph_emb[:, None, :].expand(B, H, -1)
        z = self.net(torch.cat([h_rep, future_context, g_rep], dim=-1))
        hidden = F.gelu(self.hidden_proj(z))
        pred_exp = F.softplus(self.exp_head(hidden))
        active_logits = self.active_head(hidden)
        # Enforce funnel: total >= buy_box >= in_stock without in-place ops.
        total = pred_exp[:, :, 0:1].clamp_min(0)
        buy = torch.minimum(pred_exp[:, :, 1:2].clamp_min(0), total)
        instock = torch.minimum(pred_exp[:, :, 2:3].clamp_min(0), buy)
        pred_exp = torch.cat([total, buy, instock], dim=-1)
        return pred_exp, active_logits, hidden


class ContextZGenerator(nn.Module):
    def __init__(self, d_phi=32, context_dim=2, d_z=16, horizon=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_phi + horizon * context_dim, 128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 2 * d_z),
        )
    def forward(self, phi, future_context):
        B = phi.shape[0]
        ctx = future_context.reshape(B, -1)
        out = self.net(torch.cat([phi, ctx], dim=-1))
        z_mean, z_logstd = out.chunk(2, dim=-1)
        z_std = F.softplus(z_logstd) + 1e-4
        return z_mean, z_std


class Epinet(nn.Module):
    def __init__(self, d_phi=32, d_z=16, horizon=20, prior_scale=0.3):
        super().__init__()
        self.horizon = horizon
        self.d_z = d_z
        self.prior_scale = prior_scale
        self.learnable = nn.Sequential(
            nn.Linear(d_z + d_phi, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2 * horizon * d_z),
        )
        self.prior = nn.Sequential(
            nn.Linear(d_z + d_phi, 64), nn.ReLU(),
            nn.Linear(64, 2 * horizon * d_z),
        )
        for p in self.prior.parameters():
            p.requires_grad = False
    def forward(self, phi, z):
        inp = torch.cat([z, phi], dim=-1)
        sl = self.learnable(inp).view(-1, 2 * self.horizon, self.d_z)
        sl = torch.einsum("bhd,bd->bh", sl, z)
        sp = self.prior(inp).view(-1, 2 * self.horizon, self.d_z)
        sp = torch.einsum("bhd,bd->bh", sp, z) * self.prior_scale
        out = sl + sp
        return out[:, :self.horizon], out[:, self.horizon:]


class JointDemandExposureGraphModel(nn.Module):
    def __init__(
        self,
        input_dim,
        context_dim,
        node_feat,
        pos_idx,
        comp_idx,
        d_model=32,
        d_z=16,
        horizon=20,
        graph_dim=16,
        exposure_hidden_dim=32,
        prior_scale=0.3,
        use_graph=True,
        graph_message_scale=0.10,
        detach_exposure_for_demand=True,
    ):
        super().__init__()
        self.horizon = horizon
        self.use_graph = use_graph
        self.graph_dim = graph_dim if use_graph else 0
        self.detach_exposure_for_demand = detach_exposure_for_demand
        self.encoder = SharedHistoryEncoder(input_dim, d_model, horizon)
        if use_graph:
            self.graph_encoder = DualGraphEncoder(
                node_feat=node_feat,
                pos_idx=pos_idx,
                comp_idx=comp_idx,
                graph_dim=graph_dim,
                message_scale=graph_message_scale,
            )
        else:
            self.graph_encoder = None
        self.exposure_decoder = ExposureCovariateDecoder(
            d_model=d_model,
            context_dim=context_dim,
            graph_dim=self.graph_dim,
            hidden_dim=exposure_hidden_dim,
            horizon=horizon,
        )
        # Demand augmented context receives raw future_context + graph + log exposure + p_active + exposure_hidden.
        demand_context_dim = context_dim + self.graph_dim + 3 + 3 + exposure_hidden_dim
        self.z_generator = ContextZGenerator(d_model, demand_context_dim, d_z, horizon)
        self.epinet = Epinet(d_model, d_z, horizon, prior_scale)

    def _graph_emb(self, asin_idx, device, B):
        if not self.use_graph:
            return torch.zeros(B, 0, device=device)
        return self.graph_encoder(asin_idx)

    def forward(self, x, future_context, asin_idx, nZ=8, return_aux=True):
        mu_base, alpha_base, h_t, enc_seq = self.encoder(x)
        B, H, _ = future_context.shape
        graph_emb = self._graph_emb(asin_idx, x.device, B)
        pred_exp, active_logits, exposure_hidden = self.exposure_decoder(h_t, future_context, graph_emb)
        p_active = torch.sigmoid(active_logits)

        if self.detach_exposure_for_demand:
            pred_exp_for_d = pred_exp.detach()
            p_active_for_d = p_active.detach()
            exposure_hidden_for_d = exposure_hidden.detach()
            graph_for_d = graph_emb.detach()
        else:
            pred_exp_for_d = pred_exp
            p_active_for_d = p_active
            exposure_hidden_for_d = exposure_hidden
            graph_for_d = graph_emb

        g_rep = graph_for_d[:, None, :].expand(B, H, -1) if self.use_graph else torch.zeros(B, H, 0, device=x.device)
        demand_context = torch.cat([
            future_context,
            g_rep,
            torch.log1p(pred_exp_for_d).clamp(min=0.0),
            p_active_for_d,
            exposure_hidden_for_d,
        ], dim=-1)

        phi = h_t.detach()
        z_mean, z_std = self.z_generator(phi, demand_context)
        z_reg = 0.001 * (z_mean ** 2 + z_std ** 2).mean()
        preds = []
        for _ in range(nZ):
            eps = torch.randn_like(z_mean)
            z = z_mean + z_std * eps
            mu_e, al_e = self.epinet(phi, z)
            mu = F.softplus(mu_base + mu_e)
            alpha = F.softplus(alpha_base + al_e) + 1e-4
            preds.append((mu, alpha))
        aux = {
            "pred_exp": pred_exp,
            "active_logits": active_logits,
            "p_active": p_active,
            "exposure_hidden": exposure_hidden,
            "graph_emb": graph_emb,
            "demand_context": demand_context,
        }
        return preds, z_reg, aux

    def predict(self, x, future_context, asin_idx, M=50, return_aux=False):
        self.eval()
        with torch.no_grad():
            mu_base, alpha_base, h_t, enc_seq = self.encoder(x)
            B, H, _ = future_context.shape
            graph_emb = self._graph_emb(asin_idx, x.device, B)
            pred_exp, active_logits, exposure_hidden = self.exposure_decoder(h_t, future_context, graph_emb)
            p_active = torch.sigmoid(active_logits)
            g_rep = graph_emb[:, None, :].expand(B, H, -1) if self.use_graph else torch.zeros(B, H, 0, device=x.device)
            demand_context = torch.cat([
                future_context,
                g_rep,
                torch.log1p(pred_exp).clamp(min=0.0),
                p_active,
                exposure_hidden,
            ], dim=-1)
            phi = h_t.detach()
            z_mean, z_std = self.z_generator(phi, demand_context)
            samples = []
            for _ in range(M):
                eps = torch.randn_like(z_mean)
                z = z_mean + z_std * eps
                mu_e, al_e = self.epinet(phi, z)
                mu = F.softplus(mu_base + mu_e)
                alpha = F.softplus(alpha_base + al_e) + 1e-4
                dist = torch.distributions.NegativeBinomial(
                    total_count=(1.0 / alpha).clamp(min=1e-4),
                    probs=(mu * alpha / (1 + mu * alpha)).clamp(1e-6, 1 - 1e-6),
                )
                samples.append(dist.sample().float())
            samples = torch.stack(samples, dim=1)
            p50 = samples.quantile(0.5, dim=1)
            p70 = samples.quantile(0.7, dim=1)
            p70 = torch.maximum(p70, p50)
            aux = {
                "pred_exp": pred_exp,
                "active_logits": active_logits,
                "p_active": p_active,
                "exposure_hidden": exposure_hidden,
                "graph_emb": graph_emb,
            }
        if return_aux:
            return p50, p70, aux
        return p50, p70


# =====================================================
# Losses
# =====================================================

def negbin_nll_elementwise(y, mu, alpha):
    eps = 1e-6
    r = (1.0 / alpha).clamp(min=eps)
    p = (mu * alpha / (1 + mu * alpha)).clamp(eps, 1 - eps)
    return -(
        torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
        + r * torch.log(1 - p) + y * torch.log(p)
    )


def tail_weighted_negbin_nll(y, mu, alpha, beta_tail=0.5):
    nll = negbin_nll_elementwise(y, mu, alpha)
    weight = 1.0 + beta_tail * torch.log1p(y)
    return (nll * weight).sum() / weight.sum().clamp(min=1.0)


def pinball(y, pred, q):
    d = y - pred
    return torch.mean(torch.max(q * d, (q - 1) * d))


def exposure_aux_loss(pred_exp, active_logits, y_exp):
    # Magnitude loss on log scale keeps exposure auxiliary stable.
    log_pred = torch.log1p(pred_exp)
    log_true = torch.log1p(y_exp.clamp(min=0))
    mag_loss = F.smooth_l1_loss(log_pred, log_true)
    active_true = (y_exp > 0).float()
    # Slightly emphasize in_stock active, because demand first uses effective exposure.
    bce_all = F.binary_cross_entropy_with_logits(active_logits, active_true, reduction="none")
    weights = torch.tensor([0.5, 0.7, 1.0], device=y_exp.device).view(1, 1, 3)
    active_loss = (bce_all * weights).mean()
    return mag_loss, active_loss


# =====================================================
# Train/evaluate
# =====================================================

def _to_device_batch(b, device):
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_joint(
    model,
    tr_ld,
    va_ld,
    epochs=60,
    nZ=8,
    lr=1e-3,
    lambda_q=0.05,
    beta_tail=0.5,
    lambda_z_reg=1.0,
    lambda_exp=0.10,
    lambda_active=0.10,
    patience=8,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_demand = 0.0
        tr_exp = 0.0
        for bi, b0 in enumerate(tr_ld):
            b = _to_device_batch(b0, device)
            preds, z_reg, aux = model(b["x"], b["future_context"], b["asin_idx"], nZ=nZ)
            nll_loss = sum(tail_weighted_negbin_nll(b["y"], mu, alpha, beta_tail=beta_tail) for mu, alpha in preds) / nZ
            mu_stack = torch.stack([mu for mu, _ in preds], dim=1)
            p50_train = mu_stack.quantile(0.5, dim=1)
            p70_train = torch.maximum(mu_stack.quantile(0.7, dim=1), p50_train)
            q_loss = pinball(b["y"], p50_train, 0.5) + pinball(b["y"], p70_train, 0.7)
            mag_loss, active_loss = exposure_aux_loss(aux["pred_exp"], aux["active_logits"], b["y_exp"])
            demand_loss = nll_loss + lambda_q * q_loss + lambda_z_reg * z_reg
            loss = demand_loss + lambda_exp * mag_loss + lambda_active * active_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            tr_demand += demand_loss.item()
            tr_exp += (mag_loss.item() + active_loss.item())
            if epoch == 0 and bi < 2:
                print(f"  [batch {bi}] demand_active={(b['y']>0).float().mean().item():.3f} "
                      f"exp_instock_active={(b['y_exp'][:,:,2]>0).float().mean().item():.3f} "
                      f"pred_exp_mean={aux['pred_exp'][:,:,2].mean().item():.2f}")
        sch.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b0 in va_ld:
                b = _to_device_batch(b0, device)
                p50, p70 = model.predict(b["x"], b["future_context"], b["asin_idx"], M=50)
                vl += (pinball(b["y"], p50, 0.5) + pinball(b["y"], p70, 0.7)).item()
        vl /= max(1, len(va_ld))
        improved = vl < best_val
        if improved:
            best_val = vl
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f"Epoch {epoch+1:3d} | train={tr_loss/max(1,len(tr_ld)):.4f} "
              f"| demand={tr_demand/max(1,len(tr_ld)):.4f} | exp_aux={tr_exp/max(1,len(tr_ld)):.4f} "
              f"| val={vl:.4f}" + (" *" if improved else ""))
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break
    if best_sd is not None:
        model.load_state_dict(best_sd)
    print(f"Best demand val: {best_val:.4f}")
    return model


def generate_joint_forecast_df(model, va_ld, M=100, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for b0 in va_ld:
            b = _to_device_batch(b0, device)
            p50, p70, aux = model.predict(b["x"], b["future_context"], b["asin_idx"], M=M, return_aux=True)
            pred_exp = aux["pred_exp"].detach().cpu()
            p_active = aux["p_active"].detach().cpu()
            y = b0["y"]
            y_exp = b0["y_exp"]
            for i in range(y.shape[0]):
                for h in range(y.shape[1]):
                    rows.append({
                        "asin": b0["asin"][i],
                        "order_week": pd.to_datetime(b0["target_week"][h][i]),
                        "fcst_week_index": h + 1,
                        "fbi_demand": float(y[i, h].item()),
                        "p50_amxl": float(p50.detach().cpu()[i, h].item()),
                        "p70_amxl": float(p70.detach().cpu()[i, h].item()),
                        "true_future_total_dph": float(y_exp[i, h, 0].item()),
                        "true_future_buy_box_dph": float(y_exp[i, h, 1].item()),
                        "true_future_instock": float(y_exp[i, h, 2].item()),
                        "pred_total_dph": float(pred_exp[i, h, 0].item()),
                        "pred_buy_box_dph": float(pred_exp[i, h, 1].item()),
                        "pred_instock_dph": float(pred_exp[i, h, 2].item()),
                        "p_active_total": float(p_active[i, h, 0].item()),
                        "p_active_buy_box": float(p_active[i, h, 1].item()),
                        "p_active_instock": float(p_active[i, h, 2].item()),
                        "scot_oos": float(b0["oos"][i, h].item()),
                    })
    return pd.DataFrame(rows)


def summarize_joint_results(forecast_df):
    df = forecast_df.copy()
    print("\n" + "=" * 70)
    print("JOINT MODEL SUMMARY")
    print("=" * 70)
    y = df["fbi_demand"].values
    for col in ["p50_amxl", "p70_amxl"]:
        pred = df[col].values
        mae = np.abs(y - pred).mean()
        wape = np.abs(y - pred).sum() / max(y.sum(), 1e-8)
        under = np.maximum(y - pred, 0).sum() / max(y.sum(), 1e-8)
        over = np.maximum(pred - y, 0).sum() / max(y.sum(), 1e-8)
        print(f"{col}: MAE={mae:.4f} | WAPE={wape:.4f} | under={under:.4f} | over={over:.4f}")
    for tgt, pcol in [("total", "pred_total_dph"), ("buy_box", "pred_buy_box_dph"), ("instock", "pred_instock_dph")]:
        true_col = "true_future_instock" if tgt == "instock" else f"true_future_{tgt}_dph"
        ratio = df[pcol].sum() / max(df[true_col].sum(), 1e-8)
        wape = np.abs(df[pcol] - df[true_col]).sum() / max(df[true_col].sum(), 1e-8)
        print(f"exposure {tgt}: ratio={ratio:.4f} | WAPE={wape:.4f}")
    if df["true_future_instock"].gt(0).nunique() > 1:
        try:
            auc = roc_auc_score((df["true_future_instock"] > 0).astype(int), df["p_active_instock"])
            print(f"p_active_instock AUC: {auc:.4f}")
        except Exception:
            pass
    print("=" * 70)


# =====================================================
# Main run
# =====================================================

def run_joint_demand_exposure_dualgraph(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    d_model=32,
    d_z=16,
    graph_dim=16,
    neighbor_k=10,
    exposure_hidden_dim=32,
    use_graph=True,
    graph_message_scale=0.10,
    detach_exposure_for_demand=True,
    epochs=60,
    patience=8,
    batch_size=64,
    M_eval=100,
    lr=1e-3,
    lambda_q=0.05,
    beta_tail=0.5,
    lambda_z_reg=1.0,
    lambda_exp=0.10,
    lambda_active=0.10,
    dph_cap_q=0.995,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("JOINT DEMAND-CENTERED MODEL | internal exposure covariate decoder + dual graph")
    print("=" * 80)
    print("Device:", device)
    print("Demand is main task; exposure is auxiliary covariate generator.")
    print("detach_exposure_for_demand:", detach_exposure_for_demand)

    data_small = prepare_sample_intersection(data_raw1, scot_df=scot_df, n_asins=n_asins, seed=seed)
    data, context_dim, context_cols, modeling_df = load_joint_real_data(data_small, dph_cap_q=dph_cap_q)
    all_asins = list(data.keys())

    # Build graph from historical rows before final forecast horizon cutoff, to avoid final-horizon graph leakage.
    max_week = modeling_df["order_week"].max()
    cutoff = max_week - pd.Timedelta(weeks=horizon)
    graph_df = modeling_df[modeling_df["order_week"] < cutoff].copy()
    if len(graph_df) == 0:
        graph_df = modeling_df.copy()
    node_feat, pos_idx, comp_idx, asin_to_idx, node_df = build_dual_graph_inputs(graph_df, all_asins, neighbor_k=neighbor_k)

    tr_ds = JointDemandExposureDataset(data, asin_to_idx, history=history, horizon=horizon, mode="train", val_weeks=horizon)
    va_ds = JointDemandExposureDataset(data, asin_to_idx, history=history, horizon=horizon, mode="val", val_weeks=horizon)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    print("Train samples:", len(tr_ds), "| Val samples:", len(va_ds))

    model = JointDemandExposureGraphModel(
        input_dim=34,
        context_dim=context_dim,
        node_feat=node_feat,
        pos_idx=pos_idx,
        comp_idx=comp_idx,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        graph_dim=graph_dim,
        exposure_hidden_dim=exposure_hidden_dim,
        use_graph=use_graph,
        graph_message_scale=graph_message_scale,
        detach_exposure_for_demand=detach_exposure_for_demand,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} | context_dim={context_dim} | graph_dim={graph_dim} | exp_hidden={exposure_hidden_dim}")
    print(f"loss weights: lambda_exp={lambda_exp}, lambda_active={lambda_active}, lambda_q={lambda_q}")

    train_joint(
        model, tr_ld, va_ld,
        epochs=epochs,
        nZ=8,
        lr=lr,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        lambda_z_reg=lambda_z_reg,
        lambda_exp=lambda_exp,
        lambda_active=lambda_active,
        patience=patience,
        device=device,
    )

    forecast_df = generate_joint_forecast_df(model, va_ld, M=M_eval, device=device)
    summarize_joint_results(forecast_df)
    return {
        "model": model,
        "forecast_df": forecast_df,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "data": data,
        "context_cols": context_cols,
        "node_df": node_df,
        "asin_to_idx": asin_to_idx,
    }


if __name__ == "__main__":
    print("This file defines run_joint_demand_exposure_dualgraph(...).")
    print("Run it inside your notebook with data_raw1/scot_df already loaded.")
