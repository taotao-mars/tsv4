# VERSION V7: category-only graph + GL static + optional small graph-delta head + ASIN 20w sum loss.
# V5: adds historical DEMAND profile features to graph strength / KNN node features / edge scoring.
# Only consumes data_raw1/scot_df and outputs three exposure hats for downstream demand.
# SAFE version: renamed generic prepare_data_* functions to prepare_exposure_data_*
# to avoid namespace collision with demand model functions when using %run -i.
# ============================================================
# TCN Exposure Model V3 - SINGLE-HEAD PATCH + MAGNITUDE-AWARE DUAL GRAPH + ASIN-SUM LOSS
# Patch decoder + long-horizon weighted loss + magnitude-aware positive/competitive graph + compact diagnostics:
#   - remove two-head p_active^gamma * magnitude combination
#   - predict log1p(total/buy_box/in_stock DPH) directly with one exposure head
#   - keep a small auxiliary active head only for diagnostics / representation learning
#   - keep GL diagnostics and final summary table
#   - add category_code code/frequency/unknown static features without changing run input API
#   - add ENN one-z-per-window regime conditioning WITHOUT multiplicative active gate
#   - add path-level peak/top-k/under-peak losses to protect high exposure regime
# Purpose: stabilize point exposure forecasts and learn joint 20-week exposure regimes.
# Long-run balanced preset:
#   - category_code is kept
#   - channel-specific zero loss is softened to avoid systematic underprediction
#   - mean-level penalty is slightly stronger to keep overall ratio near 1
#   - high-exposure weighting is slightly stronger to protect Q5/peak ASINs
#   - optional dual-relation DualRelationalGAT ASIN graph embedding for exposure/click relation modeling
#     positive edges capture co-movement / similar products
#     competitive edges capture same-category attention competition
#     node/edge features include GL/category + hbt + ind_top10_brand + customer_active_review_count
#       + zero/peak/transition history

#
# 改动：
#   1. HistoryEncoder 保留全序列输出 [B, 52, D]（原来只取最后一步）
#   2. Decoder 加 Cross-Attention：Q=decoder, K=V=encoder全序列
#   3. _make_future_context 加 horizon decay，anchor不再是常数
#   4. exposure_loss 加 Hurdle：BCE(occurrence) + Huber(magnitude)
#   5. 去掉 TFT / AnchorAttentionBlender / grid_search_blending
#
# 不变：
#   数据加载、ExposureDataset、评估函数、训练loop接口
#   forward(x, future_context) → log_hat [B, H, 3]
#   explicit NB sampling inference: mc_reduce="p50"/"nb_sample" uses alpha_head
# ============================================================


# ============================================================
# V8 GAT PATCH
# - Keeps category-aware KNN/competitive edge construction unchanged.
# - Replaces DualGraphSAGE mean aggregation with DualRelationalGAT attention aggregation.
# - API is backward-compatible: use_graphsage=True now enables the GAT graph encoder.
# - Recommended first ablation: use_graphsage=True, use_graph_head=False, graph_message_scale=0.03~0.05.
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# GPU / device helpers
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
print(f"Using device: {DEVICE}")
if USE_CUDA:
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

def get_device(device=None):
    if device is None:
        return DEVICE
    return torch.device(device)

def batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def dataloader_pin_memory():
    return bool(USE_CUDA)


# ============================================================
# 原有工具函数（不变）
# ============================================================

def _safe_numeric(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def fill_missing_dph_after_scot_merge(df, verbose=True):
    """
    Clean exposure DPH targets after sampling / SCOT intersection.

    Business rule: missing DPH means no observed exposure for that ASIN-week,
    so set missing total_dph / buy_box_dph / in_stock_dph to 0.
    This is done before extreme filtering and model feature construction so that
    all downstream zero-rate / GL / category diagnostics are consistent.
    """
    out = df.copy()
    dph_cols = ["total_dph", "buy_box_dph", "in_stock_dph"]
    existing = [c for c in dph_cols if c in out.columns]

    if verbose and existing:
        na_before = out[existing].isna().sum()

    for c in existing:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Optional but useful: if demand is missing, also treat as zero observed demand.
    if "fbi_demand" in out.columns:
        out["fbi_demand"] = pd.to_numeric(out["fbi_demand"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if verbose and existing:
        na_after = out[existing].isna().sum()
        filled = (na_before - na_after).astype(int)
        print("DPH null → 0 after sample/SCOT step:", filled.to_dict())

    return out

def _wape(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.sum(np.abs(y - p)) / (np.sum(np.abs(y)) + 1e-8)

def _corr(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return np.nan
    return np.corrcoef(y, p)[0, 1]

def _safe_spearman(y, p):
    y = pd.Series(np.asarray(y, dtype=float)).rank(method="average").values
    p = pd.Series(np.asarray(p, dtype=float)).rank(method="average").values
    if np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])

def _auc(y_binary, score):
    try:
        if len(np.unique(y_binary)) < 2:
            return np.nan
        return roc_auc_score(y_binary, score)
    except Exception:
        return np.nan


# ============================================================
# 数据加载（不变，完整保留）
# ============================================================

def prepare_exposure_data_from_sample(
    data_raw1, scot_df=None, n_asins=5000, seed=42,
):
    """
    直接从data_raw1采样n_asins个ASIN，不再做SCOT intersection。

    原因：SCOT intersection把5000个ASIN压缩到~3000，
    减少了训练样本量，增加了过拟合风险。
    现在直接用5000个ASIN，数据量更大，泛化更好。

    scot_df参数保留但不使用，保持接口兼容。
    """
    df = data_raw1.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])

    rng = np.random.default_rng(seed)
    unique_asins = df["asin"].dropna().unique()
    sample_asins = rng.choice(
        unique_asins,
        size=min(n_asins, len(unique_asins)),
        replace=False,
    )

    out = df[df["asin"].isin(set(sample_asins))].copy()
    out = fill_missing_dph_after_scot_merge(out, verbose=True)
    print(f"Sampled ASINs: {len(sample_asins)} | Rows: {len(out)}")
    return out


# 向后兼容：保留旧函数名
def prepare_exposure_data_from_sample_scot_intersection(
    data_raw1, scot_df=None, n_asins=5000, seed=42,
):
    return prepare_exposure_data_from_sample(data_raw1, scot_df, n_asins, seed)


def filter_extreme_asins(data_raw, q=0.99):
    df = data_raw.copy()
    stats = (
        df.groupby("asin")
        .agg(
            max_demand=("fbi_demand", "max"),
            max_total_dph=("total_dph", "max"),
            max_buy_box_dph=("buy_box_dph", "max"),
            max_instock_dph=("in_stock_dph", "max"),
        )
        .reset_index()
    )
    thresholds = {c: stats[c].quantile(q) for c in ["max_demand", "max_total_dph", "max_buy_box_dph", "max_instock_dph"]}
    keep = stats[
        (stats["max_demand"] <= thresholds["max_demand"]) &
        (stats["max_total_dph"] <= thresholds["max_total_dph"]) &
        (stats["max_buy_box_dph"] <= thresholds["max_buy_box_dph"]) &
        (stats["max_instock_dph"] <= thresholds["max_instock_dph"])
    ]["asin"]
    out = df[df["asin"].isin(set(keep))].copy()
    print(f"Extreme filter: {df['asin'].nunique()} → {out['asin'].nunique()} ASINs")
    return out


def _encode_static_features(df):
    """
    Static ASIN-level features encoding.

    新增：
      glance_view_band_cat → /6 归一化（值1-6，完全静态）
      hbt                  → head=1 / body=0
      ind_amxl_hb          → binary，直接用
      sort_type            → /3 归一化
      ind_new_asin         → binary，直接用
      ind_amxl_hb          → binary
    """
    df = df.copy()
    out_cols = []

    # ── 原有：gl_product_group / ind_top10_brand
    # ── 新增：category_code（细粒度品类；比GL更细，用于zero/seasonality分层）────
    for c in ["gl_product_group", "category_code", "ind_top10_brand"]:
        if c not in df.columns:
            continue

        raw = df[c].astype(str).fillna("MISSING").str.strip()
        raw = raw.replace({"": "MISSING", "nan": "MISSING", "None": "MISSING", "none": "MISSING"})

        # category_code 中 unknown 本身是强信号：catalog缺失/长尾/不稳定。
        # 保留为单独静态特征，尤其帮助zero判断。
        if c == "category_code":
            lower = raw.str.lower()
            df["stock_static__category_code__is_unknown"] = (
                lower.isin(["unknown", "missing", "nan", "none", ""] )
            ).astype(float)

        codes, uniques = pd.factorize(raw)
        denom = max(len(uniques) - 1, 1)
        df[f"stock_static__{c}__code"] = codes.astype(float) / denom
        freq = raw.value_counts(normalize=True)
        df[f"stock_static__{c}__freq"] = raw.map(freq).fillna(0.0).astype(float)
        out_cols.extend([f"stock_static__{c}__code", f"stock_static__{c}__freq"])

        if c == "category_code":
            out_cols.append("stock_static__category_code__is_unknown")

    # ── 新增：glance_view_band_cat（值1-6，静态）─────────────
    if "glance_view_band_cat" in df.columns:
        gv = _safe_numeric(df["glance_view_band_cat"]).clip(1, 6)
        df["stock_static__glance_view_band__norm"] = gv / 6.0
        out_cols.append("stock_static__glance_view_band__norm")

    # ── 新增：hbt（head=1 / body=0，静态）────────────────────
    if "hbt" in df.columns:
        df["stock_static__hbt__is_head"] = (
            df["hbt"].astype(str).str.lower().str.strip() == "head"
        ).astype(float)
        out_cols.append("stock_static__hbt__is_head")

    # ── 新增：ind_amxl_hb（binary，静态）─────────────────────
    if "ind_amxl_hb" in df.columns:
        df["stock_static__ind_amxl_hb"] = _safe_numeric(df["ind_amxl_hb"]).clip(0, 1)
        out_cols.append("stock_static__ind_amxl_hb")

    # ── 新增：sort_type（1/2/3，静态）────────────────────────
    if "sort_type" in df.columns:
        df["stock_static__sort_type__norm"] = (
            _safe_numeric(df["sort_type"]).clip(1, 3) / 3.0
        )
        out_cols.append("stock_static__sort_type__norm")

    # ── 新增：ind_new_asin（binary，静态）────────────────────
    if "ind_new_asin" in df.columns:
        df["stock_static__ind_new_asin"] = _safe_numeric(
            df["ind_new_asin"]
        ).clip(0, 1)
        out_cols.append("stock_static__ind_new_asin")

    return df, out_cols


def _event_thanksgiving_date(year):
    nov = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    return nov[nov.weekday == 3][3]


def _make_event_calendar(min_year, max_year):
    events = []
    for y in range(min_year - 1, max_year + 2):
        tg = _event_thanksgiving_date(y)
        events += [
            ("event_NewYear",              pd.Timestamp(f"{y}-01-01")),
            ("event_PrimeDay_proxy_July",  pd.Timestamp(f"{y}-07-15")),
            ("event_BackToSchool_proxy",   pd.Timestamp(f"{y}-08-15")),
            ("event_Thanksgiving",         tg),
            ("event_BlackFriday",          tg + pd.Timedelta(days=1)),
            ("event_CyberMonday",          tg + pd.Timedelta(days=4)),
            ("event_Christmas",            pd.Timestamp(f"{y}-12-25")),
        ]
    ev = pd.DataFrame(events, columns=["event_name", "event_date"])
    ev["event_week"] = ev["event_date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)
    return ev


def add_explicit_event_features(df, week_col="order_week", event_window_weeks=4):
    """
    改动：
      1. event_window_weeks 2 → 4（大件商品研究周期更长）
      2. 新增 pre_event_proximity：节假日前连续临近程度
         exp(-0.15 * weeks_until_event)，越近越大
      3. 新增 post_event_decay：节假日后连续衰减
         exp(-0.15 * weeks_since_event)，越远越小
         解决历史末尾是峰值导致的overbias问题
    """
    out = df.copy()
    out[week_col] = pd.to_datetime(out[week_col])
    out["week_start"] = out[week_col].dt.to_period("W-SUN").apply(lambda r: r.start_time)
    events = _make_event_calendar(out[week_col].dt.year.min(), out[week_col].dt.year.max())
    event_names = sorted(events["event_name"].unique().tolist())

    out["is_event_window"] = 0.0
    out["weeks_to_nearest_event"] = 99.0
    out["abs_weeks_to_nearest_event"] = 99.0
    out["is_pre_event"] = 0.0
    out["is_post_event"] = 0.0
    out["pre_event_proximity"] = 0.0   # 新增
    out["post_event_decay"] = 0.0      # 新增

    for ev_name in event_names:
        out[f"{ev_name}_window"] = 0.0
        out[f"{ev_name}_week_exact"] = 0.0

    for _, r in events.iterrows():
        ev_name = r["event_name"]
        ev_week = r["event_week"]
        diff = ((out["week_start"] - ev_week).dt.days / 7).round().astype(int)
        in_window = diff.abs() <= event_window_weeks
        exact_week = diff == 0
        out.loc[in_window, "is_event_window"] = 1.0
        out.loc[in_window, f"{ev_name}_window"] = 1.0
        out.loc[exact_week, f"{ev_name}_week_exact"] = 1.0
        current_abs = out["abs_weeks_to_nearest_event"].astype(float)
        new_abs = diff.abs().astype(float)
        replace = new_abs < current_abs
        out.loc[replace, "weeks_to_nearest_event"] = diff[replace].astype(float)
        out.loc[replace, "abs_weeks_to_nearest_event"] = new_abs[replace].astype(float)

    out["is_pre_event"] = ((out["weeks_to_nearest_event"] < 0) & (out["is_event_window"] > 0)).astype(float)
    out["is_post_event"] = ((out["weeks_to_nearest_event"] > 0) & (out["is_event_window"] > 0)).astype(float)

    # ── 连续衰减特征（归一化之前计算，用原始周数）──────────────
    weeks_raw = out["weeks_to_nearest_event"].astype(float)

    # 节假日前：还有8周=0.30, 还有4周=0.55, 还有1周=0.86, 当周=1.00
    weeks_until = (-weeks_raw).clip(lower=0.0)
    out["pre_event_proximity"] = np.exp(-0.15 * weeks_until)

    # 节假日后：过了1周=0.86, 过了5周=0.47, 过了10周=0.22
    weeks_since = weeks_raw.clip(lower=0.0)
    out["post_event_decay"] = np.exp(-0.15 * weeks_since)

    # 归一化（在连续特征计算之后）
    out["weeks_to_nearest_event"] = out["weeks_to_nearest_event"].clip(-20, 20) / 20.0
    out["abs_weeks_to_nearest_event"] = out["abs_weeks_to_nearest_event"].clip(0, 20) / 20.0

    event_cols = (
        [
            "is_event_window",
            "weeks_to_nearest_event",
            "abs_weeks_to_nearest_event",
            "is_pre_event",
            "is_post_event",
            "pre_event_proximity",   # 新增
            "post_event_decay",      # 新增
        ]
        + [f"{ev_name}_window" for ev_name in event_names]
        + [f"{ev_name}_week_exact" for ev_name in event_names]
    )
    return out, event_cols




# ============================================================
# DualRelationalGAT assets: ASIN graph construction + diagnostics support
# ============================================================

def _run_lengths(mask):
    """Return lengths of consecutive True runs in a boolean array."""
    arr = np.asarray(mask).astype(bool)
    runs = []
    cur = 0
    for v in arr:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def _top_share(x, frac=0.20):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) == 0 or x.sum() <= 0:
        return 0.0
    k = max(1, int(np.ceil(len(x) * frac)))
    return float(np.sort(x)[::-1][:k].sum() / (x.sum() + 1e-8))


def _last_run_length(mask, value=True):
    arr = np.asarray(mask).astype(bool)
    if len(arr) == 0:
        return 0.0
    target = bool(value)
    cnt = 0
    for v in arr[::-1]:
        if bool(v) == target:
            cnt += 1
        else:
            break
    return float(cnt)


def _build_graphsage_assets(
    df,
    graph_horizon=20,
    neighbor_k=10,
    graph_zero_weight=0.03,
    graph_level_peak_weight=2.2,
    graph_transition_weight=1.0,
    graph_static_weight=1.0,
    graph_brand_weight=0.3,
    verbose=True,
):
    """
    Build a shallow ASIN KNN graph for DualRelationalGAT.

    Important design choices:
      1. Use only history before the final forecast window per ASIN.
      2. Strengthen active-only magnitude / peak features so graph is not only a zero detector.
      3. Include ind_top10_brand as graph node feature and edge-similarity signal.
      4. Down-weight zero features in KNN similarity to avoid graph12-style underprediction.
      5. OOS is historical-only: past OOS enters encoder and graph summaries, but future
         scot_oos is NOT used as a known future covariate and no future hard mask is applied.

    Returns:
      node_features: standardized ASIN node feature matrix [N,F]
      neighbor_idx:  top-k neighbor indices [N,K]
      asin_to_idx:   mapping used by Dataset batches
      meta_df/raw_feature_df for graph diagnostics
    """
    work = df.copy()
    work["asin"] = work["asin"].astype(str)
    work["order_week"] = pd.to_datetime(work["order_week"])
    work = work.sort_values(["asin", "order_week"]).reset_index(drop=True)

    for c in ["total_dph", "buy_box_dph", "in_stock_dph", "fbi_demand", "scot_oos", "our_price",
              "customer_active_review_count", "customer_review_count", "ind_promotion", "ind_prime_week", "ind_top10_brand", "our_price", "list_price"]:
        if c in work.columns:
            work[c] = _safe_numeric(work[c]).fillna(0.0)

    rows = []
    meta_rows = []
    for asin, g0 in work.groupby("asin", sort=False):
        g0 = g0.sort_values("order_week").reset_index(drop=True)
        # avoid leakage: drop the final forecast horizon from graph statistics
        if len(g0) > graph_horizon:
            g = g0.iloc[:-int(graph_horizon)].copy()
        else:
            g = g0.copy()
        if len(g) == 0:
            g = g0.copy()

        instock = _safe_numeric(g.get("in_stock_dph", 0.0)).clip(lower=0.0).values.astype(float)
        buy = _safe_numeric(g.get("buy_box_dph", 0.0)).clip(lower=0.0).values.astype(float)
        total = _safe_numeric(g.get("total_dph", 0.0)).clip(lower=0.0).values.astype(float)
        demand = _safe_numeric(g.get("fbi_demand", 0.0)).clip(lower=0.0).values.astype(float)
        # Historical demand profile is safe here because g excludes the final forecast horizon.
        # It is used only as an ASIN-strength/popularity prior for the exposure graph.
        demand_mean = float(np.mean(demand)) if len(demand) else 0.0
        demand_median = float(np.median(demand)) if len(demand) else 0.0
        demand_q75 = float(np.quantile(demand, 0.75)) if len(demand) else 0.0
        demand_q90 = float(np.quantile(demand, 0.90)) if len(demand) else 0.0
        demand_q95 = float(np.quantile(demand, 0.95)) if len(demand) else 0.0
        demand_max = float(np.max(demand)) if len(demand) else 0.0
        demand_sum = float(np.sum(demand)) if len(demand) else 0.0
        demand_std = float(np.std(demand)) if len(demand) else 0.0
        demand_cv = demand_std / (demand_mean + 1e-8)
        oos = _safe_numeric(g.get("scot_oos", 0.0)).clip(0, 1).values.astype(float) if "scot_oos" in g.columns else np.zeros(len(g))
        # Historical-only OOS summaries. These are safe because g excludes the final forecast horizon.
        oos_bool = oos >= 0.5
        oos_rate_all = float(np.mean(oos_bool)) if len(oos_bool) else 0.0
        oos_rate_13 = float(np.mean(oos_bool[-13:])) if len(oos_bool) else 0.0
        oos_rate_26 = float(np.mean(oos_bool[-26:])) if len(oos_bool) else 0.0
        last_oos = float(oos_bool[-1]) if len(oos_bool) else 0.0
        oos_streak = float(_last_run_length(oos_bool, True)) if len(oos_bool) else 0.0
        weeks_since_last_oos = float((len(oos_bool) - 1 - np.where(oos_bool)[0][-1]) if np.any(oos_bool) else len(oos_bool)) if len(oos_bool) else 0.0

        active = instock > 0
        active50 = instock > 50
        zero = ~active
        active_prev = active[:-1] if len(active) > 1 else np.array([], dtype=bool)
        active_next = active[1:] if len(active) > 1 else np.array([], dtype=bool)
        a2z = float(np.sum(active_prev & (~active_next)) / (np.sum(active_prev) + 1e-8)) if len(active_prev) else 0.0
        z2a = float(np.sum((~active_prev) & active_next) / (np.sum(~active_prev) + 1e-8)) if len(active_prev) else 0.0
        active_runs = _run_lengths(active)
        zero_runs = _run_lengths(zero)

        pos_instock = instock[instock > 0]
        active_only_mean = float(pos_instock.mean()) if len(pos_instock) else 0.0
        active_only_q75 = float(np.quantile(pos_instock, 0.75)) if len(pos_instock) else 0.0
        active_only_q90 = float(np.quantile(pos_instock, 0.90)) if len(pos_instock) else 0.0
        active_only_q95 = float(np.quantile(pos_instock, 0.95)) if len(pos_instock) else 0.0

        instock_mean = float(np.mean(instock)) if len(instock) else 0.0
        instock_median = float(np.median(instock)) if len(instock) else 0.0
        instock_q75 = float(np.quantile(instock, 0.75)) if len(instock) else 0.0
        instock_q90 = float(np.quantile(instock, 0.90)) if len(instock) else 0.0
        instock_q95 = float(np.quantile(instock, 0.95)) if len(instock) else 0.0
        instock_max = float(np.max(instock)) if len(instock) else 0.0
        instock_std = float(np.std(instock)) if len(instock) else 0.0
        instock_cv = instock_std / (instock_mean + 1e-8)
        max_over_mean = instock_max / (instock_mean + 1e-8)
        q95_over_mean = instock_q95 / (instock_mean + 1e-8)
        active_q95_over_mean = active_only_q95 / (active_only_mean + 1e-8)

        # approximate concentration / burst features
        sorted_x = np.sort(np.asarray(instock, dtype=float))
        if len(sorted_x) > 0 and sorted_x.sum() > 0:
            n = len(sorted_x)
            gini = float((2 * np.arange(1, n + 1) @ sorted_x) / (n * sorted_x.sum() + 1e-8) - (n + 1) / n)
        else:
            gini = 0.0

        gl = str(g0["gl_product_group"].iloc[0]) if "gl_product_group" in g0.columns else "MISSING"
        cat = str(g0["category_code"].iloc[0]) if "category_code" in g0.columns else "MISSING"
        hbt = str(g0["hbt"].iloc[0]) if "hbt" in g0.columns else "MISSING"
        topbrand = float(_safe_numeric(g0["ind_top10_brand"].iloc[[0]]).iloc[0]) if "ind_top10_brand" in g0.columns else 0.0
        # V11: use our_price as the primary realized price-tier signal;
        # list_price is only a fallback for older data.
        if "our_price" in g.columns:
            price_mean = float(_safe_numeric(g.get("our_price", 0.0)).clip(lower=0.0).mean())
        elif "list_price" in g.columns:
            price_mean = float(_safe_numeric(g.get("list_price", 0.0)).clip(lower=0.0).mean())
        else:
            price_mean = 0.0
        # V15-PKG: physical comparability features for graph edges.
        # Package fields are static/slow-moving, so use historical median before origin.
        def _pkg_median(col):
            if col not in g.columns:
                return 0.0
            x = _safe_numeric(g[col]).clip(lower=0.0)
            x = x.replace([np.inf, -np.inf], np.nan).dropna()
            return float(x.median()) if len(x) else 0.0

        pkg_height = _pkg_median("pkg_height")
        pkg_length = _pkg_median("pkg_length")
        pkg_width  = _pkg_median("pkg_width")
        pkg_weight = _pkg_median("pkg_weight")
        pkg_volume = float(pkg_height * pkg_length * pkg_width) if (pkg_height > 0 and pkg_length > 0 and pkg_width > 0) else 0.0
        pkg_complete = float(pkg_height > 0 and pkg_length > 0 and pkg_width > 0 and pkg_weight > 0)

        # Prefer customer_active_review_count if present; fall back to customer_review_count.
        review_col = "customer_active_review_count" if "customer_active_review_count" in g.columns else ("customer_review_count" if "customer_review_count" in g.columns else None)
        review_last = float(_safe_numeric(g[review_col]).clip(lower=0.0).iloc[-1]) if review_col is not None and len(g) else 0.0
        promo_series = _safe_numeric(g.get("ind_promotion", 0.0)).clip(0, 1).values.astype(float) if "ind_promotion" in g.columns else np.zeros(len(g), dtype=float)
        prime_series = _safe_numeric(g.get("ind_prime_week", 0.0)).clip(0, 1).values.astype(float) if "ind_prime_week" in g.columns else np.zeros(len(g), dtype=float)
        promo_rate = float(np.mean(promo_series)) if len(promo_series) else 0.0
        prime_rate = float(np.mean(prime_series)) if len(prime_series) else 0.0

        def _last_mean(arr, w):
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                return 0.0
            return float(np.mean(arr[-min(int(w), len(arr)):]))
        def _last_zero(arr, w):
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                return 1.0
            x = arr[-min(int(w), len(arr)):]
            return float(np.mean(x <= 0)) if len(x) else 1.0
        def _safe_ratio(a, b):
            return float(np.log1p(max(a, 0.0)) - np.log1p(max(b, 0.0)))

        # V11 dynamic magnitude-rank features: short/medium/long strength + sparse-aware eligibility.
        recent4_total_mean = _last_mean(total, 4)
        recent13_total_mean = _last_mean(total, 13)
        recent26_total_mean = _last_mean(total, 26)
        long52_total_mean = _last_mean(total, 52)
        recent4_buybox_mean = _last_mean(buy, 4)
        recent13_buybox_mean = _last_mean(buy, 13)
        recent26_buybox_mean = _last_mean(buy, 26)
        long52_buybox_mean = _last_mean(buy, 52)
        recent4_instock_mean = _last_mean(instock, 4)
        recent13_instock_mean = _last_mean(instock, 13)
        recent26_instock_mean = _last_mean(instock, 26)
        long52_instock_mean = _last_mean(instock, 52)
        recent4_demand_mean = _last_mean(demand, 4)
        recent13_demand_mean = _last_mean(demand, 13)
        recent26_demand_mean = _last_mean(demand, 26)
        long52_demand_mean = _last_mean(demand, 52)

        recent13_total_zero = _last_zero(total, 13)
        recent13_buybox_zero = _last_zero(buy, 13)
        recent13_instock_zero = _last_zero(instock, 13)
        recent13_demand_zero = _last_zero(demand, 13)
        recent13_zero_mean = float(np.mean([recent13_total_zero, recent13_buybox_zero, recent13_instock_zero, recent13_demand_zero]))
        recent13_active_eligibility = 1.0 - recent13_zero_mean

        promo_current = float(promo_series[-1]) if len(promo_series) else 0.0
        prime_current = float(prime_series[-1]) if len(prime_series) else 0.0
        promo_response_strength = 0.0
        if len(promo_series) > 0 and promo_series.sum() > 0:
            pm = promo_series > 0.5
            base = ~pm
            promo_level = np.mean(np.log1p(instock[pm]) + np.log1p(demand[pm])) if pm.any() else 0.0
            base_level = np.mean(np.log1p(instock[base]) + np.log1p(demand[base])) if base.any() else 0.0
            promo_response_strength = float(np.clip(promo_level - base_level, -5.0, 5.0))

        # Event index from holiday_indicator_* and distance_* columns, plus Prime week.
        holiday_cols_g = [c for c in g.columns if str(c).startswith("holiday_indicator_")]
        distance_cols_g = [c for c in g.columns if str(c).startswith("distance_")]
        holiday_current = 0.0
        if holiday_cols_g:
            holiday_current = float(np.nanmax([float(_safe_numeric(g[c]).iloc[-1]) for c in holiday_cols_g]))
        distance_event_prox = 0.0
        if distance_cols_g:
            vals = []
            for c in distance_cols_g:
                dval = float(_safe_numeric(g[c]).iloc[-1])
                vals.append(max(0.0, 1.0 - abs(dval) / 12.0))
            distance_event_prox = float(np.nanmax(vals)) if vals else 0.0
        event_current = float(np.clip(max(holiday_current, distance_event_prox, prime_current), 0.0, 1.0))
        promo_event_current = float(np.clip(max(promo_current, event_current), 0.0, 1.0))

        recent13_total_momentum = _safe_ratio(recent13_total_mean, long52_total_mean)
        recent13_buybox_momentum = _safe_ratio(recent13_buybox_mean, long52_buybox_mean)
        recent13_instock_momentum = _safe_ratio(recent13_instock_mean, long52_instock_mean)
        recent13_demand_momentum = _safe_ratio(recent13_demand_mean, long52_demand_mean)

        rows.append({
            "asin": asin,
            # zero / active
            "instock_zero_rate": float(np.mean(instock <= 0)) if len(instock) else 1.0,
            "buybox_zero_rate": float(np.mean(buy <= 0)) if len(buy) else 1.0,
            "total_zero_rate": float(np.mean(total <= 0)) if len(total) else 1.0,
            "instock_active_rate": float(np.mean(instock > 0)) if len(instock) else 0.0,
            "instock_active50_rate": float(np.mean(active50)) if len(instock) else 0.0,
            "demand_active_rate": float(np.mean(demand > 0)) if len(demand) else 0.0,
            # Historical demand magnitude / popularity features.
            "log_demand_sum": np.log1p(demand_sum),
            "log_demand_mean": np.log1p(demand_mean),
            "log_demand_median": np.log1p(demand_median),
            "log_demand_q75": np.log1p(demand_q75),
            "log_demand_q90": np.log1p(demand_q90),
            "log_demand_q95": np.log1p(demand_q95),
            "log_demand_max": np.log1p(demand_max),
            "demand_cv": float(np.clip(demand_cv, 0, 50)),
            "oos_rate": oos_rate_all,
            "oos_rate_13": oos_rate_13,
            "oos_rate_26": oos_rate_26,
            "last_oos": last_oos,
            "log_oos_streak": np.log1p(oos_streak),
            "log_weeks_since_last_oos": np.log1p(weeks_since_last_oos),
            # level / peak: overall
            "log_instock_mean": np.log1p(instock_mean),
            "log_instock_median": np.log1p(instock_median),
            "log_instock_q75": np.log1p(instock_q75),
            "log_instock_q90": np.log1p(instock_q90),
            "log_instock_q95": np.log1p(instock_q95),
            "log_instock_max": np.log1p(instock_max),
            "instock_cv": float(np.clip(instock_cv, 0, 50)),
            "instock_gini": gini,
            "top10_share": _top_share(instock, 0.10),
            "top20_share": _top_share(instock, 0.20),
            "max_over_mean": float(np.clip(max_over_mean, 0, 100)),
            "q95_over_mean": float(np.clip(q95_over_mean, 0, 100)),
            # level / peak: active-only, key to avoid graph under active weeks
            "log_active_only_mean": np.log1p(active_only_mean),
            "log_active_only_q75": np.log1p(active_only_q75),
            "log_active_only_q90": np.log1p(active_only_q90),
            "log_active_only_q95": np.log1p(active_only_q95),
            "active_q95_over_mean": float(np.clip(active_q95_over_mean, 0, 100)),
            # buybox / total scale for funnel information
            "log_buybox_mean": np.log1p(float(np.mean(buy)) if len(buy) else 0.0),
            "log_total_mean": np.log1p(float(np.mean(total)) if len(total) else 0.0),
            # transition
            "active_to_zero_rate": a2z,
            "zero_to_active_rate": z2a,
            "log_avg_active_spell": np.log1p(float(np.mean(active_runs)) if active_runs else 0.0),
            "log_avg_zero_spell": np.log1p(float(np.mean(zero_runs)) if zero_runs else 0.0),
            "last_active_streak": np.log1p(_last_run_length(active, True)),
            "last_zero_streak": np.log1p(_last_run_length(active, False)),
            "weeks_since_last_positive": np.log1p((len(active) - 1 - np.where(active)[0][-1]) if np.any(active) else len(active)),
            # static / business
            "ind_top10_brand": topbrand,
            "log_price_mean": np.log1p(price_mean),
            # V15-PKG: package size/weight node features used as pairwise edge gaps.
            "log_pkg_height": np.log1p(pkg_height),
            "log_pkg_length": np.log1p(pkg_length),
            "log_pkg_width": np.log1p(pkg_width),
            "log_pkg_weight": np.log1p(pkg_weight),
            "log_pkg_volume": np.log1p(pkg_volume),
            "pkg_complete": pkg_complete,
            "log_review_last": np.log1p(review_last),
            "promo_rate": promo_rate,
            "prime_rate": prime_rate,
            "promo_current": promo_current,
            "prime_current": prime_current,
            "event_current": event_current,
            "promo_event_current": promo_event_current,
            "promo_response_strength": promo_response_strength,
            "recent4_total_mean_log": np.log1p(recent4_total_mean),
            "recent13_total_mean_log": np.log1p(recent13_total_mean),
            "recent26_total_mean_log": np.log1p(recent26_total_mean),
            "long52_total_mean_log": np.log1p(long52_total_mean),
            "recent4_buybox_mean_log": np.log1p(recent4_buybox_mean),
            "recent13_buybox_mean_log": np.log1p(recent13_buybox_mean),
            "recent26_buybox_mean_log": np.log1p(recent26_buybox_mean),
            "long52_buybox_mean_log": np.log1p(long52_buybox_mean),
            "recent4_instock_mean_log": np.log1p(recent4_instock_mean),
            "recent13_instock_mean_log": np.log1p(recent13_instock_mean),
            "recent26_instock_mean_log": np.log1p(recent26_instock_mean),
            "long52_instock_mean_log": np.log1p(long52_instock_mean),
            "recent4_demand_mean_log": np.log1p(recent4_demand_mean),
            "recent13_demand_mean_log": np.log1p(recent13_demand_mean),
            "recent26_demand_mean_log": np.log1p(recent26_demand_mean),
            "long52_demand_mean_log": np.log1p(long52_demand_mean),
            "recent13_total_zero_rate": recent13_total_zero,
            "recent13_buybox_zero_rate": recent13_buybox_zero,
            "recent13_instock_zero_rate": recent13_instock_zero,
            "recent13_demand_zero_rate": recent13_demand_zero,
            "recent13_zero_mean": recent13_zero_mean,
            "recent13_active_eligibility": recent13_active_eligibility,
            "recent13_total_momentum": recent13_total_momentum,
            "recent13_buybox_momentum": recent13_buybox_momentum,
            "recent13_instock_momentum": recent13_instock_momentum,
            "recent13_demand_momentum": recent13_demand_momentum,
        })
        meta_rows.append({"asin": asin, "gl_product_group": gl, "category_code": cat, "hbt": hbt, "ind_top10_brand": topbrand})

    feat = pd.DataFrame(rows).fillna(0.0)
    meta = pd.DataFrame(meta_rows)
    if len(feat) == 0:
        raise ValueError("No ASINs available to build DualGraphSAGE assets.")

    # Encode GL/category/hbt as continuous normalized codes + frequencies for node features and KNN.
    for c in ["gl_product_group", "category_code", "hbt"]:
        raw = meta[c].astype(str).fillna("MISSING")
        codes, uniques = pd.factorize(raw)
        denom = max(len(uniques) - 1, 1)
        feat[f"{c}_code"] = codes.astype(float) / denom
        freq = raw.value_counts(normalize=True)
        feat[f"{c}_freq"] = raw.map(freq).fillna(0.0).astype(float)
        if c == "category_code":
            feat["category_is_unknown"] = raw.str.lower().isin(["unknown", "missing", "nan", "none", ""]).astype(float)
        if c == "hbt":
            hbt_lower = raw.str.lower()
            feat["hbt_is_unknown"] = hbt_lower.isin(["unknown", "missing", "nan", "none", ""]).astype(float)
            feat["hbt_is_head"] = hbt_lower.str.contains("head").astype(float)
            feat["hbt_is_body"] = hbt_lower.str.contains("body").astype(float)
            feat["hbt_is_tail"] = hbt_lower.str.contains("tail").astype(float)

    # ------------------------------------------------------------
    # Magnitude-aware ASIN profile buckets for within-category graph.
    # These buckets are computed from historical-only features, because
    # g excludes the final forecast horizon above. They help distinguish
    # high / medium / low exposure ASINs inside the same category.
    # ------------------------------------------------------------
    def _safe_group_qbucket(values, q=4):
        s = pd.Series(values).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(s) <= 1 or s.nunique(dropna=False) <= 1:
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index)
        q_eff = int(min(q, max(1, s.nunique())))
        try:
            return pd.qcut(s.rank(method="first"), q=q_eff, labels=False, duplicates="drop").astype(int)
        except Exception:
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index)

    # Strength proxy for within-category magnitude buckets.
    # V5: demand is added as a historical popularity signal. HBT is intentionally NOT a main
    # strength driver because diagnostics showed its within-category direction can be unstable.
    exposure_strength = (
        0.70 * feat.get("log_instock_mean", 0.0).astype(float) +
        0.70 * feat.get("log_active_only_q95", 0.0).astype(float) +
        0.35 * feat.get("log_buybox_mean", 0.0).astype(float) +
        0.25 * feat.get("log_total_mean", 0.0).astype(float)
    )
    demand_strength = (
        0.35 * feat.get("log_demand_mean", 0.0).astype(float) +
        0.35 * feat.get("log_demand_q90", 0.0).astype(float) +
        0.20 * feat.get("log_demand_sum", 0.0).astype(float) +
        0.10 * feat.get("demand_active_rate", 0.0).astype(float)
    )
    weak_static_strength = (
        0.08 * feat.get("log_review_last", 0.0).astype(float) +
        0.05 * feat.get("ind_top10_brand", 0.0).astype(float)
    )
    feat["graph_exposure_strength"] = exposure_strength
    feat["graph_demand_strength"] = demand_strength
    feat["graph_strength_score"] = 0.60 * exposure_strength + 0.35 * demand_strength + 0.05 * weak_static_strength

    feat["magnitude_bucket"] = 0
    feat["active_bucket"] = 0
    # Bucket within category so that high/low ASINs in the same category are separated.
    cat_series = meta["category_code"].astype(str).fillna("MISSING")
    for _, idxs in cat_series.groupby(cat_series).groups.items():
        idxs = list(idxs)
        feat.loc[idxs, "magnitude_bucket"] = _safe_group_qbucket(feat.loc[idxs, "graph_strength_score"], q=4).values
        feat.loc[idxs, "active_bucket"] = _safe_group_qbucket(feat.loc[idxs, "instock_active_rate"], q=3).values

    feat["magnitude_bucket_norm"] = feat["magnitude_bucket"].astype(float) / 3.0
    feat["active_bucket_norm"] = feat["active_bucket"].astype(float) / 2.0

    # V11: sparse-aware, event-aware dynamic category rank prior.
    # This is built from history before the forecast origin and current known promo/event context.
    rank_base_cols = [
        "recent13_total_mean_log", "recent13_buybox_mean_log", "recent13_instock_mean_log", "recent13_demand_mean_log",
        "recent26_total_mean_log", "recent26_buybox_mean_log", "recent26_instock_mean_log", "recent26_demand_mean_log",
        "long52_total_mean_log", "long52_buybox_mean_log", "long52_instock_mean_log", "long52_demand_mean_log",
        "recent13_active_eligibility", "promo_current", "promo_event_current", "promo_response_strength",
        "event_current", "recent13_zero_mean", "ind_top10_brand", "log_price_mean",
    ]
    for c in rank_base_cols:
        if c not in feat.columns:
            feat[c] = 0.0

    def _pct_rank_group(s):
        s = pd.Series(s).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(s) <= 1 or s.nunique(dropna=False) <= 1:
            return pd.Series(np.full(len(s), 0.5), index=s.index)
        return s.rank(method="average", pct=True).clip(0, 1)

    for c in [
        "recent13_total_mean_log", "recent13_buybox_mean_log", "recent13_instock_mean_log", "recent13_demand_mean_log",
        "recent26_total_mean_log", "recent26_buybox_mean_log", "recent26_instock_mean_log", "recent26_demand_mean_log",
        "long52_total_mean_log", "long52_buybox_mean_log", "long52_instock_mean_log", "long52_demand_mean_log",
        "recent13_active_eligibility", "promo_response_strength", "log_price_mean",
    ]:
        feat[f"cat_rank_{c}"] = 0.5

    cat_series = meta["category_code"].astype(str).fillna("MISSING")
    for _, idxs in cat_series.groupby(cat_series).groups.items():
        idxs = list(idxs)
        for c in [
            "recent13_total_mean_log", "recent13_buybox_mean_log", "recent13_instock_mean_log", "recent13_demand_mean_log",
            "recent26_total_mean_log", "recent26_buybox_mean_log", "recent26_instock_mean_log", "recent26_demand_mean_log",
            "long52_total_mean_log", "long52_buybox_mean_log", "long52_instock_mean_log", "long52_demand_mean_log",
            "recent13_active_eligibility", "promo_response_strength", "log_price_mean",
        ]:
            feat.loc[idxs, f"cat_rank_{c}"] = _pct_rank_group(feat.loc[idxs, c]).values

    feat["cat_rank_recent13_strength"] = (
        0.28 * feat["cat_rank_recent13_total_mean_log"] +
        0.30 * feat["cat_rank_recent13_buybox_mean_log"] +
        0.28 * feat["cat_rank_recent13_instock_mean_log"] +
        0.14 * feat["cat_rank_recent13_demand_mean_log"]
    )
    feat["cat_rank_long52_strength"] = (
        0.25 * feat["cat_rank_long52_total_mean_log"] +
        0.25 * feat["cat_rank_long52_buybox_mean_log"] +
        0.25 * feat["cat_rank_long52_instock_mean_log"] +
        0.25 * feat["cat_rank_long52_demand_mean_log"]
    )
    feat["cat_rank_momentum"] = (
        0.25 * feat.get("recent13_total_momentum", 0.0).astype(float) +
        0.25 * feat.get("recent13_buybox_momentum", 0.0).astype(float) +
        0.25 * feat.get("recent13_instock_momentum", 0.0).astype(float) +
        0.25 * feat.get("recent13_demand_momentum", 0.0).astype(float)
    )
    # Normalize momentum to category percentile so it can be used as a soft prior.
    feat["cat_rank_momentum_pct"] = 0.5
    for _, idxs in cat_series.groupby(cat_series).groups.items():
        idxs = list(idxs)
        feat.loc[idxs, "cat_rank_momentum_pct"] = _pct_rank_group(feat.loc[idxs, "cat_rank_momentum"]).values

    feat["rank_zero_good"] = (1.0 - feat["recent13_zero_mean"].astype(float).clip(0, 1))
    feat["rank_promo_boost"] = (feat["promo_event_current"].astype(float).clip(0, 1) * (0.5 + 0.5 * feat["cat_rank_promo_response_strength"].astype(float).clip(0, 1)))
    feat["rank_composite_dynamic"] = (
        0.38 * feat["cat_rank_recent13_strength"] +
        0.22 * feat["cat_rank_long52_strength"] +
        0.14 * feat["cat_rank_momentum_pct"] +
        0.14 * feat["rank_zero_good"] +
        0.08 * feat["rank_promo_boost"] +
        0.04 * feat["ind_top10_brand"].astype(float).clip(0, 1)
    ).clip(0, 1)
    feat["rank_stability_abs_gap"] = (feat["cat_rank_recent13_strength"] - feat["cat_rank_long52_strength"]).abs().clip(0, 1)

    zero_cols = [
        "instock_zero_rate", "buybox_zero_rate", "total_zero_rate",
        "oos_rate", "oos_rate_13", "oos_rate_26", "last_oos",
        "log_oos_streak", "log_weeks_since_last_oos",
    ]
    level_peak_cols = [
        "log_instock_mean", "log_instock_median", "log_instock_q75", "log_instock_q90",
        "log_instock_q95", "log_instock_max", "instock_cv", "instock_gini",
        "top10_share", "top20_share", "max_over_mean", "q95_over_mean",
        "log_active_only_mean", "log_active_only_q75", "log_active_only_q90", "log_active_only_q95",
        "active_q95_over_mean", "log_buybox_mean", "log_total_mean",
    ]
    demand_cols = [
        "log_demand_sum", "log_demand_mean", "log_demand_median", "log_demand_q75",
        "log_demand_q90", "log_demand_q95", "log_demand_max", "demand_cv",
        "demand_active_rate", "graph_demand_strength"
    ]
    transition_cols = ["active_to_zero_rate", "zero_to_active_rate", "log_avg_active_spell", "log_avg_zero_spell", "last_active_streak", "last_zero_streak", "weeks_since_last_positive"]
    static_cols = ["gl_product_group_code", "gl_product_group_freq", "category_code_code", "category_code_freq", "category_is_unknown",
                   "hbt_code", "hbt_freq", "hbt_is_unknown", "hbt_is_head", "hbt_is_body", "hbt_is_tail",
                   "log_price_mean", "log_pkg_height", "log_pkg_length", "log_pkg_width", "log_pkg_weight", "log_pkg_volume", "pkg_complete",
                   "log_review_last", "promo_rate", "prime_rate"]
    brand_cols = ["ind_top10_brand"]
    node_feature_cols = list(dict.fromkeys(
        zero_cols + level_peak_cols + demand_cols + transition_cols + static_cols + brand_cols +
        ["instock_active_rate", "instock_active50_rate",
         "graph_exposure_strength", "graph_strength_score", "magnitude_bucket_norm", "active_bucket_norm",
         "recent13_total_mean_log", "recent13_buybox_mean_log", "recent13_instock_mean_log", "recent13_demand_mean_log",
         "recent26_total_mean_log", "recent26_buybox_mean_log", "recent26_instock_mean_log", "recent26_demand_mean_log",
         "long52_total_mean_log", "long52_buybox_mean_log", "long52_instock_mean_log", "long52_demand_mean_log",
         "recent13_active_eligibility", "recent13_zero_mean", "promo_current", "promo_event_current", "event_current",
         "promo_response_strength", "cat_rank_recent13_strength", "cat_rank_long52_strength", "cat_rank_momentum_pct",
         "rank_zero_good", "rank_promo_boost", "rank_composite_dynamic", "rank_stability_abs_gap"]
    ))
    for c in node_feature_cols:
        if c not in feat.columns:
            feat[c] = 0.0

    X_raw = feat[node_feature_cols].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0).values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw).astype(np.float32)

    # Weighted KNN features. Zero is down-weighted; active-only level/peak is emphasized.
    weight_map = {c: 1.0 for c in node_feature_cols}
    for c in zero_cols:
        weight_map[c] = float(graph_zero_weight)
    for c in level_peak_cols:
        weight_map[c] = float(graph_level_peak_weight)
    # Demand is useful as an ASIN popularity prior, but should not dominate exposure-specific features.
    for c in demand_cols:
        weight_map[c] = min(float(graph_level_peak_weight), 1.25)
    for c in transition_cols:
        weight_map[c] = float(graph_transition_weight)
    for c in static_cols:
        weight_map[c] = float(graph_static_weight)
    for c in brand_cols:
        weight_map[c] = float(graph_brand_weight)
    W = np.asarray([weight_map.get(c, 1.0) for c in node_feature_cols], dtype=np.float32)
    X_knn = X_std * W[None, :]

    N = X_knn.shape[0]
    K = max(1, min(int(neighbor_k), max(N - 1, 1)))

    # ------------------------------------------------------------
    # CATEGORY-ONLY magnitude-aware dual graph construction (v6).
    # GL is intentionally NOT used to select graph neighbors; GL remains a static/model feature.
    # Positive graph: same category + close exposure/demand/active buckets.
    # Competitive graph: same category + far exposure/demand/active buckets.
    # For very small categories, we keep weak self-neighbors instead of falling back to GL,
    # because GL is too coarse and can over-smooth different categories.
    # ------------------------------------------------------------
    def _fallback_neighbors(i, K):
        if N <= 1:
            return [i] * K
        # global nearest fallback, excluding self
        sims = X_knn @ X_knn[i]
        order = np.argsort(-sims)
        out = [int(j) for j in order if int(j) != int(i)]
        if not out:
            out = [int(i)]
        while len(out) < K:
            out.append(out[-1])
        return out[:K]

    cat_arr = meta["category_code"].astype(str).values
    gl_arr = meta["gl_product_group"].astype(str).values
    hbt_arr = meta["hbt"].astype(str).str.lower().values if "hbt" in meta.columns else np.array(["missing"] * N)
    top_arr = meta["ind_top10_brand"].astype(float).values if "ind_top10_brand" in meta.columns else np.zeros(N)
    mag_arr = feat["magnitude_bucket"].astype(int).values
    act_arr = feat["active_bucket"].astype(int).values
    strength_arr = feat["graph_strength_score"].astype(float).values
    strength_z = (strength_arr - np.nanmean(strength_arr)) / (np.nanstd(strength_arr) + 1e-8)
    demand_strength_arr = feat.get("graph_demand_strength", pd.Series(np.zeros(N))).astype(float).values
    demand_strength_z = (demand_strength_arr - np.nanmean(demand_strength_arr)) / (np.nanstd(demand_strength_arr) + 1e-8)

    # ------------------------------------------------------------------
    # V10 learned-relation-inspired edge scoring.
    # These are the exact business signals validated in the standalone
    # dynamic relation test: total/buybox/instock/demand + HBT + top10 brand
    # + ind_promotion + list_price.  They are historical-only here because
    # feat is built after dropping the forecast horizon.
    # ------------------------------------------------------------------
    def _z_arr(col):
        v = feat.get(col, pd.Series(np.zeros(N))).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)

    z_total_mean   = _z_arr("log_total_mean")
    z_buybox_mean  = _z_arr("log_buybox_mean")
    z_instock_mean = _z_arr("log_instock_mean")
    z_demand_mean  = _z_arr("log_demand_mean")
    z_instock_q90  = _z_arr("log_instock_q90")
    z_demand_q90   = _z_arr("log_demand_q90")
    z_price        = _z_arr("log_price_mean")
    # V15-PKG: package size/weight gaps are edge-level features. Use raw log gaps,
    # not z gaps, so thresholds/penalties remain interpretable.
    log_pkg_height = feat.get("log_pkg_height", pd.Series(np.zeros(N))).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    log_pkg_length = feat.get("log_pkg_length", pd.Series(np.zeros(N))).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    log_pkg_width  = feat.get("log_pkg_width",  pd.Series(np.zeros(N))).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    log_pkg_weight = feat.get("log_pkg_weight", pd.Series(np.zeros(N))).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    log_pkg_volume = feat.get("log_pkg_volume", pd.Series(np.zeros(N))).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    pkg_complete_arr = feat.get("pkg_complete", pd.Series(np.zeros(N))).astype(float).fillna(0.0).values
    promo_arr      = feat.get("promo_rate", pd.Series(np.zeros(N))).astype(float).fillna(0.0).values
    prime_arr      = feat.get("prime_rate", pd.Series(np.zeros(N))).astype(float).fillna(0.0).values
    instock_ar     = feat.get("instock_active_rate", pd.Series(np.zeros(N))).astype(float).fillna(0.0).values
    demand_ar      = feat.get("demand_active_rate", pd.Series(np.zeros(N))).astype(float).fillna(0.0).values
    total_zero     = feat.get("total_zero_rate", pd.Series(np.ones(N))).astype(float).fillna(1.0).values
    buybox_zero    = feat.get("buybox_zero_rate", pd.Series(np.ones(N))).astype(float).fillna(1.0).values
    instock_zero   = feat.get("instock_zero_rate", pd.Series(np.ones(N))).astype(float).fillna(1.0).values

    def _relation_components(i, cand):
        cand = np.asarray(cand, dtype=int)
        # closeness gaps: smaller => positive relation
        level_gap = (
            np.abs(z_total_mean[cand] - z_total_mean[i]) +
            np.abs(z_buybox_mean[cand] - z_buybox_mean[i]) +
            np.abs(z_instock_mean[cand] - z_instock_mean[i]) +
            np.abs(z_demand_mean[cand] - z_demand_mean[i])
        ) / 4.0
        peak_gap = (
            np.abs(z_instock_q90[cand] - z_instock_q90[i]) +
            np.abs(z_demand_q90[cand] - z_demand_q90[i])
        ) / 2.0
        active_gap = (np.abs(instock_ar[cand] - instock_ar[i]) + np.abs(demand_ar[cand] - demand_ar[i])) / 2.0
        zero_gap = (np.abs(total_zero[cand] - total_zero[i]) + np.abs(buybox_zero[cand] - buybox_zero[i]) + np.abs(instock_zero[cand] - instock_zero[i])) / 3.0
        price_gap = np.abs(z_price[cand] - z_price[i])
        # V15-PKG: physical comparability. Positive edges are heavily penalized
        # when package height/length/width/weight/volume are far apart.
        pkg_height_gap = np.abs(log_pkg_height[cand] - log_pkg_height[i])
        pkg_length_gap = np.abs(log_pkg_length[cand] - log_pkg_length[i])
        pkg_width_gap  = np.abs(log_pkg_width[cand]  - log_pkg_width[i])
        pkg_weight_gap = np.abs(log_pkg_weight[cand] - log_pkg_weight[i])
        pkg_volume_gap = np.abs(log_pkg_volume[cand] - log_pkg_volume[i])
        pkg_distance = np.sqrt(pkg_height_gap ** 2 + pkg_length_gap ** 2 + pkg_width_gap ** 2 + pkg_weight_gap ** 2)
        pkg_both_complete = ((pkg_complete_arr[cand] > 0.5) & (pkg_complete_arr[i] > 0.5)).astype(float)
        pkg_size_similar_strict = (
            (pkg_both_complete > 0.5)
            & (pkg_height_gap <= 0.35)
            & (pkg_length_gap <= 0.35)
            & (pkg_width_gap  <= 0.35)
            & (pkg_weight_gap <= 0.50)
            & (pkg_volume_gap <= 0.70)
            & (pkg_distance <= 0.90)
        ).astype(float)
        pkg_size_similar_relaxed = (
            (pkg_both_complete > 0.5)
            & (pkg_height_gap <= 0.45)
            & (pkg_length_gap <= 0.45)
            & (pkg_width_gap  <= 0.45)
            & (pkg_weight_gap <= 0.65)
            & (pkg_volume_gap <= 0.90)
            & (pkg_distance <= 1.10)
        ).astype(float)
        pkg_not_comparable = 1.0 - pkg_size_similar_relaxed
        promo_gap = np.abs(promo_arr[cand] - promo_arr[i])
        prime_gap = np.abs(prime_arr[cand] - prime_arr[i])
        same_hbt = (hbt_arr[cand] == hbt_arr[i]).astype(float)
        hbt_diff = 1.0 - same_hbt
        brand_same = (top_arr[cand] == top_arr[i]).astype(float)
        brand_diff = 1.0 - brand_same
        funnel_strength_gap = np.abs(strength_z[cand] - strength_z[i])
        demand_gap = np.abs(demand_strength_z[cand] - demand_strength_z[i])
        stronger = np.maximum(strength_z[cand] - strength_z[i], 0.0)
        stronger_demand = np.maximum(demand_strength_z[cand] - demand_strength_z[i], 0.0)
        stronger_top10 = ((top_arr[cand] > top_arr[i]).astype(float) * (stronger > 0).astype(float))
        # positive: similar funnel/demand, same HBT/brand, similar promo and price regime
        pos_score = (
            2.00
            - 1.45 * level_gap
            - 0.65 * peak_gap
            - 0.75 * active_gap
            - 0.55 * zero_gap
            - 0.45 * price_gap
            - 0.75 * pkg_distance
            - 0.45 * pkg_volume_gap
            - 0.35 * pkg_weight_gap
            + 0.35 * pkg_size_similar_strict
            - 0.65 * promo_gap
            - 0.15 * prime_gap
            + 0.42 * same_hbt
            + 0.38 * brand_same
        )
        # competitive: same category but stronger/different HBT/brand/price/promo regime.
        comp_score = (
            0.90 * funnel_strength_gap
            + 0.65 * demand_gap
            + 0.45 * stronger
            + 0.35 * stronger_demand
            + 0.50 * hbt_diff
            + 0.45 * brand_diff
            + 0.65 * stronger_top10
            + 0.35 * price_gap
            + 0.45 * promo_gap
            + 0.20 * prime_gap
            + 0.30 * pkg_size_similar_relaxed
            - 0.35 * pkg_not_comparable
            - 0.20 * pkg_distance
        )
        return pos_score, comp_score, {
            "level_gap": level_gap,
            "peak_gap": peak_gap,
            "active_gap": active_gap,
            "zero_gap": zero_gap,
            "price_gap": price_gap,
            "pkg_height_gap": pkg_height_gap,
            "pkg_length_gap": pkg_length_gap,
            "pkg_width_gap": pkg_width_gap,
            "pkg_weight_gap": pkg_weight_gap,
            "pkg_volume_gap": pkg_volume_gap,
            "pkg_distance": pkg_distance,
            "pkg_both_complete": pkg_both_complete,
            "pkg_size_similar_strict": pkg_size_similar_strict,
            "pkg_size_similar_relaxed": pkg_size_similar_relaxed,
            "pkg_not_comparable": pkg_not_comparable,
            "promo_gap": promo_gap,
            "prime_gap": prime_gap,
            "same_hbt": same_hbt,
            "hbt_diff": hbt_diff,
            "brand_same": brand_same,
            "brand_diff": brand_diff,
            "funnel_strength_gap": funnel_strength_gap,
            "demand_gap": demand_gap,
            "stronger": stronger,
            "stronger_demand": stronger_demand,
            "stronger_top10": stronger_top10,
        }

    # V14 learned-edge graph: keep same candidate pool, but expose rich edge features
    # so EdgeMLP can learn how to weight positive/competitive messages end-to-end.
    edge_feature_names = [
        "rule_pos_score", "rule_comp_score",
        "level_gap", "peak_gap", "active_gap", "zero_gap", "price_gap",
        "pkg_height_gap", "pkg_length_gap", "pkg_width_gap", "pkg_weight_gap", "pkg_volume_gap",
        "pkg_distance", "pkg_both_complete", "pkg_size_similar_strict", "pkg_size_similar_relaxed", "pkg_not_comparable",
        "promo_gap", "prime_gap",
        "same_hbt", "hbt_diff", "brand_same", "brand_diff",
        "funnel_strength_gap", "demand_gap", "stronger", "stronger_demand", "stronger_top10",
    ]

    def _edge_feature_matrix(i, cand):
        ps, cs, info = _relation_components(i, cand)
        cols = [ps, cs]
        for name in edge_feature_names[2:]:
            cols.append(info.get(name, np.zeros(len(cand), dtype=float)))
        mat = np.stack(cols, axis=1).astype(np.float32)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        return mat

    neigh_rows = []
    comp_rows = []
    pos_score_rows = []
    comp_score_rows = []
    pos_edge_feat_rows = []
    comp_edge_feat_rows = []
    all_idx = np.arange(N)
    min_category_graph_size = max(4, min(K + 1, 10))
    for i in range(N):
        # Candidate pool: STRICT same-category only.
        # Do NOT fall back to same GL here; GL is kept as a static feature only.
        same_cat = all_idx[(cat_arr == cat_arr[i]) & (all_idx != i)]
        same_gl = all_idx[(gl_arr == gl_arr[i]) & (all_idx != i)]  # diagnostics only
        cand_pos = same_cat

        # If category is too small, avoid noisy GL/global smoothing.
        if len(cand_pos) < 1:
            pos = [i] * K
            pos_scores = [0.0] * K
            pos_feats = [np.zeros(len(edge_feature_names), dtype=np.float32) for _ in range(K)]
        else:
            # Positive relation: same-category ASINs with close historical total/buybox/instock/demand,
            # similar HBT, top10-brand state, promotion regime, and list-price tier.
            pos_score, _, pos_info = _relation_components(i, cand_pos)
            pos_edge_feat_mat = _edge_feature_matrix(i, cand_pos)
            # Keep a tiny KNN similarity tie-breaker so near-identical profiles rank stably.
            sim = X_knn[cand_pos] @ X_knn[i]
            score = pos_score + 0.03 * sim
            order = np.argsort(-score)
            selected = order[:K]
            pos = [int(cand_pos[j]) for j in selected]
            pos_scores = [float(score[j]) for j in selected]
            pos_feats = [pos_edge_feat_mat[j] for j in selected]
            if not pos:
                pos = _fallback_neighbors(i, K)
                pos_scores = [0.0] * len(pos)
                pos_feats = [np.zeros(len(edge_feature_names), dtype=np.float32) for _ in pos]
            while len(pos) < K:
                pos.append(pos[-1])
                pos_scores.append(pos_scores[-1] if pos_scores else 0.0)
                pos_feats.append(pos_feats[-1].copy() if pos_feats else np.zeros(len(edge_feature_names), dtype=np.float32))
        neigh_rows.append(pos[:K])
        pos_score_rows.append(pos_scores[:K])
        pos_edge_feat_rows.append(pos_feats[:K])

        # Competitive/contrast relation: STRICT same-category but far magnitude/demand/active buckets
        # or much stronger/weaker strength. No GL fallback, to avoid cross-category smoothing.
        cand_comp = same_cat
        if len(cand_comp) < 1:
            comp = [i] * K
            comp_scores = [0.0] * K
            comp_feats = [np.zeros(len(edge_feature_names), dtype=np.float32) for _ in range(K)]
        else:
            # Competitive relation: same-category ASINs with clear stronger/weaker or regime contrast:
            # funnel/demand gap + HBT/top10 brand/list-price/promotion contrast.
            _, comp_score, comp_info = _relation_components(i, cand_comp)
            comp_edge_feat_mat = _edge_feature_matrix(i, cand_comp)
            order = np.argsort(-comp_score)
            selected = order[:K]
            comp = [int(cand_comp[j]) for j in selected]
            comp_scores = [float(comp_score[j]) for j in selected]
            comp_feats = [comp_edge_feat_mat[j] for j in selected]
            if not comp:
                comp = _fallback_neighbors(i, K)
                comp_scores = [0.0] * len(comp)
                comp_feats = [np.zeros(len(edge_feature_names), dtype=np.float32) for _ in comp]
            while len(comp) < K:
                comp.append(comp[-1])
                comp_scores.append(comp_scores[-1] if comp_scores else 0.0)
                comp_feats.append(comp_feats[-1].copy() if comp_feats else np.zeros(len(edge_feature_names), dtype=np.float32))
        comp_rows.append(comp[:K])
        comp_score_rows.append(comp_scores[:K])
        comp_edge_feat_rows.append(comp_feats[:K])

    neigh_idx = np.asarray(neigh_rows, dtype=np.int64)
    comp_idx = np.asarray(comp_rows, dtype=np.int64)
    pos_edge_score = np.asarray(pos_score_rows, dtype=np.float32) if len(pos_score_rows) else np.zeros((N, K), dtype=np.float32)
    comp_edge_score = np.asarray(comp_score_rows, dtype=np.float32) if len(comp_score_rows) else np.zeros((N, K), dtype=np.float32)
    pos_edge_features = np.asarray(pos_edge_feat_rows, dtype=np.float32) if len(pos_edge_feat_rows) else np.zeros((N, K, len(edge_feature_names)), dtype=np.float32)
    comp_edge_features = np.asarray(comp_edge_feat_rows, dtype=np.float32) if len(comp_edge_feat_rows) else np.zeros((N, K, len(edge_feature_names)), dtype=np.float32)
    # Standardize edge-feature channels across all stored candidate edges. This lets EdgeMLP train stably.
    ef_all = np.concatenate([pos_edge_features.reshape(-1, len(edge_feature_names)), comp_edge_features.reshape(-1, len(edge_feature_names))], axis=0)
    ef_mean = np.nanmean(ef_all, axis=0, keepdims=True)
    ef_std = np.nanstd(ef_all, axis=0, keepdims=True) + 1e-8
    pos_edge_features = (pos_edge_features - ef_mean.reshape(1, 1, -1)) / ef_std.reshape(1, 1, -1)
    comp_edge_features = (comp_edge_features - ef_mean.reshape(1, 1, -1)) / ef_std.reshape(1, 1, -1)
    pos_edge_features = np.nan_to_num(pos_edge_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    comp_edge_features = np.nan_to_num(comp_edge_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # Standardize edge scores before using them as GAT attention bias.
    pos_edge_score = (pos_edge_score - np.nanmean(pos_edge_score)) / (np.nanstd(pos_edge_score) + 1e-8)
    comp_edge_score = (comp_edge_score - np.nanmean(comp_edge_score)) / (np.nanstd(comp_edge_score) + 1e-8)

    asin_list = feat["asin"].astype(str).tolist()
    asin_to_idx = {a: i for i, a in enumerate(asin_list)}

    if verbose:
        print("\n" + "=" * 100)
        print("DUAL-RELATION GRAPHSAGE ASSET BUILD")
        print("=" * 100)
        print(f"Nodes: {N} | K={K} | node_feat_dim={len(node_feature_cols)}")
        print(f"Weights: zero={graph_zero_weight}, level_peak={graph_level_peak_weight}, demand<=1.25, transition={graph_transition_weight}, static={graph_static_weight}, brand={graph_brand_weight}")
        print("Key graph features V10: dynamic-relation edge scorer using total/buybox/instock/demand + HBT + top10 brand + ind_promotion + list_price")
        try:
            print("Magnitude bucket counts:", feat["magnitude_bucket"].value_counts().sort_index().to_dict())
            print("Active bucket counts:", feat["active_bucket"].value_counts().sort_index().to_dict())
        except Exception:
            pass
        try:
            nb = neigh_idx
            same_gl = []
            same_cat = []
            same_brand = []
            gl_arr = meta["gl_product_group"].astype(str).values
            cat_arr = meta["category_code"].astype(str).values
            br_arr = meta["ind_top10_brand"].astype(float).values
            for i in range(N):
                same_gl.append(np.mean(gl_arr[nb[i]] == gl_arr[i]))
                same_cat.append(np.mean(cat_arr[nb[i]] == cat_arr[i]))
                same_brand.append(np.mean(br_arr[nb[i]] == br_arr[i]))
            print(f"Positive-neighbor homophily (category-only; GL static): same_GL={np.mean(same_gl):.3f} | same_category={np.mean(same_cat):.3f} | same_top10_brand_state={np.mean(same_brand):.3f}")
            try:
                cb = comp_idx
                comp_same_cat = []
                comp_diff_hbt = []
                comp_stronger_brand = []
                hbt_arr = meta["hbt"].astype(str).values if "hbt" in meta.columns else np.array(["missing"] * N)
                for i in range(N):
                    comp_same_cat.append(np.mean(cat_arr[cb[i]] == cat_arr[i]))
                    comp_diff_hbt.append(np.mean(hbt_arr[cb[i]] != hbt_arr[i]))
                    comp_stronger_brand.append(np.mean(br_arr[cb[i]] > br_arr[i]))
                print(f"Competitive-neighbor diagnostic: same_category={np.mean(comp_same_cat):.3f} | diff_HBT={np.mean(comp_diff_hbt):.3f} | stronger_top10_brand={np.mean(comp_stronger_brand):.3f}")
                print(f"Relation edge score diagnostics: pos_score_mean={float(np.nanmean(pos_edge_score)):.3f}, pos_score_std={float(np.nanstd(pos_edge_score)):.3f} | comp_score_mean={float(np.nanmean(comp_edge_score)):.3f}, comp_score_std={float(np.nanstd(comp_edge_score)):.3f}")
            except Exception as e2:
                print(f"Competitive-neighbor diagnostic skipped: {e2}")
        except Exception as e:
            print(f"Neighbor homophily diagnostic skipped: {e}")

    raw_feature_df = pd.concat([feat[["asin"]].reset_index(drop=True), feat[node_feature_cols].reset_index(drop=True)], axis=1)

    rank_feature_cols = [
        "rank_composite_dynamic", "cat_rank_recent13_strength", "cat_rank_long52_strength",
        "cat_rank_momentum_pct", "rank_zero_good", "rank_promo_boost",
        "recent13_active_eligibility", "recent13_zero_mean", "promo_current", "promo_event_current",
        "event_current", "promo_response_strength", "rank_stability_abs_gap",
    ]
    for c in rank_feature_cols:
        if c not in feat.columns:
            feat[c] = 0.0
    rank_raw = feat[rank_feature_cols].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0).values
    # Keep [0,1]-like rank priors mostly raw; z-score only the unbounded promo_response/stability if needed.
    rank_node_features = np.nan_to_num(rank_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return {
        "node_features": X_std.astype(np.float32),
        "rank_node_features": rank_node_features,
        "rank_feature_names": rank_feature_cols,
        "neighbor_idx": neigh_idx.astype(np.int64),
        "competitive_neighbor_idx": comp_idx.astype(np.int64),
        "positive_edge_score": pos_edge_score.astype(np.float32),
        "competitive_edge_score": comp_edge_score.astype(np.float32),
        "positive_edge_features": pos_edge_features.astype(np.float32),
        "competitive_edge_features": comp_edge_features.astype(np.float32),
        "edge_feature_names": edge_feature_names,
        "asin_to_idx": asin_to_idx,
        "idx_to_asin": asin_list,
        "node_feature_names": node_feature_cols,
        "raw_feature_df": raw_feature_df,
        "meta_df": meta.reset_index(drop=True),
        "feature_groups": {
            "zero": zero_cols,
            "level_peak": level_peak_cols,
            "demand": demand_cols,
            "transition": transition_cols,
            "static": static_cols,
            "brand": brand_cols,
            "rank": rank_feature_cols,
        },
        "weights": {
            "zero": graph_zero_weight,
            "level_peak": graph_level_peak_weight,
            "transition": graph_transition_weight,
            "static": graph_static_weight,
            "brand": graph_brand_weight,
        },
    }


def load_exposure_data(data_raw, dph_cap_q=0.995, use_graphsage=False, graph_horizon=20, neighbor_k=10, graph_zero_weight=0.03, graph_level_peak_weight=2.2, graph_transition_weight=1.0, graph_static_weight=1.0, graph_brand_weight=0.3):
    df = data_raw.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    df = df.sort_values(["asin", "order_week"]).reset_index(drop=True)

    for c in ["fbi_demand", "total_dph", "buy_box_dph", "in_stock_dph"]:
        df[c] = _safe_numeric(df[c]).clip(lower=0.0)

    for c in ["total_dph", "buy_box_dph", "in_stock_dph"]:
        cap = df[c].quantile(dph_cap_q)
        df[c] = df[c].clip(upper=cap)

    df["our_price"] = _safe_numeric(df.get("our_price", 0.0)).clip(lower=0.0)
    df["scot_oos"]  = _safe_numeric(df.get("scot_oos",  0.0)).clip(0, 1)
    # IMPORTANT: scot_oos is used as historical input only. It is NOT included in
    # future_context and there is no future OOS hard mask, avoiding leakage.

    # ── 新增动态特征 ──────────────────────────────────────────
    # ind_promotion：动态binary，99.1% ASIN有变化，进active_head
    if "ind_promotion" in df.columns:
        df["ind_promotion"] = _safe_numeric(df["ind_promotion"]).clip(0, 1)
    else:
        df["ind_promotion"] = 0.0

    # ── Known future promotion context (V18) ─────────────────────────────
    # Assumption for this ablation: future promotion schedule/discount fields are known
    # at forecast origin. These columns are therefore kept in future_context for h=1..20.
    # If this is not true in production, disable/remove these columns to avoid leakage.
    df["known_future_ind_promotion"] = df["ind_promotion"].astype(float).clip(0, 1)

    if "promotion_ratio" in df.columns:
        df["known_future_promo_ratio"] = _safe_numeric(df["promotion_ratio"]).clip(lower=0.0).fillna(0.0)
    else:
        df["known_future_promo_ratio"] = 0.0

    if "promotion_amount" in df.columns:
        df["known_future_promo_amount_log"] = np.log1p(_safe_numeric(df["promotion_amount"]).clip(lower=0.0).fillna(0.0))
    else:
        df["known_future_promo_amount_log"] = 0.0

    if "promotion_pricing_amount" in df.columns:
        df["known_future_promo_price_amount_log"] = np.log1p(_safe_numeric(df["promotion_pricing_amount"]).clip(lower=0.0).fillna(0.0))
    else:
        df["known_future_promo_price_amount_log"] = 0.0

    # Promotion type can carry different effects. Encode as frequency + code-like value.
    if "promotion_type" in df.columns:
        promo_type = df["promotion_type"].astype(str).fillna("unknown")
        vc = promo_type.value_counts(normalize=True)
        type_map = {v: i + 1 for i, v in enumerate(sorted(promo_type.unique()))}
        df["known_future_promo_type_freq"] = promo_type.map(vc).fillna(0.0).astype(float)
        df["known_future_promo_type_code"] = promo_type.map(type_map).fillna(0.0).astype(float)
        # Normalize code to [0,1] to keep scale small.
        mx = max(float(df["known_future_promo_type_code"].max()), 1.0)
        df["known_future_promo_type_code"] = df["known_future_promo_type_code"] / mx
    else:
        df["known_future_promo_type_freq"] = 0.0
        df["known_future_promo_type_code"] = 0.0

    # ind_prime_week：动态binary，3.7%是PrimeDay周，进active_head
    if "ind_prime_week" in df.columns:
        df["ind_prime_week"] = _safe_numeric(df["ind_prime_week"]).clip(0, 1)
    else:
        df["ind_prime_week"] = 0.0

    # customer_active_review_count：动态，极度右偏，log变换后进mag_head
    if "customer_active_review_count" in df.columns:
        df["log_review_count"] = np.log1p(
            _safe_numeric(df["customer_active_review_count"]).clip(lower=0.0)
        )
    else:
        df["log_review_count"] = 0.0

    # ── 全局price log变换（修复：原来是per-ASIN归一化，丢失跨ASIN信息）
    # raw skew=19.6，log1p之后skew=-0.046，分布完美正态
    global_price_log = np.log1p(df["our_price"])
    # 全局标准化保留价格水平信息
    price_mean = global_price_log.mean()
    price_std  = global_price_log.std() + 1e-8
    df["our_price_log_norm"] = (global_price_log - price_mean) / price_std

    df["order_month"]  = df["order_week"].dt.month.astype(float)
    df["month_sin"]    = np.sin(2 * np.pi * df["order_month"] / 12.0)
    df["month_cos"]    = np.cos(2 * np.pi * df["order_month"] / 12.0)
    df["season_winter"] = df["order_month"].isin([12, 1, 2]).astype(float)
    df["season_spring"] = df["order_month"].isin([3, 4, 5]).astype(float)
    df["season_summer"] = df["order_month"].isin([6, 7, 8]).astype(float)
    df["season_fall"]   = df["order_month"].isin([9, 10, 11]).astype(float)

    df, explicit_event_cols = add_explicit_event_features(df, week_col="order_week")
    df, static_cols = _encode_static_features(df)

    holiday_cols  = [c for c in df.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in df.columns if c.startswith("distance_")]
    for c in holiday_cols + distance_cols:
        df[c] = _safe_numeric(df[c])

    context_cols = list(dict.fromkeys(
        # ── 动态特征（时间驱动，进active_head）──────────────
        [
            "ind_promotion",                  # kept for backward compatibility
            "known_future_ind_promotion",     # V18: known future promo flag
            "known_future_promo_ratio",       # V18: known future discount intensity
            "known_future_promo_amount_log",
            "known_future_promo_price_amount_log",
            "known_future_promo_type_freq",
            "known_future_promo_type_code",
            "ind_prime_week",
        ]
        + holiday_cols
        + distance_cols
        + explicit_event_cols
        + ["order_month", "month_sin", "month_cos",
           "season_winter", "season_spring", "season_summer", "season_fall"]
        # ── 商品特征（进mag_head）────────────────────────────
        + ["our_price_log_norm", "log_review_count"]
        + static_cols
        # ── 历史anchor──────────────────────────────────────
        + [
            "hist_total_dph_last_log",   "hist_total_dph_mean4_log",   "hist_total_dph_mean13_log",
            "hist_buy_box_dph_last_log", "hist_buy_box_dph_mean4_log", "hist_buy_box_dph_mean13_log",
            "hist_instock_dph_last_log", "hist_instock_dph_mean4_log", "hist_instock_dph_mean13_log",
            "hist_demand_last_log", "hist_demand_mean4_log", "hist_demand_mean13_log",
            "hist_demand_active_rate",
        ]
    ))

    for c in context_cols:
        if c not in df.columns:
            df[c] = 0.0

    data = {}
    for asin, g in df.groupby("asin"):
        g = g.sort_values("order_week").reset_index(drop=True)
        demand  = g["fbi_demand"].values.astype(np.float32)
        total   = g["total_dph"].values.astype(np.float32)
        buy     = g["buy_box_dph"].values.astype(np.float32)
        instock = g["in_stock_dph"].values.astype(np.float32)
        oos     = g["scot_oos"].values.astype(np.float32)

        # ── price改成全局log归一化（不再per-ASIN归一化）────
        price_log_norm = g["our_price_log_norm"].values.astype(np.float32)

        # ── encoder历史特征（9维→11维）─────────────────────
        # 新增：log_review_count（mag信号）, ind_promotion（active信号）
        week_idx = np.arange(len(g))

        # ── 月份/季节特征 ─────────────────────────────────────
        month_sin  = g["month_sin"].values.astype(np.float32)
        month_cos  = g["month_cos"].values.astype(np.float32)
        season_w   = g["season_winter"].values.astype(np.float32)
        season_su  = g["season_summer"].values.astype(np.float32)

        # ── 절假日/事件特征（如果存在）───────────────────────
        is_event   = g["is_event_window"].values.astype(np.float32) \
                     if "is_event_window" in g.columns else np.zeros(len(g), dtype=np.float32)
        pre_event  = g["pre_event_proximity"].values.astype(np.float32) \
                     if "pre_event_proximity" in g.columns else np.zeros(len(g), dtype=np.float32)
        post_event = g["post_event_decay"].values.astype(np.float32) \
                     if "post_event_decay" in g.columns else np.zeros(len(g), dtype=np.float32)
        ind_prime  = g["ind_prime_week"].values.astype(np.float32) \
                     if "ind_prime_week" in g.columns else np.zeros(len(g), dtype=np.float32)

        # ── GL静态特征（每周重复同一个值）─────────────────────
        # 让encoder学到不同GL在不同季节/月份的DPH规律
        # TCN会自动学 GL×季节 的交互，不需要手动写交叉特征
        gl_code = g["stock_static__gl_product_group__code"].values.astype(np.float32) \
                  if "stock_static__gl_product_group__code" in g.columns \
                  else np.zeros(len(g), dtype=np.float32)
        gl_freq = g["stock_static__gl_product_group__freq"].values.astype(np.float32) \
                  if "stock_static__gl_product_group__freq" in g.columns \
                  else np.zeros(len(g), dtype=np.float32)

        # ── Category静态特征：比GL更细，帮助区分同GL内部zero/peak差异 ─────
        cat_code = g["stock_static__category_code__code"].values.astype(np.float32) \
                   if "stock_static__category_code__code" in g.columns \
                   else np.zeros(len(g), dtype=np.float32)
        cat_freq = g["stock_static__category_code__freq"].values.astype(np.float32) \
                   if "stock_static__category_code__freq" in g.columns \
                   else np.zeros(len(g), dtype=np.float32)
        cat_unknown = g["stock_static__category_code__is_unknown"].values.astype(np.float32) \
                      if "stock_static__category_code__is_unknown" in g.columns \
                      else np.zeros(len(g), dtype=np.float32)

        # ── encoder历史特征（19→22维，如果有category_code）────────────────
        features = np.stack([
            np.log1p(demand),                               # 历史需求
            (demand > 0).astype(float),                     # 需求active
            np.log1p(total),                                # 历史total_dph
            np.log1p(buy),                                  # 历史buy_box_dph
            np.log1p(instock),                              # 历史instock_dph
            price_log_norm,                                 # 全局log归一化价格
            oos,                                            # 缺货信号
            np.sin(2 * np.pi * week_idx / 52.0),           # 年内周期sin
            np.cos(2 * np.pi * week_idx / 52.0),           # 年内周期cos
            g["log_review_count"].values.astype(np.float32),  # 评论数
            g["ind_promotion"].values.astype(np.float32),     # 促销标记
            month_sin,    # 月份sin
            month_cos,    # 月份cos
            season_w,     # 冬季（感恩节/圣诞）
            season_su,    # 夏季（PrimeDay/户外）
            pre_event,    # 节假日临近程度
            post_event,   # 节假日后衰减
            # ── 新增：GL品类（让encoder学GL×季节交互）────────
            gl_code,      # GL编码（办公/园艺/家具等）
            gl_freq,      # GL频率（品类大小）
            cat_code,     # category_code编码（细粒度品类）
            cat_freq,     # category_code频率（类别大小/稀疏度）
            cat_unknown,  # category_code是否unknown（catalog缺失信号）
        ], axis=1).astype(np.float32)

        data[asin] = {
            "week":           g["order_week"].values,
            "features":       features,
            "demand":         demand,
            "total_dph":      total,
            "buy_box_dph":    buy,
            "in_stock_dph":   instock,
            "future_context": g[context_cols].values.astype(np.float32),
            "context_cols":   context_cols,
        }

    enc_dim = next(iter(data.values()))["features"].shape[1] if len(data) else 0
    print(f"ASINs: {len(data)} | Context dim: {len(context_cols)} | Encoder dim: {enc_dim}")
    known_promo_cols = [c for c in context_cols if c.startswith("known_future_promo") or c == "known_future_ind_promotion"]
    print(f"V18 known future promotion columns enabled: {known_promo_cols}")
    if "category_code" in df.columns:
        n_cat = df["category_code"].astype(str).nunique()
        unk_rate = df.get("stock_static__category_code__is_unknown", pd.Series(0, index=df.index)).mean()
        print(f"Category code enabled: n_category={n_cat} | unknown_rate={unk_rate:.4f}")

    graph_assets = None
    if use_graphsage:
        graph_assets = _build_graphsage_assets(
            df,
            graph_horizon=graph_horizon,
            neighbor_k=neighbor_k,
            graph_zero_weight=graph_zero_weight,
            graph_level_peak_weight=graph_level_peak_weight,
            graph_transition_weight=graph_transition_weight,
            graph_static_weight=graph_static_weight,
            graph_brand_weight=graph_brand_weight,
            verbose=True,
        )
        # Attach ASIN index for dataset batches. Missing ASINs are assigned 0 defensively.
        asin_to_idx = graph_assets["asin_to_idx"]
        for a, dct in data.items():
            dct["asin_idx"] = int(asin_to_idx.get(str(a), 0))

    return data, len(context_cols), context_cols, graph_assets


# ============================================================
# Dataset
# 改动：_make_future_context 加 horizon decay
# ============================================================

class ExposureDataset(Dataset):
    def __init__(self, data, history=13, horizon=20, mode="train",
                 val_weeks=20, anchor_decay=0.08):
        self.samples = []
        self.data = data
        self.history = history
        self.horizon = horizon
        self.anchor_decay = anchor_decay  # 新增：控制anchor衰减速度

        for asin, d in data.items():
            T = len(d["features"])
            if mode == "train":
                starts = range(max(0, T - val_weeks - horizon - history + 1))
            else:
                s = T - history - horizon
                starts = [s] if s >= 0 else []
            for start in starts:
                self.samples.append((asin, start))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _hist_mean(arr, end, window):
        x = arr[max(0, end - window):end]
        return float(np.mean(x)) if len(x) > 0 else 0.0

    def _make_future_context(self, d, start):
        h  = self.history
        H  = self.horizon
        fc = d["future_context"][start+h:start+h+H].copy()
        cols = d["context_cols"]
        idx  = {c: i for i, c in enumerate(cols)}
        end  = start + h

        # Freeze dynamic review count at forecast origin to avoid future realized review leakage.
        # review_count=0 is a useful zero/exposure signal, but future true review_count should not be used.
        if "log_review_count" in idx and end > 0:
            fc[:, idx["log_review_count"]] = d["future_context"][end - 1, idx["log_review_count"]]

        # V18 known-promotion ablation:
        # known_future_* promotion columns are intentionally NOT frozen here.
        # They remain horizon-specific future covariates under the assumption that
        # promotion schedule / discount info is available at forecast origin.

        total   = d["total_dph"]
        buy     = d["buy_box_dph"]
        instock = d["in_stock_dph"]
        demand  = d["demand"]   # 新增

        # ── anchor随horizon衰减 + post_event校正 ──────────────
        # 两层校正：
        #   1. horizon decay：随h增大向mean13收缩（已有）
        #   2. post_event decay：如果历史末尾是节假日峰值，
        #      对last_val做校正，避免把峰值传播到所有h的anchor

        # 从future_context里读post_event_decay（第一个h的值，代表当前时刻的节假日位置）
        # post_event_decay在context_cols里，h=0时的值反映"历史末尾距节假日多远"
        post_event_col = "post_event_decay"
        if post_event_col in idx:
            # 用预测起始时刻（h=0）的post_event_decay校正last_val
            # 节假日刚过（decay≈1）→ last_val可信；节假日过了很久（decay≈0）→ last_val不可信
            current_post_decay = float(fc[0, idx[post_event_col]])
        else:
            current_post_decay = 1.0  # 没有这个特征就不校正

        for step_h in range(H):
            # horizon decay：越远越收缩到mean13
            h_decay = np.exp(-self.anchor_decay * step_h)

            for prefix, arr in [("total", total), ("buy_box", buy), ("instock", instock)]:
                mean13_val = np.log1p(self._hist_mean(arr, end, 13))
                mean4_val  = np.log1p(self._hist_mean(arr, end, 4))
                raw_last   = np.log1p(arr[end - 1]) if end > 0 else 0.0

                # post_event校正：节假日后的峰值向mean13收缩
                # current_post_decay≈1（刚过节假日）→ last_val被大幅校正
                # current_post_decay≈0（很久以前的节假日）→ last_val基本不变
                # 校正公式：corrected = last * (1-post_decay) + mean13 * post_decay
                # 注意：post_decay越大说明越靠近节假日，此时反而需要校正
                # 感恩节后1周: post_decay≈0.86 → last_val被压向mean13
                # 正常周:       post_decay≈0.05 → last_val基本不变
                post_strength = 0.5
                effective_post_decay = post_strength * current_post_decay
                last_val = (
                    raw_last * (1.0 - effective_post_decay)
                    + mean13_val * effective_post_decay
                )

                key_map = {
                    f"hist_{prefix}_dph_last_log":   h_decay * last_val  + (1 - h_decay) * mean13_val,
                    f"hist_{prefix}_dph_mean4_log":  h_decay * mean4_val + (1 - h_decay) * mean13_val,
                    f"hist_{prefix}_dph_mean13_log": mean13_val,
                }
                for col, val in key_map.items():
                    if col in idx:
                        fc[step_h, idx[col]] = val

        # ── demand anchor（所有h用同一个历史值，demand无需decay）──
        # EDA显示demand领先instock corr=0.676，加入作为近期活跃信号
        # demand没有节假日峰值校正的问题（demand本身就是真实信号）
        demand_last   = np.log1p(demand[end - 1]) if end > 0 else 0.0
        demand_mean4  = np.log1p(self._hist_mean(demand, end, 4))
        demand_mean13 = np.log1p(self._hist_mean(demand, end, 13))
        demand_active_rate = float(np.mean(demand[max(0, end-13):end] > 0)) if end > 0 else 0.0

        for step_h in range(H):
            h_decay = np.exp(-self.anchor_decay * step_h)
            # demand anchor也随h衰减（近期更可信）
            demand_anchor = h_decay * demand_last + (1 - h_decay) * demand_mean13
            for col, val in [
                ("hist_demand_last_log",    demand_anchor),
                ("hist_demand_mean4_log",   h_decay * demand_mean4  + (1 - h_decay) * demand_mean13),
                ("hist_demand_mean13_log",  demand_mean13),
                ("hist_demand_active_rate", demand_active_rate),
            ]:
                if col in idx:
                    fc[step_h, idx[col]] = val

        return fc

    def __getitem__(self, i):
        asin, start = self.samples[i]
        d = self.data[asin]
        h = self.history
        H = self.horizon

        return {
            "asin": asin,
            "target_week": [str(w)[:10] for w in d["week"][start+h:start+h+H]],
            "x":              torch.tensor(d["features"][start:start+h], dtype=torch.float32),
            "future_context": torch.tensor(self._make_future_context(d, start), dtype=torch.float32),
            "future_total_dph":    torch.tensor(d["total_dph"][start+h:start+h+H],    dtype=torch.float32),
            "future_buy_box_dph":  torch.tensor(d["buy_box_dph"][start+h:start+h+H],  dtype=torch.float32),
            "future_instock_dph":  torch.tensor(d["in_stock_dph"][start+h:start+h+H], dtype=torch.float32),
            "future_demand":       torch.tensor(d["demand"][start+h:start+h+H],        dtype=torch.float32),
            "asin_idx":            torch.tensor(int(d.get("asin_idx", 0)), dtype=torch.long),
        }


# ============================================================
# Collate function: keep target_week as [B][H]
# ============================================================

def exposure_collate(batch):
    tensor_keys = [
        "x",
        "future_context",
        "future_total_dph",
        "future_buy_box_dph",
        "future_instock_dph",
        "future_demand",
        "asin_idx",
    ]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in tensor_keys}
    out["asin"] = [b["asin"] for b in batch]
    out["target_week"] = [b["target_week"] for b in batch]
    return out


# ============================================================
# Model V2：TCN全序列Encoder + TCN Decoder + Cross-Attention
# ============================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class ExposureAwareEncoderSelfAttention(nn.Module):
    """
    Sparse / exposure-aware self-attention inside the history encoder.

    It is designed for sparse exposure series:
      - down-weight all-zero history weeks,
      - up-weight active / peak weeks from demand and DPH history,
      - keep residual + layer norm for stability.

    Expected raw input feature indices from load_exposure_data():
      0 = log1p(demand)
      2 = log1p(total_dph)
      3 = log1p(buy_box_dph)
      4 = log1p(in_stock_dph)
    """
    def __init__(self, d_model=64, n_heads=4, dropout=0.15,
                 zero_penalty=2.0, active_bias=1.0, peak_bias=1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.zero_penalty = zero_penalty
        self.active_bias = active_bias
        self.peak_bias = peak_bias

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_out, x_raw):
        B, T, D = enc_out.shape

        demand_log = x_raw[:, :, 0]
        total_log = x_raw[:, :, 2]
        buy_log = x_raw[:, :, 3]
        instock_log = x_raw[:, :, 4]

        active_score = (
            (demand_log > 0).float()
            + (total_log > 0).float()
            + (buy_log > 0).float()
            + (instock_log > 0).float()
        ).clamp(max=1.0)

        peak_level = (
            torch.expm1(demand_log).clamp(min=0.0)
            + torch.expm1(total_log).clamp(min=0.0)
            + torch.expm1(buy_log).clamp(min=0.0)
            + torch.expm1(instock_log).clamp(min=0.0)
        )
        peak_score = torch.sqrt(peak_level + 1e-6)
        peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)

        q = self.q_proj(enc_out).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(enc_out).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(enc_out).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)

        key_bias = (
            self.active_bias * active_score
            + self.peak_bias * peak_norm
            - self.zero_penalty * (1.0 - active_score)
        )
        scores = scores + key_bias[:, None, None, :]

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return self.norm(enc_out + out)


class HistoryEncoderFull(nn.Module):
    """
    TCN Encoder，输出全序列 [B, T, D]。
    TCN 后可选一层 exposure-aware self-attention，适合 0 很多的 exposure 序列。
    """
    def __init__(self, input_dim, d_model=64, n_heads=4, dropout=0.15,
                 use_self_attn=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        dilations = [1, 2, 4, 8, 13, 26]
        self.convs = nn.ModuleList([
            CausalConv1d(d_model, d_model, kernel_size=2, dilation=d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])
        self.final_norm = nn.LayerNorm(d_model)
        self.use_self_attn = use_self_attn
        self.self_attn = ExposureAwareEncoderSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            zero_penalty=2.0,
            active_bias=1.0,
            peak_bias=1.0,
        ) if use_self_attn else None

    def forward(self, x):
        h = self.input_proj(x).transpose(1, 2)

        for conv, norm in zip(self.convs, self.norms):
            z = conv(h)
            h = h + z
            h = h.transpose(1, 2)
            h = norm(h)
            h = F.gelu(h)
            h = h.transpose(1, 2)

        enc_out = self.final_norm(h.transpose(1, 2))

        if self.self_attn is not None:
            enc_out = self.self_attn(enc_out, x)

        return enc_out


class HorizonTCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.10):
        super().__init__()
        padding    = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        z   = x.transpose(1, 2)
        z   = self.drop(F.relu(self.conv1(z)))
        z   = self.drop(F.relu(self.conv2(z)))
        z   = z.transpose(1, 2)
        m   = min(z.shape[1], res.shape[1])
        return self.norm(res[:, :m, :] + z[:, :m, :])



class DualRelationalGATEncoder(nn.Module):
    """
    Lightweight dual-relation GAT encoder for ASIN graphs.

    V11 dynamic relation + rank-magnitude correction version:
      - keeps the original positive and competitive candidate edges;
      - learns attention weights over each relation separately;
      - can return self / positive / competitive messages and attention entropy
        so the downstream decoder can learn how much positive vs competitive
        graph signal to use per horizon.
    """
    def __init__(self, node_feat_dim, graph_dim=16, dropout=0.10,
                 neighbor_message_scale=0.20, n_heads=4,
                 attention_leaky_slope=0.2,
                 edge_feat_dim=0, use_learned_edge_score=True,
                 learned_edge_score_scale=1.0, rule_edge_prior_scale=0.25):
        super().__init__()
        self.neighbor_message_scale = float(neighbor_message_scale)
        self.graph_dim = int(graph_dim)
        self.n_heads = int(max(1, n_heads))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(attention_leaky_slope)
        self.edge_feat_dim = int(edge_feat_dim or 0)
        self.use_learned_edge_score = bool(use_learned_edge_score and self.edge_feat_dim > 0)
        self.learned_edge_score_scale = float(learned_edge_score_scale)
        self.rule_edge_prior_scale = float(rule_edge_prior_scale)

        self.self_proj = nn.Linear(node_feat_dim, graph_dim)
        self.pos_proj = nn.Linear(node_feat_dim, graph_dim)
        self.comp_proj = nn.Linear(node_feat_dim, graph_dim)

        self.pos_attn = nn.Parameter(torch.empty(self.n_heads, graph_dim * 2))
        self.comp_attn = nn.Parameter(torch.empty(self.n_heads, graph_dim * 2))
        nn.init.xavier_uniform_(self.pos_attn)
        nn.init.xavier_uniform_(self.comp_attn)

        self.pos_head_mix = nn.Linear(graph_dim * self.n_heads, graph_dim)
        self.comp_head_mix = nn.Linear(graph_dim * self.n_heads, graph_dim)

        if self.use_learned_edge_score:
            hidden = max(32, min(128, self.edge_feat_dim * 4))
            self.pos_edge_mlp = nn.Sequential(
                nn.Linear(self.edge_feat_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh(),
            )
            self.comp_edge_mlp = nn.Sequential(
                nn.Linear(self.edge_feat_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh(),
            )
        else:
            self.pos_edge_mlp = None
            self.comp_edge_mlp = None

        self.out = nn.Sequential(
            nn.Linear(graph_dim * 3, graph_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, graph_dim),
            nn.LayerNorm(graph_dim),
        )

    def _make_edge_bias(self, rule_edge_bias=None, edge_features=None, relation="pos"):
        bias = None
        if rule_edge_bias is not None:
            bias = self.rule_edge_prior_scale * rule_edge_bias
        if self.use_learned_edge_score and edge_features is not None:
            mlp = self.pos_edge_mlp if relation == "pos" else self.comp_edge_mlp
            learned = mlp(edge_features).squeeze(-1) * self.learned_edge_score_scale
            bias = learned if bias is None else bias + learned
        return bias

    def _gat_message(self, h_self, h_neigh, attn_vec, return_alpha=False, edge_bias=None):
        """
        h_self:  [N,G]
        h_neigh: [N,K,G]
        attn_vec:[heads,2G]
        returns msg [N,G], optionally alpha [N,K,heads].
        """
        N, K, G = h_neigh.shape
        self_rep = h_self[:, None, :].expand(N, K, G)
        pair = torch.cat([self_rep, h_neigh], dim=-1)  # [N,K,2G]

        score = torch.einsum('nkd,hd->nkh', pair, attn_vec)
        # Optional relation-score attention bias from the dynamic relation graph.
        # Shape edge_bias=[N,K]; it is standardized in graph_assets.
        if edge_bias is not None:
            score = score + edge_bias[:, :, None].to(score.device, dtype=score.dtype)
        score = self.leaky_relu(score)
        alpha = torch.softmax(score, dim=1)
        alpha_drop = self.dropout(alpha)

        msg = torch.einsum('nkh,nkg->nhg', alpha_drop, h_neigh)
        msg = msg.reshape(N, self.n_heads * G)
        if return_alpha:
            return msg, alpha
        return msg

    @staticmethod
    def _entropy_from_alpha(alpha):
        # alpha [N,K,heads] -> [N]
        eps = 1e-8
        ent = -(alpha.clamp_min(eps) * torch.log(alpha.clamp_min(eps))).sum(dim=1)  # [N,heads]
        return ent.mean(dim=1)

    def forward(self, node_features, neighbor_idx, competitive_neighbor_idx=None, return_aux=False,
                positive_edge_score=None, competitive_edge_score=None,
                positive_edge_features=None, competitive_edge_features=None):
        h_self = self.self_proj(node_features)  # [N,G]

        pos_raw = node_features[neighbor_idx]
        h_pos_neigh = self.pos_proj(pos_raw)
        pos_bias = self._make_edge_bias(positive_edge_score, positive_edge_features, relation="pos")
        if return_aux:
            pos_msg_heads, pos_alpha = self._gat_message(h_self, h_pos_neigh, self.pos_attn, return_alpha=True, edge_bias=pos_bias)
        else:
            pos_msg_heads = self._gat_message(h_self, h_pos_neigh, self.pos_attn, return_alpha=False, edge_bias=pos_bias)
            pos_alpha = None
        h_pos = self.neighbor_message_scale * self.pos_head_mix(pos_msg_heads)

        if competitive_neighbor_idx is None:
            h_comp = torch.zeros_like(h_pos)
            comp_alpha = None
        else:
            comp_raw = node_features[competitive_neighbor_idx]
            h_comp_neigh = self.comp_proj(comp_raw)
            comp_bias = self._make_edge_bias(competitive_edge_score, competitive_edge_features, relation="comp")
            if return_aux:
                comp_msg_heads, comp_alpha = self._gat_message(h_self, h_comp_neigh, self.comp_attn, return_alpha=True, edge_bias=comp_bias)
            else:
                comp_msg_heads = self._gat_message(h_self, h_comp_neigh, self.comp_attn, return_alpha=False, edge_bias=comp_bias)
                comp_alpha = None
            h_comp = self.neighbor_message_scale * self.comp_head_mix(comp_msg_heads)

        graph_emb = self.out(torch.cat([h_self, h_pos, h_comp], dim=-1))
        if not return_aux:
            return graph_emb

        out = {
            "graph_emb": graph_emb,
            "self_msg": h_self,
            "pos_msg": h_pos,
            "comp_msg": h_comp,
            "pos_norm": torch.norm(h_pos, dim=-1),
            "comp_norm": torch.norm(h_comp, dim=-1),
            "graph_norm": torch.norm(graph_emb, dim=-1),
        }
        if pos_alpha is not None:
            out["pos_attn_entropy"] = self._entropy_from_alpha(pos_alpha)
            out["pos_attn_max"] = pos_alpha.max(dim=1).values.mean(dim=1)
        if comp_alpha is not None:
            out["comp_attn_entropy"] = self._entropy_from_alpha(comp_alpha)
            out["comp_attn_max"] = comp_alpha.max(dim=1).values.mean(dim=1)
        return out

class TCNDecoderWithCrossAttn(nn.Module):
    """
    TCN Decoder + Cross-Attention + distributional exposure head.

    Why this version:
        Recent two-head runs showed unstable compensation:
            p_active too high + mag too high + gamma stuck at lower bound.
        This version removes the final p_active^gamma * magnitude gate.

    Final forecast path:
        encoder + future_context + cross-attention
            -> direct_head
            -> log_hat = log1p(total/buy_box/in_stock DPH)

    Auxiliary active head:
        Still outputs p_active for diagnostics and a small auxiliary BCE loss,
        but p_active does NOT enter the final exposure prediction.
    """
    def __init__(self, d_model, context_dim, horizon=20,
                 hidden=96, n_heads=4, dropout=0.10,
                 anchor_indices=None,
                 active_feat_indices=None,
                 mag_feat_indices=None,
                 active_feat_dim=0,
                 mag_feat_dim=0,
                 use_enn=True,
                 z_dim=8,
                 residual_scale=2.0,
                 gate_temperature=1.0):
        super().__init__()
        self.horizon = horizon
        self.anchor_indices = anchor_indices
        self.active_feat_indices = active_feat_indices
        self.mag_feat_indices = mag_feat_indices
        self.use_enn = bool(use_enn)
        self.z_dim = int(z_dim)
        self.residual_scale = float(residual_scale)
        self.gate_temperature = float(gate_temperature)

        if self.use_enn:
            self.z_proj = nn.Sequential(
                nn.Linear(self.z_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            # V19: explicit context-conditioned latent regime generator.
            # Instead of drawing z from an unconditional N(0,I), infer z from
            # the encoder summary and horizon-level known future context.
            # Because future_context already includes known promo/date/holiday and
            # graph_emb_* columns, z is conditioned on h_t, G, and c_{t+h}.
            self.z_context_net = nn.Sequential(
                nn.Linear(d_model + context_dim, max(d_model, hidden)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(d_model, hidden), max(d_model // 2, self.z_dim * 2)),
                nn.ReLU(),
            )
            self.z_mu_head = nn.Linear(max(d_model // 2, self.z_dim * 2), self.z_dim)
            self.z_logstd_head = nn.Linear(max(d_model // 2, self.z_dim * 2), self.z_dim)
        else:
            self.z_proj = None
            self.z_context_net = None
            self.z_mu_head = None
            self.z_logstd_head = None

        # future_context + horizon position encoding -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(context_dim + 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.tcn = nn.ModuleList([
            HorizonTCNBlock(hidden, dilation=1, dropout=dropout),
            HorizonTCNBlock(hidden, dilation=2, dropout=dropout),
            HorizonTCNBlock(hidden, dilation=4, dropout=dropout),
        ])

        self.dec_proj = nn.Linear(hidden, d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_norm = nn.LayerNorm(d_model)

        # Auxiliary occurrence head. With ENN, z controls the 20-week active/zero regime.
        z_extra = d_model if self.use_enn else 0
        active_in = d_model + z_extra + max(active_feat_dim, 0)
        self.active_head = nn.Sequential(
            nn.Linear(active_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

        # Direct mean exposure head. With ENN, z controls level/peak/zero regime.
        # IMPORTANT: p_active is auxiliary only and does NOT gate final predictions.
        direct_in = d_model + z_extra + max(active_feat_dim, 0) + max(mag_feat_dim, 0)
        self.direct_head = nn.Sequential(
            nn.Linear(direct_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
            nn.Tanh(),
        )

        # Distributional over-dispersion head. This is the key difference from
        # the previous direct-softplus exposure model: prediction can be sampled
        # from a count distribution, so p50 can naturally be exactly zero.
        self.alpha_head = nn.Sequential(
            nn.Linear(direct_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )

        # Patch-specific heads. These are used instead of the generic direct_head/alpha_head
        # in forward(). The purpose is to give the long horizon its own parameters,
        # so h14-h20 does not get dominated by the easier short-horizon gradients.
        def _make_patch_head():
            return nn.Sequential(
                nn.Linear(direct_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 3),
                nn.Tanh(),
            )

        def _make_patch_alpha_head():
            return nn.Sequential(
                nn.Linear(direct_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 3),
            )

        self.direct_head_short = _make_patch_head()   # h1-h5
        self.direct_head_mid   = _make_patch_head()   # h6-h13
        self.direct_head_long  = _make_patch_head()   # h14-h20
        self.alpha_head_short  = _make_patch_alpha_head()
        self.alpha_head_mid    = _make_patch_alpha_head()
        self.alpha_head_long   = _make_patch_alpha_head()

    def forward(self, enc_out, future_context, return_aux=False, z=None):
        B, H, _ = future_context.shape

        h_idx = torch.arange(H, device=future_context.device).float()
        h_norm = h_idx.view(1, H, 1).expand(B, H, 1) / max(H, 1)
        hsin = torch.sin(2 * torch.pi * h_norm)
        hcos = torch.cos(2 * torch.pi * h_norm)

        x = torch.cat([future_context, hsin, hcos], dim=-1)
        dec = self.input_proj(x)
        for block in self.tcn:
            dec = block(dec)

        q = self.dec_proj(dec)
        attn_out, attn_w = self.cross_attn(
            q, enc_out, enc_out,
            need_weights=return_aux,
        )
        z_out = self.post_norm(q + attn_out)  # [B,H,D]

        # One latent z per ASIN-window; repeat across horizon to learn joint path regime.
        # V19: when z is not explicitly provided, generate z from q_phi(z | h_t, G, c_{t+h}).
        # - h_t: pooled encoder/cross-attended history state
        # - G: graph information is already in enc_out via early fusion and in future_context via graph_emb_*
        # - c_{t+h}: known future context, including date/holiday/known promo fields
        z_emb = None
        z_mu = None
        z_logstd = None
        if self.use_enn:
            if z is None:
                enc_summary = enc_out.mean(dim=1)              # [B,D]
                ctx_summary = future_context.mean(dim=1)       # [B,C]
                z_hidden = self.z_context_net(torch.cat([enc_summary, ctx_summary], dim=-1))
                z_mu = self.z_mu_head(z_hidden)
                z_logstd = self.z_logstd_head(z_hidden).clamp(-4.0, 2.0)
                if self.training:
                    eps = torch.randn_like(z_mu)
                    z = z_mu + eps * torch.exp(z_logstd)
                else:
                    z = z_mu
            else:
                z_mu = z
                z_logstd = torch.zeros_like(z)
            z_emb = self.z_proj(z)                         # [B,D]
            z_rep = z_emb[:, None, :].expand(B, H, -1)      # [B,H,D]
        else:
            z = None
            z_rep = None

        active_parts = [z_out]
        if z_rep is not None:
            active_parts.append(z_rep)

        active_feats = None
        if self.active_feat_indices and len(self.active_feat_indices) > 0:
            active_feats = future_context[:, :, self.active_feat_indices]
            active_parts.append(active_feats)

        active_in = torch.cat(active_parts, dim=-1)
        active_logit = self.active_head(active_in)
        # Auxiliary active probability for diagnostics/loss only.
        # It does NOT multiply the final exposure prediction.
        p_active = torch.sigmoid(active_logit / max(self.gate_temperature, 1e-6))

        direct_parts = [z_out]
        if z_rep is not None:
            direct_parts.append(z_rep)
        if active_feats is not None:
            direct_parts.append(active_feats)

        mag_feats = None
        if self.mag_feat_indices and len(self.mag_feat_indices) > 0:
            mag_feats = future_context[:, :, self.mag_feat_indices]
            direct_parts.append(mag_feats)

        direct_in = torch.cat(direct_parts, dim=-1)
        # Patch decoder:
        #   h1-h5   -> short head
        #   h6-h13  -> mid head
        #   h14-h20 -> long head
        # This directly targets the observed failure mode: short horizons are good,
        # long horizons flatten / underpredict.
        res_short = self.direct_head_short(direct_in)
        res_mid   = self.direct_head_mid(direct_in)
        res_long  = self.direct_head_long(direct_in)
        alp_short = self.alpha_head_short(direct_in)
        alp_mid   = self.alpha_head_mid(direct_in)
        alp_long  = self.alpha_head_long(direct_in)

        h_patch = torch.arange(1, H + 1, device=future_context.device).view(1, H, 1)
        m_short = (h_patch <= 5).to(direct_in.dtype)
        m_mid   = ((h_patch >= 6) & (h_patch <= 13)).to(direct_in.dtype)
        m_long  = (h_patch >= 14).to(direct_in.dtype)

        residual  = res_short * m_short + res_mid * m_mid + res_long * m_long
        alpha_raw = alp_short * m_short + alp_mid * m_mid + alp_long * m_long
        alpha = F.softplus(alpha_raw) + 1e-4

        # Anchor-residual mean log forecast.
        if self.anchor_indices is not None:
            ti, bi, ii = self.anchor_indices
            anchor = torch.stack([
                future_context[:, :, ti],
                future_context[:, :, bi],
                future_context[:, :, ii],
            ], dim=-1)
            raw_log_mag = anchor + residual * self.residual_scale
        else:
            raw_log_mag = residual * self.residual_scale

        log_mag = F.softplus(raw_log_mag)
        mag_level = torch.expm1(log_mag).clamp(min=0.0)

        # NO multiplicative gate. This stays a single-head direct forecast.
        # z enters the direct head, so zero/peak/transition regimes are learned by
        # shifting the path prediction itself, not by p_active * magnitude.
        gate = torch.ones_like(p_active)
        pred_level = mag_level
        log_hat = log_mag

        if return_aux:
            nan_like = torch.full_like(log_hat, float("nan"))
            return {
                "log_hat": log_hat,             # final direct log1p prediction, no gate
                "active_logit": active_logit,
                "p_active": p_active,
                "log_mag": log_mag,             # ungated magnitude log1p prediction
                "mag_level": mag_level,
                "pred_level": pred_level,       # mean / mu on level scale
                "mu_level": pred_level,
                "alpha": alpha,                   # NB over-dispersion, same shape [B,H,3]
                "gamma": nan_like,
                "gate": gate,
                "residual": residual,
                "z": z,
                "z_mu": z_mu if z_mu is not None else torch.full((B, self.z_dim), float("nan"), device=future_context.device, dtype=future_context.dtype),
                "z_logstd": z_logstd if z_logstd is not None else torch.full((B, self.z_dim), float("nan"), device=future_context.device, dtype=future_context.dtype),
                "attn_weights": attn_w,
            }
        return log_hat


class ExposureForecastModelV2(nn.Module):
    """
    TCN全序列Encoder + Cross-Attention Decoder + single direct exposure head

    Active Head专属特征（事件/时间驱动）：
        ind_promotion, ind_prime_week, holiday/distance/event列
        order_month/season, ind_new_asin, hist_demand_active_rate

    Mag Head专属特征（商品特性驱动）：
        glance_view_band_cat, hbt, our_price_log_norm
        log_review_count, gl_product_group, category_code, ind_amxl_hb
        sort_type, hist_demand_mean13, hist_instock_mean13
    """

    ACTIVE_FEAT_COLS = [
        "ind_promotion",
        "known_future_ind_promotion",
        "known_future_promo_ratio",
        "known_future_promo_amount_log",
        "known_future_promo_price_amount_log",
        "known_future_promo_type_freq",
        "known_future_promo_type_code",
        "ind_prime_week",
        "stock_static__ind_new_asin",
        "stock_static__category_code__code",
        "stock_static__category_code__freq",
        "stock_static__category_code__is_unknown",
        "log_review_count",        # 新增：review高→active率高（零值率从75%降到22%）
        "order_month", "month_sin", "month_cos",
        "season_winter", "season_spring", "season_summer", "season_fall",
        "is_event_window", "weeks_to_nearest_event", "abs_weeks_to_nearest_event",
        "is_pre_event", "is_post_event",
        "pre_event_proximity", "post_event_decay",
        "hist_demand_active_rate",
    ]

    MAG_FEAT_COLS = [
        "stock_static__glance_view_band__norm",
        # HBT is kept as static context but should not dominate graph edges.
        "stock_static__hbt__is_head",
        "our_price_log_norm",
        # V18: known future promotion magnitude/intensity for magnitude head.
        "known_future_ind_promotion",
        "known_future_promo_ratio",
        "known_future_promo_amount_log",
        "known_future_promo_price_amount_log",
        "known_future_promo_type_freq",
        "known_future_promo_type_code",
        "log_review_count",
        # GL is a static/group prior, NOT a graph-neighbor rule.
        "stock_static__gl_product_group__code",
        "stock_static__gl_product_group__freq",
        "stock_static__category_code__code",
        "stock_static__category_code__freq",
        "stock_static__category_code__is_unknown",
        "stock_static__ind_amxl_hb",
        "stock_static__sort_type__norm",
        "stock_static__ind_top10_brand__code",
        # Direct ASIN magnitude priors. These are the main signal supported by diagnostics.
        "hist_total_dph_last_log",
        "hist_total_dph_mean4_log",
        "hist_total_dph_mean13_log",
        "hist_buy_box_dph_last_log",
        "hist_buy_box_dph_mean4_log",
        "hist_buy_box_dph_mean13_log",
        "hist_instock_dph_last_log",
        "hist_instock_dph_mean4_log",
        "hist_instock_dph_mean13_log",
        "hist_demand_last_log",
        "hist_demand_mean4_log",
        "hist_demand_mean13_log",
        "hist_demand_active_rate",
    ]

    def __init__(self, input_dim, context_dim,
                 d_model=64, horizon=20, n_heads=4, dropout=0.10,
                 context_cols=None, use_encoder_self_attn=True,
                 use_enn=True, z_dim=8, residual_scale=2.0, gate_temperature=1.0,
                 use_graphsage=False, graph_assets=None, graph_dim=16, graph_message_scale=0.10,
                 use_graph_head=False, graph_head_scale=0.05, use_graph_gate=True,
                 use_rank_correction=True, rank_correction_scale=0.03,
                 rank_horizon_min_scale=0.40,
                 use_horizon_calibration=True, horizon_calibration_scale=0.03,
                 use_gl_calibration=True, gl_calibration_scale=0.025,
                 use_category_calibration=True, category_calibration_scale=0.015,
                 category_shrinkage_k=100.0,
                 use_learned_edge_score=True, learned_edge_score_scale=1.0, rule_edge_prior_scale=0.25,
                 use_graph_encoder_fusion=True, graph_encoder_scale=0.05,
                 # Backward-compatible aliases; the old broad MLP group calibration is disabled by default.
                 use_group_calibration=False, group_calibration_scale=0.0):
        super().__init__()
        self.use_enn = use_enn
        self.z_dim = int(z_dim)
        print(f"Exposure ENN regime enabled: {use_enn} | z_dim={z_dim}")

        self.use_graphsage = bool(use_graphsage and graph_assets is not None)
        self.graph_dim = int(graph_dim) if self.use_graphsage else 0
        self.graph_context_cols = []
        self.graph_message_scale = float(graph_message_scale)
        # V7: optional graph-delta head. This is NOT a standalone exposure head.
        # It applies a small multiplicative correction on the decoder MU path:
        #     mu_final = mu_base * exp(graph_head_scale * tanh(delta_g))
        # Default is off for maximum stability; turn on only for ablation.
        self.use_graph_head = bool(use_graph_head and self.use_graphsage)
        self.graph_head_scale = float(graph_head_scale)
        self.use_graph_gate = bool(use_graph_gate and self.use_graphsage)
        self.use_rank_correction = bool(use_rank_correction and self.use_graphsage)
        self.rank_correction_scale = float(rank_correction_scale)
        # V15: keep rank prior alive at long horizons instead of letting it collapse to 0.
        # h1 uses 1.0; hH uses rank_horizon_min_scale.
        self.rank_horizon_min_scale = float(rank_horizon_min_scale)
        # V14: lightweight horizon-level plus hierarchical GL/category calibration.
        # Broad group MLP calibration was too wide; we now use:
        #   gl_delta + shrinkage(category_count) * category_delta.
        self.use_horizon_calibration = bool(use_horizon_calibration)
        self.horizon_calibration_scale = float(horizon_calibration_scale)
        self.use_gl_calibration = bool(use_gl_calibration)
        self.gl_calibration_scale = float(gl_calibration_scale)
        self.use_category_calibration = bool(use_category_calibration)
        self.category_calibration_scale = float(category_calibration_scale)
        self.category_shrinkage_k = float(category_shrinkage_k)
        # V15 learned-edge scoring: rule scores are kept as weak priors, but EdgeMLP learns edge attention bias.
        self.use_learned_edge_score = bool(use_learned_edge_score)
        self.learned_edge_score_scale = float(learned_edge_score_scale)
        self.rule_edge_prior_scale = float(rule_edge_prior_scale)
        # V15: early graph fusion lets graph/rank information shape the encoder memory,
        # not only act as late-stage correction after the decoder.
        self.use_graph_encoder_fusion = bool(use_graph_encoder_fusion and self.use_graphsage)
        self.graph_encoder_scale = float(graph_encoder_scale)
        # Keep the old broad context MLP only if explicitly requested.
        self.use_group_calibration = bool(use_group_calibration)
        self.group_calibration_scale = float(group_calibration_scale)
        if self.use_graphsage:
            node_np = graph_assets["node_features"].astype(np.float32)
            neigh_np = graph_assets["neighbor_idx"].astype(np.int64)
            comp_np = graph_assets.get("competitive_neighbor_idx", graph_assets["neighbor_idx"]).astype(np.int64)
            self.register_buffer("graph_node_features", torch.tensor(node_np, dtype=torch.float32))
            self.register_buffer("graph_neighbor_idx", torch.tensor(neigh_np, dtype=torch.long))
            self.register_buffer("graph_competitive_neighbor_idx", torch.tensor(comp_np, dtype=torch.long))
            pos_score_np = graph_assets.get("positive_edge_score", np.zeros_like(neigh_np, dtype=np.float32)).astype(np.float32)
            comp_score_np = graph_assets.get("competitive_edge_score", np.zeros_like(comp_np, dtype=np.float32)).astype(np.float32)
            self.register_buffer("graph_positive_edge_score", torch.tensor(pos_score_np, dtype=torch.float32))
            self.register_buffer("graph_competitive_edge_score", torch.tensor(comp_score_np, dtype=torch.float32))
            pos_edge_feat_np = graph_assets.get("positive_edge_features", np.zeros((neigh_np.shape[0], neigh_np.shape[1], 0), dtype=np.float32)).astype(np.float32)
            comp_edge_feat_np = graph_assets.get("competitive_edge_features", np.zeros((comp_np.shape[0], comp_np.shape[1], pos_edge_feat_np.shape[-1] if pos_edge_feat_np.ndim == 3 else 0), dtype=np.float32)).astype(np.float32)
            self.register_buffer("graph_positive_edge_features", torch.tensor(pos_edge_feat_np, dtype=torch.float32))
            self.register_buffer("graph_competitive_edge_features", torch.tensor(comp_edge_feat_np, dtype=torch.float32))
            self.edge_feature_names = graph_assets.get("edge_feature_names", [])
            self.edge_feat_dim = int(pos_edge_feat_np.shape[-1]) if pos_edge_feat_np.ndim == 3 else 0
            rank_np = graph_assets.get("rank_node_features", np.zeros((node_np.shape[0], 1), dtype=np.float32)).astype(np.float32)
            self.register_buffer("graph_rank_node_features", torch.tensor(rank_np, dtype=torch.float32))
            self.rank_feature_names = graph_assets.get("rank_feature_names", [])
            self.rank_feat_dim = int(rank_np.shape[1])

            # V14 hierarchical calibration IDs. These are ASIN-node-level ids aligned with graph nodes.
            meta_df = graph_assets.get("meta_df", None)
            if meta_df is not None and len(meta_df) == node_np.shape[0]:
                gl_vals = meta_df.get("gl_product_group", pd.Series(["MISSING"] * node_np.shape[0])).astype(str).fillna("MISSING")
                cat_vals = meta_df.get("category_code", pd.Series(["MISSING"] * node_np.shape[0])).astype(str).fillna("MISSING")
            else:
                gl_vals = pd.Series(["MISSING"] * node_np.shape[0])
                cat_vals = pd.Series(["MISSING"] * node_np.shape[0])
            gl_ids_np, gl_uniques = pd.factorize(gl_vals, sort=True)
            cat_ids_np, cat_uniques = pd.factorize(cat_vals, sort=True)
            cat_counts = pd.Series(cat_ids_np).map(pd.Series(cat_ids_np).value_counts()).astype(float).values
            cat_w = cat_counts / (cat_counts + self.category_shrinkage_k)
            self.n_gl_calib = max(int(len(gl_uniques)), 1)
            self.n_category_calib = max(int(len(cat_uniques)), 1)
            self.register_buffer("graph_gl_calib_id", torch.tensor(gl_ids_np, dtype=torch.long))
            self.register_buffer("graph_category_calib_id", torch.tensor(cat_ids_np, dtype=torch.long))
            self.register_buffer("graph_category_calib_weight", torch.tensor(cat_w, dtype=torch.float32))

            self.graph_encoder = DualRelationalGATEncoder(
                node_feat_dim=node_np.shape[1],
                graph_dim=self.graph_dim,
                dropout=dropout,
                neighbor_message_scale=graph_message_scale,
                n_heads=4,
                edge_feat_dim=self.edge_feat_dim,
                use_learned_edge_score=self.use_learned_edge_score,
                learned_edge_score_scale=self.learned_edge_score_scale,
                rule_edge_prior_scale=self.rule_edge_prior_scale,
            )
            # IMPORTANT v4: expose graph embedding as named future-context columns so the
            # magnitude patch heads can directly consume it, not only through the TCN/cross-attn path.
            self.graph_context_cols = [f"graph_emb_{i}" for i in range(self.graph_dim)]
            if context_cols is not None:
                context_cols = list(context_cols) + self.graph_context_cols
            print(f"V19 ContextZ + LearnedEdge EarlyFusion DualRelationalGAT enabled: graph_dim={self.graph_dim} | nodes={node_np.shape[0]} | node_feat_dim={node_np.shape[1]} | edge_feat_dim={self.edge_feat_dim} | msg_scale={graph_message_scale} | learned_edge={self.use_learned_edge_score} | learned_scale={self.learned_edge_score_scale} | rule_prior_scale={self.rule_edge_prior_scale}")
            print(f"Dynamic rank node features: dim={self.rank_feat_dim} | rank_correction={self.use_rank_correction} | scale={self.rank_correction_scale}")
            print(f"Category-only graph v12: sparse/event-aware dynamic rank + long-horizon floor + horizon + GL/category shrinkage calibration.")
        else:
            self.graph_encoder = None
            self.rank_feat_dim = 0
            self.rank_feature_names = []
            self.n_gl_calib = 0
            self.n_category_calib = 0
            self.use_gl_calibration = False
            self.use_category_calibration = False
            print("DualRelationalGAT disabled")

        if self.use_graph_head:
            self.graph_delta_head = nn.Sequential(
                nn.Linear(self.graph_dim, max(32, self.graph_dim * 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(32, self.graph_dim * 2), 3),
                nn.Tanh(),
            )
        else:
            self.graph_delta_head = None

        if self.use_rank_correction:
            rank_in_dim = context_dim + self.graph_dim * 3 + self.rank_feat_dim
            self.rank_correction_head = nn.Sequential(
                nn.Linear(rank_in_dim, max(64, self.graph_dim * 4)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(64, self.graph_dim * 4), 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Tanh(),
            )
        else:
            self.rank_correction_head = None

        # V14 horizon calibration: target-specific log-multiplier by forecast horizon.
        # Initialized at 0; training learns small upward/downward corrections, especially h13-h20.
        if self.use_horizon_calibration:
            self.horizon_calib = nn.Parameter(torch.zeros(int(horizon), 3))
        else:
            self.register_parameter("horizon_calib", None)

        # V14 hierarchical calibration: GL embedding + shrinked category residual embedding.
        # Define the attributes unconditionally so forward() is safe even when a branch is off.
        if self.use_graphsage and self.use_gl_calibration and self.n_gl_calib > 0:
            self.gl_calib_emb = nn.Embedding(self.n_gl_calib, 3)
            nn.init.zeros_(self.gl_calib_emb.weight)
        else:
            self.gl_calib_emb = None

        if self.use_graphsage and self.use_category_calibration and self.n_category_calib > 0:
            self.category_calib_emb = nn.Embedding(self.n_category_calib, 3)
            nn.init.zeros_(self.category_calib_emb.weight)
        else:
            self.category_calib_emb = None

        # Legacy broad group calibration: disabled by default.
        if self.use_group_calibration:
            self.group_calib_head = nn.Sequential(
                nn.Linear(context_dim, max(32, min(128, context_dim * 2))),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(32, min(128, context_dim * 2)), 3),
                nn.Tanh(),
            )
        else:
            self.group_calib_head = None

        print(f"V15 calibration: rank_horizon_min_scale={self.rank_horizon_min_scale} | "
              f"horizon_calib={self.use_horizon_calibration}, scale={self.horizon_calibration_scale} | "
              f"gl_calib={self.use_gl_calibration}, scale={self.gl_calibration_scale} | "
              f"category_calib={self.use_category_calibration}, scale={self.category_calibration_scale}, shrink_k={self.category_shrinkage_k} | "
              f"legacy_group_calib={self.use_group_calibration}, scale={self.group_calibration_scale}")

        if self.use_graphsage:
            # Horizon-level graph gates. These learn how much positive and competitive
            # graph message to use for each ASIN and each future week.
            gate_in_dim = context_dim + self.graph_dim * 3
            self.graph_gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, max(32, self.graph_dim * 4)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(32, self.graph_dim * 4), 2),
            )
            self.graph_fusion_norm = nn.LayerNorm(self.graph_dim)
        else:
            self.graph_gate_net = None
            self.graph_fusion_norm = None

        if self.use_graph_encoder_fusion:
            enc_graph_in_dim = self.graph_dim * 3 + self.rank_feat_dim
            self.graph_encoder_proj = nn.Sequential(
                nn.Linear(enc_graph_in_dim, max(d_model, self.graph_dim * 4)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(d_model, self.graph_dim * 4), d_model),
                nn.Tanh(),
            )
            self.graph_encoder_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            self.graph_encoder_norm = nn.LayerNorm(d_model)
        else:
            self.graph_encoder_proj = None
            self.graph_encoder_gate = None
            self.graph_encoder_norm = None

        print(f"V19 graph encoder fusion: {self.use_graph_encoder_fusion} | scale={self.graph_encoder_scale}")

        self.encoder = HistoryEncoderFull(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_self_attn=use_encoder_self_attn,
        )
        print(f"Encoder exposure-aware self-attn: {use_encoder_self_attn}")

        col_idx = {c: i for i, c in enumerate(context_cols)} if context_cols else {}

        # anchor indices（mean13）
        anchor_indices = None
        try:
            anchor_indices = [
                col_idx["hist_total_dph_mean13_log"],
                col_idx["hist_buy_box_dph_mean13_log"],
                col_idx["hist_instock_dph_mean13_log"],
            ]
            print(f"Anchor indices (mean13): {anchor_indices}")
        except KeyError as e:
            print(f"Warning: anchor column not found: {e}")

        # active head专属特征索引
        active_feat_indices = []
        for c in self.ACTIVE_FEAT_COLS:
            if c in col_idx:
                active_feat_indices.append(col_idx[c])
        # 加入所有holiday/distance/event列
        if context_cols:
            for i, c in enumerate(context_cols):
                if (c.startswith("holiday_indicator_") or
                    c.startswith("distance_") or
                    c.startswith("event_")):
                    if i not in active_feat_indices:
                        active_feat_indices.append(i)

        # mag head专属特征索引
        mag_feat_indices = []
        for c in self.MAG_FEAT_COLS:
            if c in col_idx:
                mag_feat_indices.append(col_idx[c])

        # v6: graph_emb_* columns are appended to future_context so the decoder TCN/cross-attn
        # can use them as light context, but they are NOT added to mag_feat_indices.
        # This avoids the previous graph-fusion failure mode where direct-to-head graph features
        # improved global ratio but worsened WAPE / ASIN-tail error.
        graph_direct_indices = []

        print(f"Active head feat dim: {len(active_feat_indices)}")
        print(f"Mag head feat dim:    {len(mag_feat_indices)}")
        print("Graph direct-to-mag-head feat dim: 0 (v7 keeps graph out of mag features)")
        print(f"Graph delta head: {self.use_graph_head} | graph_head_scale={self.graph_head_scale}")

        print("V19 context-conditioned z: z ~ q_phi(z | encoder_summary, graph_emb, known_future_context); known promo/date enter z via future_context.")

        self.decoder = TCNDecoderWithCrossAttn(
            d_model=d_model,
            context_dim=context_dim + self.graph_dim,
            horizon=horizon,
            hidden=max(96, d_model * 2),
            n_heads=n_heads,
            dropout=dropout,
            anchor_indices=anchor_indices,
            active_feat_indices=active_feat_indices,
            mag_feat_indices=mag_feat_indices,
            active_feat_dim=len(active_feat_indices),
            mag_feat_dim=len(mag_feat_indices),
            use_enn=use_enn,
            z_dim=z_dim,
            residual_scale=residual_scale,
            gate_temperature=gate_temperature,
        )

    def _apply_rank_correction_head(self, dec_out, rank_log_delta):
        """
        V11 dynamic magnitude-rank correction.
        This is a small multiplicative correction on log1p DPH:
            level_final = level_base * exp(scale * tanh(delta_rank))
        It is horizon-specific and uses sparse/event-aware dynamic rank priors.
        """
        if rank_log_delta is None:
            return dec_out

        def _adjust(base_log_hat):
            base_level = torch.expm1(base_log_hat).clamp(min=0.0)
            final_level = base_level * torch.exp(rank_log_delta)
            return torch.log1p(final_level.clamp(min=0.0))

        if isinstance(dec_out, dict):
            base_log_hat = dec_out["log_hat"]
            final_log_hat = _adjust(base_log_hat)
            final_level = torch.expm1(final_log_hat).clamp(min=0.0)
            dec_out = dict(dec_out)
            dec_out["base_log_hat_before_rank_correction"] = base_log_hat
            dec_out["rank_log_delta"] = rank_log_delta
            dec_out["rank_correction_scale"] = torch.tensor(self.rank_correction_scale, dtype=base_log_hat.dtype, device=base_log_hat.device)
            dec_out["log_hat"] = final_log_hat
            dec_out["log_mag"] = final_log_hat
            dec_out["mag_level"] = final_level
            dec_out["pred_level"] = final_level
            dec_out["mu_level"] = final_level
            return dec_out
        return _adjust(dec_out)

    def _apply_graph_delta_head(self, dec_out, g):
        """
        V7 optional graph head.
        This is a small multiplicative correction, not an additive standalone head:
            level_final = level_base * exp(scale * tanh(delta_g))
        It lets category graph adjust ASIN-level magnitude without taking over the forecast.
        """
        if (not self.use_graph_head) or (self.graph_delta_head is None) or (g is None):
            return dec_out

        delta = self.graph_delta_head(g)  # [B,3], already tanh-bounded
        log_mult = self.graph_head_scale * delta[:, None, :]  # [B,1,3]

        def _adjust_log_hat(base_log_hat):
            base_level = torch.expm1(base_log_hat).clamp(min=0.0)
            final_level = base_level * torch.exp(log_mult)
            return torch.log1p(final_level.clamp(min=0.0))

        if isinstance(dec_out, dict):
            base_log_hat = dec_out["log_hat"]
            final_log_hat = _adjust_log_hat(base_log_hat)
            final_level = torch.expm1(final_log_hat).clamp(min=0.0)
            dec_out = dict(dec_out)
            dec_out["base_log_hat_before_graph_head"] = base_log_hat
            dec_out["graph_delta"] = delta
            dec_out["graph_log_multiplier"] = log_mult.expand_as(base_log_hat)
            dec_out["graph_head_scale"] = torch.tensor(self.graph_head_scale, dtype=base_log_hat.dtype, device=base_log_hat.device)
            dec_out["log_hat"] = final_log_hat
            dec_out["log_mag"] = final_log_hat
            dec_out["mag_level"] = final_level
            dec_out["pred_level"] = final_level
            dec_out["mu_level"] = final_level
            return dec_out

        return _adjust_log_hat(dec_out)

    def forward(self, x, future_context, return_aux=False, z=None, asin_idx=None):
        enc_out = self.encoder(x)
        g = None
        rank_log_delta = None

        graph_aux = None
        if self.use_graphsage:
            if asin_idx is None:
                raise ValueError("asin_idx is required when use_graphsage=True")
            graph_all = self.graph_encoder(
                self.graph_node_features,
                self.graph_neighbor_idx,
                self.graph_competitive_neighbor_idx,
                return_aux=True,
                positive_edge_score=self.graph_positive_edge_score,
                competitive_edge_score=self.graph_competitive_edge_score,
                positive_edge_features=self.graph_positive_edge_features,
                competitive_edge_features=self.graph_competitive_edge_features,
            )
            idx = asin_idx.long()
            g = graph_all["graph_emb"][idx]       # [B,G]
            g_self = graph_all["self_msg"][idx]   # [B,G]
            g_pos = graph_all["pos_msg"][idx]     # [B,G]
            g_comp = graph_all["comp_msg"][idx]   # [B,G]
            rank_feat = self.graph_rank_node_features[idx] if hasattr(self, "graph_rank_node_features") else None

            B, H, _ = future_context.shape
            orig_future_context = future_context
            if self.use_graph_gate and self.graph_gate_net is not None:
                g_self_rep = g_self[:, None, :].expand(B, H, -1)
                g_pos_rep = g_pos[:, None, :].expand(B, H, -1)
                g_comp_rep = g_comp[:, None, :].expand(B, H, -1)
                gate_in = torch.cat([future_context, g_self_rep, g_pos_rep, g_comp_rep], dim=-1)
                gate_logits = self.graph_gate_net(gate_in)
                gates = torch.sigmoid(gate_logits)
                pos_gate = gates[..., 0:1]
                comp_gate = gates[..., 1:2]
                g_rep = self.graph_fusion_norm(g_self_rep + pos_gate * g_pos_rep + comp_gate * g_comp_rep)
            else:
                pos_gate = torch.ones(B, H, 1, device=future_context.device, dtype=future_context.dtype)
                comp_gate = torch.ones(B, H, 1, device=future_context.device, dtype=future_context.dtype)
                g_rep = g[:, None, :].expand(B, H, -1)

            if self.use_rank_correction and self.rank_correction_head is not None and rank_feat is not None:
                rank_rep = rank_feat[:, None, :].expand(B, H, -1).to(orig_future_context.device, dtype=orig_future_context.dtype)
                g_self_rep2 = g_self[:, None, :].expand(B, H, -1)
                g_pos_rep2 = g_pos[:, None, :].expand(B, H, -1)
                g_comp_rep2 = g_comp[:, None, :].expand(B, H, -1)
                rank_in = torch.cat([orig_future_context, g_self_rep2, g_pos_rep2, g_comp_rep2, rank_rep], dim=-1)
                rank_raw_delta = self.rank_correction_head(rank_in)
                # V14: deterministic long-horizon floor. The rank prior should not disappear
                # in h13-h20 because ASIN magnitude hierarchy is still useful there.
                if H > 1:
                    h_scale = torch.linspace(
                        1.0, self.rank_horizon_min_scale, H,
                        device=orig_future_context.device, dtype=orig_future_context.dtype,
                    ).view(1, H, 1)
                else:
                    h_scale = torch.ones(1, H, 1, device=orig_future_context.device, dtype=orig_future_context.dtype)
                rank_log_delta = self.rank_correction_scale * h_scale * rank_raw_delta

            # V15 early graph fusion: condition the encoder memory on self/positive/competitive
            # graph messages before the decoder cross-attends to history. This makes graph
            # information affect the whole 20-week path, instead of only late correction.
            enc_gate_mean = None
            if self.use_graph_encoder_fusion and self.graph_encoder_proj is not None:
                if rank_feat is not None:
                    rank_for_enc = rank_feat.to(enc_out.device, dtype=enc_out.dtype)
                else:
                    rank_for_enc = torch.zeros(B, 0, device=enc_out.device, dtype=enc_out.dtype)
                enc_graph_in = torch.cat([
                    g_self.to(enc_out.device, dtype=enc_out.dtype),
                    g_pos.to(enc_out.device, dtype=enc_out.dtype),
                    g_comp.to(enc_out.device, dtype=enc_out.dtype),
                    rank_for_enc,
                ], dim=-1)
                g_enc = self.graph_encoder_proj(enc_graph_in)  # [B,D]
                g_enc_rep = g_enc[:, None, :].expand(-1, enc_out.shape[1], -1)
                enc_gate = self.graph_encoder_gate(torch.cat([enc_out, g_enc_rep], dim=-1))
                enc_out = self.graph_encoder_norm(enc_out + self.graph_encoder_scale * enc_gate * g_enc_rep)
                enc_gate_mean = enc_gate.mean(dim=-1)

            future_context = torch.cat([future_context, g_rep], dim=-1)
            if return_aux:
                graph_aux = {
                    "graph_pos_gate": pos_gate.squeeze(-1),
                    "graph_comp_gate": comp_gate.squeeze(-1),
                    "graph_gate": 0.5 * (pos_gate.squeeze(-1) + comp_gate.squeeze(-1)),
                    "graph_pos_minus_comp_gate": (pos_gate.squeeze(-1) - comp_gate.squeeze(-1)),
                    "graph_encoder_gate_mean": enc_gate_mean if enc_gate_mean is not None else torch.full((B, enc_out.shape[1]), float("nan"), device=future_context.device, dtype=future_context.dtype),
                    "graph_pos_norm": graph_all["pos_norm"][idx],
                    "graph_comp_norm": graph_all["comp_norm"][idx],
                    "graph_emb_norm": graph_all["graph_norm"][idx],
                    "graph_pos_attn_entropy": graph_all.get("pos_attn_entropy", torch.full((self.graph_node_features.shape[0],), float("nan"), device=future_context.device))[idx],
                    "graph_comp_attn_entropy": graph_all.get("comp_attn_entropy", torch.full((self.graph_node_features.shape[0],), float("nan"), device=future_context.device))[idx],
                    "graph_pos_attn_max": graph_all.get("pos_attn_max", torch.full((self.graph_node_features.shape[0],), float("nan"), device=future_context.device))[idx],
                    "graph_comp_attn_max": graph_all.get("comp_attn_max", torch.full((self.graph_node_features.shape[0],), float("nan"), device=future_context.device))[idx],
                }
                if rank_feat is not None:
                    # First few rank features are fixed by _build_graphsage_assets rank_feature_cols.
                    graph_aux["rank_composite_dynamic"] = rank_feat[:, 0]
                    if rank_feat.shape[1] > 1:
                        graph_aux["rank_recent13_strength"] = rank_feat[:, 1]
                    if rank_feat.shape[1] > 2:
                        graph_aux["rank_long52_strength"] = rank_feat[:, 2]
                    if rank_feat.shape[1] > 6:
                        graph_aux["rank_active_eligibility"] = rank_feat[:, 6]
                    if rank_feat.shape[1] > 7:
                        graph_aux["rank_zero_mean"] = rank_feat[:, 7]
                    if rank_log_delta is not None:
                        graph_aux["rank_log_delta_total"] = rank_log_delta[:, :, 0]
                        graph_aux["rank_log_delta_buybox"] = rank_log_delta[:, :, 1]
                        graph_aux["rank_log_delta_instock"] = rank_log_delta[:, :, 2]

        # V15: horizon-level + hierarchical GL/category calibration are added as small
        # log-multiplier streams. Category correction is shrinked by category size.
        calib_log_delta = rank_log_delta
        B_fc, H_fc, _ = future_context.shape
        orig_fc_for_calib = future_context[:, :, :self.group_calib_head[0].in_features] if self.group_calib_head is not None else None

        if self.use_horizon_calibration and self.horizon_calib is not None:
            h_delta = self.horizon_calibration_scale * self.horizon_calib[:H_fc, :].to(future_context.device, dtype=future_context.dtype)
            h_delta = h_delta.unsqueeze(0).expand(B_fc, -1, -1)
            calib_log_delta = h_delta if calib_log_delta is None else calib_log_delta + h_delta
        else:
            h_delta = None

        if self.use_graphsage and asin_idx is not None and getattr(self, "gl_calib_emb", None) is not None:
            gl_ids = self.graph_gl_calib_id[asin_idx.long()].to(future_context.device)
            gl_raw = torch.tanh(self.gl_calib_emb(gl_ids)).to(future_context.dtype)
            gl_delta = self.gl_calibration_scale * gl_raw[:, None, :].expand(B_fc, H_fc, -1)
            calib_log_delta = gl_delta if calib_log_delta is None else calib_log_delta + gl_delta
        else:
            gl_delta = None

        if self.use_graphsage and asin_idx is not None and getattr(self, "category_calib_emb", None) is not None:
            cat_ids = self.graph_category_calib_id[asin_idx.long()].to(future_context.device)
            cat_w = self.graph_category_calib_weight[asin_idx.long()].to(future_context.device, dtype=future_context.dtype)
            cat_raw = torch.tanh(self.category_calib_emb(cat_ids)).to(future_context.dtype)
            cat_delta = self.category_calibration_scale * cat_w[:, None, None] * cat_raw[:, None, :].expand(B_fc, H_fc, -1)
            calib_log_delta = cat_delta if calib_log_delta is None else calib_log_delta + cat_delta
        else:
            cat_delta = None
            cat_w = None

        if self.use_group_calibration and self.group_calib_head is not None:
            # Optional legacy broad context calibration; disabled by default.
            gcal_delta = self.group_calibration_scale * self.group_calib_head(orig_fc_for_calib)
            calib_log_delta = gcal_delta if calib_log_delta is None else calib_log_delta + gcal_delta
        else:
            gcal_delta = None

        dec_out = self.decoder(enc_out, future_context, return_aux=return_aux, z=z)
        dec_out = self._apply_rank_correction_head(dec_out, calib_log_delta)
        dec_out = self._apply_graph_delta_head(dec_out, g)
        if return_aux and isinstance(dec_out, dict):
            if graph_aux is not None:
                dec_out.update(graph_aux)
            if calib_log_delta is not None:
                dec_out["calib_log_delta_total"] = calib_log_delta[:, :, 0]
                dec_out["calib_log_delta_buybox"] = calib_log_delta[:, :, 1]
                dec_out["calib_log_delta_instock"] = calib_log_delta[:, :, 2]
            if h_delta is not None:
                dec_out["horizon_log_delta_total"] = h_delta[:, :, 0]
                dec_out["horizon_log_delta_buybox"] = h_delta[:, :, 1]
                dec_out["horizon_log_delta_instock"] = h_delta[:, :, 2]
            if gl_delta is not None:
                dec_out["gl_log_delta_total"] = gl_delta[:, :, 0]
                dec_out["gl_log_delta_buybox"] = gl_delta[:, :, 1]
                dec_out["gl_log_delta_instock"] = gl_delta[:, :, 2]
            if cat_delta is not None:
                dec_out["category_log_delta_total"] = cat_delta[:, :, 0]
                dec_out["category_log_delta_buybox"] = cat_delta[:, :, 1]
                dec_out["category_log_delta_instock"] = cat_delta[:, :, 2]
                if cat_w is not None:
                    dec_out["category_calibration_weight"] = cat_w[:, None].expand(B_fc, H_fc)
            if gcal_delta is not None:
                dec_out["group_log_delta_total"] = gcal_delta[:, :, 0]
                dec_out["group_log_delta_buybox"] = gcal_delta[:, :, 1]
                dec_out["group_log_delta_instock"] = gcal_delta[:, :, 2]
        return dec_out


# ============================================================
# Loss：Distributional NB + Hurdle BCE + Magnitude Huber + Mean Penalty
# ============================================================

def exposure_negbin_nll_elementwise(y, mu, alpha):
    """
    Negative-binomial NLL for nonnegative exposure counts.
    y can be float-valued DPH; lgamma form is stable and works as a quasi-likelihood.
    mu is the expected exposure level and alpha is over-dispersion.
    """
    eps = 1e-6
    y = y.clamp(min=0.0)
    mu = mu.clamp(min=eps)
    alpha = alpha.clamp(min=1e-4, max=100.0)
    r = (1.0 / alpha).clamp(min=eps, max=1e6)
    p = (mu * alpha / (1.0 + mu * alpha)).clamp(eps, 1.0 - eps)
    return -(
        torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
        + r * torch.log1p(-p) + y * torch.log(p)
    )


def exposure_tail_weighted_nb_nll(true, mu, alpha, channel_weights, high_weight_alpha=0.25):
    """Tail-weighted NB quasi-NLL for exposure distribution learning."""
    nll = exposure_negbin_nll_elementwise(true, mu, alpha)
    denom = torch.log1p(true).detach().mean(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    high_w = 1.0 + high_weight_alpha * torch.log1p(true).detach() / denom
    w = channel_weights.view(1, 1, 3) * high_w
    return (nll * w).sum() / w.sum().clamp_min(1.0)

def exposure_hurdle_loss(
    log_hat,        # [B,H,3] direct log1p prediction
    true_total,     # [B,H]
    true_buy,       # [B,H]
    true_instock,   # [B,H]
    active_logit,   # [B,H,3] auxiliary occurrence logits only
    log_mag=None,   # unused; kept for interface compatibility
    alpha=None,      # [B,H,3] NB over-dispersion for distributional exposure head
    nb_weight=0.25,
    w_total=0.30,
    w_buy=0.60,
    w_instock=1.00,
    bce_weight=0.20,
    mag_weight=1.00,
    mean_weight=0.25,
    active_calib_weight=0.05,
    # Zero-aware weights. Zero mainly happens in buy_box / in_stock, not total.
    zero_weight=0.00,  # kept for backward compatibility; not used as the main zero term
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.15,
    high_weight_alpha=0.35,
    short_horizon_weight=0.8,
    mid_horizon_weight=1.2,
    long_horizon_weight=2.0,
    long_block_sum_weight=0.30,
    # ENN/path-regime terms
    path_zero_weight=0.00,
    zero_fp_weight=0.00,
    active_count_weight=0.00,
    path_sum_weight=0.05,
    asin_sum_weight=0.40,
    # Peak/path-high regime terms. These prevent zero losses from making the model too conservative.
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    zero_fp_threshold=50.0,
    zero_fp_temperature=20.0,
):
    """
    Single-head direct exposure loss with channel-specific zero awareness.

    Why this version:
      - total_dph is almost never zero in the data, so total-zero consistency alone
        does not teach the model to capture in_stock zeros.
      - buy_box_dph / in_stock_dph have meaningful zero rates that vary by GL/month.
      - The final prediction is still single-head direct; p_active is auxiliary only.

    Main terms:
      1. direct log1p Huber regression
      2. light mean scale penalty
      3. auxiliary active BCE/calibration
      4. channel-specific zero losses for buy_box and in_stock
      5. hierarchy zero consistency:
           true_total == 0   => total/buy_box/in_stock should be near 0
           true_buy_box == 0 => buy_box/in_stock should be near 0
    """
    true = torch.stack([
        true_total.clamp(min=0.0),
        true_buy.clamp(min=0.0),
        true_instock.clamp(min=0.0),
    ], dim=-1)   # [B,H,3]

    target_log = torch.log1p(true)
    tw = torch.tensor([w_total, w_buy, w_instock],
                      dtype=log_hat.dtype, device=log_hat.device).view(1, 1, 3)

    denom = target_log.detach().mean(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    high_w = 1.0 + high_weight_alpha * target_log.detach() / denom

    H = true.shape[1]
    h = torch.arange(1, H + 1, device=true.device, dtype=true.dtype).view(1, H, 1)
    # Horizon block weighting. Short horizons already perform well; we upweight
    # h14-h20 to fight long-horizon magnitude collapse.
    base_h_w = 1.0 + horizon_weight_alpha * (h / max(float(H), 1.0))
    block_h_w = torch.where(
        h <= 5,
        torch.full_like(h, float(short_horizon_weight)),
        torch.where(
            h <= 13,
            torch.full_like(h, float(mid_horizon_weight)),
            torch.full_like(h, float(long_horizon_weight)),
        ),
    )
    horizon_w = base_h_w * block_h_w
    sample_w = high_w * horizon_w

    # 1) Main direct log loss.
    log_err = F.huber_loss(log_hat, target_log, delta=1.0, reduction="none")
    direct_loss = (log_err * sample_w * tw).mean()

    # Distributional NB quasi-likelihood. This teaches over-dispersion so that
    # sparse exposure paths can produce exact-zero sample quantiles, similar to
    # the demand model's NB sampling mechanism. The mean path is still controlled
    # by log_hat / pred_level, so we do not hard-threshold or gate predictions.
    if alpha is not None and nb_weight > 0:
        mu_for_nb = torch.expm1(log_hat).clamp(min=1e-6)
        nb_loss = exposure_tail_weighted_nb_nll(
            true=true,
            mu=mu_for_nb,
            alpha=alpha,
            channel_weights=tw.view(3),
            high_weight_alpha=high_weight_alpha,
        )
    else:
        nb_loss = torch.zeros((), dtype=log_hat.dtype, device=log_hat.device)

    # Shared zero error: target log is zero when target exposure is zero.
    zero_err = F.huber_loss(log_hat, torch.zeros_like(log_hat), delta=0.5, reduction="none")

    def _masked_channel_loss(mask_2d, channel_idx, channel_weight=1.0):
        """Mask shape [B,H]. Penalize one output channel when the matching true channel is zero."""
        m = mask_2d.float().unsqueeze(-1)  # [B,H,1]
        ch = torch.zeros_like(true)
        ch[..., channel_idx] = 1.0
        weight = m * ch * sample_w * tw
        denom = weight.sum().clamp_min(1.0)
        return channel_weight * (zero_err * weight).sum() / denom

    # 2) Channel-specific zero losses.
    # total is rare-zero, keep small; buy_box/in_stock are the important channels.
    total_zero_loss = _masked_channel_loss(true_total <= 0, 0)
    buy_zero_loss = _masked_channel_loss(true_buy <= 0, 1)
    instock_zero_loss = _masked_channel_loss(true_instock <= 0, 2)

    # 3) Hierarchy zero consistency.
    # If total is zero, all channels should be near zero. This is correct but rare.
    total_zero_mask = (true_total <= 0).float().unsqueeze(-1)
    total_zero_weight_mat = total_zero_mask * sample_w * tw
    total_zero_consistency = (zero_err * total_zero_weight_mat).sum() / total_zero_weight_mat.sum().clamp_min(1.0)

    # If buy_box is zero, buy_box and in_stock should be near zero.
    # This matters more than total-zero consistency in this dataset.
    buy_zero_mask = (true_buy <= 0).float().unsqueeze(-1)
    buy_instock_selector = torch.tensor([0.0, 1.0, 1.0], dtype=log_hat.dtype, device=log_hat.device).view(1, 1, 3)
    buy_zero_weight_mat = buy_zero_mask * buy_instock_selector * sample_w * tw
    buy_zero_consistency = (zero_err * buy_zero_weight_mat).sum() / buy_zero_weight_mat.sum().clamp_min(1.0)

    zero_loss = (
        total_zero_weight * total_zero_loss
        + buy_zero_weight * buy_zero_loss
        + instock_zero_weight * instock_zero_loss
        + total_zero_consistency_weight * total_zero_consistency
        + buy_zero_consistency_weight * buy_zero_consistency
    )

    # 4) Mean scale penalty on level space, used lightly to avoid systematic over/under.
    pred_level = torch.expm1(log_hat).clamp(min=0.0)
    mean_pred = torch.log1p(pred_level.mean(dim=(0, 1)).clamp_min(1e-6))
    mean_true = torch.log1p(true.mean(dim=(0, 1)).clamp_min(1e-6))
    mean_loss = (torch.abs(mean_pred - mean_true) * tw.view(3)).mean()

    # 5) Auxiliary occurrence loss. This is deliberately small and does not gate final predictions.
    active_label = (true > 0).float()
    pos_w = torch.tensor([0.5, 0.5, 0.5],
                         dtype=log_hat.dtype,
                         device=log_hat.device).view(1, 1, 3)
    bce_raw = F.binary_cross_entropy_with_logits(
        active_logit, active_label, reduction="none"
    )
    bce = bce_raw * (1.0 - active_label) + bce_raw * active_label * pos_w
    bce_loss = (bce * sample_w * tw).mean()

    p_active = torch.sigmoid(active_logit)
    active_rate_pred = p_active.mean(dim=(0, 1))
    active_rate_true = active_label.mean(dim=(0, 1))
    active_calib_loss = (torch.abs(active_rate_pred - active_rate_true) * tw.view(3)).mean()

    # 6) Path/regime losses for ENN.
    # These target the observed failure mode: true future is zero or active->zero,
    # but the model keeps a positive floor every week.
    pred_instock = pred_level[..., 2]
    true_instock_y = true[..., 2]

    true_path_zero = (true_instock_y.sum(dim=1) <= 0).float()
    pred_path_sum = pred_instock.sum(dim=1)
    path_zero_loss = (true_path_zero * torch.log1p(pred_path_sum)).mean()

    true_zero_instock = (true_instock_y <= 0).float()
    pred_positive_soft = torch.sigmoid((pred_instock - zero_fp_threshold) / max(zero_fp_temperature, 1e-6))
    zero_fp_loss = (true_zero_instock * pred_positive_soft * horizon_w.squeeze(-1)).mean()

    true_active_count = (true_instock_y > 0).float().sum(dim=1)
    pred_active_count = pred_positive_soft.sum(dim=1)
    active_count_loss = F.smooth_l1_loss(pred_active_count, true_active_count)

    true_path_sum_log = torch.log1p(true_instock_y.sum(dim=1).clamp_min(0.0))
    pred_path_sum_log = torch.log1p(pred_path_sum.clamp_min(0.0))
    path_sum_loss = F.smooth_l1_loss(pred_path_sum_log, true_path_sum_log)

    # ASIN/window 20-week sum loss across all three exposure channels.
    # This is the main magnitude prior: it asks the single-head decoder to get
    # each ASIN's total future exposure level right, instead of only matching
    # pointwise weekly errors. This is NOT a two-head or active-gated term.
    true_sum_all_log = torch.log1p(true.sum(dim=1).clamp_min(0.0))      # [B,3]
    pred_sum_all_log = torch.log1p(pred_level.sum(dim=1).clamp_min(0.0)) # [B,3]
    asin_sum_loss = (F.smooth_l1_loss(pred_sum_all_log, true_sum_all_log, reduction="none") * tw.view(1,3)).mean()

    # Long-block sum loss: explicitly protect h14-h20 total in_stock exposure.
    # This is more stable than only increasing pointwise weights, because the
    # key failure mode is long-block level underprediction.
    long_start = 13 if H > 13 else max(H - 1, 0)  # zero-based index: h14
    true_long_sum_log = torch.log1p(true_instock_y[:, long_start:].sum(dim=1).clamp_min(0.0))
    pred_long_sum_log = torch.log1p(pred_instock[:, long_start:].sum(dim=1).clamp_min(0.0))
    long_block_sum_loss = F.smooth_l1_loss(pred_long_sum_log, true_long_sum_log)

    # 7) Peak/path-high losses for ENN.
    # These target the opposite failure mode of zero losses: peak compression.
    # Use in_stock as the main business-critical exposure channel.
    true_peak = true_instock_y.max(dim=1).values
    pred_peak = pred_instock.max(dim=1).values
    peak_loss = F.smooth_l1_loss(torch.log1p(pred_peak), torch.log1p(true_peak))

    k = int(max(1, min(int(peak_topk), true_instock_y.shape[1])))
    true_topk = torch.topk(true_instock_y, k=k, dim=1).values
    pred_topk = torch.topk(pred_instock, k=k, dim=1).values
    topk_peak_loss = F.smooth_l1_loss(torch.log1p(pred_topk), torch.log1p(true_topk))

    # High under-loss: if the target is in the high tail, underpredicting is especially costly.
    # Detach threshold so it is a data-dependent weighting, not a learned target.
    flat_true = true_instock_y.detach().reshape(-1)
    if flat_true.numel() > 0 and torch.max(flat_true) > 0:
        high_th = torch.quantile(flat_true, float(peak_quantile))
    else:
        high_th = torch.tensor(0.0, dtype=true_instock_y.dtype, device=true_instock_y.device)
    high_mask = (true_instock_y >= high_th).float() * (true_instock_y > 0).float()
    peak_under = F.relu(torch.log1p(true_instock_y) - torch.log1p(pred_instock))
    peak_under_loss = (peak_under * high_mask).sum() / high_mask.sum().clamp_min(1.0)

    return (
        nb_weight * nb_loss
        + mag_weight * direct_loss
        + mean_weight * mean_loss
        + bce_weight * bce_loss
        + active_calib_weight * active_calib_loss
        + zero_loss
        + path_zero_weight * path_zero_loss
        + zero_fp_weight * zero_fp_loss
        + active_count_weight * active_count_loss
        + path_sum_weight * path_sum_loss
        + asin_sum_weight * asin_sum_loss
        + long_block_sum_weight * long_block_sum_loss
        + peak_weight * peak_loss
        + topk_peak_weight * topk_peak_loss
        + peak_under_weight * peak_under_loss
    )

# ============================================================
# 训练
# ============================================================

def train_exposure_model_v2(
    model, tr_ld, va_ld,
    epochs=60, lr=1e-3, patience=8,
    w_total=0.30, w_buy=0.60, w_instock=1.00,
    bce_weight=0.00, mag_weight=1.00, mean_weight=0.40,
    active_calib_weight=0.00,
    nb_weight=0.25,
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.15, high_weight_alpha=0.35,
    short_horizon_weight=0.8,
    mid_horizon_weight=1.2,
    long_horizon_weight=2.0,
    long_block_sum_weight=0.30,
    path_zero_weight=0.00,
    zero_fp_weight=0.00,
    active_count_weight=0.00,
    path_sum_weight=0.05,
    asin_sum_weight=0.40,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    zero_fp_threshold=50.0,
    zero_fp_temperature=20.0,
    device=None,
):
    device = get_device(device)
    model = model.to(device)
    print(f"Training on device: {device}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_val, best_sd, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0

        for b in tr_ld:
            b = batch_to_device(b, device)
            aux = model(b["x"], b["future_context"], return_aux=True, asin_idx=b.get("asin_idx"))
            loss = exposure_hurdle_loss(
                log_hat=aux["log_hat"],
                true_total=b["future_total_dph"],
                true_buy=b["future_buy_box_dph"],
                true_instock=b["future_instock_dph"],
                active_logit=aux["active_logit"],
                log_mag=aux["log_mag"],
                alpha=aux.get("alpha", None),
                nb_weight=nb_weight,
                w_total=w_total, w_buy=w_buy, w_instock=w_instock,
                bce_weight=bce_weight, mag_weight=mag_weight,
                mean_weight=mean_weight,
                active_calib_weight=active_calib_weight,
                zero_weight=zero_weight,
                total_zero_weight=total_zero_weight,
                buy_zero_weight=buy_zero_weight,
                instock_zero_weight=instock_zero_weight,
                total_zero_consistency_weight=total_zero_consistency_weight,
                buy_zero_consistency_weight=buy_zero_consistency_weight,
                horizon_weight_alpha=horizon_weight_alpha,
                high_weight_alpha=high_weight_alpha,
                path_zero_weight=path_zero_weight,
                zero_fp_weight=zero_fp_weight,
                active_count_weight=active_count_weight,
                path_sum_weight=path_sum_weight,
                asin_sum_weight=asin_sum_weight,
                short_horizon_weight=short_horizon_weight,
                mid_horizon_weight=mid_horizon_weight,
                long_horizon_weight=long_horizon_weight,
                long_block_sum_weight=long_block_sum_weight,
                peak_weight=peak_weight,
                topk_peak_weight=topk_peak_weight,
                peak_under_weight=peak_under_weight,
                peak_topk=peak_topk,
                peak_quantile=peak_quantile,
                zero_fp_threshold=zero_fp_threshold,
                zero_fp_temperature=zero_fp_temperature,
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * b["x"].shape[0]
            tr_n   += b["x"].shape[0]

        sch.step()

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for b in va_ld:
                b = batch_to_device(b, device)
                aux = model(b["x"], b["future_context"], return_aux=True, asin_idx=b.get("asin_idx"))
                loss = exposure_hurdle_loss(
                    log_hat=aux["log_hat"],
                    true_total=b["future_total_dph"],
                    true_buy=b["future_buy_box_dph"],
                    true_instock=b["future_instock_dph"],
                    active_logit=aux["active_logit"],
                    log_mag=aux["log_mag"],
                    alpha=aux.get("alpha", None),
                    nb_weight=nb_weight,
                    w_total=w_total, w_buy=w_buy, w_instock=w_instock,
                    bce_weight=bce_weight, mag_weight=mag_weight,
                    mean_weight=mean_weight,
                    active_calib_weight=active_calib_weight,
                    zero_weight=zero_weight,
                    total_zero_weight=total_zero_weight,
                    buy_zero_weight=buy_zero_weight,
                    instock_zero_weight=instock_zero_weight,
                    total_zero_consistency_weight=total_zero_consistency_weight,
                    buy_zero_consistency_weight=buy_zero_consistency_weight,
                    horizon_weight_alpha=horizon_weight_alpha,
                    high_weight_alpha=high_weight_alpha,
                    path_zero_weight=path_zero_weight,
                    zero_fp_weight=zero_fp_weight,
                    active_count_weight=active_count_weight,
                    path_sum_weight=path_sum_weight,
                    peak_weight=peak_weight,
                    topk_peak_weight=topk_peak_weight,
                    peak_under_weight=peak_under_weight,
                    peak_topk=peak_topk,
                    peak_quantile=peak_quantile,
                    zero_fp_threshold=zero_fp_threshold,
                    zero_fp_temperature=zero_fp_temperature,
                )
                va_sum += loss.item() * b["x"].shape[0]
                va_n   += b["x"].shape[0]

        tr_loss = tr_sum / max(tr_n, 1)
        va_loss = va_sum / max(va_n, 1)
        print(f"Epoch {epoch+1:03d} | train={tr_loss:.5f} | val={va_loss:.5f}")

        if va_loss < best_val - 1e-6:
            best_val   = va_loss
            best_sd    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stop at epoch {epoch+1}. Best val={best_val:.5f}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model


# ============================================================
# 预测（输出格式与原版完全相同，多了p_active诊断列）
# ============================================================

def _nb_sample_from_mu_alpha(mu, alpha):
    """Sample from NB parameterized by mean mu and over-dispersion alpha."""
    eps = 1e-6
    mu = mu.clamp(min=eps)
    alpha = alpha.clamp(min=1e-4, max=100.0)
    total_count = (1.0 / alpha).clamp(min=1e-4, max=1e6)
    probs = (mu * alpha / (1.0 + mu * alpha)).clamp(eps, 1.0 - eps)
    dist = torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)
    return dist.sample().float()


def predict_exposure_v2(
    model,
    va_ld,
    apply_funnel_constraint=True,
    device=None,
    mc_samples=50,
    mc_reduce="mu",
    use_distributional_samples=True,
):
    """
    Predict exposure paths.

    Default mc_reduce="mu": the main pred_*_dph columns use the direct expected level
    expm1(log_hat). This is the recommended exposure covariate for the demand model.

    NB sampling is still explicit and available via mc_reduce="p50"/"nb_sample" or
    mc_reduce="nb_mean". The sampled p50 is kept as a zero/active diagnostic, but is
    usually too conservative to pass directly into demand.

    Extra diagnostics:
      pred_*_dph_mu: direct mean path expm1(log_hat)
      pred_*_dph_dist_mean: MC mean of sampled distribution
      pred_*_dph_dist_p50: MC median of sampled distribution
    """
    device = get_device(device)
    model = model.to(device)
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            b = batch_to_device(b, device)
            sample_preds, mu_preds, pacts, gates = [], [], [], []
            last_aux = None
            K = max(int(mc_samples), 1)
            for _ in range(K):
                aux = model(b["x"], b["future_context"], return_aux=True, asin_idx=b.get("asin_idx"))
                last_aux = aux
                mu_level = torch.expm1(aux["log_hat"]).clamp(min=0.0)
                mu_preds.append(mu_level)

                if use_distributional_samples and aux.get("alpha", None) is not None:
                    sample_level = _nb_sample_from_mu_alpha(mu_level, aux["alpha"])
                else:
                    sample_level = mu_level
                sample_preds.append(sample_level)
                pacts.append(aux["p_active"])
                gates.append(aux.get("gate", torch.full_like(aux["p_active"], float("nan"))))

            sample_stack = torch.stack(sample_preds, dim=0)  # [K,B,H,3]
            mu_stack = torch.stack(mu_preds, dim=0)
            pact_stack = torch.stack(pacts, dim=0)
            gate_stack = torch.stack(gates, dim=0)

            # ------------------------------------------------------------
            # IMPORTANT: inference reduction path
            # ------------------------------------------------------------
            # mu / direct / expected:
            #   use direct mean path expm1(log_hat); alpha is NOT used here.
            # mean / nb_mean / dist_mean:
            #   use Monte Carlo mean from NegativeBinomial samples.
            # median / p50 / nb_sample:
            #   use Monte Carlo median/p50 from NegativeBinomial samples.
            #
            # This makes the NB sampling path explicit; alpha_head is only
            # activated for mean/median/p50/nb_sample reductions when
            # use_distributional_samples=True.
            mu_t = mu_stack.mean(dim=0)
            dist_mean_t = sample_stack.mean(dim=0)
            dist_p50_t = sample_stack.median(dim=0).values

            reduce_key = str(mc_reduce).lower()
            if reduce_key in ["mu", "direct", "expected"]:
                pred_t = mu_t
            elif reduce_key in ["mean", "nb_mean", "dist_mean"]:
                pred_t = dist_mean_t
            elif reduce_key in ["median", "p50", "nb_sample", "sample_median"]:
                pred_t = dist_p50_t
            else:
                raise ValueError(
                    f"Unknown mc_reduce={mc_reduce}. Use one of: "
                    "'mu', 'mean'/'nb_mean', 'median'/'p50'/'nb_sample'."
                )
            pact_t = pact_stack.mean(dim=0)
            gate_t = gate_stack.median(dim=0).values

            pred = pred_t.cpu().numpy()
            mu_np = mu_t.cpu().numpy()
            dist_mean_np = dist_mean_t.cpu().numpy()
            dist_p50_np = dist_p50_t.cpu().numpy()
            pact = pact_t.cpu().numpy()
            gamma_np = last_aux.get("gamma", torch.full_like(last_aux["p_active"], float("nan"))).cpu().numpy()
            gate_np = gate_t.cpu().numpy()

            def _aux_horizon_np(name):
                v = last_aux.get(name, None)
                if v is None:
                    return np.full((b["future_instock_dph"].shape[0], b["future_instock_dph"].shape[1]), np.nan)
                return v.detach().cpu().numpy()

            def _aux_asin_np(name):
                v = last_aux.get(name, None)
                if v is None:
                    return np.full((b["future_instock_dph"].shape[0],), np.nan)
                return v.detach().cpu().numpy()

            graph_pos_gate_np = _aux_horizon_np("graph_pos_gate")
            graph_comp_gate_np = _aux_horizon_np("graph_comp_gate")
            graph_gate_np = _aux_horizon_np("graph_gate")
            graph_pos_minus_comp_gate_np = _aux_horizon_np("graph_pos_minus_comp_gate")
            graph_pos_norm_np = _aux_asin_np("graph_pos_norm")
            graph_comp_norm_np = _aux_asin_np("graph_comp_norm")
            graph_emb_norm_np = _aux_asin_np("graph_emb_norm")
            graph_pos_attn_entropy_np = _aux_asin_np("graph_pos_attn_entropy")
            graph_comp_attn_entropy_np = _aux_asin_np("graph_comp_attn_entropy")
            graph_pos_attn_max_np = _aux_asin_np("graph_pos_attn_max")
            graph_comp_attn_max_np = _aux_asin_np("graph_comp_attn_max")
            rank_composite_np = _aux_asin_np("rank_composite_dynamic")
            rank_recent13_np = _aux_asin_np("rank_recent13_strength")
            rank_long52_np = _aux_asin_np("rank_long52_strength")
            rank_active_elig_np = _aux_asin_np("rank_active_eligibility")
            rank_zero_mean_np = _aux_asin_np("rank_zero_mean")
            rank_delta_total_np = _aux_horizon_np("rank_log_delta_total")
            rank_delta_buybox_np = _aux_horizon_np("rank_log_delta_buybox")
            rank_delta_instock_np = _aux_horizon_np("rank_log_delta_instock")
            calib_delta_total_np = _aux_horizon_np("calib_log_delta_total")
            calib_delta_buybox_np = _aux_horizon_np("calib_log_delta_buybox")
            calib_delta_instock_np = _aux_horizon_np("calib_log_delta_instock")
            gl_delta_total_np = _aux_horizon_np("gl_log_delta_total")
            gl_delta_buybox_np = _aux_horizon_np("gl_log_delta_buybox")
            gl_delta_instock_np = _aux_horizon_np("gl_log_delta_instock")
            category_delta_total_np = _aux_horizon_np("category_log_delta_total")
            category_delta_buybox_np = _aux_horizon_np("category_log_delta_buybox")
            category_delta_instock_np = _aux_horizon_np("category_log_delta_instock")
            category_calib_weight_np = _aux_horizon_np("category_calibration_weight")
            horizon_delta_total_np = _aux_horizon_np("horizon_log_delta_total")
            horizon_delta_buybox_np = _aux_horizon_np("horizon_log_delta_buybox")
            horizon_delta_instock_np = _aux_horizon_np("horizon_log_delta_instock")
            group_delta_total_np = _aux_horizon_np("group_log_delta_total")
            group_delta_buybox_np = _aux_horizon_np("group_log_delta_buybox")
            group_delta_instock_np = _aux_horizon_np("group_log_delta_instock")

            if apply_funnel_constraint:
                # apply funnel to all prediction views
                for arr in (pred, mu_np, dist_mean_np, dist_p50_np):
                    arr[:, :, 1] = np.minimum(arr[:, :, 1], arr[:, :, 0])
                    arr[:, :, 2] = np.minimum(arr[:, :, 2], arr[:, :, 1])

            B, H = b["future_instock_dph"].shape
            for i in range(B):
                for h in range(H):
                    rows.append({
                        "asin":              b["asin"][i],
                        "order_week":        pd.to_datetime(b["target_week"][i][h]),
                        "horizon":           h + 1,
                        "true_total_dph":    b["future_total_dph"][i, h].item(),
                        "pred_total_dph":    pred[i, h, 0],
                        "true_buy_box_dph":  b["future_buy_box_dph"][i, h].item(),
                        "pred_buy_box_dph":  pred[i, h, 1],
                        "true_instock_dph":  b["future_instock_dph"][i, h].item(),
                        "pred_instock_dph":  pred[i, h, 2],
                        "true_demand":       b["future_demand"][i, h].item(),

                        "pred_total_dph_mu":       mu_np[i, h, 0],
                        "pred_buy_box_dph_mu":     mu_np[i, h, 1],
                        "pred_instock_dph_mu":     mu_np[i, h, 2],
                        "pred_total_dph_dist_mean":   dist_mean_np[i, h, 0],
                        "pred_buy_box_dph_dist_mean": dist_mean_np[i, h, 1],
                        "pred_instock_dph_dist_mean": dist_mean_np[i, h, 2],
                        "pred_total_dph_dist_p50":    dist_p50_np[i, h, 0],
                        "pred_buy_box_dph_dist_p50":  dist_p50_np[i, h, 1],
                        "pred_instock_dph_dist_p50":  dist_p50_np[i, h, 2],

                        "p_active_total":    pact[i, h, 0],
                        "p_active_buy_box":  pact[i, h, 1],
                        "p_active_instock":  pact[i, h, 2],
                        "gamma_total":       gamma_np[i, h, 0],
                        "gamma_buy_box":     gamma_np[i, h, 1],
                        "gamma_instock":     gamma_np[i, h, 2],
                        "gate_total":        gate_np[i, h, 0],
                        "gate_buy_box":      gate_np[i, h, 1],
                        "gate_instock":      gate_np[i, h, 2],

                        # DualGAT gating diagnostics. These are model-learned horizon-level
                        # weights for positive vs competitive graph messages.
                        "graph_pos_gate": graph_pos_gate_np[i, h],
                        "graph_comp_gate": graph_comp_gate_np[i, h],
                        "graph_gate": graph_gate_np[i, h],
                        "graph_pos_minus_comp_gate": graph_pos_minus_comp_gate_np[i, h],
                        "graph_pos_norm": graph_pos_norm_np[i],
                        "graph_comp_norm": graph_comp_norm_np[i],
                        "graph_emb_norm": graph_emb_norm_np[i],
                        "graph_pos_attn_entropy": graph_pos_attn_entropy_np[i],
                        "graph_comp_attn_entropy": graph_comp_attn_entropy_np[i],
                        "graph_pos_attn_max": graph_pos_attn_max_np[i],
                        "graph_comp_attn_max": graph_comp_attn_max_np[i],
                        "rank_composite_dynamic": rank_composite_np[i],
                        "rank_recent13_strength": rank_recent13_np[i],
                        "rank_long52_strength": rank_long52_np[i],
                        "rank_active_eligibility": rank_active_elig_np[i],
                        "rank_zero_mean": rank_zero_mean_np[i],
                        "rank_log_delta_total": rank_delta_total_np[i, h],
                        "rank_log_delta_buybox": rank_delta_buybox_np[i, h],
                        "rank_log_delta_instock": rank_delta_instock_np[i, h],
                        "calib_log_delta_total": calib_delta_total_np[i, h],
                        "calib_log_delta_buybox": calib_delta_buybox_np[i, h],
                        "calib_log_delta_instock": calib_delta_instock_np[i, h],
                        "gl_log_delta_total": gl_delta_total_np[i, h],
                        "gl_log_delta_buybox": gl_delta_buybox_np[i, h],
                        "gl_log_delta_instock": gl_delta_instock_np[i, h],
                        "category_log_delta_total": category_delta_total_np[i, h],
                        "category_log_delta_buybox": category_delta_buybox_np[i, h],
                        "category_log_delta_instock": category_delta_instock_np[i, h],
                        "category_calibration_weight": category_calib_weight_np[i, h],
                        "horizon_log_delta_total": horizon_delta_total_np[i, h],
                        "horizon_log_delta_buybox": horizon_delta_buybox_np[i, h],
                        "horizon_log_delta_instock": horizon_delta_instock_np[i, h],
                        "group_log_delta_total": group_delta_total_np[i, h],
                        "group_log_delta_buybox": group_delta_buybox_np[i, h],
                        "group_log_delta_instock": group_delta_instock_np[i, h],
                    })
    return pd.DataFrame(rows)


# ============================================================
# 评估（完全复用原版函数）
# ============================================================

def exposure_metrics(pred_df, prefix="pred"):
    specs = [
        ("total_dph",   "true_total_dph",   f"{prefix}_total_dph"),
        ("buy_box_dph", "true_buy_box_dph",  f"{prefix}_buy_box_dph"),
        ("in_stock_dph","true_instock_dph",  f"{prefix}_instock_dph"),
    ]
    rows = []
    for name, true_col, pred_col in specs:
        y = pred_df[true_col].values
        p = pred_df[pred_col].values
        rows.append({
            "target": name,
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p),
            "pred_true_ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "corr": _corr(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
            "zero_rate_true": np.mean(y <= 0),
        })
    return pd.DataFrame(rows)




def make_p50_diagnostic_view(pred_df):
    """Return a copy where main pred_* columns are replaced by NB sampled p50.
    Use only for zero/active diagnostics, not for demand handoff.
    """
    out = pred_df.copy()
    mapping = {
        "pred_total_dph": "pred_total_dph_dist_p50",
        "pred_buy_box_dph": "pred_buy_box_dph_dist_p50",
        "pred_instock_dph": "pred_instock_dph_dist_p50",
    }
    for dst, src in mapping.items():
        if src in out.columns:
            out[dst] = out[src]
    return out


def print_mu_vs_p50_quick_diagnostics(pred_df):
    print("\n" + "=" * 100)
    print("MU vs NB-P50 QUICK DIAGNOSTICS")
    print("=" * 100)
    print("\n[MU / expected level as main pred_* columns]  <-- use for demand hat")
    print(exposure_metrics(pred_df, prefix="pred").round(5).to_string(index=False))
    p50_df = make_p50_diagnostic_view(pred_df)
    print("\n[NB sampled P50 diagnostic]  <-- zero/active diagnostic only")
    print(exposure_metrics(p50_df, prefix="pred").round(5).to_string(index=False))
    cols = ["pred_total_dph_dist_p50", "pred_buy_box_dph_dist_p50", "pred_instock_dph_dist_p50"]
    cols = [c for c in cols if c in pred_df.columns]
    if cols:
        print("\nNB-P50 exact-zero share:")
        print((pred_df[cols] == 0).mean().round(5).to_string())
    return {
        "mu_metrics": exposure_metrics(pred_df, prefix="pred"),
        "p50_metrics": exposure_metrics(p50_df, prefix="pred"),
    }

def add_naive_baselines_from_loader(pred_df, va_ld, context_cols):
    idx   = {c: i for i, c in enumerate(context_cols)}
    modes = {
        "last":   {"total": "hist_total_dph_last_log",   "buy": "hist_buy_box_dph_last_log",   "instock": "hist_instock_dph_last_log"},
        "mean4":  {"total": "hist_total_dph_mean4_log",  "buy": "hist_buy_box_dph_mean4_log",  "instock": "hist_instock_dph_mean4_log"},
        "mean13": {"total": "hist_total_dph_mean13_log", "buy": "hist_buy_box_dph_mean13_log", "instock": "hist_instock_dph_mean13_log"},
    }
    rows = []
    for b in va_ld:
        fc = b["future_context"].numpy()
        B, H, _ = fc.shape
        for i in range(B):
            for h in range(H):
                row = {"asin": b["asin"][i], "order_week": pd.to_datetime(b["target_week"][i][h]), "horizon": h + 1}
                for mode, cols in modes.items():
                    row[f"pred_total_dph_{mode}"]   = np.expm1(fc[i, h, idx[cols["total"]]])
                    row[f"pred_buy_box_dph_{mode}"] = np.expm1(fc[i, h, idx[cols["buy"]]])
                    row[f"pred_instock_dph_{mode}"] = np.expm1(fc[i, h, idx[cols["instock"]]])
                rows.append(row)
    return pred_df.merge(pd.DataFrame(rows), on=["asin", "order_week", "horizon"], how="left")


def print_exposure_diagnostics(pred_df):
    print("\n" + "=" * 100)
    print("MODEL EXPOSURE METRICS")
    print("=" * 100)
    model_tbl = exposure_metrics(pred_df, prefix="pred")
    print(model_tbl.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print("BY HORIZON: IN_STOCK_DPH")
    print("=" * 100)
    rows = []
    for h, g in pred_df.groupby("horizon"):
        y = g["true_instock_dph"].values
        p = g["pred_instock_dph"].values
        rows.append({
            "horizon":    h,
            "true_mean":  np.mean(y),
            "pred_mean":  np.mean(p),
            "ratio":      np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE":       _wape(y, p),
            "underbias":  np.maximum(y - p, 0).sum() / (np.abs(y).sum() + 1e-8),
            "overbias":   np.maximum(p - y, 0).sum() / (np.abs(y).sum() + 1e-8),
            "corr":       _corr(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
        })
    by_h = pd.DataFrame(rows)
    print(by_h.round(4).to_string(index=False))

    # ── naive baseline 对比 ───────────────────────────────────
    naive_cols = {
        "naive_last":   "pred_instock_dph_last",
        "naive_mean4":  "pred_instock_dph_mean4",
        "naive_mean13": "pred_instock_dph_mean13",
    }
    available_naive = {k: v for k, v in naive_cols.items() if v in pred_df.columns}

    if available_naive:
        print("\n" + "=" * 100)
        print("MODEL VS NAIVE: IN_STOCK_DPH (overall)")
        print("=" * 100)
        comp_rows = []
        y_all = pred_df["true_instock_dph"].values
        for name, col in [("model", "pred_instock_dph")] + list(available_naive.items()):
            if col not in pred_df.columns:
                continue
            p_all = pred_df[col].values
            comp_rows.append({
                "method":     name,
                "ratio":      np.mean(p_all) / (np.mean(y_all) + 1e-8),
                "WAPE":       _wape(y_all, p_all),
                "active_AUC": _auc((y_all > 0).astype(int), p_all),
                "corr":       _corr(y_all, p_all),
            })
        print(pd.DataFrame(comp_rows).round(4).to_string(index=False))

        print("\n" + "=" * 100)
        print("MODEL VS NAIVE BY HORIZON BLOCK: IN_STOCK_DPH")
        print("=" * 100)
        pred_df["_block"] = pd.cut(
            pred_df["horizon"],
            bins=[0, 5, 12, 20],
            labels=["short_1_5", "mid_6_12", "long_13_20"],
        )
        block_rows = []
        for block, g in pred_df.groupby("_block", observed=True):
            y_b = g["true_instock_dph"].values
            for name, col in [("model", "pred_instock_dph")] + list(available_naive.items()):
                if col not in g.columns:
                    continue
                p_b = g[col].values
                block_rows.append({
                    "block":      block,
                    "method":     name,
                    "ratio":      np.mean(p_b) / (np.mean(y_b) + 1e-8),
                    "WAPE":       _wape(y_b, p_b),
                    "active_AUC": _auc((y_b > 0).astype(int), p_b),
                    "corr":       _corr(y_b, p_b),
                })
        print(pd.DataFrame(block_rows).round(4).to_string(index=False))
        pred_df.drop(columns=["_block"], inplace=True, errors="ignore")

    # ── p_active诊断 ─────────────────────────────────────────
    p_active_cols = [c for c in ["p_active_total", "p_active_buy_box", "p_active_instock"]
                     if c in pred_df.columns]
    if p_active_cols:
        print("\n" + "=" * 100)
        print("P_ACTIVE BY HORIZON (should NOT be monotonically increasing)")
        print("=" * 100)
        pa_rows = []
        for h, g in pred_df.groupby("horizon"):
            row = {"horizon": h}
            for c in p_active_cols:
                row[c] = g[c].mean()
            # 和真实active rate对比
            row["true_active_rate"] = (g["true_instock_dph"] > 0).mean()
            pa_rows.append(row)
        pa_df = pd.DataFrame(pa_rows)
        print(pa_df.round(4).to_string(index=False))

        # 快速判断
        pa_instock = pa_df["p_active_instock"].values if "p_active_instock" in pa_df.columns else None
        if pa_instock is not None:
            is_monotone = all(pa_instock[i] <= pa_instock[i+1] for i in range(len(pa_instock)-1))
            print(f"\np_active_instock monotonically increasing: {is_monotone}")
            if is_monotone:
                print("  ⚠️  Still monotone — BCE may still be too strong")
            else:
                print("  ✅  Not monotone — BCE is calibrated correctly")

    # ── gamma / gate诊断 ─────────────────────────────────────
    gamma_gate_cols = [c for c in ["gamma_instock", "gate_instock"] if c in pred_df.columns]
    if gamma_gate_cols:
        print("\n" + "=" * 100)
        print("GAMMA / GATE BY HORIZON: IN_STOCK")
        print("=" * 100)
        gg_rows = []
        for h, g in pred_df.groupby("horizon"):
            row = {"horizon": h}
            if "gamma_instock" in g.columns:
                row["gamma_instock_mean"] = g["gamma_instock"].mean()
            if "gate_instock" in g.columns:
                row["gate_instock_mean"] = g["gate_instock"].mean()
            if "p_active_instock" in g.columns:
                row["p_active_instock_mean"] = g["p_active_instock"].mean()
            row["true_active_rate"] = (g["true_instock_dph"] > 0).mean()
            gg_rows.append(row)
        print(pd.DataFrame(gg_rows).round(4).to_string(index=False))

    # ── ASIN级别诊断 ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ASIN-LEVEL 20-WEEK SUM")
    print("=" * 100)
    asin_sum = pred_df.groupby("asin").agg(
        true_sum=("true_instock_dph", "sum"),
        pred_sum=("pred_instock_dph", "sum"),
    ).reset_index()
    asin_sum["ratio"] = asin_sum["pred_sum"] / (asin_sum["true_sum"] + 1e-8)
    asin_sum["wape"]  = (asin_sum["pred_sum"] - asin_sum["true_sum"]).abs() / (asin_sum["true_sum"] + 1e-8)
    print(f"ASIN-sum Spearman: {_safe_spearman(asin_sum['true_sum'], asin_sum['pred_sum']):.4f}")
    print(f"Median ASIN ratio: {asin_sum['ratio'].median():.4f}")
    print(f"Median ASIN WAPE:  {asin_sum['wape'].median():.4f}")
    print(f"p90 ASIN WAPE:     {asin_sum['wape'].quantile(0.90):.4f}")

    # ── 快速判断总结 ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("QUICK JUDGMENT")
    print("=" * 100)
    h1  = by_h[by_h["horizon"] == 1].iloc[0]
    h20 = by_h[by_h["horizon"] == 20].iloc[0]
    print(f"h=1  ratio={h1['ratio']:.3f}  WAPE={h1['WAPE']:.3f}  AUC={h1['active_AUC']:.3f}")
    print(f"h=20 ratio={h20['ratio']:.3f}  WAPE={h20['WAPE']:.3f}  AUC={h20['active_AUC']:.3f}")
    print(f"AUC drop h1→h20: {h1['active_AUC'] - h20['active_AUC']:.3f}  (target < 0.20)")
    ratio_ok  = 0.85 <= h20["ratio"] <= 1.15
    auc_ok    = h20["active_AUC"] >= 0.70
    drop_ok   = (h1["active_AUC"] - h20["active_AUC"]) < 0.20
    print(f"\nh=20 ratio in [0.85,1.15]: {'✅' if ratio_ok else '❌'}")
    print(f"h=20 AUC >= 0.70:          {'✅' if auc_ok else '❌'}")
    print(f"AUC drop < 0.20:           {'✅' if drop_ok else '❌'}")

    # ── Final compact summary table ─────────────────────────
    print("\n" + "=" * 100)
    print("FINAL SUMMARY TABLE")
    print("=" * 100)
    final_rows = []
    model_overall = model_tbl[model_tbl["target"] == "in_stock_dph"].iloc[0]
    final_rows.append({
        "section": "overall_instock",
        "ratio": model_overall["pred_true_ratio"],
        "WAPE": model_overall["WAPE"],
        "corr": model_overall["corr"],
        "active_AUC": model_overall["active_AUC"],
        "note": "model overall",
    })
    if available_naive:
        for name, col in available_naive.items():
            p_all = pred_df[col].values
            final_rows.append({
                "section": name,
                "ratio": np.mean(p_all) / (np.mean(y_all) + 1e-8),
                "WAPE": _wape(y_all, p_all),
                "corr": _corr(y_all, p_all),
                "active_AUC": _auc((y_all > 0).astype(int), p_all),
                "note": "baseline",
            })
    final_rows.append({
        "section": "h1_instock",
        "ratio": h1["ratio"],
        "WAPE": h1["WAPE"],
        "corr": h1["corr"],
        "active_AUC": h1["active_AUC"],
        "note": "short horizon",
    })
    final_rows.append({
        "section": "h20_instock",
        "ratio": h20["ratio"],
        "WAPE": h20["WAPE"],
        "corr": h20["corr"],
        "active_AUC": h20["active_AUC"],
        "note": "long horizon",
    })
    if "p_active_instock" in pred_df.columns:
        final_rows.append({
            "section": "p_active_gap",
            "ratio": np.nan,
            "WAPE": np.nan,
            "corr": np.nan,
            "active_AUC": np.nan,
            "note": f"mean p_active - true_active = {((pred_df['p_active_instock'].mean()) - ((pred_df['true_instock_dph'] > 0).mean())):.4f}",
        })
    final_summary = pd.DataFrame(final_rows)
    print(final_summary.round(4).to_string(index=False))

    return {"model": model_tbl, "by_horizon": by_h, "final_summary": final_summary}



# ============================================================
# Encoder / Decoder diagnostics
# ============================================================

def _diagnose_encoder_decoder_performance_impl(model, va_ld, pred_df=None, max_batches=None, device=None):
    """
    Quick diagnostic for whether encoder and decoder learned useful signals.

    Encoder checks:
      - Can h_last classify future active / inactive?
      - Can h_last predict future 20-week magnitude?

    Decoder checks:
      - p_active AUC and calibration
      - active-only magnitude ratio / WAPE
      - cross-attention entropy / concentration
    """
    device = get_device(device)
    model = model.to(device)
    model.eval()

    h_list = []
    y_total_list, y_buy_list, y_instock_list = [], [], []
    p_active_list, log_mag_list, pred_list = [], [], []
    attn_rows = []

    with torch.no_grad():
        for bi, b in enumerate(va_ld):
            if max_batches is not None and bi >= max_batches:
                break
            b = batch_to_device(b, device)

            x = b["x"]
            fc = b["future_context"]
            enc_out = model.encoder(x)
            h_last = enc_out[:, -1, :]
            # Important: call the full model instead of model.decoder(...) so graph context
            # (asin_idx -> graph embedding -> augmented future_context) is included.
            aux = model(x, fc, return_aux=True, asin_idx=b.get("asin_idx"))

            pred_level = torch.expm1(aux["log_hat"]).clamp(min=0.0)
            y_stack = torch.stack([
                b["future_total_dph"],
                b["future_buy_box_dph"],
                b["future_instock_dph"],
            ], dim=-1)

            h_list.append(h_last.detach().cpu().numpy())
            y_total_list.append(y_stack[:, :, 0].detach().cpu().numpy())
            y_buy_list.append(y_stack[:, :, 1].detach().cpu().numpy())
            y_instock_list.append(y_stack[:, :, 2].detach().cpu().numpy())
            p_active_list.append(aux["p_active"].detach().cpu().numpy())
            log_mag_list.append(aux["log_mag"].detach().cpu().numpy())
            pred_list.append(pred_level.detach().cpu().numpy())

            attn = aux.get("attn_weights", None)
            if attn is not None:
                a = attn.detach().cpu().numpy()
                if a.ndim == 4:
                    a = a.mean(axis=1)  # [B,H,T]
                entropy = -(a * np.log(a + 1e-8)).sum(axis=-1)
                max_w = a.max(axis=-1)
                argmax_pos = a.argmax(axis=-1)
                attn_rows.append({
                    "batch": bi,
                    "attn_entropy_mean": float(np.mean(entropy)),
                    "attn_max_weight_mean": float(np.mean(max_w)),
                    "attn_argmax_mean_pos": float(np.mean(argmax_pos)),
                    "attn_argmax_p90_pos": float(np.quantile(argmax_pos, 0.90)),
                })

    h = np.concatenate(h_list, axis=0)
    y_total = np.concatenate(y_total_list, axis=0)
    y_buy = np.concatenate(y_buy_list, axis=0)
    y_instock = np.concatenate(y_instock_list, axis=0)
    p_active = np.concatenate(p_active_list, axis=0)
    log_mag = np.concatenate(log_mag_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)

    target_map = {
        "total": (y_total, pred[:, :, 0], p_active[:, :, 0], log_mag[:, :, 0]),
        "buy_box": (y_buy, pred[:, :, 1], p_active[:, :, 1], log_mag[:, :, 1]),
        "in_stock": (y_instock, pred[:, :, 2], p_active[:, :, 2], log_mag[:, :, 2]),
    }

    encoder_rows = []
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import roc_auc_score, r2_score
    except Exception:
        LogisticRegression = None
        Ridge = None
        roc_auc_score = None
        r2_score = None

    for name, (y, _, _, _) in target_map.items():
        active_any = (y.sum(axis=1) > 0).astype(int)
        y_sum_log = np.log1p(y.sum(axis=1))

        enc_auc = np.nan
        enc_r2 = np.nan
        enc_spearman = np.nan

        if LogisticRegression is not None and len(np.unique(active_any)) == 2:
            try:
                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(h, active_any)
                enc_auc = roc_auc_score(active_any, clf.predict_proba(h)[:, 1])
            except Exception:
                enc_auc = np.nan

        active_mask = y.sum(axis=1) > 0
        if Ridge is not None and active_mask.sum() >= 20:
            try:
                reg = Ridge(alpha=1.0)
                reg.fit(h[active_mask], y_sum_log[active_mask])
                pred_sum_log = reg.predict(h[active_mask])
                enc_r2 = r2_score(y_sum_log[active_mask], pred_sum_log)
                enc_spearman = _safe_spearman(y_sum_log[active_mask], pred_sum_log)
            except Exception:
                enc_r2 = np.nan
                enc_spearman = np.nan

        encoder_rows.append({
            "target": name,
            "future_active_rate": float(active_any.mean()),
            "encoder_active_AUC_same_val": enc_auc,
            "encoder_active_sum_R2_same_val": enc_r2,
            "encoder_active_sum_spearman_same_val": enc_spearman,
        })

    encoder_diag = pd.DataFrame(encoder_rows)

    decoder_rows = []
    by_h_rows = []

    for name, (y, p, pa, lm) in target_map.items():
        y_flat = y.reshape(-1)
        p_flat = p.reshape(-1)
        pa_flat = pa.reshape(-1)
        active_flat = (y_flat > 0).astype(int)

        active_auc = _auc(active_flat, pa_flat)
        active_mask = y_flat > 0

        decoder_rows.append({
            "target": name,
            "true_mean": float(np.mean(y_flat)),
            "pred_mean": float(np.mean(p_flat)),
            "pred_true_ratio": float(np.mean(p_flat) / (np.mean(y_flat) + 1e-8)),
            "p_active_mean": float(np.mean(pa_flat)),
            "true_active_rate": float(np.mean(active_flat)),
            "p_active_AUC": active_auc,
            "active_only_true_mean": float(np.mean(y_flat[active_mask])) if active_mask.sum() else np.nan,
            "active_only_pred_mean": float(np.mean(p_flat[active_mask])) if active_mask.sum() else np.nan,
            "active_only_ratio": float(np.mean(p_flat[active_mask]) / (np.mean(y_flat[active_mask]) + 1e-8)) if active_mask.sum() else np.nan,
            "active_only_WAPE": _wape(y_flat[active_mask], p_flat[active_mask]) if active_mask.sum() else np.nan,
        })

        H = y.shape[1]
        for hh in range(H):
            yh = y[:, hh]
            ph = p[:, hh]
            pah = pa[:, hh]
            active_h = yh > 0
            by_h_rows.append({
                "target": name,
                "horizon": hh + 1,
                "true_mean": float(np.mean(yh)),
                "pred_mean": float(np.mean(ph)),
                "ratio": float(np.mean(ph) / (np.mean(yh) + 1e-8)),
                "true_active_rate": float(np.mean(active_h)),
                "p_active_mean": float(np.mean(pah)),
                "p_active_AUC": _auc(active_h.astype(int), pah),
                "active_only_ratio": float(np.mean(ph[active_h]) / (np.mean(yh[active_h]) + 1e-8)) if active_h.sum() else np.nan,
                "active_only_WAPE": _wape(yh[active_h], ph[active_h]) if active_h.sum() else np.nan,
            })

    decoder_diag = pd.DataFrame(decoder_rows)
    decoder_by_horizon = pd.DataFrame(by_h_rows)
    attn_diag = pd.DataFrame(attn_rows)

    print("\n" + "=" * 100)
    print("ENCODER DIAGNOSTICS: can h_last read occurrence / magnitude?")
    print("=" * 100)
    print(encoder_diag.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("DECODER DIAGNOSTICS: active head + magnitude head")
    print("=" * 100)
    print(decoder_diag.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("DECODER BY HORIZON: IN_STOCK only")
    print("=" * 100)
    in_h = decoder_by_horizon[decoder_by_horizon["target"] == "in_stock"]
    print(in_h.round(4).to_string(index=False))

    if len(attn_diag) > 0:
        print("\n" + "=" * 100)
        print("CROSS-ATTENTION DIAGNOSTICS")
        print("=" * 100)
        print(attn_diag.round(4).to_string(index=False))

    return {
        "encoder_diag": encoder_diag,
        "decoder_diag": decoder_diag,
        "decoder_by_horizon": decoder_by_horizon,
        "attn_diag": attn_diag,
    }

def make_external_hat_df(pred_df, hat_source="mu"):
    """
    Build external exposure hat for the downstream demand model.

    IMPORTANT:
    In the NB-distribution exposure version, pred_*_dph is the sampled distribution
    median / p50 by default. That is useful for zero diagnostics, but it is often too
    conservative as a demand covariate.

    Default hat_source="mu" therefore uses pred_*_dph_mu as the demand covariate:
      - mu = expected exposure level / intensity
      - p50 = conservative median, useful for zero-risk diagnostics

    Options:
      hat_source="mu"        -> use pred_*_dph_mu for demand input, recommended
      hat_source="p50"       -> use pred_*_dph / sampled median
      hat_source="dist_mean" -> use pred_*_dph_dist_mean
    """
    df = pred_df.copy()

    if hat_source == "mu":
        source_cols = {
            "pred_total_dph_mu": "pred_total_dph",
            "pred_buy_box_dph_mu": "pred_buy_box_dph",
            "pred_instock_dph_mu": "pred_instock_dph",
        }
    elif hat_source == "dist_mean":
        source_cols = {
            "pred_total_dph_dist_mean": "pred_total_dph",
            "pred_buy_box_dph_dist_mean": "pred_buy_box_dph",
            "pred_instock_dph_dist_mean": "pred_instock_dph",
        }
    elif hat_source == "p50":
        source_cols = {
            "pred_total_dph": "pred_total_dph",
            "pred_buy_box_dph": "pred_buy_box_dph",
            "pred_instock_dph": "pred_instock_dph",
        }
    else:
        raise ValueError(f"Unknown hat_source={hat_source}. Use 'mu', 'p50', or 'dist_mean'.")

    missing = [c for c in source_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for hat_source={hat_source}: {missing}")

    out = df[["asin", "order_week"] + list(source_cols.keys())].copy()
    out = out.rename(columns=source_cols)

    # Safety cleanup only; normally model outputs should already be finite and nonnegative.
    pred_cols = ["pred_total_dph", "pred_buy_box_dph", "pred_instock_dph"]
    out[pred_cols] = (
        out[pred_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # Enforce funnel for demand input: total >= buy_box >= instock.
    out["pred_buy_box_dph"] = np.minimum(out["pred_buy_box_dph"], out["pred_total_dph"])
    out["pred_instock_dph"] = np.minimum(out["pred_instock_dph"], out["pred_buy_box_dph"])

    out["external_total_dph_hat_log"]    = np.log1p(out["pred_total_dph"].clip(lower=0.0))
    out["external_buy_box_dph_hat_log"]  = np.log1p(out["pred_buy_box_dph"].clip(lower=0.0))
    out["external_instock_dph_hat_log"]  = np.log1p(out["pred_instock_dph"].clip(lower=0.0))
    out["hat_source"] = hat_source
    return out


def summarize_hat_for_demand(hat, title="EXPOSURE HAT FOR DEMAND"):
    pred_cols = ["pred_total_dph", "pred_buy_box_dph", "pred_instock_dph"]
    pred_cols = [c for c in pred_cols if c in hat.columns]
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("shape:", hat.shape)
    if "hat_source" in hat.columns:
        print("hat_source:", hat["hat_source"].iloc[0])
    print("\ndescribe:")
    print(hat[pred_cols].describe())
    print("\nzero share:")
    print((hat[pred_cols] == 0).mean())
    print("\nnegative count:")
    print((hat[pred_cols] < 0).sum())
    print("\nNaN count:")
    print(hat[pred_cols].isna().sum())
    return None


# ============================================================
# 主入口
# ============================================================


# Exposure-only public alias. Use this name to avoid confusion with demand utilities.
def summarize_exposure_hat_for_demand(hat, title="EXPOSURE HAT FOR DEMAND"):
    return summarize_hat_for_demand(hat, title=title)




# ============================================================
# Compact diagnostics override for patch model
# ============================================================
def print_exposure_diagnostics(pred_df):
    """Compact diagnostics only: overall, horizon blocks, h1/h20, ASIN sum.
    Removed verbose GL/category tables and detailed p_active/gamma tables to keep
    iteration fast and easy to read.
    """
    print("\n" + "=" * 100)
    print("COMPACT EXPOSURE DIAGNOSTICS")
    print("=" * 100)

    model_tbl = exposure_metrics(pred_df, prefix="pred")
    print("\n[Overall three-hat metrics]")
    print(model_tbl.round(5).to_string(index=False))

    # Horizon blocks aligned with patch decoder.
    block_defs = [("short_h1_5", 1, 5), ("mid_h6_13", 6, 13), ("long_h14_20", 14, 20)]
    block_rows = []
    for name, lo, hi in block_defs:
        g = pred_df[(pred_df["horizon"] >= lo) & (pred_df["horizon"] <= hi)]
        if len(g) == 0:
            continue
        y = g["true_instock_dph"].values
        p = g["pred_instock_dph"].values
        block_rows.append({
            "block": name,
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p),
            "ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "underbias": np.maximum(y - p, 0).sum() / (np.abs(y).sum() + 1e-8),
            "overbias": np.maximum(p - y, 0).sum() / (np.abs(y).sum() + 1e-8),
            "corr": _corr(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
        })
    block_df = pd.DataFrame(block_rows)
    print("\n[In-stock metrics by patch block]")
    print(block_df.round(4).to_string(index=False))

    # h1/h20 only, not all 20 rows.
    edge_rows = []
    for hh in [1, 20]:
        g = pred_df[pred_df["horizon"] == hh]
        if len(g) == 0:
            continue
        y = g["true_instock_dph"].values
        p = g["pred_instock_dph"].values
        edge_rows.append({
            "horizon": hh,
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p),
            "ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "corr": _corr(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
        })
    edge_df = pd.DataFrame(edge_rows)
    print("\n[Edge horizons]")
    print(edge_df.round(4).to_string(index=False))

    asin_sum = pred_df.groupby("asin").agg(
        true_sum=("true_instock_dph", "sum"),
        pred_sum=("pred_instock_dph", "sum"),
    ).reset_index()
    asin_sum["ratio"] = asin_sum["pred_sum"] / (asin_sum["true_sum"] + 1e-8)
    asin_sum["wape"] = (asin_sum["pred_sum"] - asin_sum["true_sum"]).abs() / (asin_sum["true_sum"] + 1e-8)
    asin_summary = pd.DataFrame([{
        "ASIN_sum_spearman": _safe_spearman(asin_sum["true_sum"], asin_sum["pred_sum"]),
        "median_ASIN_ratio": asin_sum["ratio"].median(),
        "median_ASIN_WAPE": asin_sum["wape"].median(),
        "p90_ASIN_WAPE": asin_sum["wape"].quantile(0.90),
    }])
    print("\n[ASIN 20-week sum]")
    print(asin_summary.round(4).to_string(index=False))

    # Final compact judgment.
    final_rows = []
    overall = model_tbl[model_tbl["target"] == "in_stock_dph"].iloc[0]
    final_rows.append({"section":"overall_instock", "ratio":overall["pred_true_ratio"], "WAPE":overall["WAPE"], "corr":overall["corr"], "active_AUC":overall["active_AUC"]})
    for _, r in block_df.iterrows():
        final_rows.append({"section":str(r["block"]), "ratio":r["ratio"], "WAPE":r["WAPE"], "corr":r["corr"], "active_AUC":r["active_AUC"]})
    final_summary = pd.DataFrame(final_rows)
    print("\n[Final compact summary]")
    print(final_summary.round(4).to_string(index=False))

    return {"model": model_tbl, "blocks": block_df, "edge_horizons": edge_df, "asin_summary": asin_summary, "final_summary": final_summary}

def run_exposure_v2(
    data_raw1,
    scot_df=None,    # 不再使用，保留接口兼容
    n_asins=5000,
    seed=42,
    history=13,
    horizon=20,
    d_model=48,      # 64→48，减少参数防过拟合
    n_heads=4,
    batch_size=64,
    epochs=80,       # 60→80，给模型更多时间
    lr=5e-4,         # 1e-3→5e-4，更稳定
    patience=15,     # 8→15，避免过早停止
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    apply_funnel_constraint=True,
    anchor_decay=0.08,
    bce_weight=0.20,
    mag_weight=1.00,
    mean_weight=0.25,
    active_calib_weight=0.05,
    nb_weight=0.25,
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.15,
    high_weight_alpha=0.35,
    short_horizon_weight=0.8,
    mid_horizon_weight=1.2,
    long_horizon_weight=2.0,
    long_block_sum_weight=0.30,
    path_zero_weight=0.00,
    zero_fp_weight=0.00,
    active_count_weight=0.00,
    path_sum_weight=0.05,
    asin_sum_weight=0.40,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    zero_fp_threshold=50.0,
    zero_fp_temperature=20.0,
    dropout=0.20,    # 0.10→0.20，加强dropout防过拟合
    use_graphsage=False,
    neighbor_k=10,
    graph_dim=16,
    graph_message_scale=0.04,
    use_graph_encoder_fusion=True,
    graph_encoder_scale=0.05,
    use_graph_head=False,
    graph_head_scale=0.05,
    use_graph_gate=True,
    use_rank_correction=True,
    rank_correction_scale=0.03,
    rank_horizon_min_scale=0.40,
    use_horizon_calibration=True,
    horizon_calibration_scale=0.03,
    use_gl_calibration=True,
    gl_calibration_scale=0.025,
    use_category_calibration=True,
    category_calibration_scale=0.015,
    category_shrinkage_k=100.0,
    use_group_calibration=False,
    group_calibration_scale=0.0,
    graph_zero_weight=0.03,
    graph_level_peak_weight=2.2,
    graph_transition_weight=1.0,
    graph_static_weight=1.0,
    graph_brand_weight=0.3,
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V15: LEARNED EDGE + GRAPH EARLY FUSION + HORIZON/CAT CALIBRATION")
    print("Preset: dynamic relation graph + sparse/event-aware rank prior + long-horizon calibration; MU hats for demand")
    print("=" * 100)

    df = prepare_exposure_data_from_sample(data_raw1, scot_df, n_asins, seed)
    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    data, context_dim, context_cols, graph_assets = load_exposure_data(
        df, dph_cap_q=dph_cap_q,
        use_graphsage=use_graphsage, graph_horizon=horizon, neighbor_k=neighbor_k,
        graph_zero_weight=graph_zero_weight, graph_level_peak_weight=graph_level_peak_weight,
        graph_transition_weight=graph_transition_weight, graph_static_weight=graph_static_weight,
        graph_brand_weight=graph_brand_weight,
    )

    tr_ds = ExposureDataset(data, history=history, horizon=horizon,
                            mode="train", val_weeks=horizon, anchor_decay=anchor_decay)
    va_ds = ExposureDataset(data, history=history, horizon=horizon,
                            mode="val",   val_weeks=horizon, anchor_decay=anchor_decay)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=exposure_collate, pin_memory=dataloader_pin_memory())
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=exposure_collate, pin_memory=dataloader_pin_memory())

    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    input_dim = next(iter(tr_ld))["x"].shape[-1]

    model = ExposureForecastModelV2(
        input_dim=input_dim,
        context_dim=context_dim,
        d_model=d_model,
        horizon=horizon,
        n_heads=n_heads,
        dropout=dropout,
        context_cols=context_cols,
        use_encoder_self_attn=use_encoder_self_attn,
        use_graphsage=use_graphsage,
        graph_assets=graph_assets,
        graph_dim=graph_dim,
        graph_message_scale=graph_message_scale,
        use_graph_encoder_fusion=use_graph_encoder_fusion,
        graph_encoder_scale=graph_encoder_scale,
        use_graph_head=use_graph_head,
        graph_head_scale=graph_head_scale,
        use_graph_gate=use_graph_gate,
        use_rank_correction=use_rank_correction,
        rank_correction_scale=rank_correction_scale,
        rank_horizon_min_scale=rank_horizon_min_scale,
        use_horizon_calibration=use_horizon_calibration,
        horizon_calibration_scale=horizon_calibration_scale,
        use_gl_calibration=use_gl_calibration,
        gl_calibration_scale=gl_calibration_scale,
        use_category_calibration=use_category_calibration,
        category_calibration_scale=category_calibration_scale,
        category_shrinkage_k=category_shrinkage_k,
        use_group_calibration=use_group_calibration,
        group_calibration_scale=group_calibration_scale,
    )
    print(f"Input dim: {input_dim} | Context dim: {context_dim}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_exposure_model_v2(
        model=model, tr_ld=tr_ld, va_ld=va_ld,
        epochs=epochs, lr=lr, patience=patience,
        bce_weight=bce_weight, mag_weight=mag_weight, mean_weight=mean_weight,
        active_calib_weight=active_calib_weight,
        zero_weight=zero_weight,
        total_zero_weight=total_zero_weight,
        buy_zero_weight=buy_zero_weight,
        instock_zero_weight=instock_zero_weight,
        total_zero_consistency_weight=total_zero_consistency_weight,
        buy_zero_consistency_weight=buy_zero_consistency_weight,
        horizon_weight_alpha=horizon_weight_alpha, high_weight_alpha=high_weight_alpha,
        short_horizon_weight=short_horizon_weight,
        mid_horizon_weight=mid_horizon_weight,
        long_horizon_weight=long_horizon_weight,
        long_block_sum_weight=long_block_sum_weight,
        path_zero_weight=path_zero_weight,
        zero_fp_weight=zero_fp_weight,
        active_count_weight=active_count_weight,
        path_sum_weight=path_sum_weight,
        asin_sum_weight=asin_sum_weight,
        peak_weight=peak_weight,
        topk_peak_weight=topk_peak_weight,
        peak_under_weight=peak_under_weight,
        peak_topk=peak_topk,
        peak_quantile=peak_quantile,
        zero_fp_threshold=zero_fp_threshold,
        zero_fp_temperature=zero_fp_temperature,
    )

    pred_df = predict_exposure_v2(model, va_ld, apply_funnel_constraint=apply_funnel_constraint, mc_reduce="mu")
    pred_df = add_naive_baselines_from_loader(pred_df, va_ld, context_cols)
    diagnostics = print_exposure_diagnostics(pred_df)
    diagnostics_views = print_mu_vs_p50_quick_diagnostics(pred_df)
    exposure_hat_for_demand = make_external_hat_df(pred_df, hat_source="mu")
    exposure_hat_for_demand_p50 = make_external_hat_df(pred_df, hat_source="p50")
    exposure_hat_for_demand_dist_mean = make_external_hat_df(pred_df, hat_source="dist_mean")

    summarize_hat_for_demand(exposure_hat_for_demand, title="EXPOSURE HAT FOR DEMAND (MU / EXPECTED LEVEL)")

    return {
        "model": model,
        "forecast_df": pred_df,
        "diagnostics": diagnostics,
        "diagnostics_views": diagnostics_views,
        "exposure_hat_for_demand": exposure_hat_for_demand,
        "exposure_hat_for_demand_mu": exposure_hat_for_demand,
        "exposure_hat_for_demand_p50": exposure_hat_for_demand_p50,
        "exposure_hat_for_demand_dist_mean": exposure_hat_for_demand_dist_mean,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
        "graph_assets": graph_assets,
    }


# ============================================================
# ============================================================
# Rolling Backtest + SCOT Intersection Add-on
# Added after original definitions; these functions override/use the fixed ABC model above.
# ============================================================

def prepare_exposure_data_from_sample_scot_intersection(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
):
    df = data_raw1.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])

    rng = np.random.default_rng(seed)
    unique_asins = df["asin"].dropna().unique()
    sample_asins = rng.choice(
        unique_asins,
        size=min(n_asins, len(unique_asins)),
        replace=False,
    )
    sample_asin_set = set(sample_asins)

    if scot_df is None:
        out = df[df["asin"].isin(sample_asin_set)].copy()
        out = fill_missing_dph_after_scot_merge(out, verbose=True)
        print(f"Sampled ASINs: {len(sample_asin_set)} | Rows: {len(out)}")
        return out

    scot = scot_df.copy()
    scot["asin"] = scot["asin"].astype(str)
    scot_asin_set = set(scot["asin"].dropna().unique())
    intersect_asins = sorted(sample_asin_set & scot_asin_set)

    out = df[df["asin"].isin(intersect_asins)].copy()
    out = fill_missing_dph_after_scot_merge(out, verbose=True)
    print("\n" + "=" * 100)
    print("SAMPLE + SCOT INTERSECTION")
    print("=" * 100)
    print(f"Sample ASINs: {len(sample_asin_set)}")
    print(f"SCOT ASINs: {len(scot_asin_set)}")
    print(f"Intersection ASINs: {len(intersect_asins)}")
    print(f"Rows after intersection: {len(out)}")
    print("=" * 100)
    return out


class ExposureDatasetRolling(Dataset):
    def __init__(
        self,
        data,
        history=13,
        horizon=20,
        mode="train",
        val_start_offset=0,
        anchor_decay=0.08,
    ):
        self.samples = []
        self.data = data
        self.history = history
        self.horizon = horizon
        self.anchor_decay = anchor_decay
        self.val_start_offset = int(val_start_offset)

        for asin, d in data.items():
            T = len(d["features"])
            val_start = T - history - horizon - self.val_start_offset

            if mode == "train":
                starts = range(max(0, val_start))
            else:
                starts = [val_start] if val_start >= 0 and (val_start + history + horizon) <= T else []

            for start in starts:
                self.samples.append((asin, start))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _hist_mean(arr, end, window):
        x = arr[max(0, end - window):end]
        return float(np.mean(x)) if len(x) > 0 else 0.0

    def _make_future_context(self, d, start):
        h = self.history
        H = self.horizon
        fc = d["future_context"][start + h:start + h + H].copy()
        cols = d["context_cols"]
        idx = {c: i for i, c in enumerate(cols)}
        end = start + h

        total = d["total_dph"]
        buy = d["buy_box_dph"]
        instock = d["in_stock_dph"]
        demand = d["demand"]

        post_event_col = "post_event_decay"
        current_post_decay = float(fc[0, idx[post_event_col]]) if post_event_col in idx and len(fc) > 0 else 0.0
        post_strength = 0.5
        effective_post_decay = post_strength * current_post_decay

        for step_h in range(H):
            h_decay = np.exp(-self.anchor_decay * step_h)
            for prefix, arr in [("total", total), ("buy_box", buy), ("instock", instock)]:
                mean13_val = np.log1p(self._hist_mean(arr, end, 13))
                mean4_val = np.log1p(self._hist_mean(arr, end, 4))
                raw_last = np.log1p(arr[end - 1]) if end > 0 else 0.0
                last_val = raw_last * (1.0 - effective_post_decay) + mean13_val * effective_post_decay

                for col, val in [
                    (f"hist_{prefix}_dph_last_log", h_decay * last_val + (1 - h_decay) * mean13_val),
                    (f"hist_{prefix}_dph_mean4_log", h_decay * mean4_val + (1 - h_decay) * mean13_val),
                    (f"hist_{prefix}_dph_mean13_log", mean13_val),
                ]:
                    if col in idx:
                        fc[step_h, idx[col]] = val

        demand_last = np.log1p(demand[end - 1]) if end > 0 else 0.0
        demand_mean4 = np.log1p(self._hist_mean(demand, end, 4))
        demand_mean13 = np.log1p(self._hist_mean(demand, end, 13))
        demand_active_rate = float(np.mean(demand[max(0, end - 13):end] > 0)) if end > 0 else 0.0

        for step_h in range(H):
            h_decay = np.exp(-self.anchor_decay * step_h)
            for col, val in [
                ("hist_demand_last_log", h_decay * demand_last + (1 - h_decay) * demand_mean13),
                ("hist_demand_mean4_log", h_decay * demand_mean4 + (1 - h_decay) * demand_mean13),
                ("hist_demand_mean13_log", demand_mean13),
                ("hist_demand_active_rate", demand_active_rate),
            ]:
                if col in idx:
                    fc[step_h, idx[col]] = val

        return fc

    def __getitem__(self, i):
        asin, start = self.samples[i]
        d = self.data[asin]
        h = self.history
        H = self.horizon

        return {
            "asin": asin,
            "target_week": [str(w)[:10] for w in d["week"][start + h:start + h + H]],
            "x": torch.tensor(d["features"][start:start + h], dtype=torch.float32),
            "future_context": torch.tensor(self._make_future_context(d, start), dtype=torch.float32),
            "future_total_dph": torch.tensor(d["total_dph"][start + h:start + h + H], dtype=torch.float32),
            "future_buy_box_dph": torch.tensor(d["buy_box_dph"][start + h:start + h + H], dtype=torch.float32),
            "future_instock_dph": torch.tensor(d["in_stock_dph"][start + h:start + h + H], dtype=torch.float32),
            "future_demand": torch.tensor(d["demand"][start + h:start + h + H], dtype=torch.float32),
            "asin_idx": torch.tensor(int(d.get("asin_idx", 0)), dtype=torch.long),
        }


def summarize_rolling_exposure(pred_df, label="ROLLING"):
    print("\n" + "=" * 100)
    print(f"{label}: OVERALL METRICS")
    print("=" * 100)
    tbl = exposure_metrics(pred_df, prefix="pred")
    print(tbl.round(5).to_string(index=False))

    rows = []
    for (offset, h), g in pred_df.groupby(["backtest_offset", "horizon"]):
        y = g["true_instock_dph"].values
        p = g["pred_instock_dph"].values
        rows.append({
            "backtest_offset": offset,
            "horizon": h,
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p),
            "ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "underbias": np.maximum(y - p, 0).sum() / (np.abs(y).sum() + 1e-8),
            "overbias": np.maximum(p - y, 0).sum() / (np.abs(y).sum() + 1e-8),
            "corr": _corr(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
        })
    by_offset_horizon = pd.DataFrame(rows)

    rows2 = []
    for offset, g in pred_df.groupby("backtest_offset"):
        y = g["true_instock_dph"].values
        p = g["pred_instock_dph"].values
        rows2.append({
            "backtest_offset": offset,
            "n_rows": len(g),
            "n_asins": g["asin"].nunique(),
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p),
            "ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "underbias": np.maximum(y - p, 0).sum() / (np.abs(y).sum() + 1e-8),
            "overbias": np.maximum(p - y, 0).sum() / (np.abs(y).sum() + 1e-8),
            "corr": _corr(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
        })
    by_offset = pd.DataFrame(rows2)

    print("\n" + "=" * 100)
    print(f"{label}: BY BACKTEST OFFSET")
    print("=" * 100)
    print(by_offset.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print(f"{label}: BY OFFSET + HORIZON")
    print("=" * 100)
    print(by_offset_horizon.round(4).to_string(index=False))

    return {"overall": tbl, "by_offset": by_offset, "by_offset_horizon": by_offset_horizon}


def _train_one_exposure_window(
    data,
    context_dim,
    context_cols,
    history=13,
    horizon=20,
    val_start_offset=0,
    d_model=48,
    n_heads=4,
    batch_size=128,
    epochs=20,
    lr=5e-4,
    patience=5,
    apply_funnel_constraint=True,
    anchor_decay=0.08,
    bce_weight=0.20,
    mag_weight=1.00,
    mean_weight=0.25,
    active_calib_weight=0.05,
    nb_weight=0.25,
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.25,
    high_weight_alpha=0.35,
    path_zero_weight=0.00,
    zero_fp_weight=0.00,
    active_count_weight=0.00,
    path_sum_weight=0.05,
    asin_sum_weight=0.40,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    zero_fp_threshold=50.0,
    zero_fp_temperature=20.0,
    dropout=0.20,
    use_graphsage=False,
    graph_assets=None,
    graph_dim=16,
    graph_message_scale=0.04,
    use_graph_encoder_fusion=True,
    graph_encoder_scale=0.05,
    use_graph_head=False,
    graph_head_scale=0.05,
    use_graph_gate=True,
    use_rank_correction=True,
    rank_correction_scale=0.03,
    rank_horizon_min_scale=0.40,
    use_horizon_calibration=True,
    horizon_calibration_scale=0.03,
    use_gl_calibration=True,
    gl_calibration_scale=0.025,
    use_category_calibration=True,
    category_calibration_scale=0.015,
    category_shrinkage_k=100.0,
    use_group_calibration=False,
    group_calibration_scale=0.0,
    use_encoder_self_attn=True,
):
    tr_ds = ExposureDatasetRolling(
        data,
        history=history,
        horizon=horizon,
        mode="train",
        val_start_offset=val_start_offset,
        anchor_decay=anchor_decay,
    )
    va_ds = ExposureDatasetRolling(
        data,
        history=history,
        horizon=horizon,
        mode="val",
        val_start_offset=val_start_offset,
        anchor_decay=anchor_decay,
    )

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=exposure_collate, pin_memory=dataloader_pin_memory())
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=exposure_collate, pin_memory=dataloader_pin_memory())

    print("\n" + "=" * 100)
    print(f"BACKTEST OFFSET = {val_start_offset}")
    print("=" * 100)
    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise ValueError(f"Empty train/val set for val_start_offset={val_start_offset}")

    input_dim = next(iter(tr_ld))["x"].shape[-1]
    model = ExposureForecastModelV2(
        input_dim=input_dim,
        context_dim=context_dim,
        d_model=d_model,
        horizon=horizon,
        n_heads=n_heads,
        dropout=dropout,
        context_cols=context_cols,
        use_encoder_self_attn=use_encoder_self_attn,
        use_graphsage=use_graphsage,
        graph_assets=graph_assets,
        graph_dim=graph_dim,
        graph_message_scale=graph_message_scale,
        use_graph_encoder_fusion=use_graph_encoder_fusion,
        graph_encoder_scale=graph_encoder_scale,
        use_graph_head=use_graph_head,
        graph_head_scale=graph_head_scale,
        use_graph_gate=use_graph_gate,
        use_rank_correction=use_rank_correction,
        rank_correction_scale=rank_correction_scale,
        rank_horizon_min_scale=rank_horizon_min_scale,
        use_horizon_calibration=use_horizon_calibration,
        horizon_calibration_scale=horizon_calibration_scale,
        use_gl_calibration=use_gl_calibration,
        gl_calibration_scale=gl_calibration_scale,
        use_category_calibration=use_category_calibration,
        category_calibration_scale=category_calibration_scale,
        category_shrinkage_k=category_shrinkage_k,
        use_group_calibration=use_group_calibration,
        group_calibration_scale=group_calibration_scale,
    )
    print(f"Input dim: {input_dim} | Context dim: {context_dim}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_exposure_model_v2(
        model=model,
        tr_ld=tr_ld,
        va_ld=va_ld,
        epochs=epochs,
        lr=lr,
        patience=patience,
        bce_weight=bce_weight,
        mag_weight=mag_weight,
        mean_weight=mean_weight,
        active_calib_weight=active_calib_weight,
        zero_weight=zero_weight,
        total_zero_weight=total_zero_weight,
        buy_zero_weight=buy_zero_weight,
        instock_zero_weight=instock_zero_weight,
        total_zero_consistency_weight=total_zero_consistency_weight,
        buy_zero_consistency_weight=buy_zero_consistency_weight,
        horizon_weight_alpha=horizon_weight_alpha,
        high_weight_alpha=high_weight_alpha,
        path_zero_weight=path_zero_weight,
        zero_fp_weight=zero_fp_weight,
        active_count_weight=active_count_weight,
        path_sum_weight=path_sum_weight,
        asin_sum_weight=asin_sum_weight,
        peak_weight=peak_weight,
        topk_peak_weight=topk_peak_weight,
        peak_under_weight=peak_under_weight,
        peak_topk=peak_topk,
        peak_quantile=peak_quantile,
        zero_fp_threshold=zero_fp_threshold,
        zero_fp_temperature=zero_fp_temperature,
    )

    pred_df = predict_exposure_v2(model, va_ld, apply_funnel_constraint=apply_funnel_constraint, mc_reduce="mu")
    pred_df = add_naive_baselines_from_loader(pred_df, va_ld, context_cols)
    pred_df["backtest_offset"] = int(val_start_offset)

    diagnostics = print_exposure_diagnostics(pred_df)
    return {
        "model": model,
        "forecast_df": pred_df,
        "diagnostics": diagnostics,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "tr_ds": tr_ds,
        "va_ds": va_ds,
    }




# ============================================================
# DualRelationalGAT diagnostics: is graph embedding useful, and for what?
# ============================================================

def diagnose_dualgraph_signal(model, graph_assets, pred_df, target="instock", verbose=True):
    """
    Probe whether DualRelationalGAT embedding carries useful information.

    Prints/returns:
      1. neighbor homophily: does KNN graph actually connect similar GL/category/brand nodes?
      2. graph embedding probes: can graph embedding alone explain ASIN-level true 20w sum?
      3. graph norm quartiles: are high/low graph regimes associated with different true/pred levels?

    This is diagnostic only; it does not affect predictions.
    """
    if graph_assets is None or model is None or not getattr(model, "use_graphsage", False):
        if verbose:
            print("DualRelationalGAT diagnostics skipped: graph is disabled.")
        return {}

    diag = {}
    try:
        with torch.no_grad():
            node_feat = model.graph_node_features
            neigh_idx = model.graph_neighbor_idx
            comp_idx = getattr(model, "graph_competitive_neighbor_idx", neigh_idx)
            emb = model.graph_encoder(node_feat, neigh_idx, comp_idx).detach().cpu().numpy()
        idx_to_asin = graph_assets.get("idx_to_asin", [str(i) for i in range(emb.shape[0])])
        emb_cols = [f"g{i}" for i in range(emb.shape[1])]
        emb_df = pd.DataFrame(emb, columns=emb_cols)
        emb_df["asin"] = [str(a) for a in idx_to_asin]
        emb_df["graph_norm"] = np.linalg.norm(emb, axis=1)
        diag["graph_embedding_df"] = emb_df
    except Exception as e:
        if verbose:
            print(f"Graph embedding extraction failed: {e}")
        return {"error": str(e)}

    # Homophily over constructed neighbors.
    try:
        meta = graph_assets.get("meta_df", pd.DataFrame()).copy()
        nb = graph_assets["neighbor_idx"]
        gl = meta["gl_product_group"].astype(str).values if "gl_product_group" in meta.columns else None
        cat = meta["category_code"].astype(str).values if "category_code" in meta.columns else None
        br = meta["ind_top10_brand"].astype(float).values if "ind_top10_brand" in meta.columns else None
        hom = {}
        if gl is not None:
            hom["same_gl"] = float(np.mean([np.mean(gl[nb[i]] == gl[i]) for i in range(len(nb))]))
        if cat is not None:
            hom["same_category"] = float(np.mean([np.mean(cat[nb[i]] == cat[i]) for i in range(len(nb))]))
        if br is not None:
            hom["same_top10_brand_state"] = float(np.mean([np.mean(br[nb[i]] == br[i]) for i in range(len(nb))]))
        diag["neighbor_homophily"] = hom
    except Exception as e:
        diag["neighbor_homophily_error"] = str(e)

    # ASIN-level target/pred summary.
    true_col = f"true_{target}_dph" if f"true_{target}_dph" in pred_df.columns else "true_instock_dph"
    pred_col = f"pred_{target}_dph" if f"pred_{target}_dph" in pred_df.columns else "pred_instock_dph"
    if true_col not in pred_df.columns or pred_col not in pred_df.columns:
        if verbose:
            print("Graph diagnostics warning: target/pred columns not found in pred_df.")
        return diag

    asin_sum = (
        pred_df.groupby("asin")
        .agg(
            true_sum=(true_col, "sum"),
            pred_sum=(pred_col, "sum"),
            true_mean=(true_col, "mean"),
            pred_mean=(pred_col, "mean"),
            active_rate=(true_col, lambda x: np.mean(np.asarray(x) > 0)),
        )
        .reset_index()
    )
    asin_sum["asin"] = asin_sum["asin"].astype(str)
    m = emb_df.merge(asin_sum, on="asin", how="inner")
    if len(m) == 0:
        return diag

    # Ridge probe: graph embedding alone -> log true 20w sum.
    probe_rows = []
    try:
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, roc_auc_score
        X = m[emb_cols].values.astype(float)
        y = np.log1p(m["true_sum"].values.astype(float))
        if len(m) >= 50 and np.std(y) > 1e-8:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42)
            reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
            y_hat = reg.predict(X_te)
            probe_rows.append({
                "probe": "graph_emb_to_log_true_sum_R2",
                "value": float(r2_score(y_te, y_hat)),
                "note": "higher means graph embedding carries magnitude/scale information",
            })
        # active path probe
        yb = (m["active_rate"].values.astype(float) > 0.5).astype(int)
        if len(np.unique(yb)) == 2 and len(m) >= 50:
            X_tr, X_te, y_tr, y_te = train_test_split(X, yb, test_size=0.30, random_state=42, stratify=yb)
            clf = LogisticRegression(max_iter=500, C=1.0).fit(X_tr, y_tr)
            score = clf.predict_proba(X_te)[:, 1]
            probe_rows.append({
                "probe": "graph_emb_to_active_path_AUC",
                "value": float(roc_auc_score(y_te, score)),
                "note": "higher means graph embedding carries active/zero regime information",
            })
    except Exception as e:
        probe_rows.append({"probe": "graph_probe_error", "value": np.nan, "note": str(e)})

    # Correlation and quartile summary.
    m["ratio"] = m["pred_sum"] / (m["true_sum"] + 1e-8)
    try:
        spearman_norm_true = _safe_spearman(m["graph_norm"], m["true_sum"])
        spearman_norm_ratio = _safe_spearman(m["graph_norm"], m["ratio"])
        probe_rows.append({"probe": "spearman_graph_norm_true_sum", "value": float(spearman_norm_true), "note": "graph norm vs true 20w level"})
        probe_rows.append({"probe": "spearman_graph_norm_pred_true_ratio", "value": float(spearman_norm_ratio), "note": "positive/negative indicates graph norm is linked to bias"})
    except Exception:
        pass

    probe_df = pd.DataFrame(probe_rows)
    diag["graph_probe"] = probe_df

    try:
        m["graph_norm_bucket"] = pd.qcut(m["graph_norm"], q=4, duplicates="drop")
        bucket = (
            m.groupby("graph_norm_bucket")
            .agg(
                n_asins=("asin", "nunique"),
                true_sum_mean=("true_sum", "mean"),
                pred_sum_mean=("pred_sum", "mean"),
                ratio=("ratio", "median"),
                active_rate=("active_rate", "mean"),
            )
            .reset_index()
        )
        diag["graph_norm_bucket"] = bucket
    except Exception as e:
        diag["graph_norm_bucket_error"] = str(e)

    if verbose:
        print("\n" + "=" * 100)
        print("DUAL-RELATION GRAPHSAGE EFFECT DIAGNOSTICS")
        print("=" * 100)
        if "neighbor_homophily" in diag:
            print("Neighbor homophily:", {k: round(v, 4) for k, v in diag["neighbor_homophily"].items()})
        if len(probe_df):
            print("\nGraph embedding probes:")
            print(probe_df.round(4).to_string(index=False))
        if "graph_norm_bucket" in diag:
            print("\nGraph norm bucket summary:")
            print(diag["graph_norm_bucket"].round(4).to_string(index=False))
        print("\nInterpretation:")
        print("- Good active AUC but low R2 to true_sum => graph mainly learns occurrence/regime, not magnitude.")
        print("- Low ratio in high graph_norm buckets => graph message may be conservative/smoothing too much.")
        print("- If same_category/same_brand are very low, edges may be too behavior-only; if too high, graph may be too static/clustered.")

    return diag


def run_exposure_v2(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
    history=13,
    horizon=20,
    d_model=48,
    n_heads=4,
    batch_size=128,
    epochs=30,
    lr=5e-4,
    patience=6,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    apply_funnel_constraint=True,
    anchor_decay=0.08,
    bce_weight=0.20,
    mag_weight=1.00,
    mean_weight=0.25,
    active_calib_weight=0.05,
    nb_weight=0.25,
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.25,
    high_weight_alpha=0.35,
    path_zero_weight=0.00,
    zero_fp_weight=0.00,
    active_count_weight=0.00,
    path_sum_weight=0.05,
    asin_sum_weight=0.40,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    zero_fp_threshold=50.0,
    zero_fp_temperature=20.0,
    dropout=0.20,
    use_scot_intersection=True,
    val_start_offset=0,
    use_graphsage=False,
    neighbor_k=10,
    graph_dim=16,
    graph_message_scale=0.04,
    use_graph_encoder_fusion=True,
    graph_encoder_scale=0.05,
    use_graph_head=False,
    graph_head_scale=0.05,
    use_graph_gate=True,
    use_rank_correction=True,
    rank_correction_scale=0.03,
    rank_horizon_min_scale=0.40,
    use_horizon_calibration=True,
    horizon_calibration_scale=0.03,
    use_gl_calibration=True,
    gl_calibration_scale=0.025,
    use_category_calibration=True,
    category_calibration_scale=0.015,
    category_shrinkage_k=100.0,
    use_group_calibration=False,
    group_calibration_scale=0.0,
    graph_zero_weight=0.03,
    graph_level_peak_weight=2.2,
    graph_transition_weight=1.0,
    graph_static_weight=1.0,
    graph_brand_weight=0.3,
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V18: V15_PKG + KNOWN FUTURE PROMOTION CONTEXT")
    print("=" * 100)

    if use_scot_intersection:
        df = prepare_exposure_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins, seed)
    else:
        df = prepare_exposure_data_from_sample(data_raw1, scot_df, n_asins, seed)

    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    data, context_dim, context_cols, graph_assets = load_exposure_data(
        df, dph_cap_q=dph_cap_q,
        use_graphsage=use_graphsage, graph_horizon=horizon, neighbor_k=neighbor_k,
        graph_zero_weight=graph_zero_weight, graph_level_peak_weight=graph_level_peak_weight,
        graph_transition_weight=graph_transition_weight, graph_static_weight=graph_static_weight,
        graph_brand_weight=graph_brand_weight,
    )

    out = _train_one_exposure_window(
        data=data,
        context_dim=context_dim,
        context_cols=context_cols,
        history=history,
        horizon=horizon,
        val_start_offset=val_start_offset,
        d_model=d_model,
        n_heads=n_heads,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        apply_funnel_constraint=apply_funnel_constraint,
        anchor_decay=anchor_decay,
        bce_weight=bce_weight,
        mag_weight=mag_weight,
        mean_weight=mean_weight,
        active_calib_weight=active_calib_weight,
        zero_weight=zero_weight,
        total_zero_weight=total_zero_weight,
        buy_zero_weight=buy_zero_weight,
        instock_zero_weight=instock_zero_weight,
        total_zero_consistency_weight=total_zero_consistency_weight,
        buy_zero_consistency_weight=buy_zero_consistency_weight,
        horizon_weight_alpha=horizon_weight_alpha,
        high_weight_alpha=high_weight_alpha,
        path_zero_weight=path_zero_weight,
        zero_fp_weight=zero_fp_weight,
        active_count_weight=active_count_weight,
        path_sum_weight=path_sum_weight,
        asin_sum_weight=asin_sum_weight,
        peak_weight=peak_weight,
        topk_peak_weight=topk_peak_weight,
        peak_under_weight=peak_under_weight,
        peak_topk=peak_topk,
        peak_quantile=peak_quantile,
        dropout=dropout,
        use_graphsage=use_graphsage,
        graph_assets=graph_assets,
        graph_dim=graph_dim,
        graph_message_scale=graph_message_scale,
        use_graph_encoder_fusion=use_graph_encoder_fusion,
        graph_encoder_scale=graph_encoder_scale,
        use_graph_head=use_graph_head,
        graph_head_scale=graph_head_scale,
        use_graph_gate=use_graph_gate,
        use_rank_correction=use_rank_correction,
        rank_correction_scale=rank_correction_scale,
        rank_horizon_min_scale=rank_horizon_min_scale,
        use_horizon_calibration=use_horizon_calibration,
        horizon_calibration_scale=horizon_calibration_scale,
        use_gl_calibration=use_gl_calibration,
        gl_calibration_scale=gl_calibration_scale,
        use_category_calibration=use_category_calibration,
        category_calibration_scale=category_calibration_scale,
        category_shrinkage_k=category_shrinkage_k,
        use_group_calibration=use_group_calibration,
        group_calibration_scale=group_calibration_scale,
        use_encoder_self_attn=use_encoder_self_attn,
    )

    pred_df = out["forecast_df"]

    # GL-level diagnostics are useful because GL groups have different seasonal/burst patterns.
    gl_diag = diagnose_by_gl_group(pred_df, df, target="instock", min_asins=30, top_n=30)
    gl_block_diag = diagnose_by_gl_horizon_block(pred_df, df, target="instock", min_asins=30)
    gl_summary = summarize_gl_diagnostics(gl_diag, min_asins=30)
    out["diagnostics"]["gl_group"] = gl_diag
    out["diagnostics"]["gl_horizon_block"] = gl_block_diag
    out["diagnostics"]["gl_summary"] = gl_summary

    graph_diagnostics = diagnose_dualgraph_signal(out.get("model"), graph_assets, pred_df, target="instock", verbose=True) if use_graphsage else {}
    out["diagnostics"]["graph"] = graph_diagnostics

    out.update({
        "exposure_hat_for_demand": make_external_hat_df(pred_df, hat_source="mu"),
        "exposure_hat_for_demand_mu": make_external_hat_df(pred_df, hat_source="mu"),
        "exposure_hat_for_demand_p50": make_external_hat_df(pred_df, hat_source="p50"),
        "exposure_hat_for_demand_dist_mean": make_external_hat_df(pred_df, hat_source="dist_mean"),
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
        "source_df": df,
        "graph_assets": graph_assets,
        "gl_diagnostics": gl_diag,
        "gl_horizon_block_diagnostics": gl_block_diag,
        "gl_summary": gl_summary,
    })
    return out


def run_exposure_v2_rolling(
    data_raw1,
    scot_df=None,
    n_asins=1000,
    seed=42,
    history=13,
    horizon=20,
    rolling_offsets=(60, 40, 20, 0),
    d_model=48,
    n_heads=4,
    batch_size=128,
    epochs=20,
    lr=5e-4,
    patience=5,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    apply_funnel_constraint=True,
    anchor_decay=0.08,
    bce_weight=0.20,
    mag_weight=1.00,
    mean_weight=0.25,
    active_calib_weight=0.05,
    nb_weight=0.25,
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.25,
    high_weight_alpha=0.35,
    path_zero_weight=0.00,
    zero_fp_weight=0.00,
    active_count_weight=0.00,
    path_sum_weight=0.05,
    asin_sum_weight=0.40,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    zero_fp_threshold=50.0,
    zero_fp_temperature=20.0,
    dropout=0.20,
    use_scot_intersection=True,
    use_graphsage=False,
    neighbor_k=10,
    graph_dim=16,
    graph_message_scale=0.04,
    use_graph_encoder_fusion=True,
    graph_encoder_scale=0.05,
    use_graph_head=False,
    graph_head_scale=0.05,
    use_graph_gate=True,
    use_rank_correction=True,
    rank_correction_scale=0.03,
    rank_horizon_min_scale=0.40,
    use_horizon_calibration=True,
    horizon_calibration_scale=0.03,
    use_gl_calibration=True,
    gl_calibration_scale=0.025,
    use_category_calibration=True,
    category_calibration_scale=0.015,
    category_shrinkage_k=100.0,
    use_group_calibration=False,
    group_calibration_scale=0.0,
    graph_zero_weight=0.03,
    graph_level_peak_weight=2.2,
    graph_transition_weight=1.0,
    graph_static_weight=1.0,
    graph_brand_weight=0.3,
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V14: ROLLING BACKTEST + SCOT INTERSECTION + HORIZON + GL/CATEGORY SHRINKAGE CALIB")
    print("=" * 100)
    print(f"n_asins={n_asins} | history={history} | rolling_offsets={list(rolling_offsets)} | epochs={epochs} | patience={patience} | encoder_attn={use_encoder_self_attn}")

    if use_scot_intersection:
        df = prepare_exposure_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins, seed)
    else:
        df = prepare_exposure_data_from_sample(data_raw1, scot_df, n_asins, seed)

    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    data, context_dim, context_cols, graph_assets = load_exposure_data(
        df, dph_cap_q=dph_cap_q,
        use_graphsage=use_graphsage, graph_horizon=horizon, neighbor_k=neighbor_k,
        graph_zero_weight=graph_zero_weight, graph_level_peak_weight=graph_level_peak_weight,
        graph_transition_weight=graph_transition_weight, graph_static_weight=graph_static_weight,
        graph_brand_weight=graph_brand_weight,
    )

    results_by_offset = {}
    pred_list = []

    for offset in rolling_offsets:
        # V10 rolling-safe dynamic relation graph:
        # Rebuild graph at each validation origin.  For offset=40, the graph excludes
        # the final 20+40 weeks, so neighbor relations only use history before that origin.
        if use_graphsage:
            data_offset, context_dim_offset, context_cols_offset, graph_assets_offset = load_exposure_data(
                df, dph_cap_q=dph_cap_q,
                use_graphsage=True, graph_horizon=horizon + int(offset), neighbor_k=neighbor_k,
                graph_zero_weight=graph_zero_weight, graph_level_peak_weight=graph_level_peak_weight,
                graph_transition_weight=graph_transition_weight, graph_static_weight=graph_static_weight,
                graph_brand_weight=graph_brand_weight,
            )
        else:
            data_offset, context_dim_offset, context_cols_offset, graph_assets_offset = data, context_dim, context_cols, graph_assets
        try:
            res = _train_one_exposure_window(
                data=data_offset,
                context_dim=context_dim_offset,
                context_cols=context_cols_offset,
                history=history,
                horizon=horizon,
                val_start_offset=int(offset),
                d_model=d_model,
                n_heads=n_heads,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                patience=patience,
                apply_funnel_constraint=apply_funnel_constraint,
                anchor_decay=anchor_decay,
                bce_weight=bce_weight,
                mag_weight=mag_weight,
                mean_weight=mean_weight,
                active_calib_weight=active_calib_weight,
                zero_weight=zero_weight,
                total_zero_weight=total_zero_weight,
                buy_zero_weight=buy_zero_weight,
                instock_zero_weight=instock_zero_weight,
                total_zero_consistency_weight=total_zero_consistency_weight,
                buy_zero_consistency_weight=buy_zero_consistency_weight,
                horizon_weight_alpha=horizon_weight_alpha,
                high_weight_alpha=high_weight_alpha,
                path_zero_weight=path_zero_weight,
                zero_fp_weight=zero_fp_weight,
                active_count_weight=active_count_weight,
                path_sum_weight=path_sum_weight,
                asin_sum_weight=asin_sum_weight,
                short_horizon_weight=short_horizon_weight,
                mid_horizon_weight=mid_horizon_weight,
                long_horizon_weight=long_horizon_weight,
                long_block_sum_weight=long_block_sum_weight,
                peak_weight=peak_weight,
                topk_peak_weight=topk_peak_weight,
                peak_under_weight=peak_under_weight,
                peak_topk=peak_topk,
                peak_quantile=peak_quantile,
                zero_fp_threshold=zero_fp_threshold,
                zero_fp_temperature=zero_fp_temperature,
                dropout=dropout,
                use_graphsage=use_graphsage,
                graph_assets=graph_assets_offset,
                graph_dim=graph_dim,
                graph_message_scale=graph_message_scale,
                use_encoder_self_attn=use_encoder_self_attn,
            )
            results_by_offset[int(offset)] = res
            pred_list.append(res["forecast_df"])
        except Exception as e:
            print(f"[SKIP] offset={offset} failed: {e}")

    if len(pred_list) == 0:
        raise RuntimeError("All rolling backtest windows failed.")

    rolling_pred_df = pd.concat(pred_list, ignore_index=True)
    rolling_diagnostics = summarize_rolling_exposure(rolling_pred_df, label="ROLLING BACKTEST")

    latest_offset = 0 if 0 in results_by_offset else sorted(results_by_offset.keys())[-1]
    latest_pred_df = results_by_offset[latest_offset]["forecast_df"]

    # GL diagnostics for the latest prediction window and for all rolling windows.
    latest_gl_diag = diagnose_by_gl_group(latest_pred_df, df, target="instock", min_asins=30, top_n=30)
    latest_gl_block_diag = diagnose_by_gl_horizon_block(latest_pred_df, df, target="instock", min_asins=30)
    latest_gl_summary = summarize_gl_diagnostics(latest_gl_diag, min_asins=30)

    rolling_gl_diag = diagnose_by_gl_group(rolling_pred_df, df, target="instock", min_asins=30, top_n=30)
    rolling_gl_summary = summarize_gl_diagnostics(rolling_gl_diag, min_asins=30)

    rolling_diagnostics["latest_gl_group"] = latest_gl_diag
    rolling_diagnostics["latest_gl_horizon_block"] = latest_gl_block_diag
    rolling_diagnostics["latest_gl_summary"] = latest_gl_summary
    rolling_diagnostics["rolling_gl_group"] = rolling_gl_diag
    rolling_diagnostics["rolling_gl_summary"] = rolling_gl_summary

    return {
        "results_by_offset": results_by_offset,
        "rolling_forecast_df": rolling_pred_df,
        "forecast_df": latest_pred_df,
        "diagnostics": rolling_diagnostics,
        "exposure_hat_for_demand": make_external_hat_df(latest_pred_df, hat_source="mu"),
        "exposure_hat_for_demand_mu": make_external_hat_df(latest_pred_df, hat_source="mu"),
        "exposure_hat_for_demand_p50": make_external_hat_df(latest_pred_df, hat_source="p50"),
        "exposure_hat_for_demand_dist_mean": make_external_hat_df(latest_pred_df, hat_source="dist_mean"),
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
        "source_df": df,
        "graph_assets": graph_assets,
        "rolling_offsets": list(rolling_offsets),
        "gl_diagnostics": latest_gl_diag,
        "gl_horizon_block_diagnostics": latest_gl_block_diag,
        "gl_summary": latest_gl_summary,
        "rolling_gl_diagnostics": rolling_gl_diag,
        "rolling_gl_summary": rolling_gl_summary,
    }





# ============================================================
# GL diagnostics: check whether different GL groups are calibrated differently
# ============================================================

def _attach_gl_product_group(pred_df, source_df):
    """
    Attach one GL product group per ASIN to a prediction dataframe.
    This uses source_df after sampling/filtering when available.
    """
    tmp = pred_df.copy()
    tmp["asin"] = tmp["asin"].astype(str)

    if source_df is None or "gl_product_group" not in source_df.columns:
        tmp["gl_product_group"] = "MISSING"
        return tmp

    gl_map = (
        source_df[["asin", "gl_product_group"]]
        .dropna(subset=["asin"])
        .drop_duplicates("asin")
        .copy()
    )
    gl_map["asin"] = gl_map["asin"].astype(str)
    gl_map["gl_product_group"] = gl_map["gl_product_group"].astype(str).fillna("MISSING")

    tmp = tmp.merge(gl_map, on="asin", how="left")
    tmp["gl_product_group"] = tmp["gl_product_group"].astype(str).fillna("MISSING")
    return tmp


def diagnose_by_gl_group(pred_df, source_df, target="instock", min_asins=30, top_n=30):
    """
    Per-GL diagnostics for exposure forecast.

    target can be:
      - "instock" / "in_stock"
      - "buy_box"
      - "total"

    Returns a dataframe with one row per GL group.
    """
    target = str(target).lower()
    col_map = {
        "instock": ("true_instock_dph", "pred_instock_dph", "p_active_instock"),
        "in_stock": ("true_instock_dph", "pred_instock_dph", "p_active_instock"),
        "buy_box": ("true_buy_box_dph", "pred_buy_box_dph", "p_active_buy_box"),
        "buybox": ("true_buy_box_dph", "pred_buy_box_dph", "p_active_buy_box"),
        "total": ("true_total_dph", "pred_total_dph", "p_active_total"),
    }
    if target not in col_map:
        raise ValueError(f"Unknown target={target}. Use instock, buy_box, or total.")

    true_col, pred_col, p_col = col_map[target]
    tmp = _attach_gl_product_group(pred_df, source_df)

    rows = []
    for gl, g in tmp.groupby("gl_product_group", dropna=False):
        y = g[true_col].values.astype(float)
        p = g[pred_col].values.astype(float)
        active = (y > 0).astype(int)
        rows.append({
            "gl_product_group": gl,
            "n_rows": int(len(g)),
            "n_asins": int(g["asin"].nunique()),
            "true_mean": float(np.mean(y)),
            "pred_mean": float(np.mean(p)),
            "ratio": float(np.mean(p) / (np.mean(y) + 1e-8)),
            "WAPE": float(_wape(y, p)),
            "underbias": float(np.maximum(y - p, 0).sum() / (np.abs(y).sum() + 1e-8)),
            "overbias": float(np.maximum(p - y, 0).sum() / (np.abs(y).sum() + 1e-8)),
            "corr": float(_corr(y, p)) if not np.isnan(_corr(y, p)) else np.nan,
            "active_AUC": float(_auc(active, p)) if not np.isnan(_auc(active, p)) else np.nan,
            "true_active_rate": float(np.mean(y > 0)),
            "p_active_mean": float(g[p_col].mean()) if p_col in g.columns else np.nan,
            "p_active_minus_true": float(g[p_col].mean() - np.mean(y > 0)) if p_col in g.columns else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("n_asins", ascending=False).reset_index(drop=True)
    eligible = out[out["n_asins"] >= min_asins].copy()

    print("\n" + "=" * 100)
    print(f"PER-GL DIAGNOSTICS: {target.upper()} DPH")
    print("=" * 100)
    if len(out) == 0:
        print("No GL diagnostics available.")
        return out

    print("Top GL groups by ASIN count:")
    display(out.head(top_n).round(4))

    print("\n" + "=" * 100)
    print(f"GL GROUPS WITH LARGEST OVERPREDICTION (n_asins >= {min_asins})")
    print("=" * 100)
    display(eligible.sort_values("ratio", ascending=False).head(15).round(4))

    print("\n" + "=" * 100)
    print(f"GL GROUPS WITH LARGEST UNDERPREDICTION (n_asins >= {min_asins})")
    print("=" * 100)
    display(eligible.sort_values("ratio", ascending=True).head(15).round(4))

    print("\n" + "=" * 100)
    print(f"GL GROUPS WITH WORST WAPE (n_asins >= {min_asins})")
    print("=" * 100)
    display(eligible.sort_values("WAPE", ascending=False).head(15).round(4))

    return out


def diagnose_by_gl_horizon_block(pred_df, source_df, target="instock", min_asins=30):
    """
    Per-GL x horizon block diagnostics.
    This tells whether each GL is over/under mainly in short, middle, or long horizons.
    """
    target = str(target).lower()
    col_map = {
        "instock": ("true_instock_dph", "pred_instock_dph"),
        "in_stock": ("true_instock_dph", "pred_instock_dph"),
        "buy_box": ("true_buy_box_dph", "pred_buy_box_dph"),
        "buybox": ("true_buy_box_dph", "pred_buy_box_dph"),
        "total": ("true_total_dph", "pred_total_dph"),
    }
    if target not in col_map:
        raise ValueError(f"Unknown target={target}. Use instock, buy_box, or total.")

    true_col, pred_col = col_map[target]
    tmp = _attach_gl_product_group(pred_df, source_df)
    tmp["block"] = pd.cut(
        tmp["horizon"],
        bins=[0, 5, 12, 20],
        labels=["short_1_5", "mid_6_12", "long_13_20"],
    )

    rows = []
    for (gl, block), g in tmp.groupby(["gl_product_group", "block"], observed=True):
        n_asins = int(g["asin"].nunique())
        if n_asins < min_asins:
            continue
        y = g[true_col].values.astype(float)
        p = g[pred_col].values.astype(float)
        rows.append({
            "gl_product_group": gl,
            "block": str(block),
            "n_asins": n_asins,
            "n_rows": int(len(g)),
            "true_mean": float(np.mean(y)),
            "pred_mean": float(np.mean(p)),
            "ratio": float(np.mean(p) / (np.mean(y) + 1e-8)),
            "WAPE": float(_wape(y, p)),
            "underbias": float(np.maximum(y - p, 0).sum() / (np.abs(y).sum() + 1e-8)),
            "overbias": float(np.maximum(p - y, 0).sum() / (np.abs(y).sum() + 1e-8)),
            "corr": float(_corr(y, p)) if not np.isnan(_corr(y, p)) else np.nan,
            "active_AUC": float(_auc((y > 0).astype(int), p)) if not np.isnan(_auc((y > 0).astype(int), p)) else np.nan,
        })

    out = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(f"PER-GL × HORIZON BLOCK DIAGNOSTICS: {target.upper()} DPH")
    print("=" * 100)
    if len(out) == 0:
        print("No GL x block diagnostics available. Try lowering min_asins.")
        return out

    display(out.sort_values(["gl_product_group", "block"]).round(4))

    print("\n" + "=" * 100)
    print("WORST GL × BLOCK OVERPREDICTION")
    print("=" * 100)
    display(out.sort_values("ratio", ascending=False).head(20).round(4))

    print("\n" + "=" * 100)
    print("WORST GL × BLOCK UNDERPREDICTION")
    print("=" * 100)
    display(out.sort_values("ratio", ascending=True).head(20).round(4))

    return out


def summarize_gl_diagnostics(gl_diag, min_asins=30):
    """
    Compact summary to decide whether the next fix should be global calibration or GL-specific calibration.
    """
    if gl_diag is None or len(gl_diag) == 0:
        return {}
    g = gl_diag[gl_diag["n_asins"] >= min_asins].copy()
    if len(g) == 0:
        return {}

    summary = {
        "n_gl_groups": int(len(g)),
        "share_over_1p10": float((g["ratio"] > 1.10).mean()),
        "share_under_0p90": float((g["ratio"] < 0.90).mean()),
        "median_ratio": float(g["ratio"].median()),
        "weighted_ratio_by_rows": float(np.average(g["ratio"], weights=g["n_rows"])),
        "median_WAPE": float(g["WAPE"].median()),
        "median_active_AUC": float(g["active_AUC"].median()),
        "median_p_active_minus_true": float(g["p_active_minus_true"].median()) if "p_active_minus_true" in g.columns else np.nan,
    }

    print("\n" + "=" * 100)
    print("GL DIAGNOSTIC SUMMARY")
    print("=" * 100)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if summary["share_over_1p10"] > 0.50:
        print("\nJudgment: most GL groups are overpredicted → global calibration/gamma should be fixed first.")
    elif summary["share_under_0p90"] > 0.50:
        print("\nJudgment: most GL groups are underpredicted → global level/gamma may be too conservative.")
    else:
        print("\nJudgment: bias is GL-specific → consider GL-specific calibration or GL embedding next.")

    return summary

def run_exposure_v2_final_scot_5000(
    data_raw1,
    scot_df,
    seed=42,
    history=13,
    horizon=20,
    epochs=60,
    patience=10,
    batch_size=128,
    use_graphsage=False,
    neighbor_k=10,
    graph_dim=16,
    graph_message_scale=0.04,
    use_graph_encoder_fusion=True,
    graph_encoder_scale=0.05,
    use_graph_head=False,
    graph_head_scale=0.05,
    use_graph_gate=True,
    use_rank_correction=True,
    rank_correction_scale=0.03,
    rank_horizon_min_scale=0.40,
    use_horizon_calibration=True,
    horizon_calibration_scale=0.03,
    use_gl_calibration=True,
    gl_calibration_scale=0.025,
    use_category_calibration=True,
    category_calibration_scale=0.015,
    category_shrinkage_k=100.0,
    use_group_calibration=False,
    group_calibration_scale=0.0,
    graph_zero_weight=0.03,
    graph_level_peak_weight=2.2,
    graph_transition_weight=1.0,
    graph_static_weight=1.0,
    graph_brand_weight=0.3,
    use_encoder_self_attn=True,
):
    """
    Final single-window setup:
      - sample 5000 ASINs
      - intersect with SCOT ASINs
      - train on sliding windows before the final holdout
      - validate/predict the latest 20-week window only
      - return exposure_hat_for_demand for the demand model
    """
    return run_exposure_v2(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=5000,
        seed=seed,
        history=history,
        horizon=horizon,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        use_scot_intersection=True,
        val_start_offset=0,
        use_graphsage=use_graphsage,
        neighbor_k=neighbor_k,
        graph_dim=graph_dim,
        graph_message_scale=graph_message_scale,
        use_graph_encoder_fusion=use_graph_encoder_fusion,
        graph_encoder_scale=graph_encoder_scale,
        use_graph_head=use_graph_head,
        graph_head_scale=graph_head_scale,
        use_graph_gate=use_graph_gate,
        use_rank_correction=use_rank_correction,
        rank_correction_scale=rank_correction_scale,
        rank_horizon_min_scale=rank_horizon_min_scale,
        use_horizon_calibration=use_horizon_calibration,
        horizon_calibration_scale=horizon_calibration_scale,
        use_gl_calibration=use_gl_calibration,
        gl_calibration_scale=gl_calibration_scale,
        use_category_calibration=use_category_calibration,
        category_calibration_scale=category_calibration_scale,
        category_shrinkage_k=category_shrinkage_k,
        use_group_calibration=use_group_calibration,
        group_calibration_scale=group_calibration_scale,
        graph_zero_weight=graph_zero_weight,
        graph_level_peak_weight=graph_level_peak_weight,
        graph_transition_weight=graph_transition_weight,
        graph_static_weight=graph_static_weight,
        graph_brand_weight=graph_brand_weight,
        use_encoder_self_attn=use_encoder_self_attn,
    )

# ============================================================
# CLEAN DIAGNOSTICS OVERRIDE
# Keep only useful checks for the current direction:
#   1) overall exposure quality
#   2) horizon behavior
#   3) naive comparison
#   4) ASIN-level 20-week sum
#   5) compact final summary
# Category_code is already included in load_exposure_data / context features.
# Heavy encoder/decoder diagnostics are disabled to reduce noise and runtime.
# ============================================================

def print_exposure_diagnostics(pred_df):
    print("\n" + "=" * 100)
    print("MODEL EXPOSURE METRICS")
    print("=" * 100)
    model_tbl = exposure_metrics(pred_df, prefix="pred")
    print(model_tbl.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print("BY HORIZON: IN_STOCK_DPH")
    print("=" * 100)
    rows = []
    for h, g in pred_df.groupby("horizon"):
        y = g["true_instock_dph"].values
        p = g["pred_instock_dph"].values
        rows.append({
            "horizon": h,
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p),
            "ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "underbias": np.maximum(y - p, 0).sum() / (np.abs(y).sum() + 1e-8),
            "overbias": np.maximum(p - y, 0).sum() / (np.abs(y).sum() + 1e-8),
            "corr": _corr(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
        })
    by_h = pd.DataFrame(rows)
    print(by_h.round(4).to_string(index=False))

    naive_cols = {
        "naive_last": "pred_instock_dph_last",
        "naive_mean4": "pred_instock_dph_mean4",
        "naive_mean13": "pred_instock_dph_mean13",
    }
    available_naive = {k: v for k, v in naive_cols.items() if v in pred_df.columns}
    y_all = pred_df["true_instock_dph"].values

    comp_rows = []
    for name, col in [("model", "pred_instock_dph")] + list(available_naive.items()):
        if col not in pred_df.columns:
            continue
        p_all = pred_df[col].values
        comp_rows.append({
            "method": name,
            "ratio": np.mean(p_all) / (np.mean(y_all) + 1e-8),
            "WAPE": _wape(y_all, p_all),
            "active_AUC": _auc((y_all > 0).astype(int), p_all),
            "corr": _corr(y_all, p_all),
        })
    comp_tbl = pd.DataFrame(comp_rows)
    print("\n" + "=" * 100)
    print("MODEL VS NAIVE: IN_STOCK_DPH")
    print("=" * 100)
    print(comp_tbl.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("MODEL VS NAIVE BY HORIZON BLOCK: IN_STOCK_DPH")
    print("=" * 100)
    pred_df["_block"] = pd.cut(
        pred_df["horizon"],
        bins=[0, 5, 12, 20],
        labels=["short_1_5", "mid_6_12", "long_13_20"],
    )
    block_rows = []
    for block, g in pred_df.groupby("_block", observed=True):
        y_b = g["true_instock_dph"].values
        for name, col in [("model", "pred_instock_dph")] + list(available_naive.items()):
            if col not in g.columns:
                continue
            p_b = g[col].values
            block_rows.append({
                "block": block,
                "method": name,
                "ratio": np.mean(p_b) / (np.mean(y_b) + 1e-8),
                "WAPE": _wape(y_b, p_b),
                "active_AUC": _auc((y_b > 0).astype(int), p_b),
                "corr": _corr(y_b, p_b),
            })
    block_tbl = pd.DataFrame(block_rows)
    print(block_tbl.round(4).to_string(index=False))
    pred_df.drop(columns=["_block"], inplace=True, errors="ignore")

    print("\n" + "=" * 100)
    print("ASIN-LEVEL 20-WEEK SUM")
    print("=" * 100)
    asin_sum = pred_df.groupby("asin").agg(
        true_sum=("true_instock_dph", "sum"),
        pred_sum=("pred_instock_dph", "sum"),
    ).reset_index()
    asin_sum["ratio"] = asin_sum["pred_sum"] / (asin_sum["true_sum"] + 1e-8)
    asin_sum["wape"] = (asin_sum["pred_sum"] - asin_sum["true_sum"]).abs() / (asin_sum["true_sum"] + 1e-8)
    print(f"ASIN-sum Spearman: {_safe_spearman(asin_sum['true_sum'], asin_sum['pred_sum']):.4f}")
    print(f"Median ASIN ratio: {asin_sum['ratio'].median():.4f}")
    print(f"Median ASIN WAPE:  {asin_sum['wape'].median():.4f}")
    print(f"p90 ASIN WAPE:     {asin_sum['wape'].quantile(0.90):.4f}")

    h1 = by_h[by_h["horizon"] == 1].iloc[0]
    h20 = by_h[by_h["horizon"] == 20].iloc[0]
    print("\n" + "=" * 100)
    print("QUICK JUDGMENT")
    print("=" * 100)
    print(f"h=1  ratio={h1['ratio']:.3f}  WAPE={h1['WAPE']:.3f}  AUC={h1['active_AUC']:.3f}")
    print(f"h=20 ratio={h20['ratio']:.3f}  WAPE={h20['WAPE']:.3f}  AUC={h20['active_AUC']:.3f}")
    print(f"AUC drop h1→h20: {h1['active_AUC'] - h20['active_AUC']:.3f}  (target < 0.20)")

    model_overall = model_tbl[model_tbl["target"] == "in_stock_dph"].iloc[0]
    final_rows = [{
        "section": "overall_instock",
        "ratio": model_overall["pred_true_ratio"],
        "WAPE": model_overall["WAPE"],
        "corr": model_overall["corr"],
        "active_AUC": model_overall["active_AUC"],
        "note": "model overall",
    }]
    for _, r in comp_tbl.iterrows():
        if r["method"] != "model":
            final_rows.append({
                "section": r["method"],
                "ratio": r["ratio"],
                "WAPE": r["WAPE"],
                "corr": r["corr"],
                "active_AUC": r["active_AUC"],
                "note": "baseline",
            })
    final_rows += [
        {"section": "h1_instock", "ratio": h1["ratio"], "WAPE": h1["WAPE"], "corr": h1["corr"], "active_AUC": h1["active_AUC"], "note": "short horizon"},
        {"section": "h20_instock", "ratio": h20["ratio"], "WAPE": h20["WAPE"], "corr": h20["corr"], "active_AUC": h20["active_AUC"], "note": "long horizon"},
    ]
    if "p_active_instock" in pred_df.columns:
        p_gap = pred_df["p_active_instock"].mean() - (pred_df["true_instock_dph"] > 0).mean()
        final_rows.append({"section": "p_active_gap", "ratio": np.nan, "WAPE": np.nan, "corr": np.nan, "active_AUC": np.nan, "note": f"mean p_active - true_active = {p_gap:.4f}"})
    final_summary = pd.DataFrame(final_rows)
    print("\n" + "=" * 100)
    print("FINAL SUMMARY TABLE")
    print("=" * 100)
    print(final_summary.round(4).to_string(index=False))

    return {
        "model": model_tbl,
        "by_horizon": by_h,
        "model_vs_naive": comp_tbl,
        "block_vs_naive": block_tbl,
        "asin_sum": asin_sum,
        "final_summary": final_summary,
    }



def diagnose_graph_gates(pred_df, target="instock", verbose=True):
    """
    Compact diagnostics for V9 DualGAT horizon-level positive/competitive graph gates.
    It checks whether the model uses graph differently by horizon, active state,
    prediction error direction, and ASIN true-sum bucket.
    """
    required = ["graph_pos_gate", "graph_comp_gate", "graph_gate", "graph_pos_minus_comp_gate"]
    if pred_df is None or len(pred_df) == 0 or not all(c in pred_df.columns for c in required):
        if verbose:
            print("[Graph gate diagnostics] No graph gate columns found. Run V9 gated file with use_graphsage=True.")
        return {}

    true_col = f"true_{target}_dph" if target != "instock" else "true_instock_dph"
    pred_col = f"pred_{target}_dph" if target != "instock" else "pred_instock_dph"
    if true_col not in pred_df.columns or pred_col not in pred_df.columns:
        true_col, pred_col = "true_instock_dph", "pred_instock_dph"

    df = pred_df.copy()
    df["err"] = df[pred_col] - df[true_col]
    df["abs_err"] = df["err"].abs()
    df["active"] = (df[true_col] > 0).astype(int)
    df["under"] = (df["err"] < 0).astype(int)

    by_h = df.groupby("horizon").agg(
        pos_gate_mean=("graph_pos_gate", "mean"),
        comp_gate_mean=("graph_comp_gate", "mean"),
        gate_mean=("graph_gate", "mean"),
        pos_minus_comp=("graph_pos_minus_comp_gate", "mean"),
        true_mean=(true_col, "mean"),
        pred_mean=(pred_col, "mean"),
        WAPE_num=("abs_err", "sum"),
        WAPE_den=(true_col, lambda x: np.abs(x).sum() + 1e-8),
    ).reset_index()
    by_h["WAPE"] = by_h["WAPE_num"] / by_h["WAPE_den"]
    by_h = by_h.drop(columns=["WAPE_num", "WAPE_den"])

    by_active = df.groupby("active").agg(
        n=("asin", "count"),
        pos_gate_mean=("graph_pos_gate", "mean"),
        comp_gate_mean=("graph_comp_gate", "mean"),
        gate_mean=("graph_gate", "mean"),
        pos_minus_comp=("graph_pos_minus_comp_gate", "mean"),
        true_mean=(true_col, "mean"),
        pred_mean=(pred_col, "mean"),
    ).reset_index()

    by_error = df.groupby("under").agg(
        n=("asin", "count"),
        pos_gate_mean=("graph_pos_gate", "mean"),
        comp_gate_mean=("graph_comp_gate", "mean"),
        gate_mean=("graph_gate", "mean"),
        pos_minus_comp=("graph_pos_minus_comp_gate", "mean"),
        abs_err_mean=("abs_err", "mean"),
    ).reset_index()

    asin = df.groupby("asin").agg(
        true_sum=(true_col, "sum"),
        pred_sum=(pred_col, "sum"),
        pos_gate_mean=("graph_pos_gate", "mean"),
        comp_gate_mean=("graph_comp_gate", "mean"),
        gate_mean=("graph_gate", "mean"),
        pos_norm=("graph_pos_norm", "mean"),
        comp_norm=("graph_comp_norm", "mean"),
        pos_attn_entropy=("graph_pos_attn_entropy", "mean"),
        comp_attn_entropy=("graph_comp_attn_entropy", "mean"),
    ).reset_index()
    try:
        asin["true_sum_bucket"] = pd.qcut(asin["true_sum"], q=4, duplicates="drop")
        by_bucket = asin.groupby("true_sum_bucket").agg(
            n=("asin", "count"),
            true_sum_mean=("true_sum", "mean"),
            pred_sum_mean=("pred_sum", "mean"),
            pos_gate_mean=("pos_gate_mean", "mean"),
            comp_gate_mean=("comp_gate_mean", "mean"),
            gate_mean=("gate_mean", "mean"),
            pos_norm_mean=("pos_norm", "mean"),
            comp_norm_mean=("comp_norm", "mean"),
        ).reset_index()
        by_bucket["ratio"] = by_bucket["pred_sum_mean"] / (by_bucket["true_sum_mean"] + 1e-8)
    except Exception as e:
        by_bucket = pd.DataFrame({"error": [str(e)]})

    probe = pd.DataFrame([
        {"probe": "corr_pos_gate_true", "value": _corr(df["graph_pos_gate"], df[true_col])},
        {"probe": "corr_comp_gate_true", "value": _corr(df["graph_comp_gate"], df[true_col])},
        {"probe": "corr_pos_minus_comp_true", "value": _corr(df["graph_pos_minus_comp_gate"], df[true_col])},
        {"probe": "corr_graph_gate_abs_err", "value": _corr(df["graph_gate"], df["abs_err"])},
    ])

    if verbose:
        print("\n" + "=" * 100)
        print("GRAPH GATE DIAGNOSTICS")
        print("=" * 100)
        print("\n[Gate by horizon]")
        print(by_h.round(4).to_string(index=False))
        print("\n[Gate by active/zero target]")
        print(by_active.round(4).to_string(index=False))
        print("\n[Gate by error direction: under=1 means pred < true]")
        print(by_error.round(4).to_string(index=False))
        print("\n[Gate by ASIN true-sum bucket]")
        print(by_bucket.round(4).to_string(index=False))
        print("\n[Gate probes]")
        print(probe.round(4).to_string(index=False))
        print("\nInterpretation:")
        print("- pos_gate rising with active/high true_sum means positive graph is used as helpful borrowing.")
        print("- comp_gate rising in under/high-error groups may indicate competitive message is suppressing too much.")
        print("- flat gates across horizon/active buckets means graph is not yet distinguishing useful vs harmful relations.")

    return {"by_horizon": by_h, "by_active": by_active, "by_error": by_error, "by_true_sum_bucket": by_bucket, "probe": probe}

# ============================================================
# USAGE: V14 dynamic rank + horizon + GL/category shrinkage calibration; pass MU hats to demand
# ============================================================
# Run this file in Jupyter:
# %run -i exposure_model_only_nb_mu_hats_v2_DYNAMIC_RANK_HORIZON_CATCALIB_v13.py
#
# exposure_result = run_exposure_v2_final_scot_5000(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     history=13,
#     horizon=20,
#     epochs=30,
#     patience=6,
#     batch_size=128,
#
#     # Encoder / graph
#     use_encoder_self_attn=True,
#     use_graphsage=True,       # In this file this enables DualRelationalGAT, not mean GraphSAGE.
#     use_graph_gate=True,      # Learn per-horizon positive/competitive graph usage.
#     use_rank_correction=True,
#     rank_correction_scale=0.03,
#     rank_horizon_min_scale=0.40,
#     use_horizon_calibration=True,
#     horizon_calibration_scale=0.03,
#     use_gl_calibration=True,
#     gl_calibration_scale=0.025,
#     use_category_calibration=True,
#     category_calibration_scale=0.015,
#     category_shrinkage_k=100.0,
#     use_group_calibration=False,     # legacy broad MLP group calibration; keep off by default
#     group_calibration_scale=0.0,
#     neighbor_k=10,
#     graph_dim=16,
#     graph_message_scale=0.04,
#     use_graph_encoder_fusion=True,
#     graph_encoder_scale=0.05,
#
#     # First recommended run: keep graph as embedding only.
#     # Turn this on only after the embedding-only ablation improves validation/demand.
#     use_graph_head=False,
#     graph_head_scale=0.03,
#
#     # Graph feature weights used to build ASIN profiles / edges
#     graph_zero_weight=0.03,
#     graph_level_peak_weight=2.2,
#     graph_transition_weight=1.0,
#     graph_static_weight=1.0,
#     graph_brand_weight=0.3,
# )
#
# forecast_df = exposure_result["forecast_df"]
# exposure_hat_for_demand = exposure_result["exposure_hat_for_demand_mu"].copy()
# summarize_exposure_hat_for_demand(
#     exposure_hat_for_demand,
#     title="LEARNED EDGE EARLYFUSION V15 MU HAT TO PASS INTO DEMAND",
# )
#
# graph_gate_diagnostics = diagnose_graph_gates(forecast_df, target="instock", verbose=True)
#
# Then run the demand model in a separate cell/file and pass only exposure_hat_for_demand.
