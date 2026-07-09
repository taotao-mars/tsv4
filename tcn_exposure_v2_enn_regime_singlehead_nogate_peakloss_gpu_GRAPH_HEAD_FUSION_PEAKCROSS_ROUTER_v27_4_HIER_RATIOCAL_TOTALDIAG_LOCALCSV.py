# ============================================================
# TCN Exposure Model V2 - v27.8 decoder zero-attention active-protected
# v27.3 Hierarchical exposure update + category_code static features:
#   - start from v27.1 RankGate
#   - predict total_dph first
#   - predict buy_box/in_stock as learned ratios of total_dph
#   - enforce 0 <= buy_box/in_stock <= total and total≈0 => children≈0
#
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
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


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

def prepare_data_from_sample(
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
    print(f"Sampled ASINs: {len(sample_asins)} | Rows: {len(out)}")
    return out


# 向后兼容：保留旧函数名
def prepare_data_from_sample_scot_intersection(
    data_raw1, scot_df=None, n_asins=5000, seed=42,
):
    return prepare_data_from_sample(data_raw1, scot_df, n_asins, seed)


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


def load_exposure_data(data_raw, dph_cap_q=0.995):
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

    # ── 新增动态特征 ──────────────────────────────────────────
    # ind_promotion：动态binary，99.1% ASIN有变化，进active_head
    if "ind_promotion" in df.columns:
        df["ind_promotion"] = _safe_numeric(df["ind_promotion"]).clip(0, 1)
    else:
        df["ind_promotion"] = 0.0

    # ind_prime_week：动态binary，3.7%是PrimeDay周，进active_head
    if "ind_prime_week" in df.columns:
        df["ind_prime_week"] = _safe_numeric(df["ind_prime_week"]).clip(0, 1)
    else:
        df["ind_prime_week"] = 0.0

    # ── KNOWN FUTURE PROMO COVARIATES ─────────────────────────────
    # Business assumption: future promotion schedule / rate is known at forecast origin.
    # These columns are therefore allowed in future_context for horizon t+h.
    df["known_promo_index"] = df["ind_promotion"].astype(float).clip(0, 1)

    if "promotion_ratio" in df.columns:
        df["known_promo_rate"] = _safe_numeric(df["promotion_ratio"]).clip(lower=0.0)
    else:
        df["known_promo_rate"] = 0.0

    if "promotion_amount" in df.columns:
        df["known_promo_amount_log"] = np.log1p(_safe_numeric(df["promotion_amount"]).clip(lower=0.0))
    else:
        df["known_promo_amount_log"] = 0.0

    if "promotion_pricing_amount" in df.columns:
        df["known_promo_price_amount_log"] = np.log1p(_safe_numeric(df["promotion_pricing_amount"]).clip(lower=0.0))
    else:
        df["known_promo_price_amount_log"] = 0.0

    if "promotion_type" in df.columns:
        _ptype = df["promotion_type"].astype(str).fillna("NO_PROMO")
        _codes, _uniques = pd.factorize(_ptype)
        df["known_promo_type_code"] = (_codes.astype(float) / max(len(_uniques) - 1, 1))
        df.loc[df["known_promo_index"] <= 0.5, "known_promo_type_code"] = 0.0
    else:
        df["known_promo_type_code"] = 0.0

    if "pricing_type" in df.columns:
        _prtype = df["pricing_type"].astype(str).fillna("NO_PRICE_TYPE")
        _pcodes, _puniques = pd.factorize(_prtype)
        df["known_pricing_type_code"] = (_pcodes.astype(float) / max(len(_puniques) - 1, 1))
    else:
        df["known_pricing_type_code"] = 0.0

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
        ["ind_promotion", "ind_prime_week",
         "known_promo_index", "known_promo_rate",
         "known_promo_amount_log", "known_promo_price_amount_log",
         "known_promo_type_code", "known_pricing_type_code"]
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
    if "category_code" in df.columns:
        n_cat = df["category_code"].astype(str).nunique()
        unk_rate = df.get("stock_static__category_code__is_unknown", pd.Series(0, index=df.index)).mean()
        print(f"Category code enabled: n_category={n_cat} | unknown_rate={unk_rate:.4f}")
    return data, len(context_cols), context_cols


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


class TCNDecoderWithCrossAttn(nn.Module):
    """
    TCN Decoder + Cross-Attention + hierarchical exposure head.

    Why this version:
        Recent two-head runs showed unstable compensation:
            p_active too high + mag too high + gamma stuck at lower bound.
        This version removes the final p_active^gamma * magnitude gate.

    Final forecast path:
        encoder + future_context + cross-attention
            -> total_head predicts log1p(total_dph)
            -> ratio_head predicts buy_box/total and in_stock/total
            -> buy_box_dph = total_dph * learned_buy_ratio
            -> in_stock_dph = total_dph * learned_instock_ratio
        This hard-enforces:
            0 <= buy_box_dph <= total_dph
            0 <= in_stock_dph <= total_dph
            total_dph≈0 => buy_box_dph≈0 and in_stock_dph≈0

    Auxiliary active head:
        Still outputs p_active for diagnostics and a small auxiliary BCE loss,
        but p_active does NOT enter the final exposure prediction.
    """
    def __init__(self, d_model, context_dim, horizon=20,
                 hidden=96, n_heads=4, dropout=0.10,
                 anchor_indices=None,
                 active_feat_indices=None,
                 mag_feat_indices=None,
                 graph_feat_indices=None,
                 active_feat_dim=0,
                 mag_feat_dim=0,
                 graph_feat_dim=0,
                 graph_fusion_scale=0.20,
                 peak_feat_indices=None,
                 peak_feat_dim=0,
                 peak_delta_scale=0.35,
                 rank_gate_scale=0.06,
                 router_delta_scale=0.18,
                 router_num_experts=4,
                 ratio_residual_scale=0.50,
                 zero_protect_enabled=True,
                 zero_protect_threshold=0.35,
                 zero_protect_temperature=0.10,
                 zero_protect_min_gate=0.01,
                 use_decoder_zero_attn=True,
                 decoder_zero_attn_scale=0.35,
                 decoder_zero_attn_min_factor=0.60,
                 use_peak_cross_attn=True,
                 peak_cross_attn_scale=0.05,
                 use_enn=True,
                 z_dim=8,
                 residual_scale=2.0,
                 gate_temperature=1.0):
        super().__init__()
        self.horizon = horizon
        self.anchor_indices = anchor_indices
        self.active_feat_indices = active_feat_indices
        self.mag_feat_indices = mag_feat_indices
        self.graph_feat_indices = graph_feat_indices or []
        self.graph_feat_dim = int(graph_feat_dim or 0)
        self.graph_fusion_scale = float(graph_fusion_scale)
        self.peak_feat_indices = peak_feat_indices or []
        self.peak_feat_dim = int(peak_feat_dim or 0)
        self.peak_delta_scale = float(peak_delta_scale)
        self.rank_gate_scale = float(rank_gate_scale)
        self.router_delta_scale = float(router_delta_scale)
        self.router_num_experts = int(router_num_experts)
        self.ratio_residual_scale = float(ratio_residual_scale)
        self.zero_protect_enabled = bool(zero_protect_enabled)
        self.zero_protect_threshold = float(zero_protect_threshold)
        self.zero_protect_temperature = float(zero_protect_temperature)
        self.zero_protect_min_gate = float(zero_protect_min_gate)
        self.use_decoder_zero_attn = bool(use_decoder_zero_attn)
        self.decoder_zero_attn_scale = float(decoder_zero_attn_scale)
        self.decoder_zero_attn_min_factor = float(decoder_zero_attn_min_factor)
        self.use_peak_cross_attn = bool(use_peak_cross_attn)
        self.peak_cross_attn_scale = float(peak_cross_attn_scale)
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
        else:
            self.z_proj = None

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

        # v25: SPADE-style decoder-side peak cross-attention residual.
        # The standard decoder cross-attn lets every horizon query history. This extra branch
        # builds a peak-focused query from the horizon decoder state plus known future promo/
        # holiday/event and graph context, then attends over encoder states. It is zero-start
        # and low-scale, so the model initially behaves like v24 and only learns peak-specific
        # history matching if useful.
        peak_q_in_dim = d_model + max(peak_feat_dim, 0) + max(graph_feat_dim, 0) + (d_model if self.use_enn else 0)
        if self.use_peak_cross_attn:
            self.peak_query_proj = nn.Sequential(
                nn.Linear(peak_q_in_dim, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            self.peak_cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.peak_cross_gate = nn.Sequential(
                nn.Linear(peak_q_in_dim, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            self.peak_cross_norm = nn.LayerNorm(d_model)
            # Zero-start residual: v25 begins as v24.
            nn.init.zeros_(self.peak_cross_attn.out_proj.weight)
            nn.init.zeros_(self.peak_cross_attn.out_proj.bias)
            nn.init.zeros_(self.peak_cross_gate[-1].weight)
            nn.init.constant_(self.peak_cross_gate[-1].bias, -2.0)
        else:
            self.peak_query_proj = None
            self.peak_cross_attn = None
            self.peak_cross_gate = None
            self.peak_cross_norm = None

        # Graph-head fusion: graph features are ASIN/origin/horizon context, not raw TCN timesteps.
        # We project package-aware peer context to d_model, fuse with the cross-attended encoder state,
        # and also expose the graph embedding directly to final heads together with z.
        if self.graph_feat_dim > 0:
            self.graph_proj = nn.Sequential(
                nn.Linear(self.graph_feat_dim, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            self.graph_norm = nn.LayerNorm(d_model)
        else:
            self.graph_proj = None
            self.graph_norm = None

        # Shared head input extras. Must be defined before decoder-side zero attention.
        z_extra = d_model if self.use_enn else 0
        graph_extra = d_model if self.graph_feat_dim > 0 else 0

        # v27.8b decoder-side zero attention / zero context suppressor.
        # This branch explicitly retrieves historical zero-regime context from encoder outputs.
        # It is active-protected: if p_active / peak evidence is strong, suppression is near zero.
        # It is also zero-start / conservative, so the model begins close to v27.7 and learns only if useful.
        zero_ctx_extra = 3 * d_model
        zero_in_dim = d_model + zero_ctx_extra + z_extra + graph_extra + max(active_feat_dim, 0) + max(peak_feat_dim, 0)
        if self.use_decoder_zero_attn:
            self.decoder_zero_suppress_head = nn.Sequential(
                nn.Linear(zero_in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 3),
            )
            nn.init.zeros_(self.decoder_zero_suppress_head[-1].weight)
            nn.init.constant_(self.decoder_zero_suppress_head[-1].bias, -2.0)
        else:
            self.decoder_zero_suppress_head = None

        # Auxiliary occurrence head. With ENN, z controls the 20-week active/zero regime.
        active_in = d_model + z_extra + graph_extra + max(active_feat_dim, 0)
        self.active_head = nn.Sequential(
            nn.Linear(active_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

        # v27.3 hierarchical exposure head.
        # total_head predicts the total exposure log residual.
        # ratio_head predicts buy_box/total and in_stock/total as dynamic learned ratios.
        # IMPORTANT: p_active is auxiliary only and does NOT gate final predictions.
        direct_in = d_model + z_extra + graph_extra + max(active_feat_dim, 0) + max(mag_feat_dim, 0)
        self.total_head = nn.Sequential(
            nn.Linear(direct_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Tanh(),
        )
        self.ratio_head = nn.Sequential(
            nn.Linear(direct_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        # Ratio head starts as historical-ratio anchored and learns deviations only if useful.
        nn.init.zeros_(self.ratio_head[-1].weight)
        nn.init.zeros_(self.ratio_head[-1].bias)

        # SPADE-style decoder-side peak residual branch.
        # This branch is horizon-specific and driven by known future promo/holiday/event signals.
        # It does not replace the sparse history attention; it only adds a gated positive residual
        # to the log-magnitude forecast when future peak context supports it.
        peak_in = d_model + z_extra + graph_extra + max(active_feat_dim, 0) + max(mag_feat_dim, 0) + max(peak_feat_dim, 0)
        self.peak_gate_head = nn.Sequential(
            nn.Linear(peak_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        self.peak_delta_head = nn.Sequential(
            nn.Linear(peak_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )

        # v27.1: learned rank/promo gate correction.
        # This does NOT directly set exposure level. It only makes a small, zero-start
        # correction to the peak gate, so promo-adjusted rank can help open/close peak lift
        # when it explains future exposure lift.
        self.rank_gate_head = nn.Sequential(
            nn.Linear(peak_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )

        # v24: magnitude/sparsity soft router.
        # v25: decoder-side SPADE-style peak cross-attention residual.
        # This is a small residual mixture over regimes, not a replacement of the normal path.
        # Experts are interpreted diagnostically as: sparse, normal, peak, high_mag.
        # The router sees the same representation available to the final head, including
        # history encoder state, graph embedding, known future promo/event context, and z.
        self.router_gate_head = nn.Sequential(
            nn.Linear(peak_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.router_num_experts),
        )
        self.router_delta_head = nn.Sequential(
            nn.Linear(peak_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, self.router_num_experts * 3),
        )

        # Zero-start residual: v24 begins as v23 and learns routing correction only if useful.
        for _m in [self.peak_gate_head[-1], self.peak_delta_head[-1], self.rank_gate_head[-1], self.router_delta_head[-1]]:
            nn.init.zeros_(_m.weight)
            nn.init.zeros_(_m.bias)

        # Mild prior toward the normal expert at initialization; delta is zero-start, so predictions
        # are unchanged at step 0, but diagnostics are interpretable from the beginning.
        nn.init.zeros_(self.router_gate_head[-1].weight)
        nn.init.zeros_(self.router_gate_head[-1].bias)
        if self.router_num_experts >= 2:
            with torch.no_grad():
                self.router_gate_head[-1].bias[1] = 1.0

    def forward(self, enc_out, future_context, return_aux=False, z=None, x_raw=None):
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

        graph_emb = None
        graph_feats = None
        if self.graph_proj is not None and self.graph_feat_indices:
            graph_feats = future_context[:, :, self.graph_feat_indices]
            graph_emb = self.graph_proj(graph_feats)
            z_out = self.graph_norm(z_out + self.graph_fusion_scale * graph_emb)

        peak_feats = None
        if self.peak_feat_indices and len(self.peak_feat_indices) > 0:
            peak_feats = future_context[:, :, self.peak_feat_indices]

        # v26: one latent z per ASIN-window is created BEFORE peak cross-attention.
        # This makes the decoder-side peak query scenario-aware: different z samples
        # may retrieve different analogous historical exposure states.
        z_emb = None
        if self.use_enn:
            if z is None:
                z = torch.randn(B, self.z_dim, device=future_context.device, dtype=future_context.dtype)
            z_emb = self.z_proj(z)                         # [B,D]
            z_rep = z_emb[:, None, :].expand(B, H, -1)      # [B,H,D]
        else:
            z = None
            z_rep = None

        # v27.8: build decoder-side historical zero context from raw history.
        # x_raw indices follow ExposureAwareEncoderSelfAttention: 2=total, 3=buy_box, 4=in_stock log1p DPH.
        # zero_ctx_flat is a channel-wise attention pooling of encoder states over historical zero weeks.
        zero_ctx_flat = None
        hist_zero_rate_h = None
        if self.use_decoder_zero_attn and x_raw is not None and x_raw.shape[-1] >= 5:
            hist_logs = x_raw[:, :, [2, 3, 4]]
            hist_zero = (hist_logs <= 1e-6).float()              # [B,T,3]
            hist_zero_rate = hist_zero.mean(dim=1)              # [B,3]
            zero_w = hist_zero / (hist_zero.sum(dim=1, keepdim=True) + 1e-6)
            zero_ctx = torch.einsum("btc,btd->bcd", zero_w, enc_out)  # [B,3,D]
            zero_ctx_flat = zero_ctx.reshape(B, 1, -1).expand(B, H, -1)
            hist_zero_rate_h = hist_zero_rate[:, None, :].expand(B, H, -1)

        # v26 decoder-side peak cross-attention residual.
        # Q = horizon decoder state + known future peak features + graph context + ENN z scenario.
        # K,V = full encoder history. This lets future promo/holiday/peer peak context
        # explicitly retrieve analogous historical active/peak states.
        peak_cross_out = None
        peak_cross_gate = None
        peak_cross_attn_w = None
        if self.use_peak_cross_attn and self.peak_cross_attn is not None:
            peak_q_parts = [q]
            if peak_feats is not None:
                peak_q_parts.append(peak_feats)
            elif self.peak_feat_dim > 0:
                peak_q_parts.append(torch.zeros(B, H, self.peak_feat_dim, device=future_context.device, dtype=future_context.dtype))
            if graph_feats is not None:
                peak_q_parts.append(graph_feats)
            elif self.graph_feat_dim > 0:
                peak_q_parts.append(torch.zeros(B, H, self.graph_feat_dim, device=future_context.device, dtype=future_context.dtype))
            if z_rep is not None:
                peak_q_parts.append(z_rep)
            elif self.use_enn:
                peak_q_parts.append(torch.zeros(B, H, self.z_proj[-1].out_features, device=future_context.device, dtype=future_context.dtype))
            peak_q_in = torch.cat(peak_q_parts, dim=-1)
            peak_q = self.peak_query_proj(peak_q_in)
            peak_cross_out, peak_cross_attn_w = self.peak_cross_attn(
                peak_q, enc_out, enc_out,
                need_weights=return_aux,
            )
            peak_cross_gate = torch.sigmoid(self.peak_cross_gate(peak_q_in))
            z_out = self.peak_cross_norm(
                z_out + self.peak_cross_attn_scale * peak_cross_gate * peak_cross_out
            )


        active_parts = [z_out]
        if z_rep is not None:
            active_parts.append(z_rep)
        if graph_emb is not None:
            active_parts.append(graph_emb)

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
        if graph_emb is not None:
            direct_parts.append(graph_emb)
        if active_feats is not None:
            direct_parts.append(active_feats)

        mag_feats = None
        if self.mag_feat_indices and len(self.mag_feat_indices) > 0:
            mag_feats = future_context[:, :, self.mag_feat_indices]
            direct_parts.append(mag_feats)

        # peak_feats was already built above for decoder-side peak cross-attention.

        direct_in = torch.cat(direct_parts, dim=-1)
        total_residual = self.total_head(direct_in)  # [B,H,1], [-1, 1]
        ratio_residual = self.ratio_head(direct_in)  # [B,H,2], logits residual around historical ratio anchor
        residual = torch.cat([total_residual, torch.tanh(ratio_residual)], dim=-1)

        # Build peak branch input: normal/graph/z representation plus explicit future peak context.
        peak_parts = [z_out]
        if z_rep is not None:
            peak_parts.append(z_rep)
        if graph_emb is not None:
            peak_parts.append(graph_emb)
        if active_feats is not None:
            peak_parts.append(active_feats)
        if mag_feats is not None:
            peak_parts.append(mag_feats)
        if peak_feats is not None:
            peak_parts.append(peak_feats)
        peak_in = torch.cat(peak_parts, dim=-1)
        peak_gate_base = torch.sigmoid(self.peak_gate_head(peak_in))

        # v27.1 rank gate: zero-start small correction around the base peak gate.
        # rank_gate_delta is in [-rank_gate_scale, +rank_gate_scale]. At initialization
        # it is exactly 0, so v27.1 starts as v27 and learns only if useful.
        rank_gate_logit = self.rank_gate_head(peak_in)
        rank_gate_delta = torch.tanh(rank_gate_logit) * self.rank_gate_scale
        rank_gate = torch.sigmoid(rank_gate_logit)
        peak_gate = torch.clamp(peak_gate_base + rank_gate_delta, 0.0, 1.0)

        peak_delta_log = F.softplus(self.peak_delta_head(peak_in)) * self.peak_delta_scale

        # Soft magnitude/sparsity router.
        # router_weights: [B,H,K], K={sparse, normal, peak, high_mag} by convention.
        router_logits = self.router_gate_head(peak_in)
        router_weights = F.softmax(router_logits, dim=-1)
        router_delta = self.router_delta_head(peak_in).view(B, H, self.router_num_experts, 3)
        router_delta = torch.tanh(router_delta) * self.router_delta_scale
        router_residual_log = torch.sum(router_weights.unsqueeze(-1) * router_delta, dim=2)

        # v27.3 hierarchical magnitude forecast.
        # Step 1: predict total_dph in log1p space using the same anchor/peak/router mechanism.
        # Step 2: predict buy_box and in_stock as learned fractions of total_dph.
        eps = 1e-6
        if self.anchor_indices is not None:
            ti, bi, ii = self.anchor_indices
            total_anchor_log = future_context[:, :, ti:ti+1]
            buy_anchor_log = future_context[:, :, bi:bi+1]
            instock_anchor_log = future_context[:, :, ii:ii+1]
        else:
            total_anchor_log = torch.zeros(B, H, 1, device=future_context.device, dtype=future_context.dtype)
            buy_anchor_log = torch.zeros(B, H, 1, device=future_context.device, dtype=future_context.dtype)
            instock_anchor_log = torch.zeros(B, H, 1, device=future_context.device, dtype=future_context.dtype)

        raw_total_log = total_anchor_log + total_residual * self.residual_scale
        raw_total_log = raw_total_log + peak_gate[:, :, 0:1] * peak_delta_log[:, :, 0:1] + router_residual_log[:, :, 0:1]
        log_total = F.softplus(raw_total_log)
        total_level = torch.expm1(log_total).clamp(min=0.0)

        # Historical ratio anchors are computed from historical log anchors.
        # If historical total is near zero, default ratio anchor is 0.5 and the total gate still forces children to zero.
        hist_total = torch.expm1(total_anchor_log).clamp(min=0.0)
        hist_buy = torch.expm1(buy_anchor_log).clamp(min=0.0)
        hist_instock = torch.expm1(instock_anchor_log).clamp(min=0.0)
        buy_anchor_ratio = torch.where(hist_total > eps, (hist_buy / (hist_total + eps)).clamp(1e-4, 1.0 - 1e-4), torch.full_like(hist_total, 0.5))
        instock_anchor_ratio = torch.where(hist_total > eps, (hist_instock / (hist_total + eps)).clamp(1e-4, 1.0 - 1e-4), torch.full_like(hist_total, 0.5))
        ratio_anchor_logit = torch.logit(torch.cat([buy_anchor_ratio, instock_anchor_ratio], dim=-1).clamp(1e-4, 1.0 - 1e-4))

        # Learned ASIN/horizon-specific ratio deviations.
        # Use small peak/router corrections for ratio logits; these remain bounded and cannot violate hierarchy.
        ratio_logits = ratio_anchor_logit + self.ratio_residual_scale * ratio_residual
        if peak_gate is not None and peak_delta_log is not None:
            ratio_logits = ratio_logits + 0.15 * torch.tanh(peak_gate[:, :, 1:3] * peak_delta_log[:, :, 1:3])
        if router_residual_log is not None:
            ratio_logits = ratio_logits + 0.15 * torch.tanh(router_residual_log[:, :, 1:3])
        ratios = torch.sigmoid(ratio_logits).clamp(0.0, 1.0)
        buy_ratio = ratios[:, :, 0:1]
        instock_ratio = ratios[:, :, 1:2]

        # v27.8: decoder-side zero-attention suppressor.
        # It uses historical zero context retrieved from encoder states, but is active-protected:
        # high p_active or peak evidence prevents zero attention from creating underforecast.
        decoder_zero_factor = torch.ones_like(p_active)
        decoder_zero_suppress = torch.zeros_like(p_active)
        if self.use_decoder_zero_attn and self.decoder_zero_suppress_head is not None and zero_ctx_flat is not None:
            zero_parts = [z_out, zero_ctx_flat]
            if z_rep is not None:
                zero_parts.append(z_rep)
            if graph_emb is not None:
                zero_parts.append(graph_emb)
            if active_feats is not None:
                zero_parts.append(active_feats)
            if peak_feats is not None:
                zero_parts.append(peak_feats)
            zero_in = torch.cat(zero_parts, dim=-1)
            zero_raw = torch.sigmoid(self.decoder_zero_suppress_head(zero_in))
            # Active evidence protects true-active horizons from being suppressed.
            active_evidence = torch.maximum(p_active, peak_gate.detach())
            if hist_zero_rate_h is None:
                hist_zero_rate_h = torch.zeros_like(p_active)
            decoder_zero_suppress = zero_raw * hist_zero_rate_h * (1.0 - active_evidence).clamp(0.0, 1.0)
            decoder_zero_factor = 1.0 - self.decoder_zero_attn_scale * decoder_zero_suppress
            decoder_zero_factor = decoder_zero_factor.clamp(min=self.decoder_zero_attn_min_factor, max=1.0)

        # v27.7/v27.8: zero-protected final gate.
        # Goal: if the auxiliary occurrence head is very confident that a channel is zero,
        # external graph/rank/peer residuals should not lift that channel into a non-zero exposure.
        # This is NOT the old multiplicative active gate for all predictions: it is almost 1
        # when p_active is moderate/high, and only suppresses clearly zero-like cases.
        if self.zero_protect_enabled:
            temp = max(self.zero_protect_temperature, 1e-6)
            zg = torch.sigmoid((p_active - self.zero_protect_threshold) / temp)
            zero_gate = self.zero_protect_min_gate + (1.0 - self.zero_protect_min_gate) * zg
        else:
            zero_gate = torch.ones_like(p_active)
        zero_gate = zero_gate * decoder_zero_factor

        # Preserve hierarchy after zero protection:
        # total' = total * gate_total; children = total' * ratio * child_gate, so children <= total'.
        total_level = total_level * zero_gate[:, :, 0:1]
        buy_level = total_level * buy_ratio * zero_gate[:, :, 1:2]
        instock_level = total_level * instock_ratio * zero_gate[:, :, 2:3]
        pred_level = torch.cat([total_level, buy_level, instock_level], dim=-1).clamp(min=0.0)
        log_hat = torch.log1p(pred_level)
        log_mag = log_hat
        mag_level = pred_level

        # NO multiplicative active gate. p_active stays auxiliary only.
        gate = torch.ones_like(p_active)

        if return_aux:
            nan_like = torch.full_like(log_hat, float("nan"))
            return {
                "log_hat": log_hat,             # final direct log1p prediction, no gate
                "active_logit": active_logit,
                "p_active": p_active,
                "log_mag": log_mag,             # ungated magnitude log1p prediction
                "mag_level": mag_level,
                "pred_level": pred_level,
                "gamma": nan_like,
                "gate": gate,
                "zero_gate": zero_gate,
                "decoder_zero_factor": decoder_zero_factor,
                "decoder_zero_suppress": decoder_zero_suppress,
                "hist_zero_rate_h": hist_zero_rate_h if hist_zero_rate_h is not None else torch.zeros_like(p_active),
                "residual": residual,
                "peak_gate": peak_gate,
                "peak_gate_base": peak_gate_base,
                "rank_gate": rank_gate,
                "rank_gate_delta": rank_gate_delta,
                "peak_delta_log": peak_delta_log,
                "router_logits": router_logits,
                "router_weights": router_weights,
                "router_delta_log": router_delta,
                "router_residual_log": router_residual_log,
                "peak_cross_gate": peak_cross_gate,
                "peak_cross_out": peak_cross_out,
                "peak_cross_attn_weights": peak_cross_attn_w,
                "z": z,
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
        "stock_static__hbt__is_head",
        "our_price_log_norm",
        "log_review_count",
        "stock_static__gl_product_group__code",
        "stock_static__gl_product_group__freq",
        "stock_static__category_code__code",
        "stock_static__category_code__freq",
        "stock_static__category_code__is_unknown",
        "stock_static__ind_amxl_hb",
        "stock_static__sort_type__norm",
        "stock_static__ind_top10_brand__code",
        "hist_demand_mean13_log",
        "hist_instock_dph_mean13_log",
    ]

    # Horizon-specific known future peak drivers.
    # These are safe only under the business assumption that future promotion schedule/rate is known.
    PEAK_FEAT_COLS = [
        "known_promo_index",
        "known_promo_rate",
        "known_promo_amount_log",
        "known_promo_price_amount_log",
        "known_promo_type_code",
        "known_pricing_type_code",
        "ind_promotion",
        "ind_prime_week",
        "is_event_window",
        "weeks_to_nearest_event",
        "abs_weeks_to_nearest_event",
        "is_pre_event",
        "is_post_event",
        "pre_event_proximity",
        "post_event_decay",
        "month_sin",
        "month_cos",
        "season_winter",
        "season_spring",
        "season_summer",
        "season_fall",
        "graph_peer_known_promo_nextH_rate",
        "graph_peer_known_promo_long13_20_rate",
        "graph_peer_known_promo_rate_max",
        "graph_peer_known_promo_amount_log_max",
    ]

    def __init__(self, input_dim, context_dim,
                 d_model=64, horizon=20, n_heads=4, dropout=0.10,
                 context_cols=None, use_encoder_self_attn=True,
                 use_enn=True, z_dim=8, residual_scale=2.0,
                 ratio_residual_scale=0.50,
                 zero_protect_enabled=True,
                 zero_protect_threshold=0.35,
                 zero_protect_temperature=0.10,
                 zero_protect_min_gate=0.01,
                 use_decoder_zero_attn=True,
                 decoder_zero_attn_scale=0.35,
                 decoder_zero_attn_min_factor=0.60,
                 gate_temperature=1.0):
        super().__init__()
        self.use_enn = use_enn
        self.z_dim = int(z_dim)
        print(f"Exposure ENN regime enabled: {use_enn} | z_dim={z_dim}")

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

        # graph head/fusion feature indices
        graph_feat_indices = []
        if context_cols:
            for i, c in enumerate(context_cols):
                if c.startswith("graph_peer_") or c in {"graph_same_hbt_peer_rate", "graph_top10_peer_rate"}:
                    graph_feat_indices.append(i)

        # decoder-side peak residual feature indices
        peak_feat_indices = []
        for c in self.PEAK_FEAT_COLS:
            if c in col_idx and col_idx[c] not in peak_feat_indices:
                peak_feat_indices.append(col_idx[c])
        if context_cols:
            for i, c in enumerate(context_cols):
                if (c.startswith("holiday_indicator_") or
                    c.startswith("distance_") or
                    c.startswith("event_")):
                    if i not in peak_feat_indices:
                        peak_feat_indices.append(i)

        print(f"Active head feat dim: {len(active_feat_indices)}")
        print(f"Mag head feat dim:    {len(mag_feat_indices)}")
        print(f"Graph head/fusion feat dim: {len(graph_feat_indices)}")
        print(f"Peak residual feat dim: {len(peak_feat_indices)}")

        self.decoder = TCNDecoderWithCrossAttn(
            d_model=d_model,
            context_dim=context_dim,
            horizon=horizon,
            hidden=max(96, d_model * 2),
            n_heads=n_heads,
            dropout=dropout,
            anchor_indices=anchor_indices,
            active_feat_indices=active_feat_indices,
            mag_feat_indices=mag_feat_indices,
            graph_feat_indices=graph_feat_indices,
            active_feat_dim=len(active_feat_indices),
            mag_feat_dim=len(mag_feat_indices),
            graph_feat_dim=len(graph_feat_indices),
            graph_fusion_scale=0.20,
            peak_feat_indices=peak_feat_indices,
            peak_feat_dim=len(peak_feat_indices),
            peak_delta_scale=0.35,
            rank_gate_scale=0.06,
            use_enn=use_enn,
            z_dim=z_dim,
            residual_scale=residual_scale,
            ratio_residual_scale=ratio_residual_scale,
            zero_protect_enabled=zero_protect_enabled,
            zero_protect_threshold=zero_protect_threshold,
            zero_protect_temperature=zero_protect_temperature,
            zero_protect_min_gate=zero_protect_min_gate,
            use_decoder_zero_attn=use_decoder_zero_attn,
            decoder_zero_attn_scale=decoder_zero_attn_scale,
            decoder_zero_attn_min_factor=decoder_zero_attn_min_factor,
            gate_temperature=gate_temperature,
        )

    def forward(self, x, future_context, return_aux=False, z=None):
        enc_out = self.encoder(x)
        return self.decoder(enc_out, future_context, return_aux=return_aux, z=z, x_raw=x)


# ============================================================
# Loss：Hurdle BCE + Magnitude Huber + Mean Penalty
# ============================================================

def exposure_hurdle_loss(
    log_hat,        # [B,H,3] direct log1p prediction
    true_total,     # [B,H]
    true_buy,       # [B,H]
    true_instock,   # [B,H]
    active_logit,   # [B,H,3] auxiliary occurrence logits only
    log_mag=None,   # unused; kept for interface compatibility
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
    horizon_weight_alpha=0.25,
    high_weight_alpha=0.35,
    # ENN/path-regime terms
    path_zero_weight=0.08,
    zero_fp_weight=0.08,
    active_count_weight=0.05,
    path_sum_weight=0.05,
    # Peak/path-high regime terms. These prevent zero losses from making the model too conservative.
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    zero_fp_threshold=50.0,
    zero_fp_temperature=20.0,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
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
    horizon_w = 1.0 + horizon_weight_alpha * (h / max(float(H), 1.0))
    sample_w = high_w * horizon_w

    # 1) Main direct log loss.
    log_err = F.huber_loss(log_hat, target_log, delta=1.0, reduction="none")
    direct_loss = (log_err * sample_w * tw).mean()

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

    # 4b) v27.4 ratio calibration for the hierarchical child channels.
    # When total is accurate but buy_box/in_stock are systematically high, the error
    # comes from the learned child ratios. These losses keep the dynamic ratios anchored
    # without weakening the total head or zero/active discrimination.
    eps = 1e-6
    pred_total = pred_level[..., 0]
    pred_buy = pred_level[..., 1]
    pred_instock = pred_level[..., 2]
    true_total_y = true[..., 0]
    true_buy_y = true[..., 1]
    true_instock_y_for_ratio = true[..., 2]
    total_pos_mask = (true_total_y > 0).float()

    pred_buy_ratio = (pred_buy / (pred_total + eps)).clamp(0.0, 1.0)
    pred_instock_ratio = (pred_instock / (pred_total + eps)).clamp(0.0, 1.0)
    true_buy_ratio = (true_buy_y / (true_total_y + eps)).clamp(0.0, 1.0)
    true_instock_ratio = (true_instock_y_for_ratio / (true_total_y + eps)).clamp(0.0, 1.0)

    ratio_denom = total_pos_mask.sum().clamp_min(1.0)
    pred_buy_ratio_mean = (pred_buy_ratio * total_pos_mask).sum() / ratio_denom
    true_buy_ratio_mean = (true_buy_ratio * total_pos_mask).sum() / ratio_denom
    pred_instock_ratio_mean = (pred_instock_ratio * total_pos_mask).sum() / ratio_denom
    true_instock_ratio_mean = (true_instock_ratio * total_pos_mask).sum() / ratio_denom
    ratio_mean_loss = torch.abs(pred_buy_ratio_mean - true_buy_ratio_mean) + torch.abs(pred_instock_ratio_mean - true_instock_ratio_mean)

    def _active_mean_log_gap(pred_y, true_y):
        m = (true_y > 0).float()
        denom_m = m.sum().clamp_min(1.0)
        pred_mean_active = (pred_y * m).sum() / denom_m
        true_mean_active = (true_y * m).sum() / denom_m
        return torch.abs(torch.log1p(pred_mean_active.clamp_min(0.0)) - torch.log1p(true_mean_active.clamp_min(0.0)))

    active_mean_loss = (
        _active_mean_log_gap(pred_buy, true_buy_y)
        + _active_mean_log_gap(pred_instock, true_instock_y_for_ratio)
    )

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
        mag_weight * direct_loss
        + mean_weight * mean_loss
        + bce_weight * bce_loss
        + active_calib_weight * active_calib_loss
        + zero_loss
        + path_zero_weight * path_zero_loss
        + zero_fp_weight * zero_fp_loss
        + active_count_weight * active_count_loss
        + path_sum_weight * path_sum_loss
        + peak_weight * peak_loss
        + topk_peak_weight * topk_peak_loss
        + peak_under_weight * peak_under_loss
        + ratio_mean_weight * ratio_mean_loss
        + active_mean_weight * active_mean_loss
    )

# ============================================================
# 训练
# ============================================================

def train_exposure_model_v2(
    model, tr_ld, va_ld,
    epochs=60, lr=1e-3, patience=8,
    w_total=0.30, w_buy=0.60, w_instock=1.00,
    bce_weight=0.15, mag_weight=1.00, mean_weight=0.35,
    active_calib_weight=0.05,
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.25, high_weight_alpha=0.35,
    path_zero_weight=0.08,
    zero_fp_weight=0.08,
    active_count_weight=0.05,
    path_sum_weight=0.05,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    zero_fp_threshold=50.0,
    zero_fp_temperature=20.0,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
    device=None,
):
    device = get_device(device)
    model = model.to(device)
    print(f"Training on device: {device}")
    print(f"v27.4 ratio calibration | ratio_residual_scale={getattr(model.decoder, 'ratio_residual_scale', float('nan')):.3f} | ratio_mean_weight={ratio_mean_weight} | active_mean_weight={active_mean_weight}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_val, best_sd, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0

        for b in tr_ld:
            b = batch_to_device(b, device)
            aux = model(b["x"], b["future_context"], return_aux=True)
            loss = exposure_hurdle_loss(
                log_hat=aux["log_hat"],
                true_total=b["future_total_dph"],
                true_buy=b["future_buy_box_dph"],
                true_instock=b["future_instock_dph"],
                active_logit=aux["active_logit"],
                log_mag=aux["log_mag"],
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
                ratio_mean_weight=ratio_mean_weight,
                active_mean_weight=active_mean_weight,
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
                aux = model(b["x"], b["future_context"], return_aux=True)
                loss = exposure_hurdle_loss(
                    log_hat=aux["log_hat"],
                    true_total=b["future_total_dph"],
                    true_buy=b["future_buy_box_dph"],
                    true_instock=b["future_instock_dph"],
                    active_logit=aux["active_logit"],
                    log_mag=aux["log_mag"],
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

def predict_exposure_v2(model, va_ld, apply_funnel_constraint=True, device=None, mc_samples=20, mc_reduce="median", context_cols=None):
    device = get_device(device)
    model = model.to(device)
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            b = batch_to_device(b, device)
            # MC inference over ENN z. Median is more robust than mean for exposure hats,
            # because mean can be pulled up by high-regime samples.
            preds, pacts, gates, zero_gates, rank_gates, rank_gate_deltas = [], [], [], [], [], []
            last_aux = None
            K = max(int(mc_samples), 1)
            for _ in range(K):
                aux = model(b["x"], b["future_context"], return_aux=True)
                last_aux = aux
                preds.append(torch.expm1(aux["log_hat"]).clamp(min=0.0))
                pacts.append(aux["p_active"])
                gates.append(aux.get("gate", torch.full_like(aux["p_active"], float("nan"))))
                zero_gates.append(aux.get("zero_gate", torch.full_like(aux["p_active"], float("nan"))))
                rank_gates.append(aux.get("rank_gate", torch.full_like(aux["p_active"], float("nan"))))
                rank_gate_deltas.append(aux.get("rank_gate_delta", torch.full_like(aux["p_active"], float("nan"))))

            pred_stack = torch.stack(preds, dim=0)   # [K,B,H,3]
            pact_stack = torch.stack(pacts, dim=0)
            gate_stack = torch.stack(gates, dim=0)
            zero_gate_stack = torch.stack(zero_gates, dim=0)
            rank_gate_stack = torch.stack(rank_gates, dim=0)
            rank_gate_delta_stack = torch.stack(rank_gate_deltas, dim=0)

            # Keep MC quantiles so we can inspect whether exposure hats also need an upper shift.
            pred_q50_t = torch.quantile(pred_stack, 0.50, dim=0)
            pred_q70_t = torch.quantile(pred_stack, 0.70, dim=0)
            pred_q90_t = torch.quantile(pred_stack, 0.90, dim=0)

            if mc_reduce == "mean":
                pred_t = pred_stack.mean(dim=0)
                pact_t = pact_stack.mean(dim=0)
                gate_t = gate_stack.mean(dim=0)
                zero_gate_t = zero_gate_stack.mean(dim=0)
                rank_gate_t = rank_gate_stack.mean(dim=0)
                rank_gate_delta_t = rank_gate_delta_stack.mean(dim=0)
            else:
                pred_t = pred_stack.median(dim=0).values
                pact_t = pact_stack.mean(dim=0)
                gate_t = gate_stack.median(dim=0).values
                zero_gate_t = zero_gate_stack.median(dim=0).values
                rank_gate_t = rank_gate_stack.mean(dim=0)
                rank_gate_delta_t = rank_gate_delta_stack.mean(dim=0)

            pred = pred_t.cpu().numpy()
            pred_q50 = pred_q50_t.cpu().numpy()
            pred_q70 = pred_q70_t.cpu().numpy()
            pred_q90 = pred_q90_t.cpu().numpy()
            pact = pact_t.cpu().numpy()
            gamma_np = last_aux.get("gamma", torch.full_like(last_aux["p_active"], float("nan"))).cpu().numpy()
            gate_np = gate_t.cpu().numpy()
            zero_gate_np = zero_gate_t.cpu().numpy()
            rank_gate_np = rank_gate_t.cpu().numpy()
            rank_gate_delta_np = rank_gate_delta_t.cpu().numpy()

            if apply_funnel_constraint:
                for arr in [pred, pred_q50, pred_q70, pred_q90]:
                    arr[:, :, 1] = np.minimum(arr[:, :, 1], arr[:, :, 0])
                    arr[:, :, 2] = np.minimum(arr[:, :, 2], arr[:, :, 1])

            B, H = b["future_instock_dph"].shape
            ctx_np = b["future_context"].detach().cpu().numpy() if "future_context" in b else None
            ctx_idx = {c: j for j, c in enumerate(context_cols or [])}
            rank_diag_cols = [
                "graph_peer_rank_prior",
                "graph_own_known_promo_h",
                "graph_peer_known_promo_h",
                "graph_own_vs_peer_promo_delta_h",
                "graph_promo_adjusted_rank_prior_h",
                "hist_instock_dph_mean13_log",
            ]
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
                        # MC ENN quantiles for p-shift diagnostics. Demand reader still uses
                        # pred_total_dph / pred_buy_box_dph / pred_instock_dph by default.
                        "pred_total_dph_q50":    pred_q50[i, h, 0],
                        "pred_buy_box_dph_q50":  pred_q50[i, h, 1],
                        "pred_instock_dph_q50":  pred_q50[i, h, 2],
                        "pred_total_dph_q70":    pred_q70[i, h, 0],
                        "pred_buy_box_dph_q70":  pred_q70[i, h, 1],
                        "pred_instock_dph_q70":  pred_q70[i, h, 2],
                        "pred_total_dph_q90":    pred_q90[i, h, 0],
                        "pred_buy_box_dph_q90":  pred_q90[i, h, 1],
                        "pred_instock_dph_q90":  pred_q90[i, h, 2],
                        "op_p50_total_dph":      pred_q70[i, h, 0],
                        "op_p50_buy_box_dph":    pred_q70[i, h, 1],
                        "op_p50_instock_dph":    pred_q70[i, h, 2],
                        "op_p70_total_dph":      pred_q90[i, h, 0],
                        "op_p70_buy_box_dph":    pred_q90[i, h, 1],
                        "op_p70_instock_dph":    pred_q90[i, h, 2],
                        "true_demand":       b["future_demand"][i, h].item(),
                        # 诊断列
                        "p_active_total":    pact[i, h, 0],
                        "p_active_buy_box":  pact[i, h, 1],
                        "p_active_instock":  pact[i, h, 2],
                        "gamma_total":       gamma_np[i, h, 0],
                        "gamma_buy_box":     gamma_np[i, h, 1],
                        "gamma_instock":     gamma_np[i, h, 2],
                        "gate_total":        gate_np[i, h, 0],
                        "gate_buy_box":      gate_np[i, h, 1],
                        "gate_instock":      gate_np[i, h, 2],
                        "zero_gate_total":   zero_gate_np[i, h, 0],
                        "zero_gate_buy_box": zero_gate_np[i, h, 1],
                        "zero_gate_instock": zero_gate_np[i, h, 2],
                        "decoder_zero_factor_total":   decoder_zero_factor_np[i, h, 0],
                        "decoder_zero_factor_buy_box": decoder_zero_factor_np[i, h, 1],
                        "decoder_zero_factor_instock": decoder_zero_factor_np[i, h, 2],
                        "decoder_zero_suppress_total":   decoder_zero_suppress_np[i, h, 0],
                        "decoder_zero_suppress_buy_box": decoder_zero_suppress_np[i, h, 1],
                        "decoder_zero_suppress_instock": decoder_zero_suppress_np[i, h, 2],
                        "rank_gate_total":        rank_gate_np[i, h, 0],
                        "rank_gate_buy_box":      rank_gate_np[i, h, 1],
                        "rank_gate_instock":      rank_gate_np[i, h, 2],
                        "rank_gate_delta_total":  rank_gate_delta_np[i, h, 0],
                        "rank_gate_delta_buy_box":rank_gate_delta_np[i, h, 1],
                        "rank_gate_delta_instock":rank_gate_delta_np[i, h, 2],
                        **({col: float(ctx_np[i, h, ctx_idx[col]])
                            for col in rank_diag_cols
                            if ctx_np is not None and col in ctx_idx}),
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



def diagnose_rank_gate_future_lift(pred_df):
    """
    v27.1 diagnostic: rank/promo gate should explain future lift, not only absolute level.
    Uses only columns already stored in forecast_df.
    """
    need = ["true_instock_dph", "pred_instock_dph", "hist_instock_dph_mean13_log"]
    missing = [c for c in need if c not in pred_df.columns]
    if missing:
        print("\nRANK-GATE FUTURE-LIFT DIAGNOSTIC skipped. Missing columns:", missing)
        return {}

    df = pred_df.copy()
    hist = np.expm1(pd.to_numeric(df["hist_instock_dph_mean13_log"], errors="coerce").fillna(0.0).values)
    y = pd.to_numeric(df["true_instock_dph"], errors="coerce").fillna(0.0).values
    p = pd.to_numeric(df["pred_instock_dph"], errors="coerce").fillna(0.0).values
    df["true_lift_log"] = np.log1p(y) - np.log1p(hist)
    df["pred_lift_log"] = np.log1p(p) - np.log1p(hist)
    df["true_lift_positive"] = (df["true_lift_log"] > 0).astype(int)

    signal_cols = [
        "graph_peer_rank_prior",
        "graph_promo_adjusted_rank_prior_h",
        "graph_own_vs_peer_promo_delta_h",
        "rank_gate_instock",
        "rank_gate_delta_instock",
    ]
    signal_cols = [c for c in signal_cols if c in df.columns]
    rows = []
    for c in signal_cols:
        svals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ok = svals.notna() & np.isfinite(df["true_lift_log"])
        if ok.sum() < 10:
            continue
        rows.append({
            "signal": c,
            "mean": float(svals[ok].mean()),
            "std": float(svals[ok].std()),
            "spearman_with_true_lift": _safe_spearman(df.loc[ok, "true_lift_log"], svals[ok]),
            "spearman_with_pred_lift": _safe_spearman(df.loc[ok, "pred_lift_log"], svals[ok]),
            "lift_positive_AUC": _auc(df.loc[ok, "true_lift_positive"].astype(int).values, svals[ok].values),
        })
    lift_summary = pd.DataFrame(rows)

    gate_bucket_df = pd.DataFrame()
    if "rank_gate_delta_instock" in df.columns:
        x = pd.to_numeric(df["rank_gate_delta_instock"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() >= 20 and x.nunique(dropna=True) > 3:
            try:
                df["rank_gate_bucket"] = pd.qcut(x, q=5, duplicates="drop")
                b_rows = []
                for bucket, g in df.dropna(subset=["rank_gate_bucket"]).groupby("rank_gate_bucket", observed=True):
                    yy = g["true_instock_dph"].values.astype(float)
                    pp = g["pred_instock_dph"].values.astype(float)
                    b_rows.append({
                        "rank_gate_bucket": str(bucket),
                        "rows": int(len(g)),
                        "gate_delta_mean": float(g["rank_gate_delta_instock"].mean()),
                        "true_lift_log_mean": float(g["true_lift_log"].mean()),
                        "pred_lift_log_mean": float(g["pred_lift_log"].mean()),
                        "true_mean": float(np.mean(yy)),
                        "pred_mean": float(np.mean(pp)),
                        "ratio": float(np.mean(pp) / (np.mean(yy) + 1e-8)),
                        "WAPE": float(_wape(yy, pp)),
                        "active_rate": float(np.mean(yy > 0)),
                    })
                gate_bucket_df = pd.DataFrame(b_rows)
            except Exception:
                gate_bucket_df = pd.DataFrame()

    print("\n" + "=" * 100)
    print("RANK-GATE FUTURE-LIFT DIAGNOSTIC: IN_STOCK_DPH")
    print("=" * 100)
    print("Goal: rank/promo signals should explain future exposure lift over hist_mean13, not only absolute exposure level.")
    if len(lift_summary):
        print(lift_summary.round(4).to_string(index=False))
    else:
        print("No usable rank/gate lift signals.")

    print("\nRANK-GATE BUCKETS BY ROW / HORIZON")
    if len(gate_bucket_df):
        print(gate_bucket_df.round(4).to_string(index=False))
    else:
        print("No rank-gate bucket table available, likely because rank_gate_delta is still near zero or constant.")

    if len(lift_summary) and "rank_gate_delta_instock" in lift_summary["signal"].values:
        rg = lift_summary[lift_summary["signal"] == "rank_gate_delta_instock"].iloc[0]
        print(f"\nRank-gate delta signal: Spearman(true_lift)={rg['spearman_with_true_lift']:.4f}, lift_AUC={rg['lift_positive_AUC']:.4f}")
        if (rg["spearman_with_true_lift"] > 0.03) or (rg["lift_positive_AUC"] > 0.55):
            print("Judgment: rank gate is learning useful lift information.")
        else:
            print("Judgment: rank gate is not clearly useful yet; keep scale small and compare downstream demand.")

    return {"lift_summary": lift_summary, "rank_gate_buckets": gate_bucket_df}


def diagnose_total_base(pred_df):
    """Detailed diagnostics for the hierarchical base channel total_dph."""
    rows = []
    for h, g in pred_df.groupby("horizon"):
        y = g["true_total_dph"].values.astype(float)
        p = g["pred_total_dph"].values.astype(float)
        rows.append({
            "horizon": int(h),
            "true_mean": float(np.mean(y)),
            "pred_mean": float(np.mean(p)),
            "ratio": float(np.mean(p) / (np.mean(y) + 1e-8)),
            "WAPE": float(_wape(y, p)),
            "underbias": float(np.maximum(y - p, 0).sum() / (np.abs(y).sum() + 1e-8)),
            "overbias": float(np.maximum(p - y, 0).sum() / (np.abs(y).sum() + 1e-8)),
            "corr": float(_corr(y, p)),
            "active_AUC": float(_auc((y > 0).astype(int), p)),
        })
    by_h_total = pd.DataFrame(rows)

    y = pred_df["true_total_dph"].values.astype(float)
    p = pred_df["pred_total_dph"].values.astype(float)
    active_mask = y > 0
    zero_mask = y <= 0
    total_summary = pd.DataFrame([{
        "true_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "ratio": float(np.mean(p) / (np.mean(y) + 1e-8)),
        "WAPE": float(_wape(y, p)),
        "corr": float(_corr(y, p)),
        "active_AUC": float(_auc((y > 0).astype(int), p)),
        "zero_rate_true": float(np.mean(zero_mask)),
        "active_true_mean": float(np.mean(y[active_mask])) if active_mask.any() else np.nan,
        "active_pred_mean": float(np.mean(p[active_mask])) if active_mask.any() else np.nan,
        "active_only_ratio": float(np.mean(p[active_mask]) / (np.mean(y[active_mask]) + 1e-8)) if active_mask.any() else np.nan,
        "active_only_WAPE": float(_wape(y[active_mask], p[active_mask])) if active_mask.any() else np.nan,
        "zero_pred_mean": float(np.mean(p[zero_mask])) if zero_mask.any() else np.nan,
        "zero_pred_p90": float(np.quantile(p[zero_mask], 0.90)) if zero_mask.any() else np.nan,
        "zero_pred_gt_0p1_rate": float(np.mean(p[zero_mask] > 0.1)) if zero_mask.any() else np.nan,
        "zero_pred_gt_1_rate": float(np.mean(p[zero_mask] > 1.0)) if zero_mask.any() else np.nan,
    }])

    print("\n" + "=" * 100)
    print("TOTAL BASE DIAGNOSTIC")
    print("=" * 100)
    print("Overall / active-only / zero-consistency for total_dph")
    print(total_summary.round(4).to_string(index=False))

    print("\nTOTAL BY HORIZON")
    print(by_h_total.round(4).to_string(index=False))

    # Child ratio diagnostic: if total is good but child channels are high, this table will show it.
    child_rows = []
    eps = 1e-8
    total_pos = pred_df["true_total_dph"].values.astype(float) > 0
    for name, true_col, pred_col in [
        ("buy_box", "true_buy_box_dph", "pred_buy_box_dph"),
        ("in_stock", "true_instock_dph", "pred_instock_dph"),
    ]:
        true_ratio = (pred_df[true_col].values.astype(float) / (pred_df["true_total_dph"].values.astype(float) + eps))[total_pos]
        pred_ratio = (pred_df[pred_col].values.astype(float) / (pred_df["pred_total_dph"].values.astype(float) + eps))[total_pos]
        child_rows.append({
            "channel": name,
            "true_ratio_mean": float(np.mean(np.clip(true_ratio, 0, 1))) if len(true_ratio) else np.nan,
            "pred_ratio_mean": float(np.mean(np.clip(pred_ratio, 0, 1))) if len(pred_ratio) else np.nan,
            "ratio_mean_gap": float(np.mean(np.clip(pred_ratio, 0, 1)) - np.mean(np.clip(true_ratio, 0, 1))) if len(true_ratio) else np.nan,
            "child_level_ratio": float(pred_df[pred_col].mean() / (pred_df[true_col].mean() + eps)),
            "active_only_ratio": float(pred_df.loc[pred_df[true_col] > 0, pred_col].mean() / (pred_df.loc[pred_df[true_col] > 0, true_col].mean() + eps)) if (pred_df[true_col] > 0).any() else np.nan,
        })
    ratio_diag = pd.DataFrame(child_rows)
    print("\nCHILD RATIO DIAGNOSTIC: buy_box/total and in_stock/total")
    print(ratio_diag.round(4).to_string(index=False))

    return {"total_summary": total_summary, "total_by_horizon": by_h_total, "child_ratio": ratio_diag}


def diagnose_zero_protection(pred_df, threshold=1.0):
    """Threshold-based zero/active diagnostic for final exposure predictions.

    This directly answers: among true-zero rows, how often did the model lift the
    channel above a practical non-zero threshold? We print both pred-threshold and
    p_active-threshold diagnostics.
    """
    print("\n" + "=" * 100)
    print("ZERO-PROTECTION DIAGNOSTIC")
    print("=" * 100)
    rows = []
    specs = [
        ("total", "true_total_dph", "pred_total_dph", "p_active_total", "zero_gate_total", "decoder_zero_factor_total", "decoder_zero_suppress_total"),
        ("buy_box", "true_buy_box_dph", "pred_buy_box_dph", "p_active_buy_box", "zero_gate_buy_box", "decoder_zero_factor_buy_box", "decoder_zero_suppress_buy_box"),
        ("in_stock", "true_instock_dph", "pred_instock_dph", "p_active_instock", "zero_gate_instock", "decoder_zero_factor_instock", "decoder_zero_suppress_instock"),
    ]
    for name, ycol, pcol, pacol, zgcol, dzfcol, dzscol in specs:
        if ycol not in pred_df.columns or pcol not in pred_df.columns:
            continue
        y = pd.to_numeric(pred_df[ycol], errors="coerce").fillna(0.0).values.astype(float)
        p = pd.to_numeric(pred_df[pcol], errors="coerce").fillna(0.0).values.astype(float)
        pa = pd.to_numeric(pred_df[pacol], errors="coerce").fillna(np.nan).values.astype(float) if pacol in pred_df.columns else np.full_like(p, np.nan)
        zg = pd.to_numeric(pred_df[zgcol], errors="coerce").fillna(np.nan).values.astype(float) if zgcol in pred_df.columns else np.full_like(p, np.nan)
        dzf = pd.to_numeric(pred_df[dzfcol], errors="coerce").fillna(np.nan).values.astype(float) if dzfcol in pred_df.columns else np.full_like(p, np.nan)
        dzs = pd.to_numeric(pred_df[dzscol], errors="coerce").fillna(np.nan).values.astype(float) if dzscol in pred_df.columns else np.full_like(p, np.nan)
        active = y > 0
        pred_active = p > threshold
        tp = int(np.sum(pred_active & active))
        fp = int(np.sum(pred_active & (~active)))
        fn = int(np.sum((~pred_active) & active))
        tn = int(np.sum((~pred_active) & (~active)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        zero_recall = tn / max(tn + fp, 1)  # true zero kept zero
        false_active_rate_on_zero = fp / max(np.sum(~active), 1)
        rows.append({
            "target": name,
            "threshold_pred_gt": threshold,
            "true_zero_rate": float(np.mean(~active)),
            "pred_active_rate": float(np.mean(pred_active)),
            "active_precision": precision,
            "active_recall": recall,
            "active_f1": f1,
            "zero_recall": zero_recall,
            "false_active_rate_on_zero": false_active_rate_on_zero,
            "zero_pred_mean": float(np.mean(p[~active])) if np.any(~active) else np.nan,
            "zero_pred_p90": float(np.quantile(p[~active], 0.90)) if np.any(~active) else np.nan,
            "active_only_ratio": float(np.mean(p[active]) / (np.mean(y[active]) + 1e-8)) if np.any(active) else np.nan,
            "p_active_zero_mean": float(np.nanmean(pa[~active])) if np.any(~active) else np.nan,
            "p_active_active_mean": float(np.nanmean(pa[active])) if np.any(active) else np.nan,
            "zero_gate_zero_mean": float(np.nanmean(zg[~active])) if np.any(~active) else np.nan,
            "zero_gate_active_mean": float(np.nanmean(zg[active])) if np.any(active) else np.nan,
            "decoder_zero_factor_zero_mean": float(np.nanmean(dzf[~active])) if np.any(~active) else np.nan,
            "decoder_zero_factor_active_mean": float(np.nanmean(dzf[active])) if np.any(active) else np.nan,
            "decoder_zero_suppress_zero_mean": float(np.nanmean(dzs[~active])) if np.any(~active) else np.nan,
            "decoder_zero_suppress_active_mean": float(np.nanmean(dzs[active])) if np.any(active) else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        print(out.round(4).to_string(index=False))
    return out

def print_exposure_diagnostics(pred_df):
    """
    Clean exposure diagnostics.

    Kept:
      1) overall model exposure metrics
      2) by-horizon in_stock_dph metrics
      3) auxiliary p_active / gate diagnostics
      4) ASIN-level 20-week sum diagnostics
      5) quick judgment + compact final summary

    Removed for the current ablation round:
      - p-shift / exposure calibration diagnostics
      - model-vs-naive baseline diagnostics
      - GL diagnostics are not called in the main run
    """
    print("\n" + "=" * 100)
    print("MODEL EXPOSURE METRICS")
    print("=" * 100)
    model_tbl = exposure_metrics(pred_df, prefix="pred")
    print(model_tbl.round(5).to_string(index=False))

    total_base_diagnostics = diagnose_total_base(pred_df)
    zero_protection_diagnostics = diagnose_zero_protection(pred_df, threshold=1.0)

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

    # ── p_active诊断 ─────────────────────────────────────────
    p_active_cols = [c for c in ["p_active_total", "p_active_buy_box", "p_active_instock"]
                     if c in pred_df.columns]
    pa_df = pd.DataFrame()
    if p_active_cols:
        print("\n" + "=" * 100)
        print("P_ACTIVE BY HORIZON")
        print("=" * 100)
        pa_rows = []
        for h, g in pred_df.groupby("horizon"):
            row = {"horizon": h}
            for c in p_active_cols:
                row[c] = g[c].mean()
            row["true_active_rate"] = (g["true_instock_dph"] > 0).mean()
            pa_rows.append(row)
        pa_df = pd.DataFrame(pa_rows)
        print(pa_df.round(4).to_string(index=False))

        if "p_active_instock" in pa_df.columns:
            pa_instock = pa_df["p_active_instock"].values
            is_monotone = all(pa_instock[i] <= pa_instock[i+1] for i in range(len(pa_instock)-1))
            print(f"\np_active_instock monotonically increasing: {is_monotone}")

    # ── gamma / gate诊断 ─────────────────────────────────────
    gamma_gate_cols = [c for c in ["gamma_instock", "gate_instock", "rank_gate_instock", "rank_gate_delta_instock"] if c in pred_df.columns]
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
            if "rank_gate_instock" in g.columns:
                row["rank_gate_instock_mean"] = g["rank_gate_instock"].mean()
            if "rank_gate_delta_instock" in g.columns:
                row["rank_gate_delta_instock_mean"] = g["rank_gate_delta_instock"].mean()
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

    rank_diagnostics = diagnose_promo_adjusted_rank(pred_df, target="in_stock")
    rank_gate_lift_diagnostics = diagnose_rank_gate_future_lift(pred_df)

    return {
        "model": model_tbl,
        "total_base": total_base_diagnostics,
        "by_horizon": by_h,
        "p_active_by_horizon": pa_df,
        "final_summary": final_summary,
        "promo_adjusted_rank": rank_diagnostics,
        "rank_gate_future_lift": rank_gate_lift_diagnostics,
    }


# ============================================================
# Encoder / Decoder diagnostics
# ============================================================

def diagnose_encoder_decoder_performance(model, va_ld, pred_df=None, max_batches=None, device=None):
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
            aux = model.decoder(enc_out, fc, return_aux=True)

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

def make_external_hat_df(pred_df):
    """
    Build one CSV-ready exposure hat dataframe for demand.

    Demand readers use the standard level columns by default:
      pred_total_dph, pred_buy_box_dph, pred_instock_dph

    Extra MC quantile / operational p-shift columns are preserved for diagnostics,
    but they do not change demand unless the demand reader is explicitly changed to use them.
    """
    key_cols = ["asin", "order_week"]
    optional_key_cols = ["horizon"] if "horizon" in pred_df.columns else []
    hat_cols = [
        "pred_total_dph", "pred_buy_box_dph", "pred_instock_dph",
        "pred_total_dph_q50", "pred_buy_box_dph_q50", "pred_instock_dph_q50",
        "pred_total_dph_q70", "pred_buy_box_dph_q70", "pred_instock_dph_q70",
        "pred_total_dph_q90", "pred_buy_box_dph_q90", "pred_instock_dph_q90",
        "op_p50_total_dph", "op_p50_buy_box_dph", "op_p50_instock_dph",
        "op_p70_total_dph", "op_p70_buy_box_dph", "op_p70_instock_dph",
    ]
    keep = key_cols + optional_key_cols + [c for c in hat_cols if c in pred_df.columns]
    out = pred_df[keep].copy()

    # Main logs used by older demand code paths.
    out["external_total_dph_hat_log"]    = np.log1p(out["pred_total_dph"].clip(lower=0.0))
    out["external_buy_box_dph_hat_log"]  = np.log1p(out["pred_buy_box_dph"].clip(lower=0.0))
    out["external_instock_dph_hat_log"]  = np.log1p(out["pred_instock_dph"].clip(lower=0.0))

    # Optional logs for MC quantiles / upper shifted hats.
    for c in [col for col in out.columns if col.startswith("pred_") or col.startswith("op_p")]:
        if c.endswith("_log"):
            continue
        out[f"{c}_log"] = np.log1p(pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0))

    return out


# ============================================================
# 主入口
# ============================================================

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
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.25,
    high_weight_alpha=0.35,
    path_zero_weight=0.08,
    zero_fp_weight=0.08,
    active_count_weight=0.05,
    path_sum_weight=0.05,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    ratio_residual_scale=0.50,
    zero_protect_enabled=True,
    zero_protect_threshold=0.35,
    zero_protect_temperature=0.10,
    zero_protect_min_gate=0.01,
    use_decoder_zero_attn=True,
    decoder_zero_attn_scale=0.35,
    decoder_zero_attn_min_factor=0.60,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
    dropout=0.20,    # 0.10→0.20，加强dropout防过拟合
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V27.8: DECODER_ZERO_ATTN_ACTIVEPROTECTED + TOTALDIAG + SAVEHAT")
    print("Preset: v27.7 zero gate + decoder-side zero-attention, active-protected")
    print("=" * 100)

    df = prepare_data_from_sample(data_raw1, scot_df, n_asins, seed)
    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    data, context_dim, context_cols = load_exposure_data(df, dph_cap_q=dph_cap_q)

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
        ratio_residual_scale=ratio_residual_scale,
        zero_protect_enabled=zero_protect_enabled,
        zero_protect_threshold=zero_protect_threshold,
        zero_protect_temperature=zero_protect_temperature,
        zero_protect_min_gate=zero_protect_min_gate,
        use_decoder_zero_attn=use_decoder_zero_attn,
        decoder_zero_attn_scale=decoder_zero_attn_scale,
        decoder_zero_attn_min_factor=decoder_zero_attn_min_factor,
        use_encoder_self_attn=use_encoder_self_attn,
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
        path_zero_weight=path_zero_weight,
        zero_fp_weight=zero_fp_weight,
        active_count_weight=active_count_weight,
        path_sum_weight=path_sum_weight,
        peak_weight=peak_weight,
        topk_peak_weight=topk_peak_weight,
        peak_under_weight=peak_under_weight,
        peak_topk=peak_topk,
        peak_quantile=peak_quantile,
        ratio_mean_weight=ratio_mean_weight,
        active_mean_weight=active_mean_weight,
    )

    pred_df = predict_exposure_v2(model, va_ld, apply_funnel_constraint=apply_funnel_constraint, context_cols=context_cols)
    diagnostics = print_exposure_diagnostics(pred_df)
    encoder_decoder_diagnostics = diagnose_encoder_decoder_performance(model, va_ld, pred_df=pred_df)
    diagnostics["encoder_decoder"] = encoder_decoder_diagnostics
    exposure_hat_for_demand = make_external_hat_df(pred_df)

    return {
        "model": model,
        "forecast_df": pred_df,
        "diagnostics": diagnostics,
        "exposure_hat_for_demand": exposure_hat_for_demand,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
    }


# ============================================================
# 使用
# ============================================================
#
# result = run_exposure_v2(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     seed=42,
#     history=13,
#     horizon=20,
#     d_model=64,
#     n_heads=4,
#     batch_size=64,
#     epochs=60,
#     lr=1e-3,
#     patience=8,
#     anchor_decay=0.08,     # anchor衰减速度，越大远期越快收缩到mean13
#     bce_weight=1.00,       # occurrence BCE loss权重
#     mag_weight=1.00,       # magnitude Huber loss权重
#     mean_weight=0.50,      # mean scale penalty权重
# )
#
# exposure_hat_for_demand = result["exposure_hat_for_demand"]
# pred_df = result["forecast_df"]
#
# # 诊断occurrence预测质量
# print(pred_df.groupby("horizon")["p_active_instock"].mean())

# ============================================================
# Rolling Backtest + SCOT Intersection Add-on
# Added after original definitions; these functions override/use the fixed ABC model above.
# ============================================================

def prepare_data_from_sample_scot_intersection(
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
        print(f"Sampled ASINs: {len(sample_asin_set)} | Rows: {len(out)}")
        return out

    scot = scot_df.copy()
    scot["asin"] = scot["asin"].astype(str)
    scot_asin_set = set(scot["asin"].dropna().unique())
    intersect_asins = sorted(sample_asin_set & scot_asin_set)

    out = df[df["asin"].isin(intersect_asins)].copy()
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
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.25,
    high_weight_alpha=0.35,
    path_zero_weight=0.08,
    zero_fp_weight=0.08,
    active_count_weight=0.05,
    path_sum_weight=0.05,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    ratio_residual_scale=0.50,
    zero_protect_enabled=True,
    zero_protect_threshold=0.35,
    zero_protect_temperature=0.10,
    zero_protect_min_gate=0.01,
    use_decoder_zero_attn=True,
    decoder_zero_attn_scale=0.35,
    decoder_zero_attn_min_factor=0.60,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
    dropout=0.20,
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
        ratio_residual_scale=ratio_residual_scale,
        zero_protect_enabled=zero_protect_enabled,
        zero_protect_threshold=zero_protect_threshold,
        zero_protect_temperature=zero_protect_temperature,
        zero_protect_min_gate=zero_protect_min_gate,
        use_decoder_zero_attn=use_decoder_zero_attn,
        decoder_zero_attn_scale=decoder_zero_attn_scale,
        decoder_zero_attn_min_factor=decoder_zero_attn_min_factor,
        use_encoder_self_attn=use_encoder_self_attn,
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
        total_zero_consistency_weight=total_zero_consistency_weight,
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
        ratio_mean_weight=ratio_mean_weight,
        active_mean_weight=active_mean_weight,
    )

    pred_df = predict_exposure_v2(model, va_ld, apply_funnel_constraint=apply_funnel_constraint, context_cols=context_cols)
    pred_df["backtest_offset"] = int(val_start_offset)

    diagnostics = print_exposure_diagnostics(pred_df)
    encoder_decoder_diagnostics = diagnose_encoder_decoder_performance(model, va_ld, pred_df=pred_df)
    diagnostics["encoder_decoder"] = encoder_decoder_diagnostics
    return {
        "model": model,
        "forecast_df": pred_df,
        "diagnostics": diagnostics,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "tr_ds": tr_ds,
        "va_ds": va_ds,
    }


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
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.25,
    high_weight_alpha=0.35,
    path_zero_weight=0.08,
    zero_fp_weight=0.08,
    active_count_weight=0.05,
    path_sum_weight=0.05,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    ratio_residual_scale=0.50,
    zero_protect_enabled=True,
    zero_protect_threshold=0.35,
    zero_protect_temperature=0.10,
    zero_protect_min_gate=0.01,
    use_decoder_zero_attn=True,
    decoder_zero_attn_scale=0.35,
    decoder_zero_attn_min_factor=0.60,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
    dropout=0.20,
    use_scot_intersection=True,
    val_start_offset=0,
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V27.8: DECODER_ZERO_ATTN_ACTIVEPROTECTED + TOTALDIAG + LOCAL CSV")
    print("=" * 100)

    if use_scot_intersection:
        df = prepare_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins, seed)
    else:
        df = prepare_data_from_sample(data_raw1, scot_df, n_asins, seed)

    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    data, context_dim, context_cols = load_exposure_data(df, dph_cap_q=dph_cap_q)

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
        total_zero_consistency_weight=total_zero_consistency_weight,
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
        ratio_residual_scale=ratio_residual_scale,
        zero_protect_enabled=zero_protect_enabled,
        zero_protect_threshold=zero_protect_threshold,
        zero_protect_temperature=zero_protect_temperature,
        zero_protect_min_gate=zero_protect_min_gate,
        use_decoder_zero_attn=use_decoder_zero_attn,
        decoder_zero_attn_scale=decoder_zero_attn_scale,
        decoder_zero_attn_min_factor=decoder_zero_attn_min_factor,
        ratio_mean_weight=ratio_mean_weight,
        active_mean_weight=active_mean_weight,
        dropout=dropout,
        use_encoder_self_attn=use_encoder_self_attn,
    )

    pred_df = out["forecast_df"]

    # v26: GL diagnostics are intentionally not run/returned to keep the exposure run lighter.
    # Category/static/graph features are still used inside the model; only the printed GL reports
    # are removed.
    out.update({
        "exposure_hat_for_demand": make_external_hat_df(pred_df),
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
        "source_df": df,
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
    zero_weight=0.00,
    total_zero_weight=0.01,
    buy_zero_weight=0.05,
    instock_zero_weight=0.08,
    total_zero_consistency_weight=0.01,
    buy_zero_consistency_weight=0.05,
    horizon_weight_alpha=0.25,
    high_weight_alpha=0.35,
    path_zero_weight=0.08,
    zero_fp_weight=0.08,
    active_count_weight=0.05,
    path_sum_weight=0.05,
    peak_weight=0.08,
    topk_peak_weight=0.05,
    peak_under_weight=0.08,
    peak_topk=3,
    peak_quantile=0.80,
    ratio_residual_scale=0.50,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
    dropout=0.20,
    use_scot_intersection=True,
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V2: ROLLING BACKTEST + SCOT INTERSECTION")
    print("=" * 100)
    print(f"n_asins={n_asins} | history={history} | rolling_offsets={list(rolling_offsets)} | epochs={epochs} | patience={patience} | encoder_attn={use_encoder_self_attn}")

    if use_scot_intersection:
        df = prepare_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins, seed)
    else:
        df = prepare_data_from_sample(data_raw1, scot_df, n_asins, seed)

    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    data, context_dim, context_cols = load_exposure_data(df, dph_cap_q=dph_cap_q)

    results_by_offset = {}
    pred_list = []

    for offset in rolling_offsets:
        try:
            res = _train_one_exposure_window(
                data=data,
                context_dim=context_dim,
                context_cols=context_cols,
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
                dropout=dropout,
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

    # v26: GL diagnostics are intentionally not run/returned.
    return {
        "results_by_offset": results_by_offset,
        "rolling_forecast_df": rolling_pred_df,
        "forecast_df": latest_pred_df,
        "diagnostics": rolling_diagnostics,
        "exposure_hat_for_demand": make_external_hat_df(latest_pred_df),
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
        "source_df": df,
        "rolling_offsets": list(rolling_offsets),
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



# ============================================================
# Save exposure hats for demand
# ============================================================

def _parse_exposure_hat_variant_scale(variant, default_scale=1.0):
    """Parse variant names like scale_1p10 into numeric scale 1.10."""
    variant = str(variant or "base").lower().strip()
    if variant.startswith("scale_"):
        s = variant.replace("scale_", "").replace("p", ".")
        try:
            return float(s)
        except Exception:
            return float(default_scale)
    return float(default_scale)


def apply_exposure_hat_variant_for_demand(
    hat,
    hat_variant_for_demand="base",
    exposure_hat_scale=1.0,
    exposure_hat_blend_q70_weight=0.30,
    ratio_residual_scale=0.50,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
):
    """
    Select which exposure hat columns will be written into the standard demand-readable
    pred_total_dph / pred_buy_box_dph / pred_instock_dph columns.

    Default is base: NO p-shift. This is the safest option.

    Supported variants:
      base / none:
          pred_* = base model hat
      scale or scale_1p05 / scale_1p10 / scale_1p15:
          pred_* = base * scale
      blend_q70 / e / e_blend / e_blend_q70:
          pred_* = (1-w) * base + w * q70, default w=0.30
      q70:
          pred_* = q70, diagnostic/aggressive only

    The function always preserves *_base and *_blend_q70_wXX columns so the CSV can be audited.
    """
    import numpy as np
    import pandas as pd

    hat = hat.copy()
    variant = str(hat_variant_for_demand or "base").lower().strip()
    w = float(exposure_hat_blend_q70_weight)
    w = min(max(w, 0.0), 1.0)

    triples = [
        ("total", "pred_total_dph", "pred_total_dph_q70"),
        ("buy_box", "pred_buy_box_dph", "pred_buy_box_dph_q70"),
        ("instock", "pred_instock_dph", "pred_instock_dph_q70"),
    ]

    # Preserve original base hats before any experimental override.
    for name, base_col, q70_col in triples:
        if base_col in hat.columns:
            hat[f"{base_col}_base"] = pd.to_numeric(hat[base_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if q70_col in hat.columns and base_col in hat.columns:
            q70 = pd.to_numeric(hat[q70_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            base = hat[f"{base_col}_base"]
            hat[f"{base_col}_blend_q70_w{int(round(w*100)):02d}"] = ((1.0 - w) * base + w * q70).clip(lower=0.0)
            for s in [1.05, 1.10, 1.15]:
                hat[f"{base_col}_scale_{str(s).replace('.', 'p')}"] = (base * s).clip(lower=0.0)

    # Choose what demand will read from the standard pred_* columns.
    if variant in {"base", "none", "no_shift", "noshift"}:
        chosen_desc = "base / no p-shift"
        for _, base_col, _ in triples:
            if f"{base_col}_base" in hat.columns:
                hat[base_col] = hat[f"{base_col}_base"]

    elif variant.startswith("scale"):
        scale = _parse_exposure_hat_variant_scale(variant, exposure_hat_scale)
        chosen_desc = f"base * {scale:.3f}"
        for _, base_col, _ in triples:
            if f"{base_col}_base" in hat.columns:
                hat[base_col] = (hat[f"{base_col}_base"] * scale).clip(lower=0.0)

    elif variant in {"blend_q70", "e", "e_blend", "e_blend_q70", "base_q70_blend"}:
        chosen_desc = f"E blend: {(1-w):.2f} * base + {w:.2f} * q70"
        for _, base_col, q70_col in triples:
            if f"{base_col}_base" in hat.columns and q70_col in hat.columns:
                base = hat[f"{base_col}_base"]
                q70 = pd.to_numeric(hat[q70_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                hat[base_col] = ((1.0 - w) * base + w * q70).clip(lower=0.0)

    elif variant in {"q70", "upper", "pshift_q70"}:
        chosen_desc = "q70 as demand-readable pred_* columns (aggressive diagnostic)"
        for _, base_col, q70_col in triples:
            if q70_col in hat.columns:
                hat[base_col] = pd.to_numeric(hat[q70_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    else:
        chosen_desc = f"unknown variant '{variant}', fallback to base / no p-shift"
        for _, base_col, _ in triples:
            if f"{base_col}_base" in hat.columns:
                hat[base_col] = hat[f"{base_col}_base"]

    # Recompute main logs based on the chosen demand-readable pred_* columns.
    if "pred_total_dph" in hat.columns:
        hat["external_total_dph_hat_log"] = np.log1p(pd.to_numeric(hat["pred_total_dph"], errors="coerce").fillna(0.0).clip(lower=0.0))
    if "pred_buy_box_dph" in hat.columns:
        hat["external_buy_box_dph_hat_log"] = np.log1p(pd.to_numeric(hat["pred_buy_box_dph"], errors="coerce").fillna(0.0).clip(lower=0.0))
    if "pred_instock_dph" in hat.columns:
        hat["external_instock_dph_hat_log"] = np.log1p(pd.to_numeric(hat["pred_instock_dph"], errors="coerce").fillna(0.0).clip(lower=0.0))

    print("\n" + "=" * 100)
    print("EXPOSURE HAT VARIANT FOR DEMAND CSV")
    print("=" * 100)
    print(f"Selected variant: {hat_variant_for_demand} -> {chosen_desc}")
    print("Demand will read standard columns: pred_total_dph, pred_buy_box_dph, pred_instock_dph")
    print("Base and candidate columns are preserved for audit/ablation.")

    return hat


# ============================================================
# Save exposure hats for demand
# ============================================================

def save_exposure_hat_for_demand_csv(
    exposure_result_or_hat,
    csv_path="/mnt/data/exposure_v26_hat_for_demand.csv",
    hat_variant_for_demand="base",
    exposure_hat_scale=1.0,
    exposure_hat_blend_q70_weight=0.30,
    ratio_residual_scale=0.50,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
):
    """
    Save all exposure hats into one CSV so demand can read it directly later.

    Required demand-compatible columns:
      asin, order_week, pred_total_dph, pred_buy_box_dph, pred_instock_dph

    Default: no p-shift. pred_* columns are the base exposure hats.

    For ablation, set:
      hat_variant_for_demand="scale_1p10"  -> pred_* = base * 1.10
      hat_variant_for_demand="e_blend_q70" -> pred_* = 0.70 * base + 0.30 * q70
      hat_variant_for_demand="q70"         -> pred_* = q70, aggressive diagnostic

    Additional base/q50/q70/q90/blend/scale columns are preserved for diagnostics.
    """
    import os
    import pandas as pd
    import numpy as np

    if isinstance(exposure_result_or_hat, dict):
        if "exposure_hat_for_demand" not in exposure_result_or_hat:
            raise ValueError("exposure result dict must contain key 'exposure_hat_for_demand'.")
        hat = exposure_result_or_hat["exposure_hat_for_demand"].copy()
    else:
        hat = exposure_result_or_hat.copy()

    if "asin" not in hat.columns or "order_week" not in hat.columns:
        raise ValueError("Exposure hat must contain asin and order_week columns.")

    hat["asin"] = hat["asin"].astype(str)
    hat["order_week"] = pd.to_datetime(hat["order_week"])

    # Ensure standard demand-readable level columns exist.
    if "pred_total_dph" not in hat.columns and "external_total_dph_hat_log" in hat.columns:
        hat["pred_total_dph"] = np.expm1(pd.to_numeric(hat["external_total_dph_hat_log"], errors="coerce").fillna(0.0))
    if "pred_buy_box_dph" not in hat.columns and "external_buy_box_dph_hat_log" in hat.columns:
        hat["pred_buy_box_dph"] = np.expm1(pd.to_numeric(hat["external_buy_box_dph_hat_log"], errors="coerce").fillna(0.0))
    if "pred_instock_dph" not in hat.columns:
        if "pred_in_stock_dph" in hat.columns:
            hat["pred_instock_dph"] = pd.to_numeric(hat["pred_in_stock_dph"], errors="coerce").fillna(0.0)
        elif "external_instock_dph_hat_log" in hat.columns:
            hat["pred_instock_dph"] = np.expm1(pd.to_numeric(hat["external_instock_dph_hat_log"], errors="coerce").fillna(0.0))

    required = ["pred_total_dph", "pred_buy_box_dph", "pred_instock_dph"]
    missing = [c for c in required if c not in hat.columns]
    if missing:
        raise ValueError(f"Missing required demand hat columns: {missing}. Available: {hat.columns.tolist()}")

    for c in [col for col in hat.columns if col.startswith("pred_") or col.startswith("op_p")]:
        hat[c] = pd.to_numeric(hat[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    # One ASIN-week row for robust demand merge. If horizon exists, keep first because order_week
    # should already uniquely identify the horizon in the final holdout.
    agg = {}
    for c in hat.columns:
        if c in ["asin", "order_week"]:
            continue
        if c == "horizon":
            agg[c] = "first"
        elif pd.api.types.is_numeric_dtype(hat[c]):
            agg[c] = "mean"
        else:
            agg[c] = "first"
    hat = hat.groupby(["asin", "order_week"], as_index=False).agg(agg)

    # Apply selected demand-readable hat variant AFTER grouping.
    # Default is base / no shift. Candidate E is blend_q70.
    hat = apply_exposure_hat_variant_for_demand(
        hat,
        hat_variant_for_demand=hat_variant_for_demand,
        exposure_hat_scale=exposure_hat_scale,
        exposure_hat_blend_q70_weight=exposure_hat_blend_q70_weight,
    )

    preferred = [
        "asin", "order_week", "horizon",
        "pred_total_dph", "pred_buy_box_dph", "pred_instock_dph",
        "pred_total_dph_base", "pred_buy_box_dph_base", "pred_instock_dph_base",
        "pred_total_dph_blend_q70_w30", "pred_buy_box_dph_blend_q70_w30", "pred_instock_dph_blend_q70_w30",
        "pred_total_dph_scale_1p05", "pred_buy_box_dph_scale_1p05", "pred_instock_dph_scale_1p05",
        "pred_total_dph_scale_1p1", "pred_buy_box_dph_scale_1p1", "pred_instock_dph_scale_1p1",
        "pred_total_dph_scale_1p15", "pred_buy_box_dph_scale_1p15", "pred_instock_dph_scale_1p15",
        "pred_total_dph_q50", "pred_buy_box_dph_q50", "pred_instock_dph_q50",
        "pred_total_dph_q70", "pred_buy_box_dph_q70", "pred_instock_dph_q70",
        "pred_total_dph_q90", "pred_buy_box_dph_q90", "pred_instock_dph_q90",
        "op_p50_total_dph", "op_p50_buy_box_dph", "op_p50_instock_dph",
        "op_p70_total_dph", "op_p70_buy_box_dph", "op_p70_instock_dph",
        "external_total_dph_hat_log", "external_buy_box_dph_hat_log", "external_instock_dph_hat_log",
    ]
    cols = [c for c in preferred if c in hat.columns] + [c for c in hat.columns if c not in preferred]
    hat = hat[cols]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    hat.to_csv(csv_path, index=False)
    print(f"\nSaved exposure_hat_for_demand CSV: {csv_path}")
    print(f"Rows: {len(hat):,} | ASINs: {hat['asin'].nunique():,} | Weeks: {hat['order_week'].nunique():,}")
    print("Demand-compatible columns:", [c for c in ["pred_total_dph", "pred_buy_box_dph", "pred_instock_dph"] if c in hat.columns])
    print("Base backup columns:", [c for c in ["pred_total_dph_base", "pred_buy_box_dph_base", "pred_instock_dph_base"] if c in hat.columns])
    return csv_path

def run_exposure_v2_final_scot_5000(
    data_raw1,
    scot_df,
    seed=42,
    history=13,
    horizon=20,
    epochs=60,
    patience=10,
    batch_size=128,
    use_encoder_self_attn=True,
    save_hat_csv_path="exposure_hat_for_demand.csv",
    hat_variant_for_demand="base",
    exposure_hat_scale=1.0,
    exposure_hat_blend_q70_weight=0.30,
    ratio_residual_scale=0.50,
    zero_protect_enabled=True,
    zero_protect_threshold=0.35,
    zero_protect_temperature=0.10,
    zero_protect_min_gate=0.01,
    use_decoder_zero_attn=True,
    decoder_zero_attn_scale=0.35,
    decoder_zero_attn_min_factor=0.60,
    ratio_mean_weight=0.05,
    active_mean_weight=0.03,
):
    """
    Final single-window setup:
      - sample 5000 ASINs
      - intersect with SCOT ASINs
      - train on sliding windows before the final holdout
      - validate/predict the latest 20-week window only
      - return exposure_hat_for_demand for the demand model
    """
    result = run_exposure_v2(
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
        use_encoder_self_attn=use_encoder_self_attn,
        ratio_residual_scale=ratio_residual_scale,
        zero_protect_enabled=zero_protect_enabled,
        zero_protect_threshold=zero_protect_threshold,
        zero_protect_temperature=zero_protect_temperature,
        zero_protect_min_gate=zero_protect_min_gate,
        use_decoder_zero_attn=use_decoder_zero_attn,
        decoder_zero_attn_scale=decoder_zero_attn_scale,
        decoder_zero_attn_min_factor=decoder_zero_attn_min_factor,
        ratio_mean_weight=ratio_mean_weight,
        active_mean_weight=active_mean_weight,
    )
    if save_hat_csv_path is not None:
        result["exposure_hat_csv_path"] = save_exposure_hat_for_demand_csv(
            result,
            save_hat_csv_path,
            hat_variant_for_demand=hat_variant_for_demand,
            exposure_hat_scale=exposure_hat_scale,
            exposure_hat_blend_q70_weight=exposure_hat_blend_q70_weight,
        )
    return result

# ============================================================
# Usage
# ============================================================
# Final setup: 5000 sample + SCOT intersection + latest 20-week holdout.
# This AUTOSAVE version saves default CSV to: exposure_hat_for_demand.csv
# Training samples are sliding windows; validation/test is the final 20-week window.
#
# %run -i tcn_exposure_v2_single_head_direct_gl_diag.py
#
# result = run_exposure_v2_final_scot_5000(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     history=13,
#     horizon=20,
#     epochs=30,
#     patience=6,
#     batch_size=128,
#     use_encoder_self_attn=True,
# )
#
# pred_df = result["forecast_df"]
# exposure_hat_for_demand = result["exposure_hat_for_demand"]
# diagnostics = result["diagnostics"]
# gl_diag = result["gl_diagnostics"]
# gl_block_diag = result["gl_horizon_block_diagnostics"]
# gl_summary = result["gl_summary"]
#
# Optional no-attention ablation:
# result_no_attn = run_exposure_v2_final_scot_5000(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     history=13,
#     horizon=20,
#     epochs=30,
#     patience=6,
#     batch_size=128,
#     use_encoder_self_attn=False,
# )
#
# Rolling backtest is still available for robustness checks:
# result_roll = run_exposure_v2_rolling(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     history=13,
#     horizon=20,
#     rolling_offsets=(60, 40, 20, 0),
#     epochs=20,
#     patience=5,
#     batch_size=128,
#     use_scot_intersection=True,
#     use_encoder_self_attn=True,
# )


# ============================================================
# PACKAGE-AWARE ASIN RELATION GRAPH CONTEXT + KNOWN PROMO + GRAPH-HEAD-FUSION PATCH
# Added by ChatGPT: package-comparable peer graph features.
# v22: graph is no longer only side context; it is projected, fused with encoder/cross-attn state,
#      and passed directly to final active/direct heads together with ENN latent z.
# v23: add SPADE-style decoder-side peak residual branch driven by known future promo/holiday/event features.
# Design:
#   - graph neighbors = same category_code + relaxed package-size/weight comparable
#   - graph features are computed at forecast origin from historical values only
#   - future target weeks are never used to construct graph context
#   - features are injected into future_context so encoder/decoder/z can use them
# ============================================================

GRAPH_CONTEXT_COLS = [
    "graph_peer_total_mean13_log",
    "graph_peer_buybox_mean13_log",
    "graph_peer_instock_mean13_log",
    "graph_peer_demand_mean13_log",
    "graph_peer_active_rate13",
    "graph_peer_zero_rate13",
    "graph_peer_count_log",
    "graph_peer_rank_prior",
    # v27: horizon-specific promo-adjusted rank features for decoder/future_context.
    # These use only known future own promo, known future peer promo, and origin historical rank.
    "graph_own_known_promo_h",
    "graph_peer_known_promo_h",
    "graph_own_vs_peer_promo_delta_h",
    "graph_promo_adjusted_rank_prior_h",
    "graph_peer_known_promo_nextH_rate",
    "graph_peer_known_promo_long13_20_rate",
    "graph_peer_known_promo_rate_max",
    "graph_peer_known_promo_amount_log_max",
    "graph_same_hbt_peer_rate",
    "graph_top10_peer_rate",
]


def _graph_mode_str(x, default="MISSING"):
    try:
        s = pd.Series(x).astype(str).replace({"nan": default, "None": default, "": default})
        if len(s) == 0:
            return default
        return str(s.mode().iloc[0]) if len(s.mode()) else default
    except Exception:
        return default


def _graph_safe_num(x, fill=0.0):
    try:
        return pd.to_numeric(x, errors="coerce").fillna(fill)
    except Exception:
        return pd.Series(fill, index=getattr(x, 'index', None))


def _graph_build_meta_from_raw(data_raw):
    """Static product metadata used for graph neighbor construction."""
    df = data_raw.copy()
    asin_col = "asin" if "asin" in df.columns else ("ASIN" if "ASIN" in df.columns else None)
    if asin_col is None:
        return {}
    df[asin_col] = df[asin_col].astype(str)
    if "order_week" in df.columns:
        df["order_week"] = pd.to_datetime(df["order_week"], errors="coerce")
        df = df.sort_values([asin_col, "order_week"])

    for c in ["pkg_height", "pkg_length", "pkg_width", "pkg_weight", "our_price", "ind_top10_brand"]:
        if c not in df.columns:
            df[c] = np.nan if c.startswith("pkg_") else 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "category_code" not in df.columns:
        df["category_code"] = "UNKNOWN"
    if "hbt" not in df.columns:
        df["hbt"] = "MISSING"

    meta = {}
    for asin, g in df.groupby(asin_col):
        last = g.iloc[-1]
        dims = {}
        for c in ["pkg_height", "pkg_length", "pkg_width", "pkg_weight"]:
            vals = pd.to_numeric(g[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            v = float(vals.iloc[-1]) if len(vals) else np.nan
            dims[c] = v
            dims[f"log_{c}"] = float(np.log1p(max(v, 0.0))) if np.isfinite(v) else np.nan
        vol = dims.get("pkg_height", np.nan) * dims.get("pkg_length", np.nan) * dims.get("pkg_width", np.nan)
        dims["pkg_volume"] = vol
        dims["log_pkg_volume"] = float(np.log1p(max(vol, 0.0))) if np.isfinite(vol) else np.nan
        price_vals = pd.to_numeric(g["our_price"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        price = float(price_vals.iloc[-1]) if len(price_vals) else 0.0
        meta[str(asin)] = {
            "category_code": _graph_mode_str(g["category_code"], "UNKNOWN"),
            "hbt": _graph_mode_str(g["hbt"], "MISSING").lower(),
            "ind_top10_brand": float(pd.to_numeric(g["ind_top10_brand"], errors="coerce").fillna(0.0).iloc[-1]),
            "log_our_price": float(np.log1p(max(price, 0.0))),
            **dims,
        }
    return meta


def _graph_pkg_relaxed_similar(mi, mj, max_mean_log_gap=0.40, max_volume_log_gap=0.75, max_weight_log_gap=0.55):
    """Relaxed package comparability: same physical scale, not exact duplicate."""
    dim_cols = ["log_pkg_height", "log_pkg_length", "log_pkg_width", "log_pkg_weight"]
    gaps = []
    for c in dim_cols:
        a, b = mi.get(c, np.nan), mj.get(c, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            gaps.append(abs(float(a) - float(b)))
    if len(gaps) == 0:
        return False
    mean_gap = float(np.mean(gaps))
    vol_gap = abs(float(mi.get("log_pkg_volume", np.nan)) - float(mj.get("log_pkg_volume", np.nan))) \
        if np.isfinite(mi.get("log_pkg_volume", np.nan)) and np.isfinite(mj.get("log_pkg_volume", np.nan)) else mean_gap
    wt_gap = abs(float(mi.get("log_pkg_weight", np.nan)) - float(mj.get("log_pkg_weight", np.nan))) \
        if np.isfinite(mi.get("log_pkg_weight", np.nan)) and np.isfinite(mj.get("log_pkg_weight", np.nan)) else mean_gap
    return (mean_gap <= max_mean_log_gap) and (vol_gap <= max_volume_log_gap) and (wt_gap <= max_weight_log_gap)


def _graph_recent_mean(arr, end, window=13):
    x = np.asarray(arr[max(0, end-window):end], dtype=float)
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.clip(x, 0, None)))


def _graph_strength_for_asin(d, end):
    return (
        0.30 * np.log1p(_graph_recent_mean(d.get("total_dph", []), end, 13)) +
        0.25 * np.log1p(_graph_recent_mean(d.get("buy_box_dph", []), end, 13)) +
        0.25 * np.log1p(_graph_recent_mean(d.get("in_stock_dph", d.get("instock_raw", [])), end, 13)) +
        0.20 * np.log1p(_graph_recent_mean(d.get("demand", []), end, 13))
    )


def _graph_percentile_rank(value, values):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) <= 1:
        return 0.5
    return float((np.sum(vals < value) + 0.5 * np.sum(vals == value)) / len(vals))


def _graph_add_context_cols_to_data(data, context_cols, data_raw=None):
    """Append zero graph columns to every future_context and attach product metadata."""
    context_cols = list(context_cols)
    add_cols = [c for c in GRAPH_CONTEXT_COLS if c not in context_cols]
    if len(add_cols) > 0:
        for d in data.values():
            zeros = np.zeros((d["future_context"].shape[0], len(add_cols)), dtype=np.float32)
            d["future_context"] = np.concatenate([d["future_context"], zeros], axis=1)
        context_cols = context_cols + add_cols

    meta = _graph_build_meta_from_raw(data_raw) if data_raw is not None else {}
    for asin, d in data.items():
        d["context_cols"] = context_cols
        d["graph_context_idx"] = {c: context_cols.index(c) for c in GRAPH_CONTEXT_COLS if c in context_cols}
        d["graph_meta"] = meta.get(str(asin), {
            "category_code": "UNKNOWN", "hbt": "missing", "ind_top10_brand": 0.0,
            "log_our_price": 0.0, "log_pkg_height": np.nan, "log_pkg_length": np.nan,
            "log_pkg_width": np.nan, "log_pkg_weight": np.nan, "log_pkg_volume": np.nan,
        })
    return data, len(context_cols), context_cols


def _graph_known_promo_window_features(d, end, H=20):
    """Known-future promo summaries for one ASIN after forecast origin end.

    These are allowed only under the user's business assumption that promo index/rate
    are known before the forecast is made.
    """
    cols = d.get("context_cols", [])
    fc = d.get("future_context", None)
    if fc is None or len(cols) == 0:
        return 0.0, 0.0, 0.0, 0.0
    name_to_idx = {c: i for i, c in enumerate(cols)}
    sl = slice(int(end), min(int(end) + int(H), fc.shape[0]))
    if sl.start >= sl.stop:
        return 0.0, 0.0, 0.0, 0.0

    def arr(col):
        if col not in name_to_idx:
            return np.zeros((sl.stop - sl.start,), dtype=float)
        return np.asarray(fc[sl, name_to_idx[col]], dtype=float)

    promo_index = arr("known_promo_index")
    promo_rate = arr("known_promo_rate")
    promo_amt = arr("known_promo_amount_log")

    nextH_rate = float(np.mean(promo_index > 0.5)) if len(promo_index) else 0.0
    long = promo_index[12:20] if len(promo_index) > 12 else np.array([], dtype=float)
    long_rate = float(np.mean(long > 0.5)) if len(long) else 0.0
    rate_max = float(np.nanmax(promo_rate)) if len(promo_rate) else 0.0
    amt_max = float(np.nanmax(promo_amt)) if len(promo_amt) else 0.0
    if not np.isfinite(rate_max): rate_max = 0.0
    if not np.isfinite(amt_max): amt_max = 0.0
    return nextH_rate, long_rate, rate_max, amt_max


def _graph_build_neighbor_map(data, min_neighbors=3):
    asins = list(data.keys())
    by_cat = {}
    for a in asins:
        cat = data[a].get("graph_meta", {}).get("category_code", "UNKNOWN")
        by_cat.setdefault(cat, []).append(a)

    nbrs = {}
    for a in asins:
        mi = data[a].get("graph_meta", {})
        cand = []
        for b in by_cat.get(mi.get("category_code", "UNKNOWN"), []):
            if b == a:
                continue
            mj = data[b].get("graph_meta", {})
            if _graph_pkg_relaxed_similar(mi, mj):
                cand.append(b)
        if len(cand) < min_neighbors:
            cand = [b for b in by_cat.get(mi.get("category_code", "UNKNOWN"), []) if b != a]
        nbrs[a] = cand
    return nbrs



def _graph_known_promo_at_horizon(d, end, step_h):
    """
    Known future promo signal for one ASIN at horizon h.
    Uses future_context known promo columns only; no future demand/exposure target is used.
    Returns a compact [0, 1+] score dominated by promo_index, with a small contribution
    from known_promo_rate/amount when available.
    """
    cols = d.get("context_cols", [])
    fc = d.get("future_context", None)
    if fc is None or len(cols) == 0:
        return 0.0
    pos = int(end) + int(step_h)
    if pos < 0 or pos >= fc.shape[0]:
        return 0.0
    name_to_idx = {c: i for i, c in enumerate(cols)}
    promo_index = float(fc[pos, name_to_idx["known_promo_index"]]) if "known_promo_index" in name_to_idx else 0.0
    promo_rate = float(fc[pos, name_to_idx["known_promo_rate"]]) if "known_promo_rate" in name_to_idx else 0.0
    promo_amt = float(fc[pos, name_to_idx["known_promo_amount_log"]]) if "known_promo_amount_log" in name_to_idx else 0.0
    # Keep this bounded/stable. It is a rank prior feature, not the target.
    promo_score = promo_index + 0.25 * np.tanh(promo_rate) + 0.10 * np.tanh(promo_amt / 5.0)
    return float(np.clip(promo_score, 0.0, 1.35))


def _safe_get_vec_value(vec, col, default=0.0):
    try:
        k = GRAPH_CONTEXT_COLS.index(col)
        return float(vec[k])
    except Exception:
        return float(default)


def _bucketize_rank(x, n_bins=5):
    """Robust qcut helper for diagnostics."""
    import pandas as pd
    try:
        return pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        try:
            return pd.cut(x, bins=n_bins, labels=False, include_lowest=True)
        except Exception:
            return pd.Series(np.zeros(len(x), dtype=int), index=getattr(x, 'index', None))


def diagnose_promo_adjusted_rank(pred_df, target="in_stock"):
    """
    Check whether horizon-specific promo-adjusted rank is informative.

    Two views are printed:
      1) Row-level rank buckets: does higher adjusted rank correspond to higher true exposure?
      2) ASIN-level 20-week WAPE: since AMXL WAPE is often interpreted per item/path,
         aggregate each ASIN's final 20-week exposure and inspect whether errors concentrate
         in high/low future-rank groups.
    """
    import numpy as np
    import pandas as pd

    true_col = "true_instock_dph" if target in {"instock", "in_stock", "in_stock_dph"} else f"true_{target}_dph"
    pred_col = "pred_instock_dph" if target in {"instock", "in_stock", "in_stock_dph"} else f"pred_{target}_dph"
    need = [true_col, pred_col, "graph_peer_rank_prior", "graph_promo_adjusted_rank_prior_h"]
    missing = [c for c in need if c not in pred_df.columns]
    if missing:
        print("\nPROMO-ADJUSTED RANK DIAGNOSTIC skipped. Missing columns:", missing)
        return {"rank_bucket_rows": pd.DataFrame(), "asin_rank_wape": pd.DataFrame(), "rank_summary": pd.DataFrame()}

    df = pred_df.copy()
    for c in need + ["graph_own_vs_peer_promo_delta_h", "graph_own_known_promo_h", "graph_peer_known_promo_h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    y = df[true_col].values.astype(float)
    p = df[pred_col].values.astype(float)
    origin = df["graph_peer_rank_prior"].values.astype(float)
    adj = df["graph_promo_adjusted_rank_prior_h"].values.astype(float)

    row_summary = pd.DataFrame([
        {
            "rank_signal": "origin_rank_prior",
            "spearman_with_true": _safe_spearman(origin, y),
            "spearman_with_pred": _safe_spearman(origin, p),
            "active_AUC": _auc((y > 0).astype(int), origin),
        },
        {
            "rank_signal": "promo_adjusted_rank_h",
            "spearman_with_true": _safe_spearman(adj, y),
            "spearman_with_pred": _safe_spearman(adj, p),
            "active_AUC": _auc((y > 0).astype(int), adj),
        },
    ])

    df["rank_bucket"] = _bucketize_rank(df["graph_promo_adjusted_rank_prior_h"], n_bins=5)
    bucket_rows = []
    for b, g in df.groupby("rank_bucket"):
        yy = g[true_col].values.astype(float)
        pp = g[pred_col].values.astype(float)
        bucket_rows.append({
            "rank_bucket": int(b) if pd.notna(b) else -1,
            "n_rows": int(len(g)),
            "avg_origin_rank": float(g["graph_peer_rank_prior"].mean()),
            "avg_adjusted_rank": float(g["graph_promo_adjusted_rank_prior_h"].mean()),
            "avg_promo_delta": float(g["graph_own_vs_peer_promo_delta_h"].mean()) if "graph_own_vs_peer_promo_delta_h" in g.columns else np.nan,
            "true_mean": float(np.mean(yy)),
            "pred_mean": float(np.mean(pp)),
            "ratio": float(np.mean(pp) / (np.mean(yy) + 1e-8)),
            "WAPE": float(_wape(yy, pp)),
            "active_rate": float(np.mean(yy > 0)),
            "active_AUC_pred": float(_auc((yy > 0).astype(int), pp)),
        })
    bucket_df = pd.DataFrame(bucket_rows).sort_values("rank_bucket") if bucket_rows else pd.DataFrame()

    asin = df.groupby("asin").agg(
        true_sum=(true_col, "sum"),
        pred_sum=(pred_col, "sum"),
        avg_origin_rank=("graph_peer_rank_prior", "mean"),
        avg_adjusted_rank=("graph_promo_adjusted_rank_prior_h", "mean"),
        avg_promo_delta=("graph_own_vs_peer_promo_delta_h", "mean") if "graph_own_vs_peer_promo_delta_h" in df.columns else ("graph_promo_adjusted_rank_prior_h", "mean"),
        active_weeks=(true_col, lambda x: int(np.sum(np.asarray(x) > 0))),
    ).reset_index()
    asin["ratio"] = asin["pred_sum"] / (asin["true_sum"] + 1e-8)
    asin["asin_20w_wape"] = (asin["pred_sum"] - asin["true_sum"]).abs() / (asin["true_sum"] + 1e-8)
    asin["rank_bucket"] = _bucketize_rank(asin["avg_adjusted_rank"], n_bins=5)
    asin_rows = []
    for b, g in asin.groupby("rank_bucket"):
        yy = g["true_sum"].values.astype(float)
        pp = g["pred_sum"].values.astype(float)
        asin_rows.append({
            "rank_bucket": int(b) if pd.notna(b) else -1,
            "n_asins": int(len(g)),
            "avg_adjusted_rank": float(g["avg_adjusted_rank"].mean()),
            "avg_promo_delta": float(g["avg_promo_delta"].mean()),
            "true_sum_mean": float(g["true_sum"].mean()),
            "pred_sum_mean": float(g["pred_sum"].mean()),
            "ratio": float(np.mean(pp) / (np.mean(yy) + 1e-8)),
            "aggregate_WAPE": float(_wape(yy, pp)),
            "median_ASIN_WAPE": float(g["asin_20w_wape"].median()),
            "p90_ASIN_WAPE": float(g["asin_20w_wape"].quantile(0.90)),
        })
    asin_bucket_df = pd.DataFrame(asin_rows).sort_values("rank_bucket") if asin_rows else pd.DataFrame()

    print("\n" + "=" * 100)
    print("PROMO-ADJUSTED RANK DIAGNOSTIC: IN_STOCK_DPH")
    print("=" * 100)
    print("Signal usefulness: compare origin historical rank vs horizon-specific promo-adjusted rank.")
    print(row_summary.round(4).to_string(index=False))

    print("\nRANK BUCKETS BY ROW / HORIZON")
    if len(bucket_df):
        print(bucket_df.round(4).to_string(index=False))
    else:
        print("No bucket table available.")

    print("\nASIN-LEVEL 20-WEEK WAPE BY AVG PROMO-ADJUSTED RANK")
    if len(asin_bucket_df):
        print(asin_bucket_df.round(4).to_string(index=False))
    else:
        print("No ASIN-level rank table available.")

    if len(row_summary) == 2:
        d_spear = row_summary.loc[1, "spearman_with_true"] - row_summary.loc[0, "spearman_with_true"]
        d_auc = row_summary.loc[1, "active_AUC"] - row_summary.loc[0, "active_AUC"]
        print(f"\nDelta vs origin rank: Spearman(true)={d_spear:+.4f}, active_AUC={d_auc:+.4f}")
        if (d_spear > 0.01) or (d_auc > 0.01):
            print("Judgment: promo-adjusted rank is adding useful future competitive information.")
        else:
            print("Judgment: promo-adjusted rank is not clearly better than origin rank yet; keep it light/diagnostic.")

    return {"rank_bucket_rows": bucket_df, "asin_rank_wape": asin_bucket_df, "rank_summary": row_summary}

class _GraphContextMixin:
    def _init_graph_context(self, min_graph_neighbors=3):
        self.graph_neighbor_map = _graph_build_neighbor_map(self.data, min_neighbors=min_graph_neighbors)
        self._graph_context_cache = {}
        counts = [len(v) for v in self.graph_neighbor_map.values()]
        if len(counts):
            print("Package-aware graph context enabled | ASINs:", len(counts),
                  "| median neighbors:", int(np.median(counts)),
                  "| mean neighbors:", round(float(np.mean(counts)), 2),
                  "| min/max:", int(np.min(counts)), int(np.max(counts)))

    def _compute_graph_context_vec(self, asin, end):
        key = (str(asin), int(end))
        if key in self._graph_context_cache:
            return self._graph_context_cache[key]
        d_i = self.data[asin]
        idx = d_i.get("graph_context_idx", {})
        if len(idx) == 0:
            vec = np.zeros(len(GRAPH_CONTEXT_COLS), dtype=np.float32)
            self._graph_context_cache[key] = vec
            return vec
        nbrs = self.graph_neighbor_map.get(asin, [])
        if len(nbrs) == 0:
            nbrs = [asin]

        vals_total, vals_buy, vals_inst, vals_dem, active_rates = [], [], [], [], []
        strengths = []
        known_nextH_rates, known_long_rates, known_rate_maxs, known_amt_maxs = [], [], [], []
        same_hbt, top10 = [], []
        hbt_i = d_i.get("graph_meta", {}).get("hbt", "missing")
        for b in nbrs:
            d = self.data[b]
            vals_total.append(_graph_recent_mean(d.get("total_dph", []), end, 13))
            vals_buy.append(_graph_recent_mean(d.get("buy_box_dph", []), end, 13))
            vals_inst.append(_graph_recent_mean(d.get("in_stock_dph", d.get("instock_raw", [])), end, 13))
            vals_dem.append(_graph_recent_mean(d.get("demand", []), end, 13))
            x = np.asarray(d.get("in_stock_dph", d.get("instock_raw", []))[max(0, end-13):end], dtype=float)
            active_rates.append(float(np.mean(x > 0)) if len(x) else 0.0)
            strengths.append(_graph_strength_for_asin(d, end))
            kn, kl, kr, ka = _graph_known_promo_window_features(d, end, H=20)
            known_nextH_rates.append(kn)
            known_long_rates.append(kl)
            known_rate_maxs.append(kr)
            known_amt_maxs.append(ka)
            same_hbt.append(1.0 if d.get("graph_meta", {}).get("hbt", "missing") == hbt_i else 0.0)
            top10.append(float(d.get("graph_meta", {}).get("ind_top10_brand", 0.0)))

        own_strength = _graph_strength_for_asin(d_i, end)
        all_strengths = strengths + [own_strength]
        rank_prior = _graph_percentile_rank(own_strength, all_strengths)
        peer_count = max(len(nbrs), 1)
        vec_map = {
            "graph_peer_total_mean13_log": np.log1p(np.mean(vals_total) if len(vals_total) else 0.0),
            "graph_peer_buybox_mean13_log": np.log1p(np.mean(vals_buy) if len(vals_buy) else 0.0),
            "graph_peer_instock_mean13_log": np.log1p(np.mean(vals_inst) if len(vals_inst) else 0.0),
            "graph_peer_demand_mean13_log": np.log1p(np.mean(vals_dem) if len(vals_dem) else 0.0),
            "graph_peer_active_rate13": float(np.mean(active_rates) if len(active_rates) else 0.0),
            "graph_peer_zero_rate13": float(1.0 - np.mean(active_rates) if len(active_rates) else 1.0),
            "graph_peer_count_log": np.log1p(peer_count),
            "graph_peer_rank_prior": rank_prior,
            # v27 dynamic horizon-level columns are overwritten in _inject_graph_context.
            # Initialize with origin-safe defaults so context dimensions are stable.
            "graph_own_known_promo_h": 0.0,
            "graph_peer_known_promo_h": float(np.mean(known_nextH_rates) if len(known_nextH_rates) else 0.0),
            "graph_own_vs_peer_promo_delta_h": 0.0,
            "graph_promo_adjusted_rank_prior_h": rank_prior,
            "graph_peer_known_promo_nextH_rate": float(np.mean(known_nextH_rates) if len(known_nextH_rates) else 0.0),
            "graph_peer_known_promo_long13_20_rate": float(np.mean(known_long_rates) if len(known_long_rates) else 0.0),
            "graph_peer_known_promo_rate_max": float(np.max(known_rate_maxs) if len(known_rate_maxs) else 0.0),
            "graph_peer_known_promo_amount_log_max": float(np.max(known_amt_maxs) if len(known_amt_maxs) else 0.0),
            "graph_same_hbt_peer_rate": float(np.mean(same_hbt) if len(same_hbt) else 0.0),
            "graph_top10_peer_rate": float(np.mean(top10) if len(top10) else 0.0),
        }
        vec = np.array([vec_map[c] for c in GRAPH_CONTEXT_COLS], dtype=np.float32)
        self._graph_context_cache[key] = vec
        return vec

    def _compute_peer_promo_by_horizon(self, asin, end, H):
        """Average known future promo score among graph peers by horizon h."""
        key = ("peer_promo_h", str(asin), int(end), int(H))
        if key in self._graph_context_cache:
            return self._graph_context_cache[key]
        nbrs = self.graph_neighbor_map.get(asin, [])
        if len(nbrs) == 0:
            nbrs = [asin]
        out = []
        for step_h in range(int(H)):
            vals = []
            for b in nbrs:
                vals.append(_graph_known_promo_at_horizon(self.data[b], end, step_h))
            out.append(float(np.mean(vals) if len(vals) else 0.0))
        arr = np.asarray(out, dtype=np.float32)
        self._graph_context_cache[key] = arr
        return arr

    def _inject_graph_context(self, fc, d, asin, end):
        idx = d.get("graph_context_idx", {})
        if len(idx) == 0 or fc is None or len(fc) == 0:
            return fc
        base_vec = self._compute_graph_context_vec(asin, end)
        H = fc.shape[0]
        base_rank = _safe_get_vec_value(base_vec, "graph_peer_rank_prior", default=0.5)
        peer_promo_h = self._compute_peer_promo_by_horizon(asin, end, H)

        beta_delta = 0.15  # small by design: rank prior should inform, not dominate.
        for step_h in range(H):
            # keep graph prior strongest near origin but still available at long horizon
            h_decay = 0.65 + 0.35 * np.exp(-0.06 * step_h)
            own_promo_h = _graph_known_promo_at_horizon(d, end, step_h)
            peer_promo = float(peer_promo_h[step_h]) if step_h < len(peer_promo_h) else 0.0
            promo_delta = float(own_promo_h - peer_promo)
            promo_adjusted_rank = float(np.clip(base_rank + beta_delta * promo_delta, 0.0, 1.0))

            for k, col in enumerate(GRAPH_CONTEXT_COLS):
                if col not in idx:
                    continue
                if col == "graph_own_known_promo_h":
                    val = own_promo_h
                elif col == "graph_peer_known_promo_h":
                    val = peer_promo
                elif col == "graph_own_vs_peer_promo_delta_h":
                    val = promo_delta
                elif col == "graph_promo_adjusted_rank_prior_h":
                    # Do not decay adjusted rank. It is a horizon-specific competitive prior in [0,1].
                    val = promo_adjusted_rank
                else:
                    val = float(base_vec[k]) * h_decay
                fc[step_h, idx[col]] = float(val)
        return fc


# ---- Exposure-specific overrides ----
_ORIGINAL_LOAD_EXPOSURE_DATA_BEFORE_GRAPH = load_exposure_data
_ORIGINAL_EXPOSURE_DATASET_BEFORE_GRAPH = ExposureDataset
_ORIGINAL_EXPOSURE_DATASET_ROLLING_BEFORE_GRAPH = ExposureDatasetRolling


def load_exposure_data(data_raw, dph_cap_q=0.995):
    data, context_dim, context_cols = _ORIGINAL_LOAD_EXPOSURE_DATA_BEFORE_GRAPH(data_raw, dph_cap_q=dph_cap_q)
    data, context_dim, context_cols = _graph_add_context_cols_to_data(data, context_cols, data_raw=data_raw)
    print("\n" + "=" * 100)
    print("PACKAGE-AWARE RELATION GRAPH FEATURES ADDED TO EXPOSURE FUTURE_CONTEXT")
    print("Graph cols:", GRAPH_CONTEXT_COLS)
    print("New context dim:", context_dim)
    print("=" * 100)
    return data, context_dim, context_cols


class ExposureDataset(_GraphContextMixin, _ORIGINAL_EXPOSURE_DATASET_BEFORE_GRAPH):
    def __init__(self, *args, min_graph_neighbors=3, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_graph_context(min_graph_neighbors=min_graph_neighbors)

    def _make_future_context(self, d, start):
        fc = super()._make_future_context(d, start)
        asin = None
        # recover asin by object identity only for compatibility with original Dataset
        # this path is rarely used in final run; RollingDataset below passes asin directly.
        for a, dd in self.data.items():
            if dd is d:
                asin = a
                break
        if asin is None:
            return fc
        return self._inject_graph_context(fc, d, asin, start + self.history)


class ExposureDatasetRolling(_GraphContextMixin, _ORIGINAL_EXPOSURE_DATASET_ROLLING_BEFORE_GRAPH):
    def __init__(self, *args, min_graph_neighbors=3, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_graph_context(min_graph_neighbors=min_graph_neighbors)

    def _make_future_context(self, d, start):
        fc = super()._make_future_context(d, start)
        asin = None
        for a, dd in self.data.items():
            if dd is d:
                asin = a
                break
        if asin is None:
            return fc
        return self._inject_graph_context(fc, d, asin, start + self.history)


def summarize_graph_context_from_result(exposure_result):
    """Quick check that graph columns were attached and used as future_context columns."""
    cols = exposure_result.get("context_cols", []) if isinstance(exposure_result, dict) else []
    present = [c for c in GRAPH_CONTEXT_COLS if c in cols]
    print("Graph context cols present:", present)
    print("n_graph_cols:", len(present), "| context_dim:", len(cols))
    return {"graph_cols": present, "n_graph_cols": len(present), "context_dim": len(cols)}

# Usage:
# %run -i tcn_exposure_v2_enn_regime_singlehead_nogate_peakloss_gpu_GRAPH_PKG_PEER_v20.py
# exposure_result = run_exposure_v2_final_scot_5000(data_raw1=data_raw1, scot_df=scot_df, history=13, horizon=20, epochs=30, patience=6, batch_size=128)
# graph_ctx_summary = summarize_graph_context_from_result(exposure_result)
