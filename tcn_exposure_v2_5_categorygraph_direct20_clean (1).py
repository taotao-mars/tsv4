# ============================================================
# TCN Exposure Model V2
# Single-head direct update + category_code static features:
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


# ============================================================
# SAFE-FOR-DEMAND NAMESPACE PATCH
# This exposure file intentionally uses exposure-specific helper names
# (prepare_exposure_*, filter_extreme_exposure_asins, _exposure_safe_numeric)
# so it will not overwrite the demand model helpers when both files are run
# in the same Jupyter namespace via %run -i.
# Demand file is unchanged.
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

def _exposure_safe_numeric(s, fill=0.0):
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
    print(f"Sampled ASINs: {len(sample_asins)} | Rows: {len(out)}")
    return out


# 向后兼容：保留旧函数名
def prepare_exposure_data_from_sample_scot_intersection(
    data_raw1, scot_df=None, n_asins=5000, seed=42,
):
    return prepare_exposure_data_from_sample(data_raw1, scot_df, n_asins, seed)


def filter_extreme_exposure_asins(data_raw, q=0.99):
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
        gv = _exposure_safe_numeric(df["glance_view_band_cat"]).clip(1, 6)
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
        df["stock_static__ind_amxl_hb"] = _exposure_safe_numeric(df["ind_amxl_hb"]).clip(0, 1)
        out_cols.append("stock_static__ind_amxl_hb")

    # ── 新增：sort_type（1/2/3，静态）────────────────────────
    if "sort_type" in df.columns:
        df["stock_static__sort_type__norm"] = (
            _exposure_safe_numeric(df["sort_type"]).clip(1, 3) / 3.0
        )
        out_cols.append("stock_static__sort_type__norm")

    # ── 新增：ind_new_asin（binary，静态）────────────────────
    if "ind_new_asin" in df.columns:
        df["stock_static__ind_new_asin"] = _exposure_safe_numeric(
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
        df[c] = _exposure_safe_numeric(df[c]).clip(lower=0.0)

    for c in ["total_dph", "buy_box_dph", "in_stock_dph"]:
        cap = df[c].quantile(dph_cap_q)
        df[c] = df[c].clip(upper=cap)

    df["our_price"] = _exposure_safe_numeric(df.get("our_price", 0.0)).clip(lower=0.0)
    df["scot_oos"]  = _exposure_safe_numeric(df.get("scot_oos",  0.0)).clip(0, 1)

    # ── 新增动态特征 ──────────────────────────────────────────
    # ind_promotion：动态binary，99.1% ASIN有变化，进active_head
    if "ind_promotion" in df.columns:
        df["ind_promotion"] = _exposure_safe_numeric(df["ind_promotion"]).clip(0, 1)
    else:
        df["ind_promotion"] = 0.0

    # ind_prime_week：动态binary，3.7%是PrimeDay周，进active_head
    if "ind_prime_week" in df.columns:
        df["ind_prime_week"] = _exposure_safe_numeric(df["ind_prime_week"]).clip(0, 1)
    else:
        df["ind_prime_week"] = 0.0

    # customer_active_review_count：动态，极度右偏，log变换后进mag_head
    if "customer_active_review_count" in df.columns:
        df["log_review_count"] = np.log1p(
            _exposure_safe_numeric(df["customer_active_review_count"]).clip(lower=0.0)
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
        df[c] = _exposure_safe_numeric(df[c])

    context_cols = list(dict.fromkeys(
        # ── 动态特征（时间驱动，进active_head）──────────────
        ["ind_promotion", "ind_prime_week"]
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
    TCN Decoder + Cross-Attention + SINGLE direct exposure head.

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

        # Auxiliary occurrence head. With ENN, z controls the 20-week active/zero regime.
        z_extra = d_model if self.use_enn else 0
        active_in = d_model + z_extra + max(active_feat_dim, 0)
        self.active_head = nn.Sequential(
            nn.Linear(active_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

        # Direct single-head exposure head. With ENN, z controls level/peak/zero regime.
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
        z_emb = None
        if self.use_enn:
            if z is None:
                z = torch.randn(B, self.z_dim, device=future_context.device, dtype=future_context.dtype)
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
        residual = self.direct_head(direct_in)  # [-1, 1]

        # Anchor-residual magnitude log forecast.
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
                "pred_level": pred_level,
                "gamma": nan_like,
                "gate": gate,
                "residual": residual,
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

    def __init__(self, input_dim, context_dim,
                 d_model=64, horizon=20, n_heads=4, dropout=0.10,
                 context_cols=None, use_encoder_self_attn=True,
                 use_enn=True, z_dim=8, residual_scale=2.0, gate_temperature=1.0):
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

        print(f"Active head feat dim: {len(active_feat_indices)}")
        print(f"Mag head feat dim:    {len(mag_feat_indices)}")

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
            active_feat_dim=len(active_feat_indices),
            mag_feat_dim=len(mag_feat_indices),
            use_enn=use_enn,
            z_dim=z_dim,
            residual_scale=residual_scale,
            gate_temperature=gate_temperature,
        )

    def forward(self, x, future_context, return_aux=False, z=None):
        enc_out = self.encoder(x)
        return self.decoder(enc_out, future_context, return_aux=return_aux, z=z)


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

def predict_exposure_v2(model, va_ld, apply_funnel_constraint=True, device=None, mc_samples=20, mc_reduce="median"):
    device = get_device(device)
    model = model.to(device)
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            b = batch_to_device(b, device)
            # MC inference over ENN z. Median is more robust than mean for exposure hats,
            # because mean can be pulled up by high-regime samples.
            preds, pacts, gates = [], [], []
            last_aux = None
            K = max(int(mc_samples), 1)
            for _ in range(K):
                aux = model(b["x"], b["future_context"], return_aux=True)
                last_aux = aux
                preds.append(torch.expm1(aux["log_hat"]).clamp(min=0.0))
                pacts.append(aux["p_active"])
                gates.append(aux.get("gate", torch.full_like(aux["p_active"], float("nan"))))

            pred_stack = torch.stack(preds, dim=0)
            pact_stack = torch.stack(pacts, dim=0)
            gate_stack = torch.stack(gates, dim=0)
            if mc_reduce == "mean":
                pred_t = pred_stack.mean(dim=0)
                pact_t = pact_stack.mean(dim=0)
                gate_t = gate_stack.mean(dim=0)
            else:
                pred_t = pred_stack.median(dim=0).values
                pact_t = pact_stack.mean(dim=0)
                gate_t = gate_stack.median(dim=0).values

            pred = pred_t.cpu().numpy()
            pact = pact_t.cpu().numpy()
            gamma_np = last_aux.get("gamma", torch.full_like(last_aux["p_active"], float("nan"))).cpu().numpy()
            gate_np = gate_t.cpu().numpy()

            if apply_funnel_constraint:
                pred[:, :, 1] = np.minimum(pred[:, :, 1], pred[:, :, 0])
                pred[:, :, 2] = np.minimum(pred[:, :, 2], pred[:, :, 1])

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
    out = pred_df[["asin", "order_week", "pred_total_dph", "pred_buy_box_dph", "pred_instock_dph"]].copy()
    out["external_total_dph_hat_log"]    = np.log1p(out["pred_total_dph"].clip(lower=0.0))
    out["external_buy_box_dph_hat_log"]  = np.log1p(out["pred_buy_box_dph"].clip(lower=0.0))
    out["external_instock_dph_hat_log"]  = np.log1p(out["pred_instock_dph"].clip(lower=0.0))
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
    dropout=0.20,    # 0.10→0.20，加强dropout防过拟合
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V2: TCN Full-Seq Encoder + Cross-Attn + SINGLE-HEAD DIRECT")
    print("Preset: category_code + softened zero-aware loss + stronger mean-level balance")
    print("=" * 100)

    df = prepare_exposure_data_from_sample(data_raw1, scot_df, n_asins, seed)
    if remove_extreme:
        df = filter_extreme_exposure_asins(df, q=extreme_q)

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
    )

    pred_df = predict_exposure_v2(model, va_ld, apply_funnel_constraint=apply_funnel_constraint)
    pred_df = add_naive_baselines_from_loader(pred_df, va_ld, context_cols)
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
    )

    pred_df = predict_exposure_v2(model, va_ld, apply_funnel_constraint=apply_funnel_constraint)
    pred_df = add_naive_baselines_from_loader(pred_df, va_ld, context_cols)
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
    dropout=0.20,
    use_scot_intersection=True,
    val_start_offset=0,
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V2: SINGLE-HEAD DIRECT + SCOT OPTION")
    print("=" * 100)

    if use_scot_intersection:
        df = prepare_exposure_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins, seed)
    else:
        df = prepare_exposure_data_from_sample(data_raw1, scot_df, n_asins, seed)

    if remove_extreme:
        df = filter_extreme_exposure_asins(df, q=extreme_q)

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
        dropout=dropout,
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

    out.update({
        "exposure_hat_for_demand": make_external_hat_df(pred_df),
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
        "source_df": df,
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
    dropout=0.20,
    use_scot_intersection=True,
    use_encoder_self_attn=True,
):
    print("\n" + "=" * 100)
    print("EXPOSURE MODEL V2: ROLLING BACKTEST + SCOT INTERSECTION")
    print("=" * 100)
    print(f"n_asins={n_asins} | history={history} | rolling_offsets={list(rolling_offsets)} | epochs={epochs} | patience={patience} | encoder_attn={use_encoder_self_attn}")

    if use_scot_intersection:
        df = prepare_exposure_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins, seed)
    else:
        df = prepare_exposure_data_from_sample(data_raw1, scot_df, n_asins, seed)

    if remove_extreme:
        df = filter_extreme_exposure_asins(df, q=extreme_q)

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
        "exposure_hat_for_demand": make_external_hat_df(latest_pred_df),
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
        "source_df": df,
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
        use_encoder_self_attn=use_encoder_self_attn,
    )

# ============================================================
# FINAL USAGE ONLY: exposure first, then pass hat dataframe to demand
# Demand file is NOT modified. This exposure file avoids helper-name collisions.
# ============================================================
# %run -i tcn_exposure_v2_enn_regime_singlehead_nogate_peakloss_gpu_FIXED3_SAFE_FOR_DEMAND.py
#
# exposure_result = run_exposure_v2_final_scot_5000(
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
# exposure_hat_for_demand = exposure_result["exposure_hat_for_demand"].copy()
# summarize_exposure_hat_for_demand(
#     exposure_hat_for_demand,
#     title="EXPOSURE HAT TO PASS INTO DEMAND",
# )
#
# # Then run the demand file and pass exposure_hat_for_demand.
# # %run -i demand_external_exposure3_clean_3modes.py
# demand_result_all3 = run_demand_with_predicted_exposure_all3(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     exposure_result_or_hat=exposure_hat_for_demand,
#     n_asins=5000,
#     history=52,
#     horizon=20,
#     epochs=60,
#     batch_size=64,
#     M_eval=100,
# )
