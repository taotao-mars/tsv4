# Dynamic ASIN relation examples from raw data
# Goal: find examples where ASIN pair relationship changes over time inside same category / GL.
# Uses history only before each origin_week, no future leakage.

import numpy as np
import pandas as pd
from itertools import combinations


def _safe_log1p(x):
    return np.log1p(np.clip(pd.to_numeric(x, errors='coerce').fillna(0).astype(float), 0, None))


def _zscore_by_group(s, g):
    m = s.groupby(g).transform('mean')
    sd = s.groupby(g).transform('std').replace(0, np.nan)
    return ((s - m) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rank_pct_by_group(s, g, ascending=False):
    # 0 = strongest/highest if ascending=False, 1 = weakest
    r = s.groupby(g).rank(method='average', pct=True, ascending=ascending)
    return r.fillna(1.0)


def build_dynamic_asin_profiles(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    history_weeks=13,
    long_weeks=52,
    horizon_gap=0,
    sample_random_state=42,
    use_scot_intersection=True,
):
    """
    Build ASIN profiles at each origin_week using only rows <= origin_week - horizon_gap.
    Profile is one row per (origin_week, asin). It is used to compare ASIN distances dynamically.
    """
    df = data_raw1.copy()
    required = ['asin', 'order_week', 'category_code']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # normalize week as datetime if possible
    df['order_week'] = pd.to_datetime(df['order_week'])

    # SCOT intersection if provided
    if scot_df is not None and use_scot_intersection:
        sc = scot_df.copy()
        if 'asin' not in sc.columns:
            raise ValueError("scot_df must contain asin column")
        df = df[df['asin'].isin(sc['asin'].dropna().unique())].copy()

    # sample ASINs after SCOT filter
    all_asins = pd.Series(df['asin'].dropna().unique())
    if n_asins is not None and len(all_asins) > n_asins:
        all_asins = all_asins.sample(n=n_asins, random_state=sample_random_state)
    df = df[df['asin'].isin(set(all_asins))].copy()

    # fill optional columns
    for c in ['total_dph', 'buy_box_dph', 'in_stock_dph', 'fbi_demand', 'order_units']:
        if c not in df.columns:
            df[c] = 0.0
    # demand target fallback
    if 'fbi_demand' not in data_raw1.columns and 'order_units' in data_raw1.columns:
        df['fbi_demand'] = df['order_units']

    # user said use our_price not list_price
    if 'ourprice' in df.columns and 'our_price' not in df.columns:
        df['our_price'] = df['ourprice']
    if 'our_price' not in df.columns:
        if 'list_price' in df.columns:
            df['our_price'] = df['list_price']
        else:
            df['our_price'] = 0.0

    if 'ind_promotion' not in df.columns:
        # fallback from promotion_ratio/amount/type if present
        if 'promotion_ratio' in df.columns:
            df['ind_promotion'] = (pd.to_numeric(df['promotion_ratio'], errors='coerce').fillna(0) > 0).astype(int)
        elif 'promotion_amount' in df.columns:
            df['ind_promotion'] = (pd.to_numeric(df['promotion_amount'], errors='coerce').fillna(0) > 0).astype(int)
        else:
            df['ind_promotion'] = 0

    for c in ['hbt', 'ind_top10_brand', 'is_head_brand', 'brand_class', 'gl_product_group', 'gl_product_group_desc']:
        if c not in df.columns:
            df[c] = 'unknown' if c in ['hbt', 'brand_class', 'gl_product_group_desc'] else 0

    # aggregate weekly ASIN values
    agg = {
        'total_dph': 'sum',
        'buy_box_dph': 'sum',
        'in_stock_dph': 'sum',
        'fbi_demand': 'sum',
        'our_price': 'mean',
        'ind_promotion': 'max',
        'category_code': 'last',
        'gl_product_group': 'last',
        'gl_product_group_desc': 'last',
        'hbt': 'last',
        'ind_top10_brand': 'max',
        'is_head_brand': 'max',
        'brand_class': 'last',
    }
    w = df.groupby(['asin', 'order_week'], as_index=False).agg(agg)
    w = w.sort_values(['asin', 'order_week']).reset_index(drop=True)

    all_weeks = np.array(sorted(w['order_week'].unique()))
    origins = all_weeks[long_weeks + horizon_gap:] if len(all_weeks) > long_weeks + horizon_gap else all_weeks[history_weeks + horizon_gap:]
    profiles = []

    for origin in origins:
        cutoff = origin - np.timedelta64(7*horizon_gap, 'D')
        hist = w[w['order_week'] < cutoff].copy()
        if hist.empty:
            continue
        recent_cut = cutoff - np.timedelta64(7*history_weeks, 'D')
        long_cut = cutoff - np.timedelta64(7*long_weeks, 'D')
        recent = hist[hist['order_week'] >= recent_cut]
        longh = hist[hist['order_week'] >= long_cut]

        def summarize(x, prefix):
            if x.empty:
                return pd.DataFrame(columns=['asin'])
            out = x.groupby('asin').agg(
                **{
                    f'{prefix}_total_sum': ('total_dph', 'sum'),
                    f'{prefix}_buybox_sum': ('buy_box_dph', 'sum'),
                    f'{prefix}_instock_sum': ('in_stock_dph', 'sum'),
                    f'{prefix}_demand_sum': ('fbi_demand', 'sum'),
                    f'{prefix}_active_rate': ('fbi_demand', lambda s: float((s > 0).mean())),
                    f'{prefix}_zero_rate': ('fbi_demand', lambda s: float((s <= 0).mean())),
                    f'{prefix}_promo_rate': ('ind_promotion', 'mean'),
                    f'{prefix}_price_mean': ('our_price', 'mean'),
                }
            ).reset_index()
            return out

        pr = summarize(recent, 'r13')
        pl = summarize(longh, 'l52')
        static = hist.sort_values('order_week').groupby('asin').tail(1)[[
            'asin', 'category_code', 'gl_product_group', 'gl_product_group_desc',
            'hbt', 'ind_top10_brand', 'is_head_brand', 'brand_class'
        ]].drop_duplicates('asin')
        prof = static.merge(pr, on='asin', how='left').merge(pl, on='asin', how='left')
        num_cols = [c for c in prof.columns if c.startswith('r13_') or c.startswith('l52_')]
        prof[num_cols] = prof[num_cols].fillna(0.0)
        prof['origin_week'] = pd.Timestamp(origin)

        # dynamic composite rank score: high = strong
        prof['score_level'] = (
            0.25*_safe_log1p(prof['r13_total_sum']) +
            0.25*_safe_log1p(prof['r13_buybox_sum']) +
            0.25*_safe_log1p(prof['r13_instock_sum']) +
            0.25*_safe_log1p(prof['r13_demand_sum'])
        )
        prof['score_long'] = (
            0.25*_safe_log1p(prof['l52_total_sum']) +
            0.25*_safe_log1p(prof['l52_buybox_sum']) +
            0.25*_safe_log1p(prof['l52_instock_sum']) +
            0.25*_safe_log1p(prof['l52_demand_sum'])
        )
        prof['score_promo'] = prof['r13_promo_rate'].fillna(0)
        prof['score_active'] = prof['r13_active_rate'].fillna(0)
        prof['score_dynamic'] = 0.55*prof['score_level'] + 0.25*prof['score_long'] + 0.10*prof['score_promo'] + 0.10*prof['score_active']
        prof['cat_rank_pct'] = _rank_pct_by_group(prof['score_dynamic'], prof['category_code'], ascending=False)
        profiles.append(prof)

    profiles = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    return profiles


def _pair_scores(a, b):
    """Return distances and relation scores for two ASIN profile rows."""
    eps = 1e-9
    # log scale gaps using recent 13 and long 52
    feature_keys = ['total_sum', 'buybox_sum', 'instock_sum', 'demand_sum']
    gaps = {}
    for pref in ['r13', 'l52']:
        for k in feature_keys:
            ai = np.log1p(float(a.get(f'{pref}_{k}', 0) or 0))
            bj = np.log1p(float(b.get(f'{pref}_{k}', 0) or 0))
            gaps[f'{pref}_{k}_gap'] = abs(ai - bj)

    active_gap = abs(float(a.get('r13_active_rate',0))-float(b.get('r13_active_rate',0)))
    zero_gap = abs(float(a.get('r13_zero_rate',0))-float(b.get('r13_zero_rate',0)))
    promo_gap = abs(float(a.get('r13_promo_rate',0))-float(b.get('r13_promo_rate',0)))
    price_i = np.log1p(max(float(a.get('r13_price_mean',0) or 0), 0))
    price_j = np.log1p(max(float(b.get('r13_price_mean',0) or 0), 0))
    price_gap = abs(price_i - price_j)
    hbt_same = float(str(a.get('hbt','')) == str(b.get('hbt','')))
    top10_same = float(a.get('ind_top10_brand',0) == b.get('ind_top10_brand',0))
    head_same = float(a.get('is_head_brand',0) == b.get('is_head_brand',0))
    rank_gap = abs(float(a.get('cat_rank_pct',1))-float(b.get('cat_rank_pct',1)))
    score_gap = abs(float(a.get('score_dynamic',0))-float(b.get('score_dynamic',0)))

    mag_gap = np.mean([
        gaps['r13_total_sum_gap'], gaps['r13_buybox_sum_gap'], gaps['r13_instock_sum_gap'], gaps['r13_demand_sum_gap'],
        0.5*gaps['l52_total_sum_gap'], 0.5*gaps['l52_demand_sum_gap']
    ])
    behavior_gap = np.mean([active_gap, zero_gap, promo_gap, min(price_gap/2.0, 2.0), rank_gap])
    distance = 0.65*mag_gap + 0.35*behavior_gap

    # interpretable scores, not model scores
    positive_score = (
        1.8*np.exp(-distance)
        + 0.25*hbt_same + 0.15*top10_same + 0.10*head_same
        - 0.60*rank_gap - 0.25*promo_gap
    )
    competitive_score = (
        0.75*rank_gap + 0.35*score_gap + 0.25*price_gap
        + 0.25*(1-hbt_same) + 0.25*(1-top10_same)
        + 0.20*promo_gap
    )
    # normalize-ish
    positive_score = float(positive_score)
    competitive_score = float(competitive_score)

    if positive_score >= 1.05 and rank_gap <= 0.30 and distance <= 1.60:
        label = 'positive'
    elif competitive_score >= 0.85 and rank_gap >= 0.25:
        label = 'competitive'
    else:
        label = 'neutral'

    out = dict(
        distance=distance,
        mag_gap=mag_gap,
        behavior_gap=behavior_gap,
        rank_gap=rank_gap,
        score_gap=score_gap,
        positive_score=positive_score,
        competitive_score=competitive_score,
        relation_label=label,
        hbt_same=hbt_same,
        top10_same=top10_same,
        head_same=head_same,
        active_gap=active_gap,
        zero_gap=zero_gap,
        promo_gap=promo_gap,
        price_log_gap=price_gap,
        **gaps,
    )
    return out


def build_dynamic_pair_relations(
    profiles,
    max_pairs_per_category_per_origin=2000,
    min_asins_per_category=8,
    category_filter=None,
    random_state=42,
):
    """Build dynamic relation rows for ASIN pairs inside same category at each origin."""
    rng = np.random.default_rng(random_state)
    rows = []
    if profiles.empty:
        return pd.DataFrame()

    for (origin, cat), g in profiles.groupby(['origin_week', 'category_code']):
        if category_filter is not None and cat not in set(category_filter if isinstance(category_filter, (list,set,tuple)) else [category_filter]):
            continue
        if len(g) < min_asins_per_category:
            continue
        g = g.reset_index(drop=True)
        idxs = list(range(len(g)))
        pairs = list(combinations(idxs, 2))
        if len(pairs) > max_pairs_per_category_per_origin:
            sel = rng.choice(len(pairs), size=max_pairs_per_category_per_origin, replace=False)
            pairs = [pairs[int(i)] for i in sel]
        for i, j in pairs:
            a = g.iloc[i]
            b = g.iloc[j]
            s = _pair_scores(a, b)
            rows.append({
                'origin_week': origin,
                'category_code': cat,
                'gl_product_group_i': a.get('gl_product_group'),
                'gl_product_group_j': b.get('gl_product_group'),
                'gl_product_group_desc_i': a.get('gl_product_group_desc'),
                'gl_product_group_desc_j': b.get('gl_product_group_desc'),
                'asin_i': a['asin'],
                'asin_j': b['asin'],
                'hbt_i': a.get('hbt'),
                'hbt_j': b.get('hbt'),
                'top10_i': a.get('ind_top10_brand'),
                'top10_j': b.get('ind_top10_brand'),
                'score_i': a.get('score_dynamic'),
                'score_j': b.get('score_dynamic'),
                'rank_i': a.get('cat_rank_pct'),
                'rank_j': b.get('cat_rank_pct'),
                'r13_total_i': a.get('r13_total_sum'),
                'r13_total_j': b.get('r13_total_sum'),
                'r13_buybox_i': a.get('r13_buybox_sum'),
                'r13_buybox_j': b.get('r13_buybox_sum'),
                'r13_instock_i': a.get('r13_instock_sum'),
                'r13_instock_j': b.get('r13_instock_sum'),
                'r13_demand_i': a.get('r13_demand_sum'),
                'r13_demand_j': b.get('r13_demand_sum'),
                'r13_promo_i': a.get('r13_promo_rate'),
                'r13_promo_j': b.get('r13_promo_rate'),
                'r13_price_i': a.get('r13_price_mean'),
                'r13_price_j': b.get('r13_price_mean'),
                **s,
            })
    return pd.DataFrame(rows)


def find_relation_switch_examples(pair_df, top_n=30):
    """Find same ASIN pair whose relation label changes over time."""
    if pair_df.empty:
        return pd.DataFrame()
    df = pair_df.sort_values(['category_code', 'asin_i', 'asin_j', 'origin_week']).copy()
    # canonical pair id
    df['pair_id'] = df['asin_i'].astype(str) + '||' + df['asin_j'].astype(str)
    df['prev_label'] = df.groupby('pair_id')['relation_label'].shift(1)
    df['prev_origin_week'] = df.groupby('pair_id')['origin_week'].shift(1)
    df['prev_distance'] = df.groupby('pair_id')['distance'].shift(1)
    df['prev_positive_score'] = df.groupby('pair_id')['positive_score'].shift(1)
    df['prev_competitive_score'] = df.groupby('pair_id')['competitive_score'].shift(1)
    sw = df[df['prev_label'].notna() & (df['relation_label'] != df['prev_label'])].copy()
    if sw.empty:
        return sw
    sw['delta_distance'] = sw['distance'] - sw['prev_distance']
    sw['delta_pos_score'] = sw['positive_score'] - sw['prev_positive_score']
    sw['delta_comp_score'] = sw['competitive_score'] - sw['prev_competitive_score']
    # prefer positive<->competitive switches and large score changes
    sw['switch_importance'] = (
        ((sw['prev_label'].isin(['positive','competitive'])) & (sw['relation_label'].isin(['positive','competitive']))).astype(int) * 10
        + sw['delta_distance'].abs().fillna(0)
        + sw['delta_pos_score'].abs().fillna(0)
        + sw['delta_comp_score'].abs().fillna(0)
    )
    cols = [
        'category_code','gl_product_group_i','gl_product_group_desc_i','asin_i','asin_j',
        'prev_origin_week','origin_week','prev_label','relation_label',
        'prev_distance','distance','delta_distance',
        'prev_positive_score','positive_score','delta_pos_score',
        'prev_competitive_score','competitive_score','delta_comp_score',
        'rank_i','rank_j','rank_gap','r13_total_i','r13_total_j','r13_buybox_i','r13_buybox_j',
        'r13_instock_i','r13_instock_j','r13_demand_i','r13_demand_j','r13_promo_i','r13_promo_j',
        'r13_price_i','r13_price_j','hbt_i','hbt_j','top10_i','top10_j',
        'mag_gap','behavior_gap','promo_gap','price_log_gap','active_gap','zero_gap'
    ]
    cols = [c for c in cols if c in sw.columns]
    return sw.sort_values('switch_importance', ascending=False)[cols].head(top_n).reset_index(drop=True)


def find_anchor_neighbor_switch_examples(pair_df, top_n=30, relation='positive'):
    """
    Find examples: for same anchor ASIN, best positive/competitive neighbor changes over time.
    Example: ASIN1's positive neighbor changes from ASIN2 to ASIN3.
    """
    if pair_df.empty:
        return pd.DataFrame()
    df = pair_df.copy()
    # make directed rows so each pair contributes both anchors
    left = df.rename(columns={'asin_i':'anchor_asin','asin_j':'neighbor_asin', 'score_i':'anchor_score','score_j':'neighbor_score', 'rank_i':'anchor_rank','rank_j':'neighbor_rank'})
    right = df.rename(columns={'asin_j':'anchor_asin','asin_i':'neighbor_asin', 'score_j':'anchor_score','score_i':'neighbor_score', 'rank_j':'anchor_rank','rank_i':'neighbor_rank'})
    d = pd.concat([left, right], ignore_index=True)
    if relation == 'positive':
        cand = d[d['relation_label'] == 'positive'].copy()
        score_col = 'positive_score'
    elif relation == 'competitive':
        cand = d[d['relation_label'] == 'competitive'].copy()
        score_col = 'competitive_score'
    else:
        cand = d.copy()
        score_col = 'positive_score'
    if cand.empty:
        return cand
    cand = cand.sort_values(['origin_week','anchor_asin', score_col], ascending=[True, True, False])
    best = cand.groupby(['origin_week','anchor_asin']).head(1).copy()
    best = best.sort_values(['anchor_asin','origin_week'])
    best['prev_neighbor_asin'] = best.groupby('anchor_asin')['neighbor_asin'].shift(1)
    best['prev_origin_week'] = best.groupby('anchor_asin')['origin_week'].shift(1)
    best['prev_relation_label'] = best.groupby('anchor_asin')['relation_label'].shift(1)
    best['prev_score'] = best.groupby('anchor_asin')[score_col].shift(1)
    changed = best[best['prev_neighbor_asin'].notna() & (best['neighbor_asin'] != best['prev_neighbor_asin'])].copy()
    if changed.empty:
        return changed
    changed['score_change'] = changed[score_col] - changed['prev_score']
    cols = [
        'category_code','gl_product_group_i','gl_product_group_desc_i','anchor_asin',
        'prev_origin_week','origin_week','prev_neighbor_asin','neighbor_asin',
        'relation_label', score_col, 'prev_score','score_change',
        'distance','rank_gap','anchor_rank','neighbor_rank',
        'r13_total_i','r13_total_j','r13_buybox_i','r13_buybox_j','r13_instock_i','r13_instock_j','r13_demand_i','r13_demand_j',
        'r13_promo_i','r13_promo_j','r13_price_i','r13_price_j','hbt_i','hbt_j','top10_i','top10_j',
    ]
    cols = [c for c in cols if c in changed.columns]
    return changed.sort_values('score_change', key=lambda s: s.abs(), ascending=False)[cols].head(top_n).reset_index(drop=True)


def run_dynamic_relation_example_search(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    history_weeks=13,
    long_weeks=52,
    max_pairs_per_category_per_origin=1500,
    min_asins_per_category=8,
    top_n=30,
    category_filter=None,
    random_state=42,
    save_prefix='dynamic_asin_relation_examples',
):
    print("="*90)
    print("DYNAMIC ASIN RELATION EXAMPLE SEARCH")
    print("Uses historical raw data only before each origin_week.")
    print("Looks for same-category ASIN-pair relation switches over time.")
    print("="*90)

    profiles = build_dynamic_asin_profiles(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        history_weeks=history_weeks,
        long_weeks=long_weeks,
        sample_random_state=random_state,
    )
    print(f"Profiles shape: {profiles.shape}")
    print(f"Origins: {profiles['origin_week'].nunique() if not profiles.empty else 0}, ASINs: {profiles['asin'].nunique() if not profiles.empty else 0}, categories: {profiles['category_code'].nunique() if not profiles.empty else 0}")

    pair_df = build_dynamic_pair_relations(
        profiles,
        max_pairs_per_category_per_origin=max_pairs_per_category_per_origin,
        min_asins_per_category=min_asins_per_category,
        category_filter=category_filter,
        random_state=random_state,
    )
    print(f"Pair relation rows: {pair_df.shape}")
    if pair_df.empty:
        print("No pair rows found. Try lowering min_asins_per_category or increasing n_asins.")
        return {'profiles': profiles, 'pair_df': pair_df}

    print("\nRelation label distribution:")
    print(pair_df['relation_label'].value_counts(dropna=False))
    print("\nTop categories by pair rows:")
    print(pair_df.groupby(['category_code','gl_product_group_i']).size().sort_values(ascending=False).head(15))

    switches = find_relation_switch_examples(pair_df, top_n=top_n)
    pos_neighbor_switches = find_anchor_neighbor_switch_examples(pair_df, top_n=top_n, relation='positive')
    comp_neighbor_switches = find_anchor_neighbor_switch_examples(pair_df, top_n=top_n, relation='competitive')

    print("\n" + "="*90)
    print("PAIR LABEL SWITCH EXAMPLES: same ASIN_i / ASIN_j changes relation over time")
    print("="*90)
    if switches.empty:
        print("No label switches found. Try increasing max_pairs_per_category_per_origin or top_n.")
    else:
        print(switches.head(top_n).to_string(index=False))

    print("\n" + "="*90)
    print("ANCHOR POSITIVE NEIGHBOR SWITCH EXAMPLES: ASIN1 positive neighbor changes from ASIN2 to ASIN3")
    print("="*90)
    if pos_neighbor_switches.empty:
        print("No positive-neighbor switches found.")
    else:
        print(pos_neighbor_switches.head(top_n).to_string(index=False))

    print("\n" + "="*90)
    print("ANCHOR COMPETITIVE NEIGHBOR SWITCH EXAMPLES")
    print("="*90)
    if comp_neighbor_switches.empty:
        print("No competitive-neighbor switches found.")
    else:
        print(comp_neighbor_switches.head(top_n).to_string(index=False))

    # save csvs
    try:
        pair_path = f"{save_prefix}_pair_relations.csv"
        switch_path = f"{save_prefix}_pair_switches.csv"
        pos_path = f"{save_prefix}_positive_neighbor_switches.csv"
        comp_path = f"{save_prefix}_competitive_neighbor_switches.csv"
        pair_df.to_csv(pair_path, index=False)
        switches.to_csv(switch_path, index=False)
        pos_neighbor_switches.to_csv(pos_path, index=False)
        comp_neighbor_switches.to_csv(comp_path, index=False)
        print("\nSaved CSVs:")
        print(pair_path)
        print(switch_path)
        print(pos_path)
        print(comp_path)
    except Exception as e:
        print(f"CSV save skipped due to error: {e}")

    return {
        'profiles': profiles,
        'pair_df': pair_df,
        'pair_switches': switches,
        'positive_neighbor_switches': pos_neighbor_switches,
        'competitive_neighbor_switches': comp_neighbor_switches,
    }


# =========================
# USAGE IN JUPYTER
# =========================
# %run -i dynamic_asin_relation_examples_rawdata_v1.py
#
# dynamic_relation_result = run_dynamic_relation_example_search(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     history_weeks=13,
#     long_weeks=52,
#     max_pairs_per_category_per_origin=1500,
#     min_asins_per_category=8,
#     top_n=30,
#     category_filter=None,   # or set to a category_code like 149
#     save_prefix='dynamic_asin_relation_examples',
# )
#
# pair_switches = dynamic_relation_result['pair_switches']
# positive_neighbor_switches = dynamic_relation_result['positive_neighbor_switches']
# competitive_neighbor_switches = dynamic_relation_result['competitive_neighbor_switches']
# pair_df = dynamic_relation_result['pair_df']
