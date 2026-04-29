"""
utils/data_load.py  –  centralised data loading for Japan CPI analysis

Loads level_1.csv / level_2.csv / level_3.csv, extracts embedded weights,
computes growth rates and forward targets for regression.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LEVEL_DIR, START_DATE, HORIZON, COMPOSITE_COLS, SPECIAL_COLS
from utils.smart_imputation import smart_impute


HEADLINE_COL = 'All items'


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL-BASED DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_level_data(
    level: int,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Load a ``level_{level}.csv`` file, extracting embedded basket weights.

    These CSVs have a ``Weights`` row as the first data row, followed by
    YYYYMM-indexed CPI index levels.

    Parameters
    ----------
    level : int
        Hierarchy level (1, 2, or 3).
    start_date : str or None
        If provided, filter to rows >= this date.

    Returns
    -------
    df : pd.DataFrame
        Datetime-indexed, all-numeric CPI index levels.
    weights : dict[str, float]
        Column name -> basket weight (out of 10 000).
    """
    # Accept either naming convention: level_3.csv or level3.csv.
    path = LEVEL_DIR / f'level_{level}.csv'
    if not path.exists():
        path = LEVEL_DIR / f'level{level}.csv'
    raw = pd.read_csv(path, dtype=str)

    # first column is the date / label column
    date_col = raw.columns[0]
    cat_cols = raw.columns[1:]

    # extract the Weights row
    mask = raw[date_col].str.strip() == 'Weights'
    weights_row = raw.loc[mask, cat_cols].iloc[0]
    weights = {col: float(weights_row[col]) for col in cat_cols}

    # remaining rows are data (drop blank trailing rows)
    data = raw.loc[~mask].copy()
    data = data.dropna(subset=[date_col])
    data = data[data[date_col].str.strip().astype(bool)]
    data[date_col] = pd.to_datetime(data[date_col].str.strip(), format='%Y%m')
    data = (data.set_index(date_col)
                .replace('-', np.nan)
                .apply(pd.to_numeric, errors='coerce'))
    data.index.name = 'Date'

    if start_date:
        data = data[data.index >= start_date].copy()

    return data, weights


# ══════════════════════════════════════════════════════════════════════════════
#  GROWTH RATES & TARGET
# ══════════════════════════════════════════════════════════════════════════════

def compute_growth_3m3m(df: pd.DataFrame) -> pd.DataFrame:
    """3-month-over-3-month annualised growth rates (x 4 x 100)."""
    return df.apply(lambda s: ((s / s.shift(3)) - 1) * 4 * 100)


def compute_forward_target(headline_growth: pd.Series,
                           horizon: int = HORIZON) -> pd.Series:
    """
    12-month forward average of headline growth as regression target.

    target_t = mean(headline_{t+1}, ..., headline_{t+horizon})
    """
    target = pd.Series(np.nan, index=headline_growth.index)
    for i in range(len(headline_growth) - horizon):
        window = headline_growth.iloc[i + 1 : i + 1 + horizon]
        if window.notna().all():
            target.iloc[i] = window.mean()
    return target


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK SERIES
# ══════════════════════════════════════════════════════════════════════════════

_BENCHMARK_COLS = [
    'All items, less fresh food',
    'All items, less food (less alcoholic beverages) and energy',
    'All items, less fresh food and energy',   # used as 4th regressor in Xbm+
]


def load_benchmark_series(start_date: str = START_DATE) -> pd.DataFrame:
    """
    Return 3m/3m growth rates for the two core composite series only.

    'All items' is intentionally excluded — use the level's own headline
    series (from growth) to avoid data vintage mismatches around tax events.
    Only the 2 needed columns are loaded and imputed — not the full 700+.
    """
    path = LEVEL_DIR / 'level_3_full.csv'
    raw = pd.read_csv(path, dtype=str)
    date_col = raw.columns[0]

    # keep only the columns we need
    keep = [c for c in _BENCHMARK_COLS if c in raw.columns]
    raw = raw[[date_col] + keep]

    mask = raw[date_col].str.strip() == 'Weights'
    data = raw.loc[~mask].copy()
    data = data.dropna(subset=[date_col])
    data = data[data[date_col].str.strip().astype(bool)]
    data[date_col] = pd.to_datetime(data[date_col].str.strip(), format='%Y%m')
    data = (data.set_index(date_col)
                .replace('-', np.nan)
                .apply(pd.to_numeric, errors='coerce'))
    data.index.name = 'Date'
    if start_date:
        data = data[data.index >= start_date]

    df_imp, _ = smart_impute(data, strategy='auto', verbose=False)
    return compute_growth_3m3m(df_imp)


# ══════════════════════════════════════════════════════════════════════════════
#  TRIMMED-MEAN INFLATION (constructed from disaggregated level3.csv)
# ══════════════════════════════════════════════════════════════════════════════

def load_trimmed_mean_3m3m(
    start_date: str = START_DATE,
    trim_pct: float = 10.0,
    level3_file: str = 'level3.csv',
) -> pd.Series:
    """
    Construct a Japan trimmed-mean inflation series from `level3.csv` (the
    cleaned 647 disaggregated subindices with embedded basket weights). No
    external data required.

    Per month, components are sorted by m/m (annualised) % change and the
    bottom and top `trim_pct`% of the basket weight are dropped; the basket-
    weighted mean of the remaining components is the trimmed-mean inflation
    rate. The resulting m/m-annualised series is then 3-month-MA smoothed to
    yield a 3m/3m-equivalent benchmark series — comparable to the other 3m/3m
    Xbm regressors. (For small log-returns, mean of 3 consecutive m/m
    annualised rates ≈ 3m/3m annualised; exact in continuous time.)

    Default `trim_pct=10` matches the Bank of Japan's 10% trimmed-mean CPI.
    """
    path = LEVEL_DIR / level3_file
    raw = pd.read_csv(path, dtype=str)
    date_col = raw.columns[0]
    cat_cols = raw.columns[1:]

    mask = raw[date_col].str.strip() == 'Weights'
    weights = pd.to_numeric(raw.loc[mask, cat_cols].iloc[0],
                            errors='coerce').astype(float)

    data = raw.loc[~mask].copy()
    data = data.dropna(subset=[date_col])
    data = data[data[date_col].str.strip().astype(bool)]
    data[date_col] = pd.to_datetime(data[date_col].str.strip(), format='%Y%m')
    data = (data.set_index(date_col)
                .replace('-', np.nan)
                .apply(pd.to_numeric, errors='coerce'))
    data.index.name = 'Date'

    # drop headline aggregate so trimming is over true subcomponents only
    drop = [c for c in (HEADLINE_COL,) if c in data.columns]
    data = data.drop(columns=drop, errors='ignore')
    weights = weights.drop(labels=drop, errors='ignore')

    df_imp, _ = smart_impute(data, strategy='auto', verbose=False)
    growth_mm = ((df_imp / df_imp.shift(1)) - 1.0) * 12.0 * 100.0

    cols = [c for c in growth_mm.columns
            if c in weights.index and weights[c] > 0 and np.isfinite(weights[c])]
    G = growth_mm[cols].to_numpy()
    w = weights.reindex(cols).to_numpy()
    w = w / w.sum()

    trim = trim_pct / 100.0
    out = np.full(G.shape[0], np.nan)
    for t in range(G.shape[0]):
        valid = np.isfinite(G[t])
        if not valid.any():
            continue
        g = G[t, valid]
        wv = w[valid]
        wv = wv / wv.sum()
        order = np.argsort(g)
        g_s, w_s = g[order], wv[order]
        cum = np.cumsum(w_s)
        keep = (cum > trim) & (cum < 1.0 - trim)
        if keep.any():
            ww = w_s[keep] / w_s[keep].sum()
            out[t] = float(np.dot(ww, g_s[keep]))

    tm_mm = pd.Series(out, index=growth_mm.index, name=f'Trimmed mean ({int(trim_pct)}%)')
    tm_3m = tm_mm.rolling(3, min_periods=3).mean()

    if start_date:
        tm_3m = tm_3m[tm_3m.index >= start_date]
    return tm_3m


# ══════════════════════════════════════════════════════════════════════════════
#  FULL REGRESSION DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def prepare_regression_data(
    level: int = 2,
    start_date: str = START_DATE,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DatetimeIndex, pd.DataFrame]:
    """
    Load level CSV -> impute -> 3m/3m growth rates -> 12m forward target.

    Parameters
    ----------
    level      : int        which level file to load (1, 2, or 3)
    start_date : str        regression window start
    horizon    : int        months-ahead target

    Returns
    -------
    X        : ndarray (T, K)    component 3m/3m growth rates
    y        : ndarray (T,)      12m forward average of headline 3m/3m
    w_prior  : ndarray (K,)      prior weights normalised to sum=1
    features : list[str]         component column names
    dates    : DatetimeIndex     corresponding dates
    growth   : DataFrame         full growth-rate frame (incl. headline)
    """
    df, weights_dict = load_level_data(level, start_date)

    # impute missing values
    df_imp, _ = smart_impute(df, strategy='auto', verbose=False)

    # 3m/3m growth rates
    growth = compute_growth_3m3m(df_imp)

    # forward target from headline (first column = 'All items')
    headline_col = df.columns[0]
    target = compute_forward_target(growth[headline_col], horizon)

    # component columns = everything except headline and known composites/aggregates
    # Also drop pandas auto-named placeholder columns from trailing-comma CSVs.
    _exclude = set(COMPOSITE_COLS) | set(SPECIAL_COLS) | {headline_col}
    features = [c for c in df.columns
                if c not in _exclude and not c.startswith('Unnamed:')]

    valid = target.notna() & growth[headline_col].notna()
    X_df = growth[features][valid]

    if X_df.isna().any().any():
        X_df = X_df.fillna(X_df.median())

    # prior weights from embedded Weights row, normalised to sum=1
    w_raw = np.array([weights_dict[c] for c in features], dtype=float)
    w_prior = w_raw / w_raw.sum()

    return X_df.values, target[valid].values, w_prior, features, X_df.index, growth
