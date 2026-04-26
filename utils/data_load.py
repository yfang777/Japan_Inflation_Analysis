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
    path = LEVEL_DIR / f'level_{level}.csv'
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
    'All items',
    'All items, less fresh food',
    'All items, less food (less alcoholic beverages) and energy',
]


def load_benchmark_series(start_date: str = START_DATE) -> pd.DataFrame:
    """
    Return 3m/3m growth rates for headline + two core composite series.

    These composites exist in level_3.csv regardless of which level is
    used for the regression, so benchmarks always work.
    """
    df, _ = load_level_data(3, start_date)
    df_imp, _ = smart_impute(df, strategy='auto', verbose=False)
    growth = compute_growth_3m3m(df_imp)
    cols = [c for c in _BENCHMARK_COLS if c in growth.columns]
    return growth[cols]


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
    _exclude = set(COMPOSITE_COLS) | set(SPECIAL_COLS) | {headline_col}
    features = [c for c in df.columns if c not in _exclude]

    valid = target.notna() & growth[headline_col].notna()
    X_df = growth[features][valid]

    if X_df.isna().any().any():
        X_df = X_df.fillna(X_df.median())

    # prior weights from embedded Weights row, normalised to sum=1
    w_raw = np.array([weights_dict[c] for c in features], dtype=float)
    w_prior = w_raw / w_raw.sum()

    return X_df.values, target[valid].values, w_prior, features, X_df.index, growth
