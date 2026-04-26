"""
utils/data_load.py  –  centralised data loading for Japan CPI analysis

Provides raw data loading, basket weight lookup, growth-rate computation,
forward-target construction, and a one-call regression data pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LEVEL_DIR, START_DATE, HORIZON
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
        Column name → basket weight (out of 10 000).
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

    # remaining rows are data
    data = raw.loc[~mask].copy()
    data[date_col] = pd.to_datetime(data[date_col].str.strip(), format='%Y%m')
    data = (data.set_index(date_col)
                .replace('-', np.nan)
                .apply(pd.to_numeric, errors='coerce'))
    data.index.name = 'Date'

    if start_date:
        data = data[data.index >= start_date].copy()

    return data, weights


# ══════════════════════════════════════════════════════════════════════════════
#  BASKET WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def load_basket_weights(features: list[str] | None = None,
                        include_rent: bool = False) -> np.ndarray | dict[str, float]:
    """
    Official basket weights from the Japan Statistics Bureau.

    Parameters
    ----------
    features : list[str] or None
        If provided, return an ndarray of weights for these features,
        renormalised to sum to 1 (regression use-case).
        If None, return a dict {english_name: basket_share} for the
        46 active components (EDA use-case).
    include_rent : bool
        Only used when features is None.  If True, include Rent in the dict.

    Returns
    -------
    np.ndarray or dict[str, float]
    """
    ow = pd.read_csv(WEIGHTS_CSV)
    jpn_to_raw = dict(zip(ow['Category_Name'], ow['Weight']))
    total = jpn_to_raw['総合']

    if features is not None:
        raw = np.array([
            jpn_to_raw.get(EN_TO_JPN.get(col, ''), np.nan)
            for col in features
        ])
        if np.isnan(raw).any():
            missing = [features[i] for i in np.where(np.isnan(raw))[0]]
            print(f'  Warning: no official weight for {missing}; using equal fallback.')
            raw = np.where(np.isnan(raw), np.nanmean(raw), raw)
        shares = raw / total
        return shares / shares.sum()
    else:
        cols = COMPONENT_COLS + (['Rent'] if include_rent else [])
        return {
            eng: jpn_to_raw[EN_TO_JPN[eng]] / total
            for eng in cols
            if EN_TO_JPN.get(eng) in jpn_to_raw
        }


# ══════════════════════════════════════════════════════════════════════════════
#  GROWTH RATES & TARGET
# ══════════════════════════════════════════════════════════════════════════════

def compute_growth_3m3m(df: pd.DataFrame) -> pd.DataFrame:
    """3-month-over-3-month annualised growth rates (× 4 × 100)."""
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
#  FULL REGRESSION DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def prepare_regression_data(
    start_date: str = START_DATE,
    horizon: int = HORIZON,
    features: list[str] = FEATURES,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    """
    Load raw CPI index → impute → 3m/3m growth rates → 12m forward target.

    Returns
    -------
    X      : ndarray (T, K)       component 3m/3m growth rates
    y      : ndarray (T,)         12m forward average of headline 3m/3m
    dates  : DatetimeIndex        corresponding dates
    growth : DataFrame            full growth-rate frame (incl. composites for benchmarks)
    """
    df = load_raw_data(start_date)

    # columns to impute: features + benchmark composites
    benchmark_cols = [
        'All items',
        'All items, less fresh food',
        'All items, less food (less alcoholic beverages) and energy',
    ]
    cols_to_impute = list(dict.fromkeys(features + benchmark_cols))
    df_imp, _ = smart_impute(df[cols_to_impute], strategy='auto', verbose=False)

    growth = compute_growth_3m3m(df_imp)
    target = compute_forward_target(growth['All items'], horizon)

    valid = target.notna() & growth['All items'].notna()
    X_df = growth[features][valid]

    if X_df.isna().any().any():
        X_df = X_df.fillna(X_df.median())

    return X_df.values, target[valid].values, X_df.index, growth
