"""
regression/benchmarks.py  –  naive and OLS benchmark models

Three naive predictors (random walk, core ex fresh food, supercore)
plus an unconditional-mean floor and an OLS combination benchmark.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_TRAIN


# ══════════════════════════════════════════════════════════════════════════════
#  NAIVE BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def compute_benchmarks(growth: pd.DataFrame,
                       oos_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Naive benchmarks aligned to OOS dates.

    Each benchmark uses only information available at time t to predict y_t
    (the 12-month forward average of headline inflation).

    Three predictors:
      Random walk         — current 3m/3m headline (most naive)
      Core ex fresh food  — Japan's standard published core
      Core ex food&energy — the 'supercore'
    """
    col_map = {
        'Random walk':           'All items',
        'Core (ex fresh food)':  'All items, less fresh food',
        'Core (ex food&energy)': 'All items, less food (less alcoholic beverages) and energy',
    }
    bm = {name: growth[col].reindex(oos_dates)
          for name, col in col_map.items()
          if col in growth.columns}
    return pd.DataFrame(bm)


def compute_mean_benchmark(y: np.ndarray, dates: pd.DatetimeIndex,
                           oos_dates: pd.DatetimeIndex,
                           min_train: int = MIN_TRAIN) -> pd.Series:
    """
    Unconditional mean benchmark: predict mean(y[0:t]) at each OOS step.
    This is the R²=0 floor — any model that can't beat this is useless.
    """
    records = []
    for t, date in enumerate(dates):
        if date not in oos_dates:
            continue
        idx = np.where(dates == date)[0][0]
        records.append({'date': date, 'predicted': float(y[:idx].mean())})
    return pd.DataFrame(records).set_index('date')['predicted']


def compute_ols_benchmark(growth: pd.DataFrame, y: np.ndarray,
                          dates: pd.DatetimeIndex,
                          oos_dates: pd.DatetimeIndex,
                          min_train: int = MIN_TRAIN) -> pd.Series:
    """
    OLS regression benchmark — the paper's Xbm_t.

    At each OOS step t, fit OLS on expanding window:
        y = b1*headline + b2*core_ex_ff + b3*core_ex_fe
    Then predict y[t] using current-period rates.
    """
    col_map = {
        'headline': 'All items',
        'core_ff':  'All items, less fresh food',
        'core_fe':  'All items, less food (less alcoholic beverages) and energy',
    }
    X_bm = np.column_stack([
        growth[col].reindex(dates).ffill().values
        for col in col_map.values()
    ])

    records = []
    for date in oos_dates:
        t = np.where(dates == date)[0][0]
        if t < min_train:
            continue
        X_tr, y_tr = X_bm[:t], y[:t]
        beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        records.append({'date': date, 'predicted': float(X_bm[t] @ beta)})
    return pd.DataFrame(records).set_index('date')['predicted']
