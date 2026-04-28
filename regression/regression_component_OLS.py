"""
regression/regression_component_OLS.py  –  unconstrained OLS baseline

Simple ordinary least squares on level-3 component 3m/3m growth rates,
predicting the same 12-month-forward headline inflation target used by
the assemblage models.

Differences from regression_component.py:
  - No simplex constraint (w ≥ 0, Σw = 1) — coefficients are unconstrained,
    can be negative, and need not sum to anything in particular.
  - No prior shrinkage, no λ regularisation, no CV.
  - Missing growth values are replaced by 0 (skip smart_impute entirely).
  - Includes an intercept.

Note: with k=636 components and only T=420 observations, the system is
under-determined; np.linalg.lstsq returns the minimum-norm solution. The
in-sample fit will look ~perfect; OOS is the meaningful number.

Run:  python regression/regression_component_OLS.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    START_DATE, HORIZON, MIN_TRAIN, OOS_STEP,
    COMPOSITE_COLS, SPECIAL_COLS,
)
from utils.data_load import (
    load_level_data, load_benchmark_series,
    compute_growth_3m3m, compute_forward_target,
)
from regression.benchmarks import (
    compute_benchmarks, compute_mean_benchmark, compute_ols_benchmark,
)
from regression.evaluation import metrics


def prepare_data_simple(level: int = 3):
    """Load level CSV → 3m/3m growth → fillna(0) → 12m forward target."""
    df, _weights = load_level_data(level, START_DATE)

    growth = compute_growth_3m3m(df)               # NaN where shift(3) NaN
    headline_col = df.columns[0]
    target = compute_forward_target(growth[headline_col], HORIZON)

    _exclude = set(COMPOSITE_COLS) | set(SPECIAL_COLS) | {headline_col}
    features = [c for c in df.columns
                if c not in _exclude and not c.startswith('Unnamed:')]

    valid = target.notna() & growth[headline_col].notna()
    X_df  = growth[features][valid].fillna(0.0)    # <-- the only imputation
    return X_df.values, target[valid].values, features, X_df.index, growth


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain OLS with intercept via least squares. Returns β of length k+1."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def main(level: int = 3) -> None:
    print('=' * 65)
    print(f'SIMPLE OLS  –  Level {level}  (fillna 0, no constraints, no λ)')
    print('=' * 65)

    print('\n[1/4] Loading data...')
    X, y, features, dates, growth = prepare_data_simple(level)
    print(f'  Observations: {len(y)}  ({dates[0]:%Y-%m} – {dates[-1]:%Y-%m})')
    print(f'  Features:     {X.shape[1]} components')
    print(f'  Target mean:  {y.mean():.3f}%   std: {y.std():.3f}%')
    print(f'  X range:      [{X.min():.2f}, {X.max():.2f}]   any NaN: {np.isnan(X).any()}')

    print('\n[2/4] In-sample OLS (full data)...')
    t0 = time.perf_counter()
    beta = fit_ols(X, y)
    y_hat = predict(beta, X)
    t_in = time.perf_counter() - t0
    m_in = metrics(y, y_hat)
    print(f'  fit time: {t_in*1000:.1f} ms')
    print(f'  RMSE={m_in["RMSE"]:.4f}  MAE={m_in["MAE"]:.4f}  R2={m_in["R2"]:.4f}')
    print(f'  intercept={beta[0]:+.4f}   ||β||_2 (slopes) = {np.linalg.norm(beta[1:]):.4f}')
    top = np.argsort(np.abs(beta[1:]))[::-1][:5]
    print('  Top 5 |β| coefficients:')
    for i in top:
        print(f'    {features[i]:<55}  β = {beta[1:][i]:+10.4f}')

    print(f'\n[3/4] Rolling OOS (expanding window, MIN_TRAIN={MIN_TRAIN})...')
    t0 = time.perf_counter()
    recs = []
    for t in range(MIN_TRAIN, len(X), OOS_STEP):
        beta_t = fit_ols(X[:t], y[:t])
        recs.append({
            'date':      dates[t],
            'actual':    y[t],
            'predicted': float(predict(beta_t, X[t:t + 1])[0]),
        })
    oos = pd.DataFrame(recs).set_index('date')
    t_oos = time.perf_counter() - t0
    m_oos = metrics(oos['actual'].values, oos['predicted'].values)
    print(f'  {len(oos)} OOS predictions in {t_oos:.1f}s ({t_oos/len(oos)*1000:.0f} ms/fit)')
    print(f'  RMSE={m_oos["RMSE"]:.4f}  MAE={m_oos["MAE"]:.4f}  R2={m_oos["R2"]:.4f}')

    print('\n[4/4] Comparison vs benchmarks...')
    bm_growth = load_benchmark_series().combine_first(growth)
    bm_df = compute_benchmarks(bm_growth, oos.index)
    bm_df['Unconditional mean']        = compute_mean_benchmark(y, dates, oos.index)
    bm_df['OLS (headline+core+super)'] = compute_ols_benchmark(bm_growth, y, dates, oos.index)

    rows = [('OLS (level-3, fillna 0)', m_oos)]
    for col in bm_df.columns:
        rows.append((col, metrics(oos['actual'].values, bm_df[col].values)))

    print('\n' + '=' * 65)
    print('OOS SCORECARD')
    print('=' * 65)
    print(f'  {"Model":<32}  {"RMSE":>7}  {"MAE":>7}  {"R2":>7}')
    print('  ' + '-' * 59)
    for name, m in rows:
        marker = ' <--' if name.startswith('OLS (level-3') else ''
        print(f'  {name:<32}  {m["RMSE"]:>7.4f}  {m["MAE"]:>7.4f}  {m["R2"]:>7.4f}{marker}')
    print('=' * 65)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    main(level=args.level)
