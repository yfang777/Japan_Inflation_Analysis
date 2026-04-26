"""
regression/regression_rank.py  –  Albacoreranks (rank-space assemblage)

Instead of weighting components by identity, sort them low→high at each t
and learn which percentile of the cross-sectional distribution is predictive
of future headline inflation.  Fused-ridge penalty encourages smooth weights
across adjacent ranks.  Mean constraint replaces sum-to-1.

    min_w  mean(y - Ow)²  +  λ Σ(w_{r+1} - w_r)²   [fused ridge]
    s.t.   w >= 0,  Ō'w = ȳ                          [mean constraint]

Run:  python regression/regression_rank.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_TRAIN, OOS_STEP, N_CV_FOLDS, LAMBDA_GRID_RANKS, ROLLING_WINDOW, PLOTS_DIR,
)
from utils.data_load import prepare_regression_data, load_benchmark_series
from regression.benchmarks import (
    compute_benchmarks, compute_mean_benchmark, compute_ols_benchmark,
)
from regression.evaluation import print_scorecard
from regression.figures import fig_ranks_weights, fig_oos, set_plots_dir


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_rank_matrix(X: np.ndarray) -> np.ndarray:
    """Sort component growth rates low→high at each t.  Shape unchanged (T, K)."""
    return np.sort(X, axis=1)


def _fit_ranks_single(O: np.ndarray, y: np.ndarray, lam: float) -> dict:
    """
    Assemblage in rank space.

    min_w  mean(y - Ow)²  +  λ Σ(w_{r+1} - w_r)²   [fused ridge]
    s.t.   w >= 0,  Ō'w = ȳ                          [mean constraint]
    """
    K      = O.shape[1]
    O_mean = O.mean(axis=0)
    y_mean = float(y.mean())

    def objective(w):
        return np.mean((y - O @ w) ** 2) + lam * float(np.sum(np.diff(w) ** 2))

    res = minimize(
        objective,
        x0=np.ones(K) / K,
        method='SLSQP',
        bounds=[(0, None)] * K,
        constraints={'type': 'eq', 'fun': lambda w: float(O_mean @ w) - y_mean},
        options={'maxiter': 2000, 'ftol': 1e-10},
    )
    w     = res.x
    y_hat = O @ w
    ss_r  = np.sum((y - y_hat) ** 2)
    ss_t  = np.sum((y - y.mean()) ** 2)
    return {
        'weights':   w,
        'fitted':    y_hat,
        'lambda':    lam,
        'rmse':      float(np.sqrt(np.mean((y - y_hat) ** 2))),
        'mae':       float(np.mean(np.abs(y - y_hat))),
        'r2':        float(1 - ss_r / ss_t) if ss_t > 0 else 0.0,
        'n_nonzero': int((w > 1e-4).sum()),
        'converged': res.success,
    }


def _cv_mses_ranks(O: np.ndarray, y: np.ndarray,
                   lambdas: np.ndarray,
                   n_folds: int = N_CV_FOLDS) -> np.ndarray:
    """Expanding-window CV for rank-space model."""
    n         = len(y)
    min_tr    = n // (n_folds + 1)
    fold_size = n // n_folds
    mse_mat   = np.full((len(lambdas), n_folds), np.nan)

    for li, lam in enumerate(lambdas):
        for f in range(n_folds):
            t_end  = min_tr + f * fold_size
            t_test = min(t_end + fold_size, n)
            if t_end >= n:
                continue
            r  = _fit_ranks_single(O[:t_end], y[:t_end], lam)
            pv = O[t_end:t_test] @ r['weights']
            mse_mat[li, f] = np.mean((y[t_end:t_test] - pv) ** 2)

    return np.nanmean(mse_mat, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#  OUT-OF-SAMPLE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def rolling_oos_ranks(X: np.ndarray, y: np.ndarray,
                      dates: pd.DatetimeIndex,
                      min_train: int = MIN_TRAIN,
                      step: int = OOS_STEP,
                      lambdas: np.ndarray = LAMBDA_GRID_RANKS,
                      window: int = ROLLING_WINDOW) -> tuple:
    """
    Rolling OOS for Albacoreranks.

    Builds order-statistic matrix O from X, then runs rank-space model
    with fused-ridge penalty and rolling window.
    Lambda fixed from first min_train months — no look-ahead.
    Returns (oos_df, oos_lambda).
    """
    O = build_rank_matrix(X)
    win_str = f'rolling {window}m' if window else 'expanding'
    print(f'\n  [Ranks] Selecting OOS lambda from first {min_train} months [{win_str}]...')
    cv_mse     = _cv_mses_ranks(O[:min_train], y[:min_train], lambdas, n_folds=5)
    oos_lambda = lambdas[int(np.argmin(cv_mse))]
    print(f'  [Ranks] OOS λ = {oos_lambda:.5f}')

    steps   = range(min_train, len(O), step)
    records = []
    for i, t in enumerate(steps):
        t_start = max(0, t - window) if window else 0
        r       = _fit_ranks_single(O[t_start:t], y[t_start:t], oos_lambda)
        y_pred  = float(O[t] @ r['weights'])
        records.append({'date': dates[t], 'actual': y[t], 'predicted': y_pred})
        if (i + 1) % 60 == 0:
            print(f'    {i+1}/{len(steps)} ...')

    df_oos           = pd.DataFrame(records).set_index('date')
    df_oos['error']  = df_oos['predicted'] - df_oos['actual']
    print(f'  [Ranks] OOS complete: {len(df_oos)} predictions')
    return df_oos, oos_lambda


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(level: int = 2) -> None:
    print('=' * 65)
    print(f'ALBACORERANKS  –  Rank-Space Assemblage (level {level})')
    print('=' * 65)

    # ── data ──────────────────────────────────────────────────────────────────
    print('\n[1/3] Loading and preparing data...')
    X, y, w_prior, features, dates, growth = prepare_regression_data(level=level)
    print(f'  Observations: {len(y)}  ({dates[0]:%Y-%m} – {dates[-1]:%Y-%m})')
    print(f'  Features:     {X.shape[1]} components')
    print(f'  Target mean:  {y.mean():.3f}%  std: {y.std():.3f}%')

    # ── out-of-sample: ranks rolling ──────────────────────────────────────────
    print('\n[2/3] OOS – rolling 20-year window...')
    oos_ranks_df, ranks_lam = rolling_oos_ranks(X, y, dates)

    # ── in-sample rank weights (last full window) ─────────────────────────────
    O_full     = build_rank_matrix(X)
    t_start    = max(0, len(X) - ROLLING_WINDOW)
    r_ranks_is = _fit_ranks_single(O_full[t_start:], y[t_start:], ranks_lam)
    print(f'  Ranks in-sample R2 (last window): {r_ranks_is["r2"]:.4f}')

    # ── benchmarks ────────────────────────────────────────────────────────────
    print('\n[3/3] Computing benchmarks...')
    bm_growth = load_benchmark_series().combine_first(growth)
    bm_df = compute_benchmarks(bm_growth, oos_ranks_df.index)
    bm_df['Unconditional mean']        = compute_mean_benchmark(y, dates, oos_ranks_df.index)
    bm_df['OLS (headline+core+super)'] = compute_ols_benchmark(bm_growth, y, dates, oos_ranks_df.index)

    # ── scorecard ─────────────────────────────────────────────────────────────
    insample_info = {
        'rmse': r_ranks_is['rmse'],
        'mae':  r_ranks_is['mae'],
        'r2':   r_ranks_is['r2'],
        'n_nonzero': r_ranks_is['n_nonzero'],
        'best_lambda': ranks_lam,
    }
    print_scorecard(insample_info, oos_ranks_df, bm_df,
                    our_models={'Ranks (rolling 20y)'},
                    primary_label='Ranks (rolling 20y)',
                    features=features)

    # ── figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    set_plots_dir(PLOTS_DIR / f'level{level}_rank')
    fig_ranks_weights(r_ranks_is['weights'], ranks_lam)
    fig_oos(oos_ranks_df, bm_df)

    print(f'\nDone. Figures saved to {PLOTS_DIR.resolve()}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    main(level=args.level)
