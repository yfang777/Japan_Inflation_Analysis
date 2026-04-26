"""
regression/regression_component.py  –  Albacorecomps (component-space assemblage)

Assemblage regression that finds optimal component weights to predict
12-month forward headline inflation, shrinking toward official basket weights.

    min_w  mean(y - Xw)²  +  λ ||w - w_prior||²
    s.t.   w >= 0,  Σw = 1

Run:  python regression/regression_component.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_TRAIN, OOS_STEP, N_CV_FOLDS, LAMBDA_GRID, ROLLING_WINDOW, PLOTS_DIR,
)
from utils.data_load import prepare_regression_data, load_benchmark_series
from regression.benchmarks import (
    compute_benchmarks, compute_mean_benchmark, compute_ols_benchmark,
)
from regression.evaluation import print_scorecard
from regression.figures import fig_weights, fig_lambda_cv, fig_insample, fig_oos, set_plots_dir


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _fit_single(X: np.ndarray, y: np.ndarray,
                lam: float, w_prior: np.ndarray,
                x0: np.ndarray = None,
                nonneg: bool = True) -> dict:
    """
    Fit assemblage for a single lambda.

    min_w  mean(y - Xw)²  +  λ ||w - w_prior||²
    s.t.   Σw = 1  [and w >= 0 if nonneg=True]
    """
    k = X.shape[1]

    def objective(w):
        return np.mean((y - X @ w) ** 2) + lam * np.sum((w - w_prior) ** 2)

    res = minimize(
        objective,
        x0=w_prior.copy() if x0 is None else x0,
        method='SLSQP',
        bounds=[(0, None)] * k if nonneg else None,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
        options={'maxiter': 2000, 'ftol': 1e-10},
    )
    w     = res.x
    y_hat = X @ w
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


def _cv_mses(X: np.ndarray, y: np.ndarray,
             lambdas: np.ndarray, w_prior: np.ndarray,
             n_folds: int = N_CV_FOLDS,
             nonneg: bool = True) -> np.ndarray:
    """
    Expanding-window time-series CV.  Returns mean CV MSE per lambda.
    Parallelised over lambdas — all (lambda, fold) pairs are independent.
    """
    n         = len(y)
    min_tr    = n // (n_folds + 1)
    fold_size = n // n_folds

    def _one(lam, f):
        t_end  = min_tr + f * fold_size
        t_test = min(t_end + fold_size, n)
        if t_end >= n:
            return (lam, f, np.nan)
        r  = _fit_single(X[:t_end], y[:t_end], lam, w_prior, nonneg=nonneg)
        pv = X[t_end:t_test] @ r['weights']
        return (lam, f, float(np.mean((y[t_end:t_test] - pv) ** 2)))

    jobs = [(lam, f) for lam in lambdas for f in range(n_folds)]
    out  = Parallel(n_jobs=-1, backend='loky')(delayed(_one)(lam, f) for lam, f in jobs)

    lam_to_idx = {lam: i for i, lam in enumerate(lambdas)}
    mse_matrix = np.full((len(lambdas), n_folds), np.nan)
    for lam, f, mse in out:
        mse_matrix[lam_to_idx[lam], f] = mse
    return np.nanmean(mse_matrix, axis=1)


def train(X: np.ndarray, y: np.ndarray, w_prior: np.ndarray,
          lambdas: np.ndarray = LAMBDA_GRID,
          n_folds: int = N_CV_FOLDS) -> dict:
    """Select lambda via time-series CV, then fit on full data."""
    print(f'  CV over {len(lambdas)} lambdas ({n_folds}-fold expanding window)...')
    cv_mse      = _cv_mses(X, y, lambdas, w_prior, n_folds)
    best_idx    = int(np.argmin(cv_mse))
    best_lam    = lambdas[best_idx]
    print(f'  Best λ = {best_lam:.5f}  (CV RMSE = {np.sqrt(cv_mse[best_idx]):.4f})')

    result = _fit_single(X, y, best_lam, w_prior)
    result['cv_mse']      = cv_mse
    result['lambda_grid'] = lambdas
    result['best_lambda'] = best_lam
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  OUT-OF-SAMPLE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def rolling_oos(X: np.ndarray, y: np.ndarray,
                w_prior: np.ndarray, dates: pd.DatetimeIndex,
                min_train: int = MIN_TRAIN,
                step: int = OOS_STEP,
                lambdas: np.ndarray = LAMBDA_GRID,
                window: int = None,
                nonneg: bool = True,
                n_jobs: int = -1) -> tuple:
    """
    Rolling OOS for Albacorecomps.

    Lambda fixed from first min_train months — no look-ahead.
    window=None → expanding (all history); window=240 → rolling 20-year.
    Returns (oos_df, oos_lambda).
    """
    win_str = f'rolling {window}m' if window else 'expanding'
    print(f'\n  Selecting OOS lambda from first {min_train} months [{win_str}]...')
    cv_mse     = _cv_mses(X[:min_train], y[:min_train], lambdas, w_prior, n_folds=5, nonneg=nonneg)
    oos_lambda = lambdas[int(np.argmin(cv_mse))]
    print(f'  OOS λ = {oos_lambda:.5f}')

    steps = list(range(min_train, len(X), step))

    def _predict(t):
        t_start = max(0, t - window) if window else 0
        r = _fit_single(X[t_start:t], y[t_start:t], oos_lambda, w_prior, nonneg=nonneg)
        return {'date': dates[t], 'actual': y[t], 'predicted': float(X[t] @ r['weights'])}

    records = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_predict)(t) for t in steps)

    df_oos           = pd.DataFrame(records).set_index('date')
    df_oos['error']  = df_oos['predicted'] - df_oos['actual']
    print(f'  OOS complete: {len(df_oos)} predictions')
    return df_oos, oos_lambda


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(level: int = 2) -> None:
    print('=' * 65)
    print(f'ALBACORECOMPS  –  Component-Space Assemblage (level {level})')
    print('=' * 65)

    # ── data ──────────────────────────────────────────────────────────────────
    print('\n[1/5] Loading and preparing data...')
    X, y, w_prior, features, dates, growth = prepare_regression_data(level=level)
    print(f'  Observations: {len(y)}  ({dates[0]:%Y-%m} – {dates[-1]:%Y-%m})')
    print(f'  Features:     {X.shape[1]} components')
    print(f'  Target mean:  {y.mean():.3f}%  std: {y.std():.3f}%')

    # ── prior weights ─────────────────────────────────────────────────────────
    print('\n[2/5] Prior weights (from CSV)...')
    print(f'  Prior range: [{w_prior.min():.4f}, {w_prior.max():.4f}]  sum={w_prior.sum():.6f}')
    top3 = np.argsort(w_prior)[::-1][:3]
    for i in top3:
        print(f'  Top prior: {features[i]:<45} {w_prior[i]*100:.2f}%')

    # ── in-sample training ────────────────────────────────────────────────────
    print('\n[3/5] In-sample training (full data)...')
    insample = train(X, y, w_prior)
    print(f'  In-sample RMSE: {insample["rmse"]:.4f}  R2: {insample["r2"]:.4f}')
    print(f'  Non-zero weights: {insample["n_nonzero"]}/{len(features)}')

    top5_idx = np.argsort(insample['weights'])[::-1][:5]
    print('  Top 5 weights:')
    for i in top5_idx:
        print(f'    {features[i]:<45}  opt={insample["weights"][i]*100:.2f}%'
              f'  prior={w_prior[i]*100:.2f}%')

    # ── out-of-sample: expanding ──────────────────────────────────────────────
    print('\n[4/5] OOS – expanding window...')
    oos_exp_df, _ = rolling_oos(X, y, w_prior, dates, window=None)

    # ── out-of-sample: rolling ────────────────────────────────────────────────
    print('\n[5/5] OOS – rolling 20-year window...')
    oos_roll_df, _ = rolling_oos(X, y, w_prior, dates, window=ROLLING_WINDOW)

    # ── benchmarks ────────────────────────────────────────────────────────────
    print('\nComputing benchmarks...')
    bm_growth = load_benchmark_series().combine_first(growth)
    bm_df = compute_benchmarks(bm_growth, oos_exp_df.index)
    bm_df['Unconditional mean']        = compute_mean_benchmark(y, dates, oos_exp_df.index)
    bm_df['OLS (headline+core+super)'] = compute_ols_benchmark(bm_growth, y, dates, oos_exp_df.index)

    extra_oos = {'Comps (rolling 20y)': oos_roll_df}

    # ── scorecard ─────────────────────────────────────────────────────────────
    our_models = {'Comps (expanding)', 'Comps (rolling 20y)'}
    print_scorecard(insample, oos_exp_df, bm_df,
                    extra_oos=extra_oos, our_models=our_models,
                    features=features)

    # ── figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    set_plots_dir(PLOTS_DIR / f'level{level}_component')
    fig_weights(insample, w_prior, features=features)
    fig_lambda_cv(insample)
    fig_insample(insample, dates, y)
    fig_oos(oos_exp_df, bm_df, extra_oos=extra_oos)

    print(f'\nDone. Figures saved to {PLOTS_DIR.resolve()}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    main(level=args.level)
