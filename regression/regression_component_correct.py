"""
regression/regression_component_correct.py – Albacorecomps (R-parity build)

Component-space assemblage matching R's `nonneg.ridge.sum1()` exactly
(Assemblage/R/assemblage_v240228.R:604-698). Finds optimal component weights
to predict 12-month forward headline inflation, shrinking toward official
basket weights with a per-feature sd-weighted L2 penalty.

    min_w  ||y − Xw||²  +  λ · Σⱼ sd(xⱼ) · (wⱼ − w_priorⱼ)²
    s.t.   w >= 0,  Σw = 1

Differences vs. the original Python port (regression_component.py):
  • SSR loss instead of MSE                  (matches R's sum-of-squares)
  • Per-feature sd-weighted penalty           (was identity)
  • 10-fold contiguous-block leave-one-out CV (was expanding-window k-fold)
  • Per-window CV re-tuning in OOS            (was frozen λ from first window)

NOTE: switching from MSE to SSR rescales λ by a factor of n. Any LAMBDA_GRID
calibrated for the original script will likely be too small here; re-tune.

Run:  python regression/regression_component_correct.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_TRAIN, OOS_STEP, LAMBDA_GRID, ROLLING_WINDOW, PLOTS_DIR,
)
from utils.data_load import prepare_regression_data, load_benchmark_series
from regression.benchmarks import (
    compute_benchmarks, compute_mean_benchmark, compute_ols_benchmark,
)
from regression.evaluation import print_scorecard
from regression.figures import fig_weights, fig_lambda_cv, fig_insample, fig_oos, set_plots_dir


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL  –  matches R::nonneg.ridge.sum1
# ══════════════════════════════════════════════════════════════════════════════

def _fit_single(X: np.ndarray, y: np.ndarray,
                lam: float, w_prior: np.ndarray,
                x0: np.ndarray = None) -> dict:
    """
    Fit assemblage for a single λ via convex programming (CVXPY ≈ R's CVXR).

        min_w  ||y − Xw||²  +  λ · Σⱼ sd(xⱼ) · (wⱼ − w_priorⱼ)²
        s.t.   w >= 0,  Σw = 1

    Mirrors R's nonneg.ridge.sum1 (CVXR loss, line 661): SSR fit term plus a
    ridge penalty whose per-coefficient strength is the standard deviation of
    that feature, shrinking toward `w_prior`.

    Solved with CVXPY+OSQP (interior-point on the simplex), not quadprog: with
    k > n the Hessian is rank-deficient by k-n directions. quadprog's active-
    set method tolerates that poorly and reports spurious "constraints
    inconsistent" errors; CVXPY/OSQP handles PSD problems on bounded sets
    natively, just like R/CVXR does. x0 is accepted for API compatibility.
    """
    n, k = X.shape
    s = X.std(axis=0, ddof=1)                          # per-feature sd; R: apply(x.in, 2, sd)

    w = cp.Variable(k)
    loss = cp.sum_squares(y - X @ w) \
         + lam * cp.sum(cp.multiply(s, cp.square(w - w_prior)))
    constraints = [w >= 0, cp.sum(w) == 1]
    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)

    if w.value is None:
        # OSQP can occasionally fail on pathological scaling — retry with SCS.
        prob.solve(solver=cp.SCS, verbose=False)

    w = np.asarray(w.value).flatten()
    w = np.clip(w, 0.0, None)                          # scrub tiny negatives
    w = w / w.sum() if w.sum() > 0 else w              # re-normalise after clip

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
        'converged': prob.status in ('optimal', 'optimal_inaccurate'),
    }


def _cv_mses(X: np.ndarray, y: np.ndarray,
             lambdas: np.ndarray, w_prior: np.ndarray,
             n_folds: int = 10) -> np.ndarray:
    """
    10-fold contiguous-block leave-one-out CV — matches R's nonneg.ridge.sum1.

    R builds a fold vector
        fd = c(rep(1, n/10), rep(2, n/10), ..., rep(10, remainder))
    of contiguous time blocks, then for each fold f trains on `fd != f` and
    validates on `fd == f`. With 10 ordered blocks this is *not* expanding-
    window: e.g. fold 1 trains on the future to predict the past. Replicated
    verbatim to keep parity with the R reference.
    """
    n = len(y)
    fold_size = n // n_folds

    fd = np.empty(n, dtype=int)
    for f in range(n_folds - 1):
        fd[f * fold_size : (f + 1) * fold_size] = f
    fd[(n_folds - 1) * fold_size :] = n_folds - 1     # remainder folded in

    mse_matrix = np.full((len(lambdas), n_folds), np.nan)
    for li, lam in enumerate(lambdas):
        for f in range(n_folds):
            tr = fd != f
            te = fd == f
            r  = _fit_single(X[tr], y[tr], lam, w_prior)
            pv = X[te] @ r['weights']
            mse_matrix[li, f] = np.mean((y[te] - pv) ** 2)

    return np.nanmean(mse_matrix, axis=1)


def train(X: np.ndarray, y: np.ndarray, w_prior: np.ndarray,
          lambdas: np.ndarray = LAMBDA_GRID,
          n_folds: int = 10) -> dict:
    """Select λ via 10-fold contiguous-block CV, then fit on full data."""
    print(f'  CV over {len(lambdas)} lambdas ({n_folds}-fold contiguous-block)...')
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
#  OUT-OF-SAMPLE  –  per-window re-CV, matching R::assemblage.estimation.RW
# ══════════════════════════════════════════════════════════════════════════════

def rolling_oos(X: np.ndarray, y: np.ndarray,
                w_prior: np.ndarray, dates: pd.DatetimeIndex,
                min_train: int = MIN_TRAIN,
                step: int = OOS_STEP,
                lambdas: np.ndarray = LAMBDA_GRID,
                window: int = None,
                n_folds: int = 10) -> tuple:
    """
    Rolling OOS for Albacorecomps with per-window CV re-tuning.

    For each rolling-window iteration, λ is re-selected from the in-window
    training data via 10-fold contiguous-block CV (mirroring R's
    assemblage.estimation.RW, which calls nonneg.ridge.sum1 — and therefore
    re-runs CV — on every window).

    window=None → expanding (all history); window=240 → rolling 20-year.
    Returns (oos_df, chosen_lambdas).
    """
    win_str = f'rolling {window}m' if window else 'expanding'
    print(f'\n  Rolling OOS [{win_str}], re-CV per window...')

    steps = list(range(min_train, len(X), step))
    records = []
    chosen_lambdas = []
    w0 = w_prior.copy()
    for i, t in enumerate(steps):
        t_start = max(0, t - window) if window else 0
        Xw, yw  = X[t_start:t], y[t_start:t]

        cv_mse = _cv_mses(Xw, yw, lambdas, w_prior, n_folds=n_folds)
        lam_t  = lambdas[int(np.argmin(cv_mse))]
        r      = _fit_single(Xw, yw, lam_t, w_prior, x0=w0)
        w0     = r['weights']
        y_pred = float(X[t] @ r['weights'])

        records.append({'date': dates[t], 'actual': y[t], 'predicted': y_pred})
        chosen_lambdas.append(lam_t)
        if (i + 1) % 60 == 0:
            print(f'    {i+1}/{len(steps)} ...')

    df_oos          = pd.DataFrame(records).set_index('date')
    df_oos['error'] = df_oos['predicted'] - df_oos['actual']
    print(f'  OOS complete: {len(df_oos)} predictions  '
          f'(λ range across windows: {min(chosen_lambdas):.5f} – {max(chosen_lambdas):.5f})')
    return df_oos, chosen_lambdas


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(level: int = 2) -> None:
    print('=' * 65)
    print(f'ALBACORECOMPS  –  Component-Space Assemblage (level {level})')
    print('                  R-parity build  (matches nonneg.ridge.sum1)')
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
    set_plots_dir(PLOTS_DIR / f'level{level}_component_correct')
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
