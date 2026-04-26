"""
regression/regression_all.py  –  unified run: Comps + Ranks for one level

Runs both assemblage variants and prints a single scorecard:
  - Albacorecomps  expanding window
  - Albacorecomps  rolling 20-year window
  - Albacoreranks  rolling 20-year window
  + benchmarks: random walk, core ex FF, core ex F&E, unconditional mean, OLS

Run:  python regression/regression_all.py --level 2
"""

import sys
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_TRAIN, OOS_STEP, N_CV_FOLDS, LAMBDA_GRID, LAMBDA_GRID_RANKS,
    ROLLING_WINDOW, PLOTS_DIR,
)
from utils.data_load import prepare_regression_data, load_benchmark_series
from regression.regression_component import rolling_oos, train
from regression.regression_rank import rolling_oos_ranks, build_rank_matrix, _fit_ranks_single
from regression.benchmarks import (
    compute_benchmarks, compute_mean_benchmark, compute_ols_benchmark,
)
from regression.evaluation import print_scorecard, save_scorecard
from regression.figures import (
    fig_weights, fig_lambda_cv, fig_insample, fig_oos,
    fig_ranks_weights, fig_prediction_quality, set_plots_dir,
)


def main(level: int = 2, step: int = OOS_STEP) -> None:
    print('=' * 65)
    print(f'ASSEMBLAGE REGRESSION  –  Level {level}  (Comps + Ranks)  step={step}')
    print('=' * 65)

    # ── data ──────────────────────────────────────────────────────────────────
    print('\n[1/5] Loading data...')
    X, y, w_prior, features, dates, growth = prepare_regression_data(level=level)
    print(f'  Observations: {len(y)}  ({dates[0]:%Y-%m} – {dates[-1]:%Y-%m})')
    print(f'  Features:     {X.shape[1]} components')
    print(f'  Target mean:  {y.mean():.3f}%  std: {y.std():.3f}%')

    # ── comps: in-sample + OOS ─────────────────────────────────────────────────
    print('\n[2/5] Albacorecomps – in-sample...')
    insample = train(X, y, w_prior)
    print(f'  In-sample RMSE: {insample["rmse"]:.4f}  R2: {insample["r2"]:.4f}')

    print('\n[3-4/5] OOS – comps (expanding + rolling) + ranks in parallel...')
    # level 3 has 700+ features — give each model all cores (GIL released by SLSQP)
    n_jobs_each = -1 if level == 3 else max(1, (os.cpu_count() or 4) // 3)
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_exp  = pool.submit(rolling_oos, X, y, w_prior, dates,
                             window=None,           step=step, nonneg=True, n_jobs=n_jobs_each)
        f_roll = pool.submit(rolling_oos, X, y, w_prior, dates,
                             window=ROLLING_WINDOW, step=step, nonneg=True, n_jobs=n_jobs_each)
        f_rank = pool.submit(rolling_oos_ranks, X, y, dates,
                             step=step, n_jobs=n_jobs_each)
    oos_exp_df,  _         = f_exp.result()
    oos_roll_df, _         = f_roll.result()
    oos_rank_df, ranks_lam = f_rank.result()

    # ── in-sample rank weights (last full window) ─────────────────────────────
    O_full     = build_rank_matrix(X)
    t_start    = max(0, len(X) - ROLLING_WINDOW)
    r_ranks_is = _fit_ranks_single(O_full[t_start:], y[t_start:], ranks_lam)

    # ── benchmarks ────────────────────────────────────────────────────────────
    print('\n[5/5] Computing benchmarks...')
    bm_growth = load_benchmark_series().combine_first(growth)
    bm_df = compute_benchmarks(bm_growth, oos_exp_df.index)
    bm_df['Unconditional mean']        = compute_mean_benchmark(y, dates, oos_exp_df.index)
    bm_df['OLS (headline+core+super)'] = compute_ols_benchmark(bm_growth, y, dates, oos_exp_df.index)

    # ── unified scorecard ─────────────────────────────────────────────────────
    extra_oos = {
        'Comps (rolling 10y)': oos_roll_df,
        'Ranks (rolling 10y)': oos_rank_df,
    }
    our_models = {'Comps (expanding)', 'Comps (rolling 10y)', 'Ranks (rolling 10y)'}

    rows = print_scorecard(
        insample, oos_exp_df, bm_df,
        extra_oos=extra_oos,
        our_models=our_models,
        primary_label='Comps (expanding)',
        features=features,
    )

    # ── save results ──────────────────────────────────────────────────────────
    results_dir = PLOTS_DIR.parent / 'results'
    save_scorecard(rows, insample,
                   results_dir / f'level{level}_scorecard.csv',
                   features=features)

    # ── figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    out_dir = PLOTS_DIR / f'level{level}'
    set_plots_dir(out_dir)

    fig_weights(insample, w_prior, features=features)
    fig_lambda_cv(insample)
    fig_insample(insample, dates, y)
    fig_ranks_weights(r_ranks_is['weights'], ranks_lam)

    extra_oos_fig = {
        'Comps (rolling 10y)': oos_roll_df,
        'Ranks (rolling 10y)': oos_rank_df,
    }
    fig_oos(oos_exp_df, bm_df, extra_oos=extra_oos_fig)
    fig_prediction_quality(oos_exp_df, extra_oos=extra_oos_fig)

    print(f'\nDone. Figures saved to {out_dir.resolve()}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--step',  type=int, default=OOS_STEP,
                        help='OOS step size in months (1=monthly, 3=quarterly)')
    args = parser.parse_args()
    main(level=args.level, step=args.step)
