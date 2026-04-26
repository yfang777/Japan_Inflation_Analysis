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
from regression.evaluation import print_scorecard
from regression.figures import (
    fig_weights, fig_lambda_cv, fig_insample, fig_oos,
    fig_ranks_weights, set_plots_dir,
)


def main(level: int = 2) -> None:
    print('=' * 65)
    print(f'ASSEMBLAGE REGRESSION  –  Level {level}  (Comps + Ranks)')
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

    print('\n[3/5] Albacorecomps – OOS (expanding + rolling)...')
    oos_exp_df,  _ = rolling_oos(X, y, w_prior, dates, window=None)
    oos_roll_df, _ = rolling_oos(X, y, w_prior, dates, window=ROLLING_WINDOW)

    # ── ranks: OOS ────────────────────────────────────────────────────────────
    print('\n[4/5] Albacoreranks – OOS (rolling)...')
    oos_rank_df, ranks_lam = rolling_oos_ranks(X, y, dates)

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
        'Comps (rolling 20y)': oos_roll_df,
        'Ranks (rolling 20y)': oos_rank_df,
    }
    our_models = {'Comps (expanding)', 'Comps (rolling 20y)', 'Ranks (rolling 20y)'}

    print_scorecard(
        insample, oos_exp_df, bm_df,
        extra_oos=extra_oos,
        our_models=our_models,
        primary_label='Comps (expanding)',
        features=features,
    )

    # ── figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    out_dir = PLOTS_DIR / f'level{level}'
    set_plots_dir(out_dir)

    fig_weights(insample, w_prior, features=features)
    fig_lambda_cv(insample)
    fig_insample(insample, dates, y)
    fig_ranks_weights(r_ranks_is['weights'], ranks_lam)

    extra_oos_fig = {
        'Comps (rolling 20y)': oos_roll_df,
        'Ranks (rolling 20y)': oos_rank_df,
    }
    fig_oos(oos_exp_df, bm_df, extra_oos=extra_oos_fig)

    print(f'\nDone. Figures saved to {out_dir.resolve()}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    main(level=args.level)
