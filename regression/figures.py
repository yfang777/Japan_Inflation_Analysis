"""
regression/figures.py  –  all regression visualisation functions

Produces 5 figure types:
  1. Optimised weights vs. prior (horizontal bar)
  2. Lambda CV curve
  3. In-sample fit (time series)
  4. OOS predictions vs. benchmarks + rolling RMSE
  5. Rank-space weights by percentile
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PLOTS_DIR, FEATURES, GROUPS, GROUP_COLORS, COL_TO_GROUP, MPL_RCPARAMS

PLOTS_DIR.mkdir(exist_ok=True)
plt.rcParams.update(MPL_RCPARAMS)


def _save(fig, name):
    fig.savefig(PLOTS_DIR / name, bbox_inches='tight')
    print(f'  saved {name}')


def fig_weights(result: dict, w_prior: np.ndarray) -> plt.Figure:
    """Optimised weights vs. prior for each component, sorted descending."""
    w_opt   = result['weights']
    idx     = np.argsort(w_opt)[::-1]
    labels  = [FEATURES[i] for i in idx]
    opt_s   = w_opt[idx]
    prior_s = w_prior[idx]
    colors  = [
        GROUP_COLORS['Rent'] if l == 'Rent'
        else GROUP_COLORS.get(COL_TO_GROUP.get(l, 'Other'), '#7f8c8d')
        for l in labels
    ]

    fig, ax = plt.subplots(figsize=(9, 13))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, opt_s * 100, color=colors, edgecolor='white',
            linewidth=0.4, label='Optimized weight')
    ax.scatter(prior_s * 100, y_pos, color='#e67e22', zorder=5,
               s=25, label='Prior (basket share)', marker='D')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Weight (%)')
    ax.set_title(
        f'Assemblage Weights vs. Prior\n'
        f'λ = {result["best_lambda"]:.4f} | '
        f'{result["n_nonzero"]}/{len(FEATURES)} non-zero | '
        f'in-sample R2 = {result["r2"]:.3f}',
        fontsize=11, fontweight='bold',
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, 'reg_fig1_weights.png')
    return fig


def fig_lambda_cv(result: dict) -> plt.Figure:
    """CV RMSE curve showing lambda selection."""
    lambdas = result['lambda_grid']
    cv_rmse = np.sqrt(result['cv_mse'])
    best    = result['best_lambda']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(lambdas, cv_rmse, color='#2c3e50', lw=2)
    ax.axvline(best, color='#e74c3c', lw=1.5, linestyle='--',
               label=f'Best λ = {best:.4f}')
    ax.set_xlabel('λ  (log scale)')
    ax.set_ylabel('CV RMSE')
    ax.set_title('Lambda Selection – Time-Series Cross-Validation',
                 fontweight='bold')
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, 'reg_fig2_lambda_cv.png')
    return fig


def fig_insample(result: dict, dates: pd.DatetimeIndex,
                 y: np.ndarray) -> plt.Figure:
    """In-sample fit: actual 12m forward headline vs. assemblage fitted values."""
    fitted = result['fitted']

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(dates, y,      color='#2c3e50', lw=1.8, label='Actual (12m fwd)')
    ax.plot(dates, fitted, color='#e74c3c', lw=1.5,
            linestyle='--', label=f'Assemblage fit  (R2={result["r2"]:.3f})')
    ax.axhline(0, color='black', lw=0.6, linestyle='--', alpha=0.4)
    ax.set_ylabel('3m/3m annualized %')
    ax.set_title('In-Sample: Assemblage vs. Actual 12m Forward Headline Inflation',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, 'reg_fig3_insample.png')
    return fig


def fig_ranks_weights(weights: np.ndarray, lam: float) -> plt.Figure:
    """Weight assigned to each rank (rank 1 = lowest 3m/3m, rank K = highest)."""
    K     = len(weights)
    ranks = np.arange(1, K + 1)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(ranks, weights * 100, color='#3498db', edgecolor='white', linewidth=0.4)
    ax.axhline(100 / K, color='#e74c3c', lw=1.5, linestyle='--',
               label=f'Uniform 1/K = {100/K:.1f}%')
    ax.set_xlabel('Rank (1 = lowest 3m/3m component, K = highest)')
    ax.set_ylabel('Weight (%)')
    ax.set_title(f'Albacoreranks: Weight by Rank Position  |  λ = {lam:.4f}',
                 fontweight='bold')
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, 'reg_fig5_ranks_weights.png')
    return fig


def fig_oos(oos_df: pd.DataFrame, bm_df: pd.DataFrame,
            extra_oos: dict = None) -> plt.Figure:
    """OOS predictions vs. benchmarks vs. actual (2-panel: time series + rolling RMSE)."""
    actual = oos_df['actual']

    bm_styles = {
        'Random walk':           ('#95a5a6', ':',  1.2),
        'Core (ex fresh food)':  ('#27ae60', '--', 1.5),
        'Core (ex food&energy)': ('#3498db', '-.', 1.5),
    }

    extra_styles = {
        'Comps (rolling 20y)': ('#e74c3c', '--', 1.5),
        'Ranks (rolling 20y)': ('#8e44ad', '-',  1.8),
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # ── top panel: time series ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(actual.index, actual.values,
            color='#2c3e50', lw=2.2, label='Actual (12m fwd)')
    ax.plot(oos_df.index, oos_df['predicted'].values,
            color='#e74c3c', lw=1.5, linestyle=':', label='Comps (expanding)')
    if extra_oos:
        for name, df in extra_oos.items():
            c, ls, lw = extra_styles.get(name, ('#7f8c8d', '--', 1.2))
            ax.plot(df.index, df['predicted'].values, color=c, linestyle=ls,
                    lw=lw, label=name)
    for name, (c, ls, lw) in bm_styles.items():
        if name in bm_df.columns:
            ax.plot(bm_df.index, bm_df[name].values,
                    color=c, linestyle=ls, lw=lw, label=name)
    ax.axhline(0, color='black', lw=0.6, linestyle='--', alpha=0.4)
    ax.set_ylabel('3m/3m annualized %')
    ax.set_title('Out-of-Sample Forecast vs. Benchmarks (12m Forward Headline)',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=9, ncol=2)

    # ── bottom panel: rolling 36-month RMSE ───────────────────────────────────
    ax = axes[1]
    window = 36
    series_list = [
        ('Comps (expanding)', oos_df['predicted'], '#e74c3c', ':',  1.5),
    ]
    if extra_oos:
        for name, df in extra_oos.items():
            c, ls, lw = extra_styles.get(name, ('#7f8c8d', '--', 1.2))
            series_list.append((name, df['predicted'], c, ls, lw))
    for name, (c, ls, lw) in bm_styles.items():
        if name in bm_df.columns:
            series_list.append((name, bm_df[name], c, ls, lw))

    for name, pred, c, ls, lw in series_list:
        err2 = (actual - pred.reindex(actual.index)) ** 2
        roll_rmse = err2.rolling(window).mean().apply(np.sqrt)
        ax.plot(roll_rmse.index, roll_rmse.values,
                color=c, linestyle=ls, lw=lw, label=name)

    ax.set_ylabel(f'Rolling {window}m RMSE')
    ax.set_title('Rolling OOS RMSE – Assemblage vs. Benchmarks', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    _save(fig, 'reg_fig4_oos.png')
    return fig
