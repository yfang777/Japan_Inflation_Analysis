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

from config import PLOTS_DIR, GROUPS, GROUP_COLORS, COL_TO_GROUP, MPL_RCPARAMS

PLOTS_DIR.mkdir(exist_ok=True)
plt.rcParams.update(MPL_RCPARAMS)

_active_plots_dir = PLOTS_DIR


def set_plots_dir(path: Path) -> None:
    """Override the output directory for all subsequent figure saves."""
    global _active_plots_dir
    _active_plots_dir = Path(path)
    _active_plots_dir.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fig.savefig(_active_plots_dir / name, bbox_inches='tight')
    print(f'  saved {_active_plots_dir / name}')


def fig_weights(result: dict, w_prior: np.ndarray,
                features: list[str] = None) -> plt.Figure:
    """Optimised weights vs. prior for each component, sorted descending."""
    w_opt   = result['weights']
    idx     = np.argsort(w_opt)[::-1]
    if features is None:
        features = [f'comp_{i}' for i in range(len(w_opt))]
    labels  = [features[i] for i in idx]
    opt_s   = w_opt[idx]
    prior_s = w_prior[idx]
    colors  = [
        GROUP_COLORS.get('Rent', '#8e44ad') if l == 'Rent'
        else GROUP_COLORS.get(COL_TO_GROUP.get(l, 'Other'), '#7f8c8d')
        for l in labels
    ]

    n_comps = len(labels)
    fig_height = max(6, n_comps * 0.28)
    fig, ax = plt.subplots(figsize=(9, fig_height))
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
        f'{result["n_nonzero"]}/{len(features)} non-zero | '
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


def fig_prediction_quality(oos_df: pd.DataFrame,
                            extra_oos: dict = None) -> plt.Figure:
    """
    2-panel figure showing prediction quality:
      Left:  scatter actual vs predicted (45° = perfect)
      Right: prediction errors over time with ±1 std band
    """
    actual = oos_df['actual']

    model_styles = {
        'Comps (rolling 10y)': ('#e74c3c', 'o',  0.6),
        'Ranks (rolling 10y)': ('#8e44ad', 's',  0.6),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: scatter ─────────────────────────────────────────────────────────
    ax = axes[0]
    lim = max(abs(actual.min()), abs(actual.max())) * 1.1
    ax.axline((0, 0), slope=1, color='black', lw=1, linestyle='--', alpha=0.5,
              label='Perfect forecast')
    ax.scatter(actual, oos_df['predicted'], color='#e74c3c',
               s=12, alpha=0.5, label='Comps (expanding)')
    if extra_oos:
        for name, df in extra_oos.items():
            c, mk, al = model_styles.get(name, ('#7f8c8d', 'o', 0.4))
            pred = df['predicted'].reindex(actual.index).dropna()
            ax.scatter(actual.reindex(pred.index), pred,
                       color=c, marker=mk, s=12, alpha=al, label=name)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Actual 12m forward inflation (%)')
    ax.set_ylabel('Predicted (%)')
    ax.set_title('Actual vs. Predicted (OOS)', fontweight='bold')
    ax.legend(frameon=False, fontsize=8)

    # ── right: errors over time ────────────────────────────────────────────────
    ax = axes[1]
    err = oos_df['predicted'] - actual
    std = float(err.std())
    ax.fill_between(err.index, -std, std, alpha=0.15, color='#95a5a6',
                    label=f'±1 std ({std:.2f}%)')
    ax.plot(err.index, err.values, color='#e74c3c', lw=1,
            alpha=0.8, label='Comps (expanding) error')
    if extra_oos:
        extra_colors = {'Comps (rolling 10y)': '#c0392b', 'Ranks (rolling 10y)': '#8e44ad'}
        for name, df in extra_oos.items():
            c = extra_colors.get(name, '#7f8c8d')
            e = df['predicted'].reindex(actual.index) - actual
            ax.plot(e.index, e.values, color=c, lw=1, alpha=0.7, label=f'{name} error')
    ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
    ax.set_ylabel('Prediction error (%)')
    ax.set_title('OOS Prediction Errors Over Time', fontweight='bold')
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, 'reg_fig6_prediction_quality.png')
    return fig
