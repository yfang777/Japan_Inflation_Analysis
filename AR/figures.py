"""AR/figures.py — coefficient bar chart for AR_ranks(12)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from AR.config_ar import PLOTS, SUBSAMPLES
from AR.data_ar import build_ar_dataset
from AR.models import fit_ar_ranks


def _fit_full(ds: dict, h: int,
              date_lo: pd.Timestamp | None = None,
              date_hi: pd.Timestamp | None = None) -> tuple[float, np.ndarray, int]:
    R   = ds['R']
    tgt = ds['targets'][h]
    common = R.dropna().index.intersection(tgt.dropna().index)
    if date_lo is not None:
        common = common[common >= date_lo]
    if date_hi is not None:
        common = common[common <= date_hi]
    a, b = fit_ar_ranks(R.loc[common].values, tgt.loc[common].values)
    return a, b, len(common)


def plot_coef_bars(ds: dict, out_path: Path,
                   horizons: tuple[int, ...] = (1, 12)) -> None:
    fig, axes = plt.subplots(1, len(horizons), figsize=(11, 4), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons):
        alpha, beta, n = _fit_full(ds, h)
        x = np.arange(1, 13)
        ax.bar(x, beta, color='C0', edgecolor='k', lw=0.4)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'$R_{{{i}}}$' for i in x], fontsize=9)
        ax.set_title(f'h = {h}  (n = {n})')
        if ax is axes[0]:
            ax.set_ylabel(r'$\beta_r$')
        ax.text(0.02, 0.98,
                f'$\\alpha = {alpha:.3f}$\n$\\sum_r \\beta_r = {beta.sum():.3f}$',
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.suptitle(f'AR_ranks(12) coefficients — Japan {ds["headline_col"]} (full sample)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_subsample_bars(ds: dict, out_path: Path, h: int = 12) -> None:
    fig, axes = plt.subplots(1, len(SUBSAMPLES), figsize=(11, 4), sharey=True)
    for ax, (label, lo, hi) in zip(axes, SUBSAMPLES):
        a, b, n = _fit_full(ds, h, pd.Timestamp(lo), pd.Timestamp(hi))
        x = np.arange(1, 13)
        ax.bar(x, b, color='C2', edgecolor='k', lw=0.4)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'$R_{{{i}}}$' for i in x], fontsize=9)
        ax.set_title(f'{label}  (n = {n})')
        if ax is axes[0]:
            ax.set_ylabel(r'$\beta_r$')
        ax.text(0.02, 0.98,
                f'$\\alpha = {a:.3f}$\n$\\sum_r \\beta_r = {b.sum():.3f}$',
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.suptitle(f'AR_ranks(12) coefficients by subsample, h = {h}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ds = build_ar_dataset()
    out1 = PLOTS / 'coef_bars.png'
    plot_coef_bars(ds, out1)
    print(f'  plot -> {out1}')
    out2 = PLOTS / 'coef_bars_subsamples.png'
    plot_subsample_bars(ds, out2, h=12)
    print(f'  plot -> {out2}')


if __name__ == '__main__':
    main()
