"""AR/robustness.py — Phase 5 robustness checks for AR_ranks.

1. Subsample stability: refit on 1995-2012 vs 2013-2025; compare coefficient bars.
2. Consumption-tax dummies: compare AR_ranks(12) at h=12 with three treatments
   of the 1997-04, 2014-04, 2019-10 spikes — (raw / drop / interpolate).
3. Series choice: rerun the full horizon table on core CPI ('All items, less
   fresh food').
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from AR.config_ar import PLOTS, TAX_HIKE_DATES
from AR.data_ar import build_ar_dataset, load_headline
from AR.models import fit_ar_ranks
from AR.figures import plot_subsample_bars
from AR.horizon_table import run as run_horizon_table

OUT_DIR = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════════════════
#  tax-shock variants
# ══════════════════════════════════════════════════════════════════════════════

def _interp_log_cpi(cpi: pd.Series) -> pd.Series:
    """Replace tax-impact CPI levels with linear interpolation in log space."""
    out = cpi.astype(float).copy()
    for d in TAX_HIKE_DATES:
        ts = pd.Timestamp(d)
        if ts in out.index:
            out.loc[ts] = np.nan
    return np.exp(np.log(out).interpolate(method='time'))


def _fit_ar_ranks(ds: dict, h: int) -> tuple[float, np.ndarray, int]:
    R   = ds['R']
    tgt = ds['targets'][h]
    common = R.dropna().index.intersection(tgt.dropna().index)
    a, b = fit_ar_ranks(R.loc[common].values, tgt.loc[common].values)
    return a, b, len(common)


def tax_robustness_table(h: int = 12) -> pd.DataFrame:
    """Compare β across three tax-shock treatments for AR_ranks(12) at horizon h.

    Variants:
      - raw:         no special treatment.
      - drop:        SA the original CPI, then mask π_t = NaN at the impact
                     months so any L/R row touching them is dropped at fit.
      - interpolate: replace CPI levels at impact months with log-linear
                     interpolation before SA (smooths the spike out of π).
    """
    cpi = load_headline()
    cpi_interp = _interp_log_cpi(cpi)

    variants = {
        'raw':         build_ar_dataset(cpi_override=cpi, verbose=False),
        'drop':        build_ar_dataset(cpi_override=cpi,
                                        pi_mask_dates=TAX_HIKE_DATES, verbose=False),
        'interpolate': build_ar_dataset(cpi_override=cpi_interp, verbose=False),
    }

    rows = []
    for name, ds in variants.items():
        a, b, n = _fit_ar_ranks(ds, h)
        rec = {
            'variant': name,
            'n_obs':   n,
            'alpha':   a,
            'sum_b':   float(b.sum()),
            'b_top':   float(b[-1]),
            'b_bot':   float(b[0]),
        }
        for j in range(12):
            rec[f'b_R{j+1}'] = float(b[j])
        rows.append(rec)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  driver
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print('\n── 1. Subsample stability ─────────────────────────────────────────')
    ds = build_ar_dataset()
    out_sub = PLOTS / 'coef_bars_subsamples.png'
    plot_subsample_bars(ds, out_sub, h=12)
    print(f'  plot -> {out_sub}')

    print('\n── 2. Consumption-tax robustness (AR_ranks, h=12) ─────────────────')
    tab = tax_robustness_table(h=12)
    print(tab.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    out_tax = OUT_DIR / 'robustness_tax.csv'
    tab.to_csv(out_tax, index=False)
    print(f'  CSV  -> {out_tax}')

    print('\n── 3. Series choice — core CPI ────────────────────────────────────')
    run_horizon_table(headline_col='All items, less fresh food',
                      suffix='_corecpi',
                      title_suffix=' (core)')


if __name__ == '__main__':
    main()
