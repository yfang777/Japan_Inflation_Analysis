"""AR/data_ar.py — Japan CPI series construction for univariate AR models.

Loads headline CPI, applies seasonal adjustment, builds:
    pi_t       : 100 * dlog CPI_t^SA              (monthly inflation, %)
    L_t        : (pi_{t-1}, ..., pi_{t-12})       (lag matrix, T x 12)
    R_t        : sort(L_t) ascending              (rank matrix, T x 12)
    yoy_t      : (1/12) sum_{j=0..11} pi_{t-j}    (current 12m YoY at t)
    y_{t+h}    : (1/12) sum_{j=1..12} pi_{t+h-j+1}  (forward target)

Run standalone:  python AR/data_ar.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LEVEL_DIR
from utils.data_load import load_level_data
from AR.config_ar import HORIZONS, PLOTS, TAX_HIKE_DATES

HEADLINE = 'All items'


# ══════════════════════════════════════════════════════════════════════════════
#  series loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_level3_composite(col: str, start_date: str) -> pd.Series:
    """Load a composite headline (e.g. 'All items, less fresh food') from level_3_full.csv."""
    raw = pd.read_csv(LEVEL_DIR / 'level_3_full.csv', dtype=str)
    date_col = raw.columns[0]
    if col not in raw.columns:
        raise KeyError(f'{col!r} not in level_3_full.csv columns')
    raw = raw[[date_col, col]]
    mask = raw[date_col].str.strip() == 'Weights'
    data = raw.loc[~mask].copy()
    data = data.dropna(subset=[date_col])
    data = data[data[date_col].str.strip().astype(bool)]
    data[date_col] = pd.to_datetime(data[date_col].str.strip(), format='%Y%m')
    data = (data.set_index(date_col)
                .replace('-', np.nan)
                .apply(pd.to_numeric, errors='coerce'))
    if start_date:
        data = data[data.index >= start_date]
    return data[col].dropna()


def load_headline(headline_col: str = HEADLINE,
                  start_date: str = '1970-01-01') -> pd.Series:
    """Return the monthly CPI level series for the chosen headline."""
    if headline_col == HEADLINE:
        df, _ = load_level_data(level=1, start_date=start_date)
        return df[HEADLINE].dropna()
    return _load_level3_composite(headline_col, start_date)


# ══════════════════════════════════════════════════════════════════════════════
#  seasonal adjustment
# ══════════════════════════════════════════════════════════════════════════════

def _seasonal_adjust(log_cpi: pd.Series) -> tuple[pd.Series, str]:
    """Try X-13 first, fall back to STL on log levels."""
    s = log_cpi.copy()
    s.index = pd.DatetimeIndex(s.index, freq='MS')
    try:
        from statsmodels.tsa.x13 import x13_arima_analysis
        res = x13_arima_analysis(s)
        return res.seasadj, 'x13'
    except Exception:
        pass
    from statsmodels.tsa.seasonal import STL
    stl = STL(s, period=12, robust=True).fit()
    return s - stl.seasonal, 'stl'


# ══════════════════════════════════════════════════════════════════════════════
#  dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def build_ar_dataset(headline_col: str = HEADLINE,
                     start_date: str = '1970-01-01',
                     cpi_override: pd.Series | None = None,
                     pi_mask_dates: list[str] | None = None,
                     verbose: bool = True) -> dict:
    """
    Build the full AR dataset.

    Parameters
    ----------
    pi_mask_dates : list of date strings, optional
        After SA and computing pi, set pi_t = NaN at these dates. Use this for
        the tax-shock 'drop' robustness variant: the corresponding rows of
        L/R/yoy will then drop out automatically when fitting.
    """
    cpi = cpi_override.dropna() if cpi_override is not None \
        else load_headline(headline_col, start_date)

    log_cpi = np.log(cpi)
    sa_log, method = _seasonal_adjust(log_cpi)
    if verbose:
        print(f'  series={headline_col!r}  obs={len(cpi)}  '
              f'range={cpi.index[0]:%Y-%m}..{cpi.index[-1]:%Y-%m}  SA={method}')

    pi = (100.0 * sa_log.diff()).rename('pi')

    if pi_mask_dates:
        for d in pi_mask_dates:
            ts = pd.Timestamp(d)
            if ts in pi.index:
                pi.loc[ts] = np.nan

    L_cols = [pi.shift(j).rename(f'lag{j}') for j in range(1, 13)]
    L = pd.concat(L_cols, axis=1)

    R_vals = np.sort(L.values, axis=1)
    R = pd.DataFrame(R_vals, index=L.index,
                     columns=[f'r{j}' for j in range(1, 13)])

    yoy = pi.rolling(window=12).mean().rename('yoy')

    targets = {h: yoy.shift(-h).rename(f'y_h{h}') for h in HORIZONS}

    return dict(cpi=cpi, log_cpi=log_cpi, sa_log=sa_log, pi=pi,
                L=L, R=R, yoy=yoy, targets=targets, sa_method=method,
                headline_col=headline_col)


# ══════════════════════════════════════════════════════════════════════════════
#  diagnostic plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_diagnostic(ds: dict, out_path: Path,
                    plot_start: str = '1985-01-01') -> None:
    """Trim leading STL edge artifacts; the modelling sample starts in 1995."""
    import matplotlib.pyplot as plt
    pi  = ds['pi'].loc[plot_start:]
    yoy = ds['yoy'].loc[plot_start:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(pi.index, pi.values, lw=0.7, color='C0')
    ax1.axhline(0, color='k', lw=0.4)
    ax1.set_title(f"Monthly inflation $\\pi_t$ (SA via {ds['sa_method']}), "
                  f"Japan {ds['headline_col']}")
    ax1.set_ylabel('% (m/m, SA)')

    ax2.plot(yoy.index, yoy.values, lw=1.0, color='C1')
    ax2.axhline(0, color='k', lw=0.4)
    ax2.set_title('12-month YoY average of $\\pi_t$')
    ax2.set_ylabel('%')

    for d in TAX_HIKE_DATES:
        for ax in (ax1, ax2):
            ax.axvline(pd.Timestamp(d), color='r', alpha=0.35, ls='--', lw=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ds = build_ar_dataset()
    out = PLOTS / 'inflation_diagnostic.png'
    plot_diagnostic(ds, out)
    print(f'  pi:  {ds["pi"].first_valid_index():%Y-%m} .. {ds["pi"].last_valid_index():%Y-%m}')
    print(f'  yoy: {ds["yoy"].first_valid_index():%Y-%m} .. {ds["yoy"].last_valid_index():%Y-%m}')
    print(f'  plot -> {out}')


if __name__ == '__main__':
    main()
