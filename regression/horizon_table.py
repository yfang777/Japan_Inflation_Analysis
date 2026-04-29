"""
regression/horizon_table.py — Forecasting performance table

Sweep:  level ∈ {1, 2}  ×  horizon h ∈ {3, 12, 24}.

For each (level, h) cell, fits Albacorecomps and Albacoreranks on an
expanding window from 1990, then computes RMSE in two out-of-sample
periods — 2010m1–2019m12 and 2020m1–2024m12 — relative to Xbm (OLS on
[headline, core_ff, core_fe] with intercept). The denominator is always
1.00 by construction and is not displayed.

Reported benchmark rows:
    • Xbm (w0=0)        — same regressors as Xbm, no intercept
    • Xbm+              — adds 'All items, less fresh food and energy'
    • Xbm+ (w0=0)       — same, no intercept

Outputs (alongside this file):
    horizon_table.csv     — flat numeric grid
    horizon_table.tex     — booktabs LaTeX
    (also printed to stdout)

Run:  python regression/horizon_table.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_TRAIN
from utils.data_load import prepare_regression_data, load_benchmark_series
from regression.regression_component import rolling_oos as rolling_oos_comps
from regression.regression_rank import rolling_oos_ranks


# ── grid ──────────────────────────────────────────────────────────────────────

PERIODS = [
    ('2010m1-2019m12', '2010-01-01', '2019-12-31'),
    ('2020m1-2024m12', '2020-01-01', '2024-12-31'),
]
HORIZONS = [3, 12, 24]
LEVELS   = [1, 2]

XBM_COLS = [
    'All items',
    'All items, less fresh food',
    'All items, less food (less alcoholic beverages) and energy',
]
XBMPLUS_COLS = XBM_COLS + ['All items, less fresh food and energy']

OUT_DIR = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════════════════
#  NON-NEGATIVE COMBINATION OF BENCHMARK SERIES
#
#  Same simplex constraint as the component-space assemblage applied to the
#  small set of benchmark aggregates:
#       min ‖y − α − Xb w‖²   s.t.   w ≥ 0,  Σw = 1   (α free if intercept)
# ══════════════════════════════════════════════════════════════════════════════

import cvxpy as cp


def _nn_fit(Xb: np.ndarray, y: np.ndarray, intercept: bool) -> tuple[float, np.ndarray]:
    _, k = Xb.shape
    w = cp.Variable(k)
    if intercept:
        a = cp.Variable()
        resid = y - Xb @ w - a
    else:
        a = None
        resid = y - Xb @ w

    prob = cp.Problem(cp.Minimize(cp.sum_squares(resid)),
                      [w >= 0, cp.sum(w) == 1])
    prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        prob.solve(solver=cp.SCS, verbose=False)

    wv = np.clip(np.asarray(w.value).flatten(), 0.0, None)
    if wv.sum() > 0:
        wv = wv / wv.sum()
    av = float(a.value) if intercept else 0.0
    return av, wv


def _nn_oos(growth: pd.DataFrame, y: np.ndarray,
            dates: pd.DatetimeIndex, oos_dates: pd.DatetimeIndex,
            cols: list[str], intercept: bool,
            min_train: int = MIN_TRAIN) -> pd.Series:
    """Expanding-window non-negative-combination forecast at each OOS date."""
    missing = [c for c in cols if c not in growth.columns]
    if missing:
        raise KeyError(f'missing benchmark cols: {missing}')

    Xb = np.column_stack([growth[c].reindex(dates).ffill().values for c in cols])

    out = []
    for date in oos_dates:
        t = np.where(dates == date)[0][0]
        if t < min_train:
            continue
        a, w = _nn_fit(Xb[:t], y[:t], intercept)
        out.append({'date': date, 'predicted': float(a + Xb[t] @ w)})
    return pd.DataFrame(out).set_index('date')['predicted']


def all_benchmarks(growth, y, dates, oos_dates) -> dict:
    return {
        'Xbm':         _nn_oos(growth, y, dates, oos_dates, XBM_COLS,     True),
        'Xbm (w0=0)':  _nn_oos(growth, y, dates, oos_dates, XBM_COLS,     False),
        'Xbm+':        _nn_oos(growth, y, dates, oos_dates, XBMPLUS_COLS, True),
        'Xbm+ (w0=0)': _nn_oos(growth, y, dates, oos_dates, XBMPLUS_COLS, False),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CELL DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def run_cell(level: int, horizon: int) -> dict:
    print(f'\n══ Level {level}, h={horizon} ══════════════════════════════════════════════════')
    X, y, w_prior, _features, dates, growth = prepare_regression_data(
        level=level, horizon=horizon)
    print(f'  data: {len(y)} obs ({dates[0]:%Y-%m}–{dates[-1]:%Y-%m}), K={X.shape[1]}')

    print('  → Albacorecomps (expanding)…')
    oos_c, _ = rolling_oos_comps(X, y, w_prior, dates, window=None)

    print('  → Albacoreranks (expanding)…')
    oos_r, _ = rolling_oos_ranks(X, y, dates, window=None)

    print('  → benchmarks…')
    bm_growth = load_benchmark_series().combine_first(growth)
    bms = all_benchmarks(bm_growth, y, dates, oos_c.index)

    return {
        'K':       X.shape[1],
        'actual':  oos_c['actual'],
        'comps':   oos_c['predicted'],
        'ranks':   oos_r['predicted'].reindex(oos_c.index),
        **bms,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC + GRID
# ══════════════════════════════════════════════════════════════════════════════

def rmse(actual: pd.Series, pred: pd.Series, start: str, end: str) -> float:
    mask = (pred.index >= start) & (pred.index <= end)
    a, p = actual[mask].values, pred[mask].values
    valid = ~(np.isnan(a) | np.isnan(p))
    return float(np.sqrt(np.mean((a[valid] - p[valid]) ** 2))) if valid.any() else np.nan


def build_grid(cells: dict) -> tuple[dict, dict]:
    """Returns (grid, Ks). grid is {(section, model_name): {(period, h): value}}."""
    grid = {}
    Ks = {l: cells[(l, HORIZONS[0])]['K'] for l in LEVELS}

    def fill(section, name, key, level=None):
        row = {}
        for plabel, pstart, pend in PERIODS:
            for h in HORIZONS:
                cell = cells[(level if level is not None else LEVELS[0], h)]
                num = rmse(cell['actual'], cell[key],   pstart, pend)
                den = rmse(cell['actual'], cell['Xbm'], pstart, pend)
                row[(plabel, h)] = num / den if den else np.nan
        grid[(section, name)] = row

    for level in LEVELS:
        section = f'Level {level} (K={Ks[level]})'
        fill(section, 'Albacorecomps', 'comps', level=level)
        fill(section, 'Albacoreranks', 'ranks', level=level)

    fill('Benchmarks', 'Xbm (w0=0)',  'Xbm (w0=0)')
    fill('Benchmarks', 'Xbm+',        'Xbm+')
    fill('Benchmarks', 'Xbm+ (w0=0)', 'Xbm+ (w0=0)')

    return grid, Ks


def to_dataframe(grid: dict) -> pd.DataFrame:
    rows = []
    for (section, name), vals in grid.items():
        rec = {'section': section, 'model': name}
        for (plabel, h), v in vals.items():
            rec[f'{plabel} | h={h}'] = v
        rows.append(rec)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def print_table(grid: dict) -> None:
    cols = [(p[0], h) for p in PERIODS for h in HORIZONS]
    width = 30 + 9 * len(cols)

    print()
    print('=' * width)
    print('Forecasting Performance of Albacore for Japan')
    print('  RMSE relative to Xbm with intercept (= 1.00, omitted)')
    print('=' * width)

    h1 = f'{"":<28}'
    block = 9 * len(HORIZONS)
    for plabel, _, _ in PERIODS:
        h1 += f' {plabel:^{block}}'
    print(h1)

    h2 = f'{"":<28}'
    for _ in PERIODS:
        for h in HORIZONS:
            h2 += f' {"h="+str(h):>8}'
    print(h2)
    print('-' * width)

    current = None
    for (section, name), vals in grid.items():
        if section != current:
            print(f'\n{section}')
            current = section
        line = f'  {name:<26}'
        for k in cols:
            v = vals[k]
            line += f' {v:>8.2f}' if not np.isnan(v) else f' {"--":>8}'
        print(line)
    print('=' * width)


def _to_latex_name(name: str) -> str:
    if name == 'Xbm':         return r'$X^{bm}_t$'
    if name == 'Xbm (w0=0)':  return r'$X^{bm}_t$, ($w_0{=}0$)'
    if name == 'Xbm+':        return r'$X^{bm+}_t$'
    if name == 'Xbm+ (w0=0)': return r'$X^{bm+}_t$, ($w_0{=}0$)'
    return name


def save_latex(grid: dict, path: Path) -> None:
    n_h = len(HORIZONS)
    n_cols = 1 + n_h * len(PERIODS)
    cspec = 'l' + 'c' * (n_h * len(PERIODS))

    L = []
    L.append(r'\begin{table}[htbp]')
    L.append(r'\centering')
    L.append(r'\caption{Forecasting Performance of Albacore for Japan}')
    L.append(r'\label{tab:horizon_table}')
    L.append(rf'\begin{{tabular}}{{{cspec}}}')
    L.append(r'\toprule')

    L.append(' & ' + ' & '.join(
        rf'\multicolumn{{{n_h}}}{{c}}{{{p[0].replace("-", "--")}}}'
        for p in PERIODS) + r' \\')
    L.append(' '.join(
        rf'\cmidrule(lr){{{2 + i * n_h}-{1 + (i + 1) * n_h}}}'
        for i in range(len(PERIODS))))
    L.append(r'$h \to$ & ' + ' & '.join(
        f'{h}' for _ in PERIODS for h in HORIZONS) + r' \\')
    L.append(r'\midrule')

    current = None
    for (section, name), vals in grid.items():
        if section != current:
            L.append(rf'\multicolumn{{{n_cols}}}{{l}}{{\textit{{{section}}}}} \\')
            current = section
        cells = ' & '.join(
            (f'{vals[(p[0], h)]:.2f}' if not np.isnan(vals[(p[0], h)]) else '--')
            for p in PERIODS for h in HORIZONS)
        L.append(f'{_to_latex_name(name)} & {cells} \\\\')

    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(
        r'\begin{tablenotes}\footnotesize'
        r'\item RMSE relative to $X^{bm}_t = [\textsc{All items}, '
        r'\text{core}_{\text{ff}}, \text{core}_{\text{fe}}]$ with intercept. '
        r'$X^{bm+}_t$ adds \textit{All items, less fresh food and energy} as '
        r'a 4th regressor. $w_0{=}0$ denotes no intercept. Expanding window '
        r'from 1990. Albacorecomps and Albacoreranks evaluated at levels 1 '
        r'and 2 of the Japan CPI hierarchy.'
        r'\end{tablenotes}')
    L.append(r'\end{table}')

    Path(path).write_text('\n'.join(L) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cells = {}
    for level in LEVELS:
        for h in HORIZONS:
            cells[(level, h)] = run_cell(level, h)

    grid, Ks = build_grid(cells)
    df = to_dataframe(grid)

    print_table(grid)

    csv_path = OUT_DIR / 'horizon_table.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n  CSV   → {csv_path}')

    tex_path = OUT_DIR / 'horizon_table.tex'
    save_latex(grid, tex_path)
    print(f'  LaTeX → {tex_path}')


if __name__ == '__main__':
    main()
