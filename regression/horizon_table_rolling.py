"""
regression/horizon_table_rolling.py — paper-aligned horizon × period RMSE
table for the standard Albacore variants:

    • Albacorecomps   — regression.regression_component
                        (component-space assemblage, quadprog QP, fixed λ)
    • Albacoreranks   — regression.regression_rank
                        (rank-space assemblage, fused-ridge mean-constrained)

Evaluation grid (matches regression/horizon_table_correct.py):
    levels      {1, 2}                    — Japan CPI hierarchy
    horizons    {1, 3, 6, 12, 24}
    window      ROLLING 20 years          — ROLLING_WINDOW = 240 months
    periods     2010m1–2019m12  and  2020m1–2024m12

Benchmarks (Xbm) — non-negative simplex combination of aggregates:
    Xbm        = [headline, core_fe, trimmed-mean(10%)] with free intercept
    Xbm (w0=0) = same regressors, no intercept
    Xbm+       = adds 'less fresh food and energy' as a 4th regressor
    Xbm+ (w0=0)= same, no intercept

The trimmed-mean series is constructed in-house from the disaggregated
level3.csv via utils.data_load.load_trimmed_mean_3m3m (BoJ-style 10%).

Run modes:
    python horizon_table_rolling.py                       — sequential
    python horizon_table_rolling.py --single-cell L H     — one cell → pickle
    python horizon_table_rolling.py --aggregate           — pickles → CSV+LaTeX

Output (alongside this file):
    rolling_20y/
        horizon_table.{csv,tex}
        cells/standard_l{level}_h{horizon}.pkl
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import START_DATE, MIN_TRAIN, ROLLING_WINDOW
from utils.data_load import (
    prepare_regression_data, load_benchmark_series, load_trimmed_mean_3m3m,
)
from regression.regression_component import rolling_oos as rolling_oos_comps
from regression.regression_rank import rolling_oos_ranks
from regression.horizon_table import rmse, _to_latex_name


# ── grid ──────────────────────────────────────────────────────────────────────

PERIODS = [
    ('2010m1-2019m12', '2010-01-01', '2019-12-31'),
    ('2020m1-2024m12', '2020-01-01', '2024-12-31'),
]
HORIZONS = [1, 3, 6, 12, 24]
LEVELS   = [1, 2]
WINDOW   = ROLLING_WINDOW

TRIM_PCT     = 10.0
TRIMMED_NAME = f'Trimmed mean ({int(TRIM_PCT)}%)'

XBM_COLS = [
    'All items',
    'All items, less food (less alcoholic beverages) and energy',
    TRIMMED_NAME,
]
XBMPLUS_COLS = XBM_COLS + ['All items, less fresh food and energy']

OUT_DIR  = Path(__file__).parent / 'rolling_20y'
CELL_DIR = OUT_DIR / 'cells'


# ══════════════════════════════════════════════════════════════════════════════
#  NN BENCHMARK FITTING (rolling window)
# ══════════════════════════════════════════════════════════════════════════════

def _nn_fit(Xb: np.ndarray, y: np.ndarray, intercept: bool) -> tuple[float, np.ndarray]:
    """min ‖y − α − Xb w‖²   s.t.  w ≥ 0,  Σw = 1   (α free if intercept)."""
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
            min_train: int = MIN_TRAIN, window: int = WINDOW) -> pd.Series:
    """Rolling-window NN combination forecast at each OOS date."""
    missing = [c for c in cols if c not in growth.columns]
    if missing:
        raise KeyError(f'missing benchmark cols: {missing}')

    Xb = np.column_stack([growth[c].reindex(dates).ffill().values for c in cols])
    out = []
    for date in oos_dates:
        t = np.where(dates == date)[0][0]
        if t < min_train:
            continue
        t_start = max(0, t - window) if window else 0
        a, w = _nn_fit(Xb[t_start:t], y[t_start:t], intercept)
        out.append({'date': date, 'predicted': float(a + Xb[t] @ w)})
    return pd.DataFrame(out).set_index('date')['predicted']


def all_benchmarks_nn(growth, y, dates, oos_dates) -> dict:
    return {
        'Xbm':         _nn_oos(growth, y, dates, oos_dates, XBM_COLS,     True),
        'Xbm (w0=0)':  _nn_oos(growth, y, dates, oos_dates, XBM_COLS,     False),
        'Xbm+':        _nn_oos(growth, y, dates, oos_dates, XBMPLUS_COLS, True),
        'Xbm+ (w0=0)': _nn_oos(growth, y, dates, oos_dates, XBMPLUS_COLS, False),
    }


def _bm_growth_with_trim(growth: pd.DataFrame) -> pd.DataFrame:
    bm = load_benchmark_series().combine_first(growth)
    tm = load_trimmed_mean_3m3m(start_date=START_DATE, trim_pct=TRIM_PCT)
    bm[TRIMMED_NAME] = tm.reindex(bm.index)
    return bm


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CELL DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def run_cell(level: int, horizon: int, rank_n_jobs: int = -1) -> dict:
    print(f'\n══ Level {level}, h={horizon} ══════════════════════════════════════════════')
    X, y, w_prior, _f, dates, growth = prepare_regression_data(
        level=level, horizon=horizon)
    print(f'  data: {len(y)} obs ({dates[0]:%Y-%m}–{dates[-1]:%Y-%m}), K={X.shape[1]}, '
          f'window=rolling {WINDOW}m')

    print('  → Albacorecomps (rolling 20y, fixed λ)…')
    oos_c, _ = rolling_oos_comps(X, y, w_prior, dates, window=WINDOW)

    print('  → Albacoreranks (rolling 20y)…')
    oos_r, _ = rolling_oos_ranks(X, y, dates, window=WINDOW, n_jobs=rank_n_jobs)

    print('  → benchmarks (NN combiner; Xbm = headline + core_fe + trimmed-mean)…')
    bm_growth = _bm_growth_with_trim(growth)
    bms = all_benchmarks_nn(bm_growth, y, dates, oos_c.index)

    return {
        'K':       X.shape[1],
        'actual':  oos_c['actual'],
        'comps':   oos_c['predicted'],
        'ranks':   oos_r['predicted'].reindex(oos_c.index),
        **bms,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PICKLE I/O FOR PARALLEL ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _cell_path(level: int, horizon: int) -> Path:
    return CELL_DIR / f'standard_l{level}_h{horizon}.pkl'


def run_single_cell(level: int, horizon: int) -> None:
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    cell = run_cell(level, horizon, rank_n_jobs=1)
    out = _cell_path(level, horizon)
    out.write_bytes(pickle.dumps(cell))
    print(f'  → wrote {out}')


def load_cells() -> dict:
    cells = {}
    for level in LEVELS:
        for h in HORIZONS:
            p = _cell_path(level, h)
            if not p.exists():
                raise FileNotFoundError(f'missing: {p}')
            cells[(level, h)] = pickle.loads(p.read_bytes())
    return cells


# ══════════════════════════════════════════════════════════════════════════════
#  GRID + OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def build_grid(cells: dict) -> tuple[dict, dict]:
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


def print_table(grid: dict, title: str) -> None:
    cols = [(p[0], h) for p in PERIODS for h in HORIZONS]
    width = 30 + 9 * len(cols)
    print()
    print('=' * width)
    print(title)
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


def save_latex(grid: dict, path: Path,
               caption: str, label: str, comps_note: str) -> None:
    n_h = len(HORIZONS)
    n_cols = 1 + n_h * len(PERIODS)
    cspec = 'l' + 'c' * (n_h * len(PERIODS))

    L = []
    L.append(r'\begin{table}[htbp]')
    L.append(r'\centering')
    L.append(rf'\caption{{{caption}}}')
    L.append(rf'\label{{{label}}}')
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
        rf'\item RMSE relative to $X^{{bm}}_t = [\text{{All items}}, '
        r'\text{core}_{\text{fe}}, \text{trim}_{10\%}]$, a non-negative '
        r'combination with free intercept ($w \ge 0$, $\sum w = 1$). '
        r'$X^{bm+}_t$ adds \textit{All items, less fresh food and energy} as '
        r'a 4th regressor. $w_0 = 0$ denotes no intercept. The trimmed-mean '
        r'series is constructed in-house from level3.csv (BoJ-style 10\% '
        rf'symmetric weighted trim). Rolling 20-year window. {comps_note}'
        r'\end{tablenotes}')
    L.append(r'\end{table}')

    Path(path).write_text('\n'.join(L) + '\n')


def emit_outputs(cells: dict) -> None:
    grid, _ = build_grid(cells)
    df = to_dataframe(grid)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    title = 'Forecasting Performance — Albacore (rolling 20y, paper-aligned)'
    print_table(grid, title)

    csv = OUT_DIR / 'horizon_table.csv'
    df.to_csv(csv, index=False)
    print(f'\n  CSV   → {csv}')

    tex = OUT_DIR / 'horizon_table.tex'
    save_latex(
        grid, tex,
        caption='Forecasting Performance: Albacore for Japan',
        label='tab:horizon_table_rolling',
        comps_note=(
            r'Standard Albacore variants from \texttt{regression/regression\_component.py} '
            r'and \texttt{regression/regression\_rank.py}.'
        ),
    )
    print(f'  LaTeX → {tex}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main_sequential() -> None:
    cells = {}
    for level in LEVELS:
        for h in HORIZONS:
            cells[(level, h)] = run_cell(level, h)
    emit_outputs(cells)


def main_aggregate() -> None:
    cells = load_cells()
    emit_outputs(cells)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--single-cell', nargs=2, metavar=('LEVEL', 'HORIZON'),
                    help='run one cell and pickle it')
    ap.add_argument('--aggregate', action='store_true',
                    help='read pickled cells and emit CSV/LaTeX')
    args = ap.parse_args()

    if args.single_cell:
        l, h = args.single_cell
        run_single_cell(int(l), int(h))
    elif args.aggregate:
        main_aggregate()
    else:
        main_sequential()
