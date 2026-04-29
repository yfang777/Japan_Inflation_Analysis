"""
regression/horizon_table_correct.py — horizon × period RMSE tables for two
component-space assemblage variants, evaluated on a 20-YEAR ROLLING WINDOW
(matches the paper's Section 3 evaluation protocol).

Variants:
    1. correct          — R-parity build (matches R::nonneg.ridge.sum1):
                          SSR loss, per-feature sd-weighted ridge, 10-fold
                          contiguous-block CV, λ re-tuned per rolling-window
                          step (shrunk 7-point grid).
    2. no_restriction   — drops w ≥ 0 (keeps Σw = 1), bypasses smart_impute
                          with fillna(0), λ fixed from first-window CV.

Evaluation grid (paper-aligned):
    levels      {1, 2}                  (Japan CPI hierarchy stand-in for US 2/3/6)
    horizons    {1, 3, 6, 12, 24}
    window      ROLLING 20 years        (= ROLLING_WINDOW = 240 months)
    periods     2010m1–2019m12   and   2020m1–2024m12

Benchmarks (Xbm) — paper-aligned:
    Xbm        = NN combination of [headline, core_fe, trimmed-mean(10%)]
                 with free intercept; weights ≥ 0, Σw = 1.
    Xbm (w0=0) = same regressors, no intercept.
    Xbm+       = adds 'less fresh food and energy' as 4th regressor.
    Xbm+ (w0=0)= same, no intercept.

Trimmed-mean inflation is constructed in-house from the disaggregated
level3.csv (647 subindices, BoJ-style 10% symmetric weighted trim) — no
external data series required. See utils.data_load.load_trimmed_mean_3m3m.

Documented deviation from the paper (rank pipeline):
    The paper computes Albacoreranks from MONTH-OVER-MONTH growth rates and
    then smooths the order-statistic series with a 3-month MA. We instead
    rank the 3m/3m growth rates directly (skipping the explicit 3m MA, since
    3m/3m is itself a smoothed measure). Under continuous-time / log returns
    these are identical up to a scalar (see Roger 1997); for small monthly
    growth they are extremely close. The current code inherits this from
    regression.regression_rank.build_rank_matrix.

Run modes:
    python horizon_table_correct.py                       — sequential, end-to-end
    python horizon_table_correct.py --single-cell V L H   — one cell → pickle (parallel)
    python horizon_table_correct.py --aggregate           — pickle dir → CSV + LaTeX

Output (alongside this file):
    rolling_20y/
        horizon_table_correct.{csv,tex}
        horizon_table_no_restriction.{csv,tex}
        cells/{variant}_l{level}_h{horizon}.pkl
"""

import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

sys.path.insert(0, str(Path(__file__).parent.parent))   # project root

from config import (
    START_DATE, COMPOSITE_COLS, SPECIAL_COLS,
    MIN_TRAIN, ROLLING_WINDOW,
)
from utils.data_load import (
    prepare_regression_data, load_benchmark_series,
    load_level_data, compute_growth_3m3m, compute_forward_target,
    load_trimmed_mean_3m3m,
)
from regression.regression_rank import rolling_oos_ranks
from regression.horizon_table import rmse, _to_latex_name
from regression import regression_component_correct as rc_correct
from regression import regression_component_no_restriction as rc_norestrict


# ── grid ──────────────────────────────────────────────────────────────────────

PERIODS = [
    ('2010m1-2019m12', '2010-01-01', '2019-12-31'),
    ('2020m1-2024m12', '2020-01-01', '2024-12-31'),
]
HORIZONS = [1, 3, 6, 12, 24]
LEVELS   = [1, 2]
WINDOW   = ROLLING_WINDOW           # 240 months = 20-year rolling window

LAMBDA_GRID_CORRECT = np.logspace(-4, 3, 7)

# Trimmed-mean column name — must match what load_trimmed_mean_3m3m produces.
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
#  NON-NEGATIVE COMBINATION OF BENCHMARK SERIES
#
#  Mirrors the paper's "Similar to our main setup, we include the different
#  series in a nonnegative regression" — i.e. the same simplex constraint as
#  the component-space assemblage applied to a small set of benchmark series.
# ══════════════════════════════════════════════════════════════════════════════

def _nn_fit(Xb: np.ndarray, y: np.ndarray, intercept: bool) -> tuple[float, np.ndarray]:
    """min ||y − α − Xb w||²   s.t.  w ≥ 0,  Σw = 1   (α free if intercept)."""
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
    """Build benchmark growth frame: officially-published cores ⊕ headline-from-
    level ⊕ in-house trimmed-mean."""
    bm = load_benchmark_series().combine_first(growth)
    tm = load_trimmed_mean_3m3m(start_date=START_DATE, trim_pct=TRIM_PCT)
    bm[TRIMMED_NAME] = tm.reindex(bm.index)
    return bm


# ══════════════════════════════════════════════════════════════════════════════
#  DATA — horizon-parameterised variant of prepare_regression_data_fillna0
# ══════════════════════════════════════════════════════════════════════════════

def prepare_norestrict(level: int, horizon: int):
    """Mirror of rc_norestrict.prepare_regression_data_fillna0 with `horizon`
    as a parameter (the original hardcodes config.HORIZON)."""
    df, weights_dict = load_level_data(level, START_DATE)
    growth       = compute_growth_3m3m(df)
    headline_col = df.columns[0]
    target       = compute_forward_target(growth[headline_col], horizon)

    _exclude = set(COMPOSITE_COLS) | set(SPECIAL_COLS) | {headline_col}
    features = [c for c in df.columns
                if c not in _exclude and not c.startswith('Unnamed:')]

    valid = target.notna() & growth[headline_col].notna()
    X_df  = growth[features][valid].fillna(0.0)

    w_raw   = np.array([weights_dict[c] for c in features], dtype=float)
    w_prior = w_raw / w_raw.sum()

    return X_df.values, target[valid].values, w_prior, features, X_df.index, growth


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CELL DRIVERS
# ══════════════════════════════════════════════════════════════════════════════

def _pack(K, oos_c, oos_r, bms):
    return {
        'K':       K,
        'actual':  oos_c['actual'],
        'comps':   oos_c['predicted'],
        'ranks':   oos_r['predicted'].reindex(oos_c.index),
        **bms,
    }


def run_cell_correct(level: int, horizon: int, rank_n_jobs: int = -1) -> dict:
    print(f'\n══ [correct] Level {level}, h={horizon} ════════════════════════════════════')
    X, y, w_prior, _f, dates, growth = prepare_regression_data(level=level, horizon=horizon)
    print(f'  data: {len(y)} obs ({dates[0]:%Y-%m}–{dates[-1]:%Y-%m}), K={X.shape[1]}, '
          f'λ-grid={len(LAMBDA_GRID_CORRECT)} pts, window=rolling {WINDOW}m')

    print('  → Albacorecomps (correct, rolling 20y, per-window CV)…')
    oos_c, _ = rc_correct.rolling_oos(X, y, w_prior, dates,
                                       lambdas=LAMBDA_GRID_CORRECT,
                                       window=WINDOW, n_folds=10)

    print('  → Albacoreranks (rolling 20y)…')
    oos_r, _ = rolling_oos_ranks(X, y, dates, window=WINDOW, n_jobs=rank_n_jobs)

    print('  → benchmarks (NN combiner; Xbm = headline + core_fe + trimmed-mean)…')
    bm_growth = _bm_growth_with_trim(growth)
    bms = all_benchmarks_nn(bm_growth, y, dates, oos_c.index)
    return _pack(X.shape[1], oos_c, oos_r, bms)


def run_cell_norestrict(level: int, horizon: int, rank_n_jobs: int = -1) -> dict:
    print(f'\n══ [no_restriction] Level {level}, h={horizon} ══════════════════════════════')
    X, y, w_prior, _f, dates, growth = prepare_norestrict(level=level, horizon=horizon)
    print(f'  data: {len(y)} obs ({dates[0]:%Y-%m}–{dates[-1]:%Y-%m}), K={X.shape[1]}, '
          f'window=rolling {WINDOW}m')

    print('  → Albacorecomps (no_restriction, rolling 20y, fixed λ)…')
    oos_c, _ = rc_norestrict.rolling_oos(X, y, w_prior, dates, window=WINDOW)

    print('  → Albacoreranks (rolling 20y)…')
    oos_r, _ = rolling_oos_ranks(X, y, dates, window=WINDOW, n_jobs=rank_n_jobs)

    print('  → benchmarks (NN combiner; Xbm = headline + core_fe + trimmed-mean)…')
    bm_growth = _bm_growth_with_trim(growth)
    bms = all_benchmarks_nn(bm_growth, y, dates, oos_c.index)
    return _pack(X.shape[1], oos_c, oos_r, bms)


# ══════════════════════════════════════════════════════════════════════════════
#  PICKLE I/O FOR PARALLEL ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _cell_path(variant: str, level: int, horizon: int) -> Path:
    return CELL_DIR / f'{variant}_l{level}_h{horizon}.pkl'


def run_single_cell(variant: str, level: int, horizon: int) -> None:
    """Run one cell and dump its pickle. For shell-orchestrated parallelism."""
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    fn = run_cell_correct if variant == 'correct' else run_cell_norestrict
    cell = fn(level, horizon, rank_n_jobs=1)   # serial inside the worker
    out = _cell_path(variant, level, horizon)
    out.write_bytes(pickle.dumps(cell))
    print(f'  → wrote {out}')


def load_cells(variant: str) -> dict:
    """Load all (level, horizon) pickles for a variant. Errors if any missing."""
    cells = {}
    for level in LEVELS:
        for h in HORIZONS:
            p = _cell_path(variant, level, h)
            if not p.exists():
                raise FileNotFoundError(f'missing: {p}')
            cells[(level, h)] = pickle.loads(p.read_bytes())
    return cells


# ══════════════════════════════════════════════════════════════════════════════
#  GRID + OUTPUT
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
        rf'\item RMSE relative to $X^{{bm}}_t$, a non-negative combination '
        r'(weights $\ge 0$, $\sum w = 1$, free intercept) of '
        r'$[\textsc{All items}, \text{core}_{\text{fe}}, '
        r'\text{trimmed mean (10\%)}]$. $X^{bm+}_t$ adds \textit{All items, '
        r'less fresh food and energy} as a 4th regressor. $w_0{=}0$ denotes '
        r'no intercept. Trimmed mean is constructed in-house from level-3 '
        rf'subindices (BoJ-style 10\% symmetric weighted trim). Estimation '
        rf'uses a {WINDOW}-month rolling window. {comps_note} '
        r'\textit{Albacoreranks note}: ranks computed on 3m/3m growth '
        r'(rather than the paper''s m/m + 3m-MA pipeline) — equivalent '
        r'under log returns and effectively identical for small monthly '
        r'growth rates.'
        r'\end{tablenotes}')
    L.append(r'\end{table}')

    Path(path).write_text('\n'.join(L) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _emit_outputs(cells: dict, variant: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, _ = build_grid(cells)
    df = to_dataframe(grid)
    if variant == 'correct':
        title  = 'Forecasting Performance — Albacorecomps (R-parity, rolling 20y)'
        csv    = OUT_DIR / 'horizon_table_correct.csv'
        tex    = OUT_DIR / 'horizon_table_correct.tex'
        cap    = (r'Forecasting Performance: Albacore$_{\text{comps,correct}}$ '
                  r'for Japan (rolling 20y window)')
        label  = 'tab:horizon_table_correct_rolling'
        note   = (
            r'The component-space variant uses an R-parity build with '
            r'sum-of-squared-residuals loss, a per-feature standard-deviation-'
            r'weighted ridge penalty toward the basket prior, and 10-fold '
            r'contiguous-block CV re-tuning of $\lambda$ at every rolling '
            r'window (shrunk 7-point grid).'
        )
    else:
        title  = 'Forecasting Performance — Albacorecomps (no $w \\ge 0$, fillna 0, rolling 20y)'
        csv    = OUT_DIR / 'horizon_table_no_restriction.csv'
        tex    = OUT_DIR / 'horizon_table_no_restriction.tex'
        cap    = (r'Forecasting Performance: Albacore$_{\text{comps,no\_restriction}}$ '
                  r'for Japan (rolling 20y window)')
        label  = 'tab:horizon_table_no_restriction_rolling'
        note   = (
            r'The component-space variant here drops the $w \ge 0$ inequality '
            r'(retaining only $\sum w = 1$, so weights may be negative) and '
            r'replaces \texttt{smart\_impute} with \texttt{fillna(0)} for '
            r'missing 3m/3m growth rates. $\lambda$ is selected once on the '
            r'first-window CV and held fixed thereafter.'
        )
    print_table(grid, title)
    df.to_csv(csv, index=False); print(f'\n  CSV   → {csv}')
    save_latex(grid, tex, caption=cap, label=label, comps_note=note)
    print(f'  LaTeX → {tex}')


def main_sequential() -> None:
    for variant in ('correct', 'no_restriction'):
        cells = {}
        fn = run_cell_correct if variant == 'correct' else run_cell_norestrict
        for level in LEVELS:
            for h in HORIZONS:
                cells[(level, h)] = fn(level, h)
        _emit_outputs(cells, variant)


def main_aggregate() -> None:
    for variant in ('correct', 'no_restriction'):
        cells = load_cells(variant)
        _emit_outputs(cells, variant)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--single-cell', nargs=3, metavar=('VARIANT', 'LEVEL', 'HORIZON'),
                    help='run one cell and pickle it')
    ap.add_argument('--aggregate', action='store_true',
                    help='read pickled cells and emit CSV/LaTeX')
    args = ap.parse_args()

    if args.single_cell:
        v, l, h = args.single_cell
        if v not in ('correct', 'no_restriction'):
            sys.exit(f'unknown variant: {v}')
        run_single_cell(v, int(l), int(h))
    elif args.aggregate:
        main_aggregate()
    else:
        main_sequential()
