"""AR/horizon_table.py — RW-normalised RMSE table (paper's Table 3 analogue)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from AR.config_ar import HORIZONS, WINDOWS
from AR.data_ar import build_ar_dataset
from AR.oos import run_oos

OUT_DIR = Path(__file__).parent

MODEL_ORDER = ['AR_lags', 'AR+_lags', 'AR_ranks', 'AR(1) on YoY']


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred,   dtype=float)
    valid = ~(np.isnan(a) | np.isnan(p))
    return float(np.sqrt(np.mean((a[valid] - p[valid]) ** 2))) \
        if valid.any() else np.nan


def build_grid(oos: pd.DataFrame) -> dict:
    """Returns {(window_label, model, h): rmse / rmse_RW}."""
    grid = {}
    for wlabel, wstart, wend in WINDOWS:
        ws, we = pd.Timestamp(wstart), pd.Timestamp(wend)
        win = oos[(oos['date'] >= ws) & (oos['date'] <= we)]
        for h in HORIZONS:
            cell = win[win['horizon'] == h]
            rw   = cell[cell['model'] == 'RW']
            den  = rmse(rw['actual'].values, rw['forecast'].values)
            for m in MODEL_ORDER:
                sub = cell[cell['model'] == m]
                num = rmse(sub['actual'].values, sub['forecast'].values)
                grid[(wlabel, m, h)] = (num / den) \
                    if den and not np.isnan(den) else np.nan
    return grid


def grid_to_df(grid: dict) -> pd.DataFrame:
    rows = []
    for m in MODEL_ORDER:
        rec = {'model': m}
        for wlabel, *_ in WINDOWS:
            for h in HORIZONS:
                rec[f'{wlabel} | h={h}'] = grid[(wlabel, m, h)]
        rows.append(rec)
    return pd.DataFrame(rows)


def print_table(grid: dict, title: str = 'AR family on Japan CPI') -> None:
    n_h = len(HORIZONS)
    width = 22 + (9 * n_h + 1) * len(WINDOWS)
    print()
    print('=' * width)
    print(f'{title}  —  RMSE / RMSE(RW)   (RW = 1.00 by construction)')
    print('=' * width)

    h1 = f'{"":<20}'
    block = 9 * n_h
    for wlabel, *_ in WINDOWS:
        h1 += f' {wlabel:^{block}}'
    print(h1)

    h2 = f'{"":<20}'
    for _ in WINDOWS:
        for h in HORIZONS:
            h2 += f' {"h="+str(h):>8}'
    print(h2)
    print('-' * width)

    for m in MODEL_ORDER:
        line = f'  {m:<18}'
        for wlabel, *_ in WINDOWS:
            for h in HORIZONS:
                v = grid[(wlabel, m, h)]
                line += f' {v:>8.2f}' if not np.isnan(v) else f' {"--":>8}'
        print(line)
    print('=' * width)


def _to_latex_name(m: str) -> str:
    return {
        'AR_lags':       r'$\text{AR}_{\text{lags}}(12)$',
        'AR+_lags':      r'$\text{AR}^{+}_{\text{lags}}(12)$',
        'AR_ranks':      r'$\text{AR}_{\text{ranks}}(12)$',
        'AR(1) on YoY':  r'AR(1) on YoY',
    }[m]


def save_latex(grid: dict, out_path: Path,
               caption: str, label: str) -> None:
    n_h = len(HORIZONS)
    n_w = len(WINDOWS)
    n_cols = 1 + n_h * n_w
    cspec = 'l' + 'c' * (n_h * n_w)

    col_min = {}
    for wlabel, *_ in WINDOWS:
        for h in HORIZONS:
            vals = [grid[(wlabel, m, h)] for m in MODEL_ORDER]
            valid = [v for v in vals if not np.isnan(v)]
            col_min[(wlabel, h)] = min(valid) if valid else np.nan

    L = []
    L.append(r'\begin{table}[htbp]')
    L.append(r'\centering')
    L.append(rf'\caption{{{caption}}}')
    L.append(rf'\label{{{label}}}')
    L.append(rf'\begin{{tabular}}{{{cspec}}}')
    L.append(r'\toprule')
    L.append(' & ' + ' & '.join(
        rf'\multicolumn{{{n_h}}}{{c}}{{{w[0]}}}' for w in WINDOWS) + r' \\')
    L.append(' '.join(
        rf'\cmidrule(lr){{{2 + i * n_h}-{1 + (i + 1) * n_h}}}'
        for i in range(n_w)))
    L.append(r'$h \to$ & ' + ' & '.join(
        f'{h}' for _ in WINDOWS for h in HORIZONS) + r' \\')
    L.append(r'\midrule')

    for m in MODEL_ORDER:
        cells = []
        for wlabel, *_ in WINDOWS:
            for h in HORIZONS:
                v = grid[(wlabel, m, h)]
                if np.isnan(v):
                    cells.append('--')
                elif not np.isnan(col_min[(wlabel, h)]) \
                        and abs(v - col_min[(wlabel, h)]) < 1e-8:
                    cells.append(rf'\textbf{{{v:.2f}}}')
                else:
                    cells.append(f'{v:.2f}')
        L.append(f'{_to_latex_name(m)} & ' + ' & '.join(cells) + r' \\')

    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(
        r'\begin{tablenotes}\footnotesize'
        r'\item RMSE relative to the random-walk forecast '
        r'($\hat{y}^{\text{RW}}_{t+h} = $ current 12-month YoY at $t$, '
        r'$=1.00$ by construction, omitted). Column minimum bolded. '
        r'Expanding-window OOS, training start 1995-01.'
        r'\end{tablenotes}')
    L.append(r'\end{table}')

    Path(out_path).write_text('\n'.join(L) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
#  driver
# ══════════════════════════════════════════════════════════════════════════════

def run(headline_col: str = 'All items',
        suffix: str = '',
        title_suffix: str = '') -> dict:
    print(f'\n══ horizon table — series={headline_col!r} ══════════════════════════')
    ds  = build_ar_dataset(headline_col=headline_col)
    print('  running expanding-window OOS …')
    oos = run_oos(ds)
    grid = build_grid(oos)

    title = f'AR family on Japan {headline_col}{title_suffix}'
    print_table(grid, title=title)

    csv_path = OUT_DIR / f'horizon_table{suffix}.csv'
    grid_to_df(grid).to_csv(csv_path, index=False)
    print(f'  CSV   -> {csv_path}')

    tex_path = OUT_DIR / f'horizon_table{suffix}.tex'
    save_latex(grid, tex_path,
               caption=f'Forecasting Performance: {title}',
               label=f'tab:ar_horizon{suffix}')
    print(f'  LaTeX -> {tex_path}')

    oos_path = OUT_DIR / f'oos{suffix}.csv'
    oos.to_csv(oos_path, index=False)
    print(f'  OOS   -> {oos_path}')

    return {'ds': ds, 'oos': oos, 'grid': grid}


def main() -> None:
    run(headline_col='All items', suffix='', title_suffix='')


if __name__ == '__main__':
    main()
