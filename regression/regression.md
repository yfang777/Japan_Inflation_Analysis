# regression/

Assemblage regression for Japan CPI, following the methodology in
Goulet Coulombe, Galipaud, Goulet, Stevanović (2024), *Maximally
Forward-Looking Core Inflation*. Component or rank weights are chosen
to maximise predictive power for future headline inflation, shrunk
toward the official basket prior.

## Models

### Albacore$_{\text{comps}}$ — component-space assemblage

```
min_w  mean(y - X w)²  +  λ ‖w − w_prior‖²
s.t.   w ≥ 0,  Σw = 1
```

Weights live on the simplex; the prior is the basket share. λ is selected
by expanding-window time-series CV. Implemented in
[`regression_component.py`](regression_component.py) (quadprog QP solver).

### Albacore$_{\text{ranks}}$ — rank-space assemblage

```
min_w  mean(y - O w)²  +  λ Σ (w_{r+1} − w_r)²        (fused ridge)
s.t.   w ≥ 0,  Ō' w = ȳ                                (mean constraint)
```

`O` is the row-wise sorted growth-rate matrix; weights are tied to
percentile positions rather than named components. The mean constraint
replaces the simplex. Implemented in
[`regression_rank.py`](regression_rank.py) (SLSQP, parallelised over λ).

### Benchmarks

Section-3-style benchmarks: a non-negative simplex combination of a small
set of aggregate series, fit under the same constraint as Albacore so
that any performance gap reflects the choice of regressors.

| Benchmark | Regressors |
|---|---|
| $X^{bm}_t$ | headline + core (less food & energy) + in-house 10 % trimmed mean |
| $X^{bm}_t$, $w_0 = 0$ | same regressors, no intercept |
| $X^{bm+}_t$ | adds *less fresh food and energy* as a fourth regressor |
| $X^{bm+}_t$, $w_0 = 0$ | same, no intercept |

The 10 % symmetric weighted trimmed-mean series is constructed in-house
from `data_clean/level3.csv` via
`utils.data_load.load_trimmed_mean_3m3m` — no external series required.

## Files

### Models

| File | Description |
|---|---|
| [`regression_component.py`](regression_component.py) | Standard Albacore$_{\text{comps}}$. Quadprog QP, expanding-window CV picks λ on the first window then holds it fixed. |
| [`regression_component_correct.py`](regression_component_correct.py) | R-parity build matching `nonneg.ridge.sum1()`: SSR loss, per-feature sd-weighted ridge, 10-fold contiguous-block CV, λ re-tuned every rolling window. CVXPY/OSQP. |
| [`regression_component_no_restriction.py`](regression_component_no_restriction.py) | Drops the `w ≥ 0` inequality (keeps Σw = 1 only); replaces `smart_impute` with `fillna(0)`. Heavy-shrinkage λ grid (1e3–1e7). |
| [`regression_component_OLS.py`](regression_component_OLS.py) | Plain OLS on level-3 components with intercept and `fillna(0)`. No constraint, no λ — minimum-norm least squares. |
| [`regression_rank.py`](regression_rank.py) | Albacore$_{\text{ranks}}$. Sorted-component matrix with fused-ridge penalty and mean constraint. |

### Forecast tables

| File | Sample / window | Variants |
|---|---|---|
| [`horizon_table.py`](horizon_table.py) | Expanding window from 1990 | `comps`, `ranks` (+ benchmarks) |
| [`horizon_table_rolling.py`](horizon_table_rolling.py) | Rolling 20-year window (paper protocol) | `comps`, `ranks` (+ benchmarks) |
| [`horizon_table_correct.py`](horizon_table_correct.py) | Rolling 20-year window | `correct`, `no_restriction` (+ benchmarks) |

All three sweep `level ∈ {1, 2}` × `horizon ∈ {1, 3, 6, 12, 24}` and
report RMSE relative to $X^{bm}_t$ for the periods 2010m1–2019m12 and
2020m1–2024m12.

### Helpers

| File | Description |
|---|---|
| [`benchmarks.py`](benchmarks.py) | Naive benchmarks (random walk, core ex fresh food, core ex food & energy), unconditional mean, OLS combiner. |
| [`evaluation.py`](evaluation.py) | RMSE / MAE / R² metrics and the `print_scorecard` formatter. |
| [`figures.py`](figures.py) | Plot helpers: weights vs prior, λ-CV curve, in-sample fit, OOS time series + rolling RMSE, rank-space weight bars. |

### Outputs

```
rolling_20y/
├── description.tex                 # paper-style narrative + final table
├── horizon_table.{csv,tex}         # standard variants (comps + ranks)
├── horizon_table_correct.{csv,tex}         # R-parity variant
├── horizon_table_no_restriction.{csv,tex}  # unconstrained variant
├── cells/                          # pickled per-cell results
│   ├── standard_l{1,2}_h{1,3,6,12,24}.pkl
│   ├── correct_l{1,2}_h{1,3,6,12,24}.pkl
│   └── no_restriction_l{1,2}_h{1,3,6,12,24}.pkl
└── cell_logs/                      # stdout from per-cell runs
```

The `cells/*.pkl` artefacts let you parallelise the slow per-cell runs
across processes (e.g. on a cluster) and aggregate later — see the
`--single-cell` / `--aggregate` flags below.

## Running

### Single model, single level

```bash
# Standard component-space, level 2
python regression/regression_component.py --level 2

# Rank-space, level 1
python regression/regression_rank.py --level 1

# OLS baseline on level 3
python regression/regression_component_OLS.py --level 3
```

Each `regression_component*.py` script writes figures into
`plots/level{level}_component[_correct|_no_restriction]/`;
`regression_rank.py` writes into `plots/level{level}_rank/`.

### Horizon × period tables (paper Table)

```bash
# Sequential, end-to-end (slow — runs every cell in this process)
python regression/horizon_table_rolling.py
python regression/horizon_table_correct.py

# Parallel: pickle one cell at a time, aggregate at the end
python regression/horizon_table_rolling.py --single-cell 2 12
python regression/horizon_table_correct.py --single-cell correct 2 12
# ...repeat for every (level, horizon) — easy to fan out via shell loop / SLURM
python regression/horizon_table_rolling.py --aggregate
python regression/horizon_table_correct.py --aggregate
```

The full-sample (expanding-window) variant is
[`horizon_table.py`](horizon_table.py); rerun with no flags to refresh
`horizon_table.{csv,tex}`.

## Configuration

Run-wide knobs live in `../config.py`:

| Constant | Default | Meaning |
|---|---|---|
| `START_DATE` | `'1990-01-01'` | Regression sample start |
| `HORIZON` | `12` | Months-ahead averaging for the target |
| `MIN_TRAIN` | `120` | First OOS observation index |
| `OOS_STEP` | `1` | OOS step size (1 = monthly) |
| `ROLLING_WINDOW` | `240` | 20-year window for rolling OOS |
| `LAMBDA_GRID` | `np.logspace(-4, 3, 15)` | λ grid for `comps` |
| `LAMBDA_GRID_RANKS` | `np.logspace(-4, 3, 15)` | λ grid for `ranks` |
| `N_CV_FOLDS` | `5` | CV folds for `comps` / `ranks` |
| `COMPOSITE_COLS`, `SPECIAL_COLS` | – | Aggregate / derived columns excluded from features |
| `PLOTS_DIR` | `plots/` | Figure output root |

## Dependencies

`numpy`, `pandas`, `scipy`, `matplotlib`, `quadprog`, `cvxpy`, `joblib`.
See `../environment.yml`.
