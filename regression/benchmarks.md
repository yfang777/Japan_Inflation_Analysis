# regression/benchmarks.py

Naive and OLS benchmark models for comparing assemblage regression performance.

## Functions

| Function | Description |
|---|---|
| `compute_benchmarks(growth, oos_dates)` | Three naive predictors aligned to OOS dates: random walk (current headline 3m/3m), core ex fresh food, core ex food & energy |
| `compute_mean_benchmark(y, dates, oos_dates, min_train)` | Unconditional expanding-mean predictor. The R²=0 floor -- any model that can't beat this is useless. |
| `compute_ols_benchmark(growth, y, dates, oos_dates, min_train)` | OLS regression on headline + core + supercore (the paper's Xbm_t). Expanding-window, re-estimated at each OOS step. |

## Benchmark Descriptions

| Benchmark | Predictor at time t |
|---|---|
| Random walk | Current headline 3m/3m growth rate |
| Core (ex fresh food) | Current core CPI 3m/3m (Japan's standard published core) |
| Core (ex food & energy) | Current supercore 3m/3m |
| Unconditional mean | Expanding mean of y up to time t |
| OLS | Fitted value from OLS on 3 core measures, trained on [0, t) |
