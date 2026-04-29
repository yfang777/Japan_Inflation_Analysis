# data_clean/

Japanese CPI index data and the supporting metadata used by the regression
pipeline. Source: **Statistics Bureau of Japan**, Consumer Price Index, 2020
base. Monthly, January 1970 – present.

## Files

| File | Shape | Purpose |
|---|---|---|
| `level_1.csv` | 11 cols | Headline + 9 major sectors |
| `level_2.csv` | 49 cols | Headline + 47 mid-level components |
| `level_3_full.csv` | 745 cols | Headline + 744 granular items (raw) |
| `level3.csv` | 639 cols | Cleaned level-3: 106 aggregate / reprint columns dropped, leaves only true subindices |
| `level_3_column_names.csv` | 4 cols | JP-EN translation table for the level-3 series |
| `official_weights_national.csv` | – | Reference: official basket weights as published |
| `japan_inflation_structured_check.csv` | – | Provenance check used while cleaning |

The first column of every level file is `year_month` (`YYYYMM`); the second
is always **`All items`** (headline CPI, weight 10 000).

## CSV format (levels 1 / 2 / 3)

```
year_month   All items   <component_1>   <component_2>   ...
Weights      10000       <w_1>           <w_2>           ...
197001       <index>     <index>         <index>         ...
197002       ...
...
```

- **Header (row 0)**: column names. English in level 1 / 2 / `level3.csv`.
  The raw `level_3_full.csv` ships with garbled headers — use
  `level_3_column_names.csv` (`english` column) to fix them.
- **`Weights` row (row 1)**: official 2020-base basket weights, integer
  basis 10 000 (so headline = 10 000). Divide by 10 000 for basket shares;
  for regression priors, drop the headline row and renormalise components
  to sum to 1. The loader at `utils/data_load.py:load_level_data` handles
  this automatically.
- **Data rows (row 2+)**: monthly CPI index values. Missing observations
  are encoded as `-` and converted to `NaN` on load.

## `level3.csv` (cleaned subindex panel)

`level_3_full.csv` mixes true subindices with category aggregates,
reprinted summary series, and derived special indices (e.g. *Energy*,
*All items, less fresh food and energy*). For the trimmed-mean and
component pipelines we need only the true subindices.

`level3.csv` is the cleaned panel: 639 columns of disaggregated items,
no aggregate or reprint columns. The 106 dropped headers are listed in
[level3_notes.md](level3_notes.md) by category (food, housing, etc.).

## Loading from Python

`utils/data_load.py` is the single entry point. Typical use:

```python
from utils.data_load import (
    prepare_regression_data,    # load → impute → growth → forward target
    load_benchmark_series,      # 3m/3m of Japan core composites
    load_trimmed_mean_3m3m,     # in-house BoJ-style 10% trimmed mean
)

X, y, w_prior, features, dates, growth = prepare_regression_data(level=2)
```

`prepare_regression_data`:
1. Reads `level_{level}.csv` and parses the embedded `Weights` row.
2. Imputes missing values via `utils/smart_imputation.smart_impute`
   (auto-classified per series; see `utils/smart_imputation.md`).
3. Computes 3m/3m annualised growth rates: `((x / x.shift(3)) - 1) * 4 * 100`.
4. Builds the regression target as the 12-month forward average of
   headline 3m/3m (configurable via `config.HORIZON`).
5. Returns `(X, y, w_prior, features, dates, growth)` with priors
   normalised to sum to 1.

`load_trimmed_mean_3m3m` constructs a Bank-of-Japan-style 10 % symmetric
weighted trimmed-mean inflation series directly from `level3.csv` —
no external data dependency. It is used as one of the regressors in the
$X^{bm}_t$ benchmark (see `regression/regression.md`).

## Level summary

| Level | K (components) | Used for |
|---|---|---|
| 1 | 10 | Sector-level Albacore (paper Table) |
| 2 | 47 | Mid-level Albacore (paper Table) |
| 3 | 639 (cleaned) | Trimmed-mean construction; auxiliary OLS baseline |

Levels 1 and 2 carry the published Albacore experiments. Level 3 is
deeper than the published Japan series and is used to build the
trimmed-mean benchmark and to stress-test the unconstrained OLS variant.
