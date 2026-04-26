# utils/data_load.py

Centralised data loading for all CPI levels. Reads level_1/2/3.csv files, extracts embedded weights, computes growth rates and forward targets.

## Functions

| Function | Returns | Description |
|---|---|---|
| `load_level_data(level, start_date)` | `(DataFrame, dict)` | Read a level CSV, extract weights row, parse dates. Returns data + weights dict. |
| `compute_growth_3m3m(df)` | `DataFrame` | 3m/3m annualised growth rates |
| `compute_forward_target(headline, horizon)` | `Series` | 12-month forward average of headline growth |
| `prepare_regression_data(level, start_date, horizon)` | `(X, y, w_prior, features, dates, growth)` | Full pipeline: load -> impute -> growth -> target. Returns arrays ready for model fitting. |

## Usage

```python
from utils.data_load import prepare_regression_data

# Level 2 (47 components, default)
X, y, w_prior, features, dates, growth = prepare_regression_data(level=2)

# Level 1 (9 sectors)
X, y, w_prior, features, dates, growth = prepare_regression_data(level=1)
```

## Data Pipeline

```
load_level_data(level)
  -> smart_impute()
  -> compute_growth_3m3m()
  -> compute_forward_target()
  -> filter valid rows
  -> return (X, y, w_prior, features, dates, growth)
```
