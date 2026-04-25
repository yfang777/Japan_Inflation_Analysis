# utils/data_load.py

Centralised data loading and preparation for the regression pipeline. Consolidates data I/O that was previously duplicated across scripts.

## Functions

| Function | Returns | Description |
|---|---|---|
| `load_raw_data(start_date)` | `DataFrame` | Read CPI index CSV, parse dates, convert to numeric. Optionally filter by start date. |
| `load_basket_weights(features, include_rent)` | `ndarray` or `dict` | Official basket weights. Pass `features` list for regression (returns normalised ndarray); pass `None` for EDA (returns dict). |
| `compute_growth_3m3m(df)` | `DataFrame` | 3-month-over-3-month annualised growth rates: `((s/s.shift(3)) - 1) * 4 * 100` |
| `compute_forward_target(headline, horizon)` | `Series` | 12-month forward average of headline growth as regression target |
| `prepare_regression_data(start_date, horizon, features)` | `(X, y, dates, growth)` | Full pipeline: load → impute → growth rates → target. Returns arrays ready for model fitting. |

## Data Pipeline

```
load_raw_data()
  → smart_impute()           (from utils/smart_imputation.py)
  → compute_growth_3m3m()
  → compute_forward_target()
  → filter valid rows
  → return (X, y, dates, growth)
```
