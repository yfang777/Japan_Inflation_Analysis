# utils/smart_imputation.py

Intelligent missing-value imputation for economic time series. Auto-classifies each series by its statistical characteristics and applies a tailored imputation method.

## Classification

| Series Type | Detection Rule | Imputation Method |
|---|---|---|
| **Stable** | Coefficient of variation < 0.15 | Growth-rate interpolation: `v[t] = v[last] * (1+g)^gap` |
| **Seasonal** | 12-lag autocorrelation > 0.7 | YoY pattern matching: same-month-previous-year, adjusted for trend |
| **Trending** | Fallback | Local linear trend: `polyfit(degree=1)` on recent 6 points |

After the primary method, a two-layer fallback ensures no NaN survives:
1. Forward-fill then backward-fill
2. Median fill (or 0 if median is also NaN)

## Functions

| Function | Description |
|---|---|
| `detect_series_type(series, cv_threshold, trend_pvalue)` | Classify as `'stable'`, `'seasonal'`, or `'trending'` |
| `impute_stable_series(series, growth_window)` | Growth-rate interpolation |
| `impute_seasonal_series(series, seasonal_period)` | YoY pattern matching with trend adjustment |
| `impute_trending_series(series, trend_window)` | Local linear trend extrapolation |
| `smart_impute(df, strategy, verbose)` | Main function: auto-classify + impute all columns. Returns `(imputed_df, series_types)`. |

## Usage

```python
from utils.smart_imputation import smart_impute

df_imputed, types = smart_impute(df, strategy='auto', verbose=True)
```
