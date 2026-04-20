# Regression Scripts

Assemblage regression for Japan CPI inflation, following the methodology in Gaglianone & Guillén (2024). Both scripts implement the same regression pipeline but differ in how missing data is handled before estimation.

## Scripts

| Script | Imputation Method | Output Files |
|--------|-------------------|--------------|
| `assemblage_smart_imputation.py` | Auto-detects series type (stable / seasonal / trending) and applies growth-rate interpolation, YoY pattern matching, or local linear trend | `assemblage_weights_smart.csv`, `assemblage_predictions_smart.csv` |
| `assemblage_yoy_forwardfill.py` | Fills each missing value with the same month from the previous year (up to 5 years back), then plain forward-fill / backward-fill | `assemblage_weights_yoy_ffill.csv`, `assemblage_predictions_yoy_ffill.csv` |

## Usage

```bash
# From project root
python regression/assemblage_smart_imputation.py
python regression/assemblage_yoy_forwardfill.py
```

## Pipeline Overview

1. **Data loading** — Reads `data_clean/japan_inflation_structured_check.csv`
2. **Growth rates** — 3-month-over-3-month annualised growth rates
3. **Imputation** — Method-specific missing-data treatment (see table above)
4. **Target construction** — 12-month forward average of headline CPI inflation
5. **Prior weights** — Official CPI basket weights from Japan Statistics Bureau
6. **Lambda selection** — R-matched grid (`exp(c(0, 0.5, seq(1,9,17))) * lambda.1se`) with expanding-window time-series CV
7. **Estimation** — Constrained optimisation (weights >= 0, sum to 1) with SD-weighted ridge penalty
8. **Out-of-sample evaluation** — Expanding-window rolling forecasts

## Output

All output CSVs are written to `data_clean/`. Compare the two sets of results to assess the sensitivity of the assemblage weights and forecasts to the imputation strategy.
