# cpi_analysis.py

Exploratory data analysis and visualisation of Japan's CPI structure, basket weights, and inflation dynamics.

## Usage

```bash
python cpi_analysis.py
```

Produces 6 PNG figures in `plots/` and prints a text summary to stdout.

## Functions

### Data

| Function | Description |
|---|---|
| `load_data()` | Read CPI index CSV, parse dates, convert to numeric |
| `load_basket_weights(include_rent)` | Official basket weights as `{eng_name: share}` dict |

### Growth Rates

| Function | Description |
|---|---|
| `yoy(s)` | Year-over-year % change (12-month lag) |
| `g3m3m(s)` | 3-month-over-3-month annualised |
| `mom(s)` | Month-over-month % change |

### Analysis

| Function | Description |
|---|---|
| `contributions_yoy(df, weights)` | Laspeyres contribution decomposition (percentage points) |
| `print_summary(df, weights)` | Text report: data range, missing values, recent inflation |

### Figures

| Function | Output | Description |
|---|---|---|
| `fig_core_measures(df)` | `fig1_core_measures_yoy.png` | Headline vs. published core measures (YoY, 1990--present) |
| `fig_recent_3m3m(df)` | `fig2_recent_3m3m.png` | Recent 3m/3m dynamics (2019--present) |
| `fig_basket_weights(weights)` | `fig3_basket_weights.png` | Horizontal bar chart with group colours, Rent hatched |
| `fig_contributions(df)` | `fig4_contributions.png` | Stacked YoY contribution by group (2018+) |
| `fig_rent_deepdive(df)` | `fig5_rent_deepdive.png` | 2-panel: Rent vs. headline YoY + volatility comparison |
| `fig_volatility_scatter(df, weights)` | `fig6_weight_vs_volatility.png` | Basket weight vs. 3m/3m volatility scatter |
