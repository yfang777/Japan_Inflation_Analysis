# Japan Inflation Analysis

Assemblage regression for Japan CPI, following the methodology in Goulet Coulombe et al. (2024), *"Maximally Forward-Looking Core Inflation."* The method constructs optimally weighted core inflation measures by finding component weights that are maximally predictive of future headline inflation.

## Data

- **Source:** Japan Statistics Bureau CPI index (2020 base)
- **Period:** January 1970 -- January 2026 (673 monthly observations, 57 sub-components)
- **Primary file:** `data_clean/japan_inflation_structured_check.csv`
- **Basket weights:** `data_clean/official_weights_national.csv`

## Directory Structure

```
Japan_Inflation_Analysis/
├── config.py                    Central configuration (paths, column taxonomy, run settings)
├── cpi_analysis.py              EDA and visualisation (core measures, basket weights, contributions)
├── environment.yml              Conda environment specification
│
├── data_clean/                  CPI index data and official weights
│   ├── japan_inflation_structured_check.csv
│   ├── official_weights_national.csv
│   └── level_1.csv, level_2.csv, level_3.csv
│
├── utils/
│   ├── data_load.py             Centralised data loading and growth-rate computation
│   └── smart_imputation.py      Intelligent missing-value imputation (stable/seasonal/trending)
│
├── regression/
│   ├── regression_component.py  Albacorecomps — component-space assemblage regression
│   ├── regression_rank.py       Albacoreranks — rank-space assemblage regression
│   ├── benchmarks.py            Naive and OLS benchmark models
│   ├── evaluation.py            Metrics (RMSE, MAE, R²) and scorecard printing
│   └── figures.py               All regression visualisations (5 figure types)
│
└── plots/                       Output figures
```

## How to Run

```bash
# EDA and visualisation
python cpi_analysis.py

# Component-space assemblage regression (full pipeline)
python regression/regression_component.py

# Rank-space assemblage regression (full pipeline)
python regression/regression_rank.py
```

Each regression script runs independently: loads data, trains, evaluates out-of-sample, computes benchmarks, prints a scorecard, and saves figures to `plots/`.

## Methodology

### Albacorecomps (component space)

Finds optimal component weights w that minimise:

    min_w  mean(y - Xw)²  +  λ ||w - w_prior||²
    s.t.   w >= 0,  Σw = 1

where X is the T x 47 matrix of component 3m/3m growth rates, y is the 12-month forward average headline 3m/3m, and w_prior is the official basket weight vector.

### Albacoreranks (rank space)

Sorts components low-to-high at each t and learns which percentile of the cross-sectional distribution predicts future inflation:

    min_w  mean(y - Ow)²  +  λ Σ(w_{r+1} - w_r)²   [fused ridge]
    s.t.   w >= 0,  Ō'w = ȳ

## Dependencies

See `environment.yml`. Core: pandas, numpy, scipy, scikit-learn, matplotlib, statsmodels.

## Reference

Goulet Coulombe, P., Barrette, C., et al. (2024). *Maximally Forward-Looking Core Inflation.*
