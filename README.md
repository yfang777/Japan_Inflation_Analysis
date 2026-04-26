# Japan Inflation Analysis

Assemblage regression for Japan CPI, following the methodology in Goulet Coulombe et al. (2024), *"Maximally Forward-Looking Core Inflation."* The method constructs optimally weighted core inflation measures by finding component weights that are maximally predictive of future headline inflation.

## Data

- **Source:** Japan Statistics Bureau CPI index (2020 base)
- **Period:** January 1970 -- present (monthly)
- **Three granularity levels:**

| Level | Components | Description |
|---|---|---|
| 1 | 9 | Major CPI sectors (Housing, Transport, Education, etc.) |
| 2 | 47 | Mid-level components (Cereals, Rent, Electricity, etc.) |
| 3 | 744 | Granular items (Tuna, Cabbage, Gasoline, etc.) |

Each level file embeds its own weights in the first row. See `data_clean/README.md` for details.

## Directory Structure

```
Japan_Inflation_Analysis/
├── config.py                    Central configuration (paths, run settings)
├── cpi_analysis.py              EDA and visualisation
├── environment.yml              Conda environment specification
│
├── data_clean/
│   ├── level_1.csv              9 sectors + headline
│   ├── level_2.csv              47 components + headline
│   ├── level_3.csv              744 items + headline (needs column name fix)
│   ├── level_3_column_names.csv JP-EN translation for level_3 columns
│   └── README.md                Data format documentation
│
├── utils/
│   ├── data_load.py             Level-based data loading and regression pipeline
│   └── smart_imputation.py      Intelligent missing-value imputation
│
├── regression/
│   ├── regression_component.py  Albacorecomps (component-space assemblage)
│   ├── regression_rank.py       Albacoreranks (rank-space assemblage)
│   ├── benchmarks.py            Naive and OLS benchmark models
│   ├── evaluation.py            Metrics and scorecard printing
│   └── figures.py               All regression visualisations
│
└── plots/                       Output figures
```

## How to Run

```bash
# Component-space regression on level 2 (default)
python regression/regression_component.py

# Component-space regression on level 1 (9 sectors)
python regression/regression_component.py --level 1

# Rank-space regression on level 2
python regression/regression_rank.py

# Rank-space regression on level 1
python regression/regression_rank.py --level 1

# EDA and visualisation
python cpi_analysis.py
```

Each level runs independently with its own components and weights.

## Methodology

### Albacorecomps (component space)

    min_w  mean(y - Xw)^2  +  lambda ||w - w_prior||^2
    s.t.   w >= 0,  sum(w) = 1

### Albacoreranks (rank space)

    min_w  mean(y - Ow)^2  +  lambda sum((w_{r+1} - w_r)^2)
    s.t.   w >= 0,  mean(O)'w = mean(y)

## Dependencies

See `environment.yml`. Core: pandas, numpy, scipy, matplotlib.

## Reference

Goulet Coulombe, P., Barrette, C., et al. (2024). *Maximally Forward-Looking Core Inflation.*
