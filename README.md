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

Each level file embeds its own weights in the first row. See
[`data_clean/data.md`](data_clean/data.md) for details.

## Directory Structure

```
Japan_Inflation_Analysis/
├── config.py                    Central configuration (paths, run settings)
├── cpi_analysis.py              EDA and visualisation
├── environment.yml              Conda environment specification
│
├── data_clean/                  CPI panels (see data.md)
├── utils/                       Data loading + smart imputation
├── regression/                  Albacore models, benchmarks, horizon tables
│                                (see regression.md)
└── AR/                          Autoregressive baselines
```

## How to Run

```bash
# Component-space regression on level 2 (default)
python regression/regression_component.py
python regression/regression_component.py --level 1

# Rank-space regression
python regression/regression_rank.py --level 2

# Forecast horizon × period tables (paper protocol, rolling 20y)
python regression/horizon_table_rolling.py
python regression/horizon_table_correct.py

# EDA
python cpi_analysis.py
```

Per-directory READMEs ([`data_clean/data.md`](data_clean/data.md),
[`regression/regression.md`](regression/regression.md)) describe the data
format, models, and command-line options in detail.

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
