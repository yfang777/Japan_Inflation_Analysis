# config.py

Central configuration for the Japan CPI analysis project. All column classifications, path definitions, basket mappings, display settings, and regression hyperparameters live here so that structural changes stay in one place.

## Paths

| Variable | Description |
|---|---|
| `ROOT` | Project root directory |
| `DATA_FILE` | Primary CPI index CSV (`data_clean/japan_inflation_structured_check.csv`) |
| `WEIGHTS_CSV` | Official basket weights CSV (`data_clean/official_weights_national.csv`) |
| `PLOTS_DIR` | Output directory for figures (`plots/`) |

## Column Taxonomy

The 57 CSV columns are classified into four groups:

| Type | Count | Variable | Treatment |
|---|---|---|---|
| Composite | 6 | `COMPOSITE_COLS` | Never use as features (derived from headline) |
| Special / 別掲 | 4 | `SPECIAL_COLS` | Exclude (cross-cutting aggregates that overlap with components) |
| True components | 46 | `COMPONENT_COLS` | Regression features |
| Rent | 1 | `ISOLATED` | Included in `FEATURES` but documented as problematic (~85% imputed rent) |

## Mappings

| Variable | Description |
|---|---|
| `EN_TO_JPN` | English → Japanese name mapping (47 entries) for official weight lookup |
| `GROUPS` | 11 product-category groups for visualisation |
| `GROUP_COLORS` | Hex colours per group |
| `COL_TO_GROUP` | Reverse lookup: component → group name |

## Regression Settings

| Variable | Default | Description |
|---|---|---|
| `FEATURES` | 47 | `COMPONENT_COLS + ['Rent']` |
| `START_DATE` | `'1990-01-01'` | Regression window start |
| `HORIZON` | 12 | Months-ahead forecast target |
| `MIN_TRAIN` | 120 | Minimum training window (10 years) for OOS |
| `OOS_STEP` | 1 | Rolling OOS step size |
| `N_CV_FOLDS` | 10 | Expanding-window CV folds |
| `LAMBDA_GRID` | `logspace(-4, 3, 40)` | Component-space regularisation grid |
| `LAMBDA_GRID_RANKS` | `logspace(0, 5, 40)` | Rank-space regularisation grid |
| `ROLLING_WINDOW` | 240 | 20-year rolling window for OOS |
| `MPL_RCPARAMS` | dict | Shared matplotlib style defaults |
