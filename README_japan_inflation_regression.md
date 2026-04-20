# Assemblage Regression — Japan Inflation

**Notebook:** `japan_inflation_regression.ipynb`

## Overview

This notebook implements the **Assemblage Regression** methodology from Goulet Coulombe et al. (2024), *"Maximally Forward-Looking Core Inflation"*, applied to Japan's CPI data. The method constructs an optimally weighted core inflation measure by finding subcomponent weights that are maximally predictive of future headline inflation.

## Methodology

The assemblage regression solves:

$$\hat{w} = \arg\min_w \sum_{t=1}^{T-h} \left(\pi_{t+1:t+h} - w'\Pi_t\right)^2 + \lambda \|w - w_{\text{headline}}\|^2$$

where:
- $\pi_{t+1:t+h}$ is the average headline inflation from $t+1$ to $t+h$
- $\Pi_t$ is the vector of component-level inflation at time $t$
- $w_{\text{headline}}$ are official CPI basket weights (prior)
- $\lambda$ controls shrinkage toward the headline weights

### Key Implementation Details (Matching R Reference Code)

1. **SD-weighted penalty:** The shrinkage term is weighted by the standard deviation of each component, following the original R implementation
2. **Official CPI weights as priors:** Loaded from Japan Statistics Bureau data via `load_official_weights` module
3. **Cross-validation:** Used to select the optimal $\lambda$
4. **Forecast horizon:** 12 months ahead (default)

## Pipeline

### 1. Data Preparation
- Load CPI index data from `data_clean/japan_inflation_structured_check.csv`
- Compute 3-month annualized growth rates
- **Smart imputation** of missing values (see below)
- Select granular components, excluding aggregate measures to avoid multicollinearity
- Analysis period: 1990 onward

### Smart Imputation (`smart_imputation.py`)

The regression requires complete data across all CPI sub-components. Rather than naive forward-fill/backward-fill, a custom imputation module auto-classifies each series and applies a tailored strategy:

| Series Type | Detection Rule | Method |
|---|---|---|
| **Stable** | Coefficient of variation < 0.15 | **Growth rate interpolation** — computes average growth from recent valid values, then compounds forward: `v[t] = v[last] × (1 + g)^gap` |
| **Seasonal** | 12-month-lag autocorrelation > 0.7 | **YoY pattern matching** — uses same month from previous year, adjusted by the year-over-year growth trend |
| **Trending** | Fallback (high variation, no strong seasonality) | **Local linear trend** — fits `polyfit(degree=1)` on up to 6 recent valid points and extrapolates |

After the primary method, a two-layer fallback ensures no NaN survives:
1. Forward-fill then backward-fill
2. Median fill (or 0 if median is also NaN)

**Note:** This imputation design is an ad-hoc heuristic — it is not derived from a specific published method. Each individual technique (CV-based classification, compound growth fill, YoY seasonal fill, local linear extrapolation) is standard, but the combined pipeline has no formal reference. Consider validating empirically (mask known values, impute, measure error) or citing general imputation literature (e.g., Little & Rubin, 2019) if used in a formal write-up.

### 2. Target Construction
- Forward-looking target: average headline inflation from $t+1$ to $t+12$

### 3. Model Training
- Assemblage regression with SD-weighted ridge penalty
- Prior weights from official CPI basket
- Lambda selected via cross-validation

### 4. Evaluation
- **In-sample:** R², RMSE, fitted vs. actual plots
- **Out-of-sample:** Rolling window validation (minimum 10-year training window)
- **Comparison** with standard core measures:
  - Core CPI (ex fresh food)
  - Core-core CPI (ex fresh food & energy)
- **Real-time tracking:** How well the assemblage core tracks current headline inflation

### 5. Weight Analysis
- Ranked component weights visualization (top 20)
- Comparison of optimized weights vs. official CPI basket weights

## Output Files

Saved to `data_clean/`:

| File | Contents |
|------|----------|
| `assemblage_weights.csv` | Optimized component weights and shrinkage SDs |
| `assemblage_predictions.csv` | Fitted values vs. actual 12-month forward inflation |

## Dependencies

- pandas, numpy, matplotlib, seaborn
- scikit-learn (Ridge, Lasso, metrics)
- scipy (optimize)
- Custom modules:
  - `smart_imputation.py` — intelligent missing value imputation (in notebook directory)
  - `load_official_weights` — loads and matches official CPI basket weights (in `scripts_regression/`)

## Reference

Goulet Coulombe, P., Barrette, C., et al. (2024). *Maximally Forward-Looking Core Inflation.* [GitHub](https://github.com/ChristopheBarrette/Assemblage)
