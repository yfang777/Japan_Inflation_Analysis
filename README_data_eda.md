# Exploratory Data Analysis — Japan Inflation Data

**Notebook:** `data_eda.ipynb`

## Overview

This notebook performs a comprehensive exploratory data analysis on Japan's Consumer Price Index (CPI) data, covering **673 monthly observations** (January 1970 – January 2026) across **57 CPI sub-components**. The analysis characterizes the distributional, temporal, and relational properties of inflation rates to inform downstream modeling decisions.

## Data

- **Source file:** `data_clean/japan_inflation_structured_check.csv`
- **Granularity:** Monthly CPI index levels by sub-component
- **Growth rates:** 3-month-over-3-month annualized rates, computed as:

$$\text{Growth Rate}_t = \left(\frac{P_t}{P_{t-3}} - 1\right) \times \frac{12}{3} \times 100$$

## Analysis Sections

### 1. Descriptive Statistics
Basic summary statistics (mean, std, percentiles) and higher-order moments (skewness, kurtosis) for key indicators including:
- All items
- All items, less fresh food
- All items, less fresh food & energy
- Food, Housing, Fuel/light/water, Transportation/communication, Education

### 2. Distribution Analysis
- Histograms with density overlay
- Kernel Density Estimation (KDE)
- Q-Q plots (normality assessment)
- Box plots

### 3. Normality Tests
Five formal tests applied to each key indicator:

| Test | Null Hypothesis |
|------|----------------|
| Shapiro-Wilk | Data is normally distributed |
| Kolmogorov-Smirnov | Data follows specified distribution |
| Anderson-Darling | Data follows specified distribution |
| Jarque-Bera | Skewness = 0 and excess kurtosis = 0 |
| D'Agostino-Pearson | Data is normally distributed |

### 4. Stationarity Tests
- **ADF (Augmented Dickey-Fuller):** H0 = unit root exists (non-stationary)
- **KPSS:** H0 = series is stationary

Both tests are applied to confirm or reject stationarity of growth rate series.

### 5. Autocorrelation Analysis
- ACF and PACF plots (40 lags) for key indicators
- **Ljung-Box test** at lags 10, 20, 30 to test for white noise (H0 = no autocorrelation)

### 6. Correlation Analysis
- **Pearson** correlation matrix and heatmap
- **Spearman** rank correlation matrix and heatmap
- Extraction of highly correlated pairs (|r| >= 0.7)

### 7. Outlier Detection
- **IQR method** (1.5× IQR bounds) with counts and time-series visualization of outliers
- **Z-score method** (threshold = 3)

### 8. Time Series Features
- 12-month rolling mean and rolling standard deviation (volatility)
- Monthly seasonality decomposition (average and std by calendar month)

## Output Files

All results are saved to `data_clean/`:

| File | Contents |
|------|----------|
| `eda_comprehensive_summary.csv` | Per-indicator summary: mean, std, skewness, kurtosis, normality, stationarity, outlier ratio |
| `eda_stationarity_tests.csv` | ADF and KPSS test statistics and conclusions |
| `eda_outlier_detection.csv` | IQR-based outlier counts and bounds |
| `eda_correlation_matrix.csv` | Pearson correlation matrix |

## Dependencies

- pandas, numpy, matplotlib, seaborn
- scipy (stats, normality tests)
- statsmodels (ADF, KPSS, Ljung-Box, ACF/PACF)
