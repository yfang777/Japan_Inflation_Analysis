"""
Smart Imputation for Time Series Data

This module implements intelligent imputation strategies for economic time series:
1. Growth rate interpolation for stable series
2. Year-over-year (YoY) pattern matching for seasonal series
3. Trend-based imputation for non-stable series

Author: Data Pipeline
Date: 2026-03-29
"""

import pandas as pd
import numpy as np
from typing import Literal


def detect_series_type(series: pd.Series,
                       cv_threshold: float = 0.15,
                       trend_pvalue: float = 0.05) -> Literal['stable', 'seasonal', 'trending']:
    """
    Detect if a series is stable, seasonal, or trending.

    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    cv_threshold : float
        Coefficient of variation threshold for stability (default: 0.15)
    trend_pvalue : float
        P-value threshold for trend detection (default: 0.05)

    Returns:
    --------
    str : 'stable', 'seasonal', or 'trending'
    """
    # Remove NaN for analysis
    clean_series = series.dropna()

    if len(clean_series) < 12:  # Not enough data
        return 'stable'

    # Calculate coefficient of variation (CV)
    mean_val = clean_series.mean()
    std_val = clean_series.std()

    if mean_val == 0:
        cv = 0
    else:
        cv = abs(std_val / mean_val)

    # Check for seasonality using year-over-year correlation
    if len(clean_series) >= 24:
        try:
            # Calculate YoY correlation (12-month lag)
            yoy_corr = clean_series.corr(clean_series.shift(12))

            # If strong YoY correlation, it's seasonal
            if abs(yoy_corr) > 0.7:
                return 'seasonal'
        except:
            pass

    # Check if stable (low variation)
    if cv < cv_threshold:
        return 'stable'

    # Otherwise, consider it trending
    return 'trending'


def impute_stable_series(series: pd.Series,
                         growth_window: int = 3) -> pd.Series:
    """
    Impute missing values in a stable series using recent growth rates.

    For stable series, we calculate the average growth rate from recent
    non-missing values and apply it to fill gaps.

    Parameters:
    -----------
    series : pd.Series
        Series with missing values
    growth_window : int
        Number of periods to use for growth rate calculation

    Returns:
    --------
    pd.Series : Imputed series
    """
    result = series.copy()

    for idx in range(len(result)):
        if pd.isna(result.iloc[idx]):
            # Look backwards for the last valid value
            last_valid_idx = None
            for j in range(idx - 1, -1, -1):
                if pd.notna(result.iloc[j]):
                    last_valid_idx = j
                    break

            if last_valid_idx is not None:
                # Calculate growth rate from recent valid values
                growth_rates = []
                for k in range(max(0, last_valid_idx - growth_window), last_valid_idx):
                    if pd.notna(result.iloc[k]) and pd.notna(result.iloc[k+1]):
                        if result.iloc[k] != 0:
                            gr = (result.iloc[k+1] - result.iloc[k]) / result.iloc[k]
                            growth_rates.append(gr)

                if growth_rates:
                    avg_growth = np.mean(growth_rates)
                    periods_gap = idx - last_valid_idx

                    # Apply compound growth
                    last_value = result.iloc[last_valid_idx]
                    result.iloc[idx] = last_value * (1 + avg_growth) ** periods_gap
                else:
                    # No growth rate available, use last value
                    result.iloc[idx] = result.iloc[last_valid_idx]

    return result


def impute_seasonal_series(series: pd.Series,
                           seasonal_period: int = 12) -> pd.Series:
    """
    Impute missing values using year-over-year (YoY) patterns.

    For seasonal series, we use the same period from previous years,
    adjusted for the trend if necessary.

    Parameters:
    -----------
    series : pd.Series
        Series with missing values (must have DatetimeIndex)
    seasonal_period : int
        Seasonal period (default: 12 for monthly data)

    Returns:
    --------
    pd.Series : Imputed series
    """
    result = series.copy()

    for idx in range(len(result)):
        if pd.isna(result.iloc[idx]):
            # Look for same period in previous year
            yoy_idx = idx - seasonal_period

            if yoy_idx >= 0 and pd.notna(result.iloc[yoy_idx]):
                # Use YoY value as base
                base_value = result.iloc[yoy_idx]

                # Calculate trend adjustment from nearby values
                # Look at the growth from 2 years ago to 1 year ago
                two_years_ago = idx - 2 * seasonal_period
                if two_years_ago >= 0 and pd.notna(result.iloc[two_years_ago]):
                    # Calculate YoY growth rate
                    if result.iloc[two_years_ago] != 0:
                        yoy_growth = (result.iloc[yoy_idx] - result.iloc[two_years_ago]) / result.iloc[two_years_ago]
                        # Apply same growth
                        result.iloc[idx] = base_value * (1 + yoy_growth)
                    else:
                        result.iloc[idx] = base_value
                else:
                    # No trend information, just use YoY value
                    result.iloc[idx] = base_value

    return result


def impute_trending_series(series: pd.Series,
                           trend_window: int = 6) -> pd.Series:
    """
    Impute missing values in a trending series using local linear trend.

    For trending series, we fit a local linear trend to recent data
    and extrapolate to fill gaps.

    Parameters:
    -----------
    series : pd.Series
        Series with missing values
    trend_window : int
        Number of periods to use for trend estimation

    Returns:
    --------
    pd.Series : Imputed series
    """
    result = series.copy()

    for idx in range(len(result)):
        if pd.isna(result.iloc[idx]):
            # Look backwards for recent valid values
            recent_values = []
            recent_indices = []

            for j in range(idx - 1, max(-1, idx - trend_window - 10), -1):
                if pd.notna(result.iloc[j]):
                    recent_values.append(result.iloc[j])
                    recent_indices.append(j)
                    if len(recent_values) >= trend_window:
                        break

            if len(recent_values) >= 2:
                # Fit linear trend to recent values
                recent_values = np.array(recent_values[::-1])
                recent_indices = np.array(recent_indices[::-1])

                # Simple linear regression
                X = recent_indices - recent_indices[0]
                y = recent_values

                # Calculate slope
                if len(X) > 1:
                    slope = np.polyfit(X, y, 1)[0]
                    intercept = recent_values[0]

                    # Extrapolate
                    x_new = idx - recent_indices[0]
                    result.iloc[idx] = intercept + slope * x_new
                else:
                    # Only one value, use it directly
                    result.iloc[idx] = recent_values[0]

    return result


def smart_impute(df: pd.DataFrame,
                 strategy: str = 'auto',
                 verbose: bool = True) -> pd.DataFrame:
    """
    Intelligently impute missing values based on series characteristics.

    This function automatically detects whether each column is stable,
    seasonal, or trending, and applies the appropriate imputation method.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with missing values
    strategy : str
        Imputation strategy:
        - 'auto': Automatically detect series type and choose method
        - 'stable': Use growth rate method for all series
        - 'seasonal': Use YoY method for all series
        - 'trending': Use trend method for all series
    verbose : bool
        Print imputation details

    Returns:
    --------
    pd.DataFrame : Imputed dataframe
    """
    result = df.copy()

    if verbose:
        print("Smart Imputation Process")
        print("=" * 60)
        print(f"Strategy: {strategy}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Total missing values before: {df.isnull().sum().sum()}\n")

    series_types = {}

    for col in df.columns:
        if df[col].isnull().any():
            if strategy == 'auto':
                series_type = detect_series_type(df[col])
            else:
                series_type = strategy

            series_types[col] = series_type

            # Apply appropriate imputation
            if series_type == 'stable':
                result[col] = impute_stable_series(result[col])
            elif series_type == 'seasonal':
                result[col] = impute_seasonal_series(result[col])
            elif series_type == 'trending':
                result[col] = impute_trending_series(result[col])

            # Final fallback: forward fill then backward fill
            result[col] = result[col].ffill().bfill()

            # If still NaN, use median
            if result[col].isnull().any():
                median_val = result[col].median()
                if pd.isna(median_val):
                    median_val = 0
                result[col] = result[col].fillna(median_val)

    if verbose:
        print("\nSeries Type Distribution:")
        type_counts = pd.Series(series_types).value_counts()
        for stype, count in type_counts.items():
            print(f"  {stype.capitalize()}: {count} columns")

        print(f"\nTotal missing values after: {result.isnull().sum().sum()}")
        print("=" * 60)

    return result, series_types


if __name__ == "__main__":
    # Test the imputation methods
    print("Smart Imputation Module Loaded Successfully!")
    print("\nAvailable functions:")
    print("  - detect_series_type(): Detect if series is stable/seasonal/trending")
    print("  - impute_stable_series(): Impute using growth rates")
    print("  - impute_seasonal_series(): Impute using YoY patterns")
    print("  - impute_trending_series(): Impute using local trends")
    print("  - smart_impute(): Main function for automatic imputation")
