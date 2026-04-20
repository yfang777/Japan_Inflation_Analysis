"""
Assemblage Regression for Japan Inflation — YoY Forward-Fill Imputation

Uses a simple imputation strategy: for each missing value, fill with the
value from the same month one year earlier (12-month seasonal forward-fill).
If no same-month-last-year value exists, fall back to plain forward-fill,
then backward-fill.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.load_official_weights import load_official_weights, match_weights_to_components

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data_clean" / "japan_inflation_structured_check.csv"
HEADLINE_COL = "All items"
HORIZON = 12          # months ahead
START_DATE = "1990-01-01"
CV_FOLDS = 10

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calculate_growth_rate(series: pd.Series, periods: int = 3) -> pd.Series:
    """Annualised periods-over-periods growth rate."""
    return ((series / series.shift(periods)) - 1) * (12 / periods) * 100


def create_forward_target(series: pd.Series, horizon: int = 12) -> pd.Series:
    """Average inflation from t+1 to t+h."""
    target = pd.Series(index=series.index, dtype=float)
    for i in range(len(series) - horizon):
        target.iloc[i] = series.iloc[i + 1 : i + 1 + horizon].mean()
    return target


def get_cpi_weights(components, method="equal", region="national"):
    """Return prior weight vector (and optionally matching info)."""
    n = len(components)
    if method == "equal":
        return np.ones(n) / n

    print(f"Loading official CPI weights ({region})...")
    official_weights = load_official_weights(use_tokyo=(region == "tokyo"))
    matched_df = match_weights_to_components(official_weights, components)

    n_matched = matched_df["Matched"].sum()
    print(f"  Matched {n_matched}/{len(matched_df)} components")

    weights = matched_df["Official_Weight"].values
    n_unmatched = np.isnan(weights).sum()
    if n_unmatched > 0:
        remaining = 1.0 - np.nansum(weights)
        fallback = remaining / n_unmatched if n_unmatched else 0
        weights = np.where(np.isnan(weights), fallback, weights)

    weights = weights / weights.sum()
    return weights, matched_df


# ---------------------------------------------------------------------------
# YoY forward-fill imputation
# ---------------------------------------------------------------------------

def yoy_forwardfill(df: pd.DataFrame, seasonal_period: int = 12,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Fill missing values with the same-month-last-year value, then plain
    forward-fill / backward-fill as fallback.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex (monthly frequency expected)
    seasonal_period : lag to use (12 = same month last year)
    verbose : print summary

    Returns
    -------
    DataFrame with no missing values
    """
    result = df.copy()
    total_before = result.isnull().sum().sum()

    for col in result.columns:
        if not result[col].isnull().any():
            continue

        # Pass 1: fill from same month last year (repeat up to 5 years back)
        for _ in range(5):
            still_missing = result[col].isnull()
            if not still_missing.any():
                break
            result[col] = result[col].fillna(result[col].shift(seasonal_period))

        # Pass 2: plain forward-fill then backward-fill
        result[col] = result[col].ffill().bfill()

        # Pass 3: if still NaN (entire column was empty), fill with 0
        if result[col].isnull().any():
            result[col] = result[col].fillna(0)

    total_after = result.isnull().sum().sum()
    if verbose:
        print(f"YoY Forward-Fill Imputation")
        print(f"  Missing before: {total_before}")
        print(f"  Missing after : {total_after}")

    return result


# ---------------------------------------------------------------------------
# Assemblage regression core
# ---------------------------------------------------------------------------

def _fit_assemblage_single(X, y, lambda_reg, w_prior, shrinkw):
    """Core optimisation matching R's nonneg.ridge.sum1."""
    n_features = X.shape[1]

    def objective(w):
        residuals = y - X @ w
        mse = np.mean(residuals ** 2)
        penalty = lambda_reg * np.sum(shrinkw * (w - w_prior) ** 2)
        return mse + penalty

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, None)] * n_features
    w0 = np.ones(n_features) / n_features

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})

    w_opt = result.x
    y_pred = X @ w_opt
    return {
        "weights": w_opt,
        "fitted_values": y_pred,
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "r2": 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2),
        "success": result.success,
        "message": result.message,
        "shrinkw": shrinkw,
    }


def assemblage_regression_v2(X, y, lambda_reg=1.0, w_prior=None,
                              cv_folds=10, lambda_custom=None):
    """Assemblage regression with time-series CV (expanding window)."""
    n_samples, n_features = X.shape

    if w_prior is None:
        w_prior = np.ones(n_features) / n_features

    shrinkw = np.std(X, axis=0, ddof=1)

    if lambda_custom is not None:
        lambda_grid = np.array(lambda_custom)
    elif isinstance(lambda_reg, (list, np.ndarray)):
        lambda_grid = np.array(lambda_reg)
    else:
        lambda_grid = np.array([lambda_reg])

    if len(lambda_grid) > 1:
        print(f"  Time-series {cv_folds}-fold CV on {len(lambda_grid)} lambdas ...")
        min_train = n_samples // (cv_folds + 1)
        test_size = n_samples // cv_folds

        mse_cv = np.zeros((len(lambda_grid), cv_folds))
        for li, lam in enumerate(lambda_grid):
            for fold in range(cv_folds):
                train_end = min_train + fold * test_size
                test_end = min(train_end + test_size, n_samples)
                if train_end >= n_samples or test_end > n_samples:
                    continue
                res = _fit_assemblage_single(X[:train_end], y[:train_end],
                                             lam, w_prior, shrinkw)
                y_val = y[train_end:test_end]
                y_pred_val = X[train_end:test_end] @ res["weights"]
                mse_cv[li, fold] = np.mean((y_val - y_pred_val) ** 2)

        mean_mse = np.array([np.mean([x for x in row if x > 0]) for row in mse_cv])
        best_idx = np.argmin(mean_mse)
        best_lambda = lambda_grid[best_idx]
        print(f"  Best lambda: {best_lambda:.6f} (CV MSE: {mean_mse[best_idx]:.6f})")
    else:
        best_lambda = lambda_grid[0]

    result = _fit_assemblage_single(X, y, best_lambda, w_prior, shrinkw)
    result["best_lambda"] = best_lambda
    if len(lambda_grid) > 1:
        result["cv_mse"] = mean_mse
        result["lambda_grid"] = lambda_grid
    return result


# ---------------------------------------------------------------------------
# Lambda grid (R-pattern)
# ---------------------------------------------------------------------------

def make_lambda_grid(lambda_1se):
    """Generate R-matched lambda grid: exp(c(0,0.5,seq(1,9,17))) * lambda.1se."""
    seq_vals = np.linspace(1, 9, 17)
    exp_inputs = np.concatenate([[0, 0.5], seq_vals])
    return np.exp(exp_inputs) * lambda_1se


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # ---- Load data --------------------------------------------------------
    print("=" * 70)
    print("ASSEMBLAGE REGRESSION — YoY FORWARD-FILL IMPUTATION")
    print("=" * 70)

    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["YearMonth"], format="%Y-%m")
    data = data.set_index("Date").drop("YearMonth", axis=1)
    data = data.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
    print(f"Loaded data: {data.shape}, {data.index.min()} – {data.index.max()}")

    # ---- Growth rates -----------------------------------------------------
    data_growth = data.apply(lambda x: calculate_growth_rate(x, periods=3))

    # ---- Subset & YoY forward-fill ----------------------------------------
    data_reg = data_growth[data_growth.index >= START_DATE].copy()
    data_reg_imputed = yoy_forwardfill(data_reg, seasonal_period=12, verbose=True)

    # ---- Component selection ----------------------------------------------
    exclude_kw = ["All items"]
    valid_components = [c for c in data_reg_imputed.columns
                        if not any(k in c for k in exclude_kw)]
    print(f"Components: {len(valid_components)}")

    # ---- Target -----------------------------------------------------------
    y_target = create_forward_target(data_reg_imputed[HEADLINE_COL], horizon=HORIZON)
    valid_idx = ~y_target.isnull() & ~data_reg_imputed[HEADLINE_COL].isnull()

    X = data_reg_imputed.loc[valid_idx, valid_components].values
    y = y_target[valid_idx].values
    dates = data_reg_imputed.index[valid_idx]
    print(f"Regression matrix: X {X.shape}, y {y.shape}")

    if np.isnan(X).any():
        print("WARNING: NaN remaining in X — replacing with 0")
        X = np.nan_to_num(X, nan=0.0)

    # ---- Prior weights (official) -----------------------------------------
    w_prior, _ = get_cpi_weights(valid_components, method="official")

    # ---- Preliminary CV for lambda.1se ------------------------------------
    print("\nPreliminary CV for lambda.1se ...")
    prelim = assemblage_regression_v2(X, y,
                                      lambda_reg=np.exp(np.linspace(-3, 2, 10)),
                                      w_prior=w_prior, cv_folds=5)
    lambda_1se = prelim["best_lambda"]
    print(f"lambda.1se = {lambda_1se:.6f}")

    # ---- Full training with R-matched grid --------------------------------
    lambda_grid = make_lambda_grid(lambda_1se)
    print(f"\nTraining with {len(lambda_grid)} lambdas ...")
    result = assemblage_regression_v2(X, y, lambda_reg=lambda_grid,
                                      w_prior=w_prior, cv_folds=CV_FOLDS)

    print(f"\n{'=' * 60}")
    print(f"Best Lambda : {result['best_lambda']:.6f}")
    print(f"RMSE        : {result['rmse']:.4f}")
    print(f"R²          : {result['r2']:.4f}")
    print(f"Non-zero wts: {np.sum(result['weights'] > 0.001)}/{len(result['weights'])}")
    print(f"{'=' * 60}")

    # ---- Weights table ----------------------------------------------------
    weights_df = pd.DataFrame({
        "Component": valid_components,
        "Weight": result["weights"],
        "Shrinkage_SD": result["shrinkw"],
    }).sort_values("Weight", ascending=False)

    print("\nTop 20 components by weight:")
    print(weights_df.head(20).to_string(index=False))

    # ---- Out-of-sample evaluation -----------------------------------------
    print("\n--- Out-of-sample (expanding window) ---")
    min_train = min(20 * 12, int(len(X) * 0.7))
    min_train = max(10 * 12, min_train)
    step = 6 if (len(X) - min_train) < 60 else 12

    oos_preds, oos_actuals, oos_dates = [], [], []
    if min_train < len(X):
        for t in range(min_train, len(X), step):
            res_oos = assemblage_regression_v2(X[:t], y[:t],
                                               lambda_reg=lambda_grid,
                                               w_prior=w_prior, cv_folds=5)
            oos_preds.append((X[t:t+1] @ res_oos["weights"])[0])
            oos_actuals.append(y[t])
            oos_dates.append(dates[t])

        oos_rmse = np.sqrt(mean_squared_error(oos_actuals, oos_preds))
        oos_r2 = 1 - (np.sum((np.array(oos_actuals) - np.array(oos_preds)) ** 2)
                       / np.sum((np.array(oos_actuals) - np.mean(oos_actuals)) ** 2))
        print(f"OOS periods : {len(oos_preds)}")
        print(f"OOS RMSE    : {oos_rmse:.4f}")
        print(f"OOS R²      : {oos_r2:.4f}")
    else:
        print("Not enough data for OOS evaluation.")

    # ---- Save results -----------------------------------------------------
    out_dir = PROJECT_ROOT / "data_clean"
    out_dir.mkdir(exist_ok=True)

    weights_df.to_csv(out_dir / "assemblage_weights_yoy_ffill.csv", index=False)
    pd.DataFrame({
        "Date": dates,
        "Actual_12m_Forward": y,
        "Albacore_Prediction": result["fitted_values"],
    }).to_csv(out_dir / "assemblage_predictions_yoy_ffill.csv", index=False)

    print("\nSaved:")
    print(f"  {out_dir / 'assemblage_weights_yoy_ffill.csv'}")
    print(f"  {out_dir / 'assemblage_predictions_yoy_ffill.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
