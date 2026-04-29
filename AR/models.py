"""AR/models.py — five univariate forecasters.

All ``fit_*`` return ``(alpha, beta)``; ``predict`` is shared.

For NNLS-with-intercept models (AR+_lags, AR_ranks) the intercept is left
unconstrained while beta >= 0. We achieve this by demeaning X and y on the
training sample, fitting NNLS on the demeaned problem, then backing out
alpha = y_bar - X_bar @ beta.
"""

import numpy as np
from scipy.optimize import nnls


# ── core estimators ──────────────────────────────────────────────────────────

def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """OLS with intercept. X is (T, K); returns (alpha, beta)."""
    Xc = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return float(coef[0]), coef[1:]


def fit_nnls_with_intercept(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """NNLS for beta with unconstrained intercept (via demeaning)."""
    Xb = X.mean(axis=0)
    yb = float(y.mean())
    beta, _ = nnls(X - Xb, y - yb)
    alpha = yb - float(Xb @ beta)
    return alpha, beta


def predict(X: np.ndarray, alpha: float, beta: np.ndarray) -> np.ndarray:
    return alpha + X @ beta


# ── named forecasters ────────────────────────────────────────────────────────

def forecast_rw(yoy_t: float) -> float:
    return float(yoy_t)


def fit_ar1_yoy(yoy_train: np.ndarray, y_train: np.ndarray):
    return fit_ols(yoy_train.reshape(-1, 1), y_train)


def fit_ar_lags(L_train: np.ndarray, y_train: np.ndarray):
    return fit_ols(L_train, y_train)


def fit_ar_lags_plus(L_train: np.ndarray, y_train: np.ndarray):
    return fit_nnls_with_intercept(L_train, y_train)


def fit_ar_ranks(R_train: np.ndarray, y_train: np.ndarray):
    return fit_nnls_with_intercept(R_train, y_train)
