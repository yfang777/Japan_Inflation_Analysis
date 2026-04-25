"""
regression.py  –  Assemblage regression for Japan CPI forward inflation

Fixes vs. original notebook:
  - 46 clean components (Rent isolated, 別掲 special cols excluded) via config.py
  - Flat penalty: λ||w − w_prior||²  (shrinkw removed — see MODEL section)
  - Single-stage wide lambda grid; no two-stage anchoring
  - OOS step=1, lambda fixed from pre-OOS window only (no look-ahead)
  - Benchmarks: random walk, core ex fresh food, core ex food & energy

Run:  python regression.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_FILE, WEIGHTS_CSV, PLOTS_DIR,
    COMPONENT_COLS, EN_TO_JPN, GROUP_COLORS, GROUPS,
)
from utils.smart_imputation import smart_impute

PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

COL_TO_GROUP = {col: grp for grp, cols in GROUPS.items() for col in cols}

# 46 clean components + Rent (isolated in config but included here for comparison)
FEATURES = COMPONENT_COLS + ['Rent']

# ── run settings ───────────────────────────────────────────────────────────────
START_DATE  = '1990-01-01'
HORIZON     = 12                     # months ahead
MIN_TRAIN   = 120                    # 10-year minimum OOS training window
OOS_STEP    = 1                      # rolling step size (months)
N_CV_FOLDS     = 10
LAMBDA_GRID       = np.logspace(-4, 3, 40) # comps: penalises deviation from basket prior
LAMBDA_GRID_RANKS = np.logspace(0, 5, 40)  # ranks: penalises non-smoothness across ranks
ROLLING_WINDOW    = 240                    # 20-year rolling window (paper's default)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(start_date=START_DATE, horizon=HORIZON):
    """
    Load raw CPI index → impute → 3m/3m growth rates → 12m forward target.

    Returns
    -------
    X      : ndarray (T, 47)   component 3m/3m growth rates
    y      : ndarray (T,)      12m forward average of headline 3m/3m
    dates  : DatetimeIndex     corresponding dates
    growth : DataFrame         full growth-rate frame (incl. composites for benchmarks)
    """
    # load
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['YearMonth'], format='%Y-%m')
    df = (df.set_index('Date')
            .drop('YearMonth', axis=1)
            .replace('-', np.nan)
            .apply(pd.to_numeric, errors='coerce'))
    df = df[df.index >= start_date].copy()

    # impute — run on components + composite cols needed for benchmarks
    benchmark_cols = [
        'All items',
        'All items, less fresh food',
        'All items, less food (less alcoholic beverages) and energy',
    ]
    cols_to_impute = list(dict.fromkeys(FEATURES + benchmark_cols))
    df_imp, _ = smart_impute(df[cols_to_impute], strategy='auto', verbose=False)

    # 3m/3m annualized growth rates
    growth = df_imp.apply(lambda s: ((s / s.shift(3)) - 1) * 4 * 100)

    # 12-month forward average headline as regression target
    headline = growth['All items']
    target = pd.Series(np.nan, index=headline.index)
    for i in range(len(headline) - horizon):
        window = headline.iloc[i + 1 : i + 1 + horizon]
        if window.notna().all():
            target.iloc[i] = window.mean()

    # valid rows: both target and headline growth are available
    valid = target.notna() & headline.notna()

    X_df = growth[FEATURES][valid]

    # final safety fill — should be zero after imputation, but guard anyway
    if X_df.isna().any().any():
        X_df = X_df.fillna(X_df.median())

    return X_df.values, target[valid].values, X_df.index, growth


# ══════════════════════════════════════════════════════════════════════════════
#  PRIOR WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def load_prior_weights() -> np.ndarray:
    """
    Official basket weights for the 46 active components.
    Denominator: 総合 raw weight (correct — not inflated by aggregate rows).
    Renormalized to sum to 1 since Rent (18.33%) is excluded.
    """
    ow = pd.read_csv(WEIGHTS_CSV)
    jpn_to_raw = dict(zip(ow['Category_Name'], ow['Weight']))
    total = jpn_to_raw['総合']

    raw = np.array([
        jpn_to_raw.get(EN_TO_JPN.get(col, ''), np.nan)
        for col in FEATURES
    ])
    if np.isnan(raw).any():
        missing = [FEATURES[i] for i in np.where(np.isnan(raw))[0]]
        print(f'  Warning: no official weight for {missing}; using equal fallback.')
        raw = np.where(np.isnan(raw), np.nanmean(raw), raw)

    shares = raw / total
    return shares / shares.sum()   # renorm to 1 (Rent excluded)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
#
# Penalty choice: flat  λ||w − w_prior||²
#
# The original notebook used shrinkw-weighted penalty: λ × σᵢ × (wᵢ − w̃ᵢ)²
# This made it 31× cheaper to deviate from Rent's prior (σ=1.09) than from
# Fruits' prior (σ=34.5). That's what let Rent absorb 57% weight.
# Even with Rent removed, shrinkw still rewards parking weight on low-variance
# series. A flat penalty treats every component equally — no built-in bias
# toward stable or volatile components.

def _fit_single(X: np.ndarray, y: np.ndarray,
                lam: float, w_prior: np.ndarray) -> dict:
    """
    Fit assemblage for a single lambda.

    min_w  mean(y − Xw)²  +  λ ||w − w_prior||²
    s.t.   w ≥ 0,  Σw = 1
    """
    k = X.shape[1]

    def objective(w):
        return np.mean((y - X @ w) ** 2) + lam * np.sum((w - w_prior) ** 2)

    res = minimize(
        objective,
        x0=w_prior.copy(),
        method='SLSQP',
        bounds=[(0, None)] * k,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
        options={'maxiter': 2000, 'ftol': 1e-10},
    )
    w     = res.x
    y_hat = X @ w
    ss_r  = np.sum((y - y_hat) ** 2)
    ss_t  = np.sum((y - y.mean()) ** 2)
    return {
        'weights':   w,
        'fitted':    y_hat,
        'lambda':    lam,
        'rmse':      float(np.sqrt(np.mean((y - y_hat) ** 2))),
        'mae':       float(np.mean(np.abs(y - y_hat))),
        'r2':        float(1 - ss_r / ss_t) if ss_t > 0 else 0.0,
        'n_nonzero': int((w > 1e-4).sum()),
        'converged': res.success,
    }


def _cv_mses(X: np.ndarray, y: np.ndarray,
             lambdas: np.ndarray, w_prior: np.ndarray,
             n_folds: int = N_CV_FOLDS) -> np.ndarray:
    """
    Expanding-window time-series CV. Returns mean CV MSE per lambda.
    No shuffling — each fold trains on [0, t] and validates on [t, t+k].
    """
    n          = len(y)
    min_tr     = n // (n_folds + 1)
    fold_size  = n // n_folds
    mse_matrix = np.full((len(lambdas), n_folds), np.nan)

    for li, lam in enumerate(lambdas):
        for f in range(n_folds):
            t_end  = min_tr + f * fold_size
            t_test = min(t_end + fold_size, n)
            if t_end >= n:
                continue
            r   = _fit_single(X[:t_end], y[:t_end], lam, w_prior)
            pv  = X[t_end:t_test] @ r['weights']
            mse_matrix[li, f] = np.mean((y[t_end:t_test] - pv) ** 2)

    return np.nanmean(mse_matrix, axis=1)


def train(X: np.ndarray, y: np.ndarray, w_prior: np.ndarray,
          lambdas: np.ndarray = LAMBDA_GRID,
          n_folds: int = N_CV_FOLDS) -> dict:
    """
    Select lambda via time-series CV, then fit on full data.
    """
    print(f'  CV over {len(lambdas)} lambdas ({n_folds}-fold expanding window)...')
    cv_mse      = _cv_mses(X, y, lambdas, w_prior, n_folds)
    best_idx    = int(np.argmin(cv_mse))
    best_lam    = lambdas[best_idx]
    print(f'  Best λ = {best_lam:.5f}  (CV RMSE = {np.sqrt(cv_mse[best_idx]):.4f})')

    result = _fit_single(X, y, best_lam, w_prior)
    result['cv_mse']      = cv_mse
    result['lambda_grid'] = lambdas
    result['best_lambda'] = best_lam
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ALBACORERANKS  –  rank-space assemblage (supervised trimming)
# ══════════════════════════════════════════════════════════════════════════════
#
# Instead of weighting components by identity, sort them low→high at each t
# and learn which percentile of the distribution is predictive of future
# headline inflation. Fused-ridge penalty encourages smooth weights across
# adjacent ranks. Mean constraint replaces sum-to-1.

def build_rank_matrix(X: np.ndarray) -> np.ndarray:
    """Sort component growth rates low→high at each t. Shape unchanged (T, K)."""
    return np.sort(X, axis=1)


def _fit_ranks_single(O: np.ndarray, y: np.ndarray, lam: float) -> dict:
    """
    Assemblage in rank space.

    min_w  mean(y − Ow)²  +  λ Σ(w_{r+1} − w_r)²   [fused ridge]
    s.t.   w ≥ 0,  Ō'w = ȳ   [mean constraint, not sum-to-1]
    """
    K      = O.shape[1]
    O_mean = O.mean(axis=0)
    y_mean = float(y.mean())

    def objective(w):
        return np.mean((y - O @ w) ** 2) + lam * float(np.sum(np.diff(w) ** 2))

    res = minimize(
        objective,
        x0=np.ones(K) / K,
        method='SLSQP',
        bounds=[(0, None)] * K,
        constraints={'type': 'eq', 'fun': lambda w: float(O_mean @ w) - y_mean},
        options={'maxiter': 2000, 'ftol': 1e-10},
    )
    w     = res.x
    y_hat = O @ w
    ss_r  = np.sum((y - y_hat) ** 2)
    ss_t  = np.sum((y - y.mean()) ** 2)
    return {
        'weights':   w,
        'fitted':    y_hat,
        'lambda':    lam,
        'rmse':      float(np.sqrt(np.mean((y - y_hat) ** 2))),
        'r2':        float(1 - ss_r / ss_t) if ss_t > 0 else 0.0,
        'n_nonzero': int((w > 1e-4).sum()),
        'converged': res.success,
    }


def _cv_mses_ranks(O: np.ndarray, y: np.ndarray,
                   lambdas: np.ndarray,
                   n_folds: int = N_CV_FOLDS) -> np.ndarray:
    """Expanding-window CV for rank-space model."""
    n         = len(y)
    min_tr    = n // (n_folds + 1)
    fold_size = n // n_folds
    mse_mat   = np.full((len(lambdas), n_folds), np.nan)

    for li, lam in enumerate(lambdas):
        for f in range(n_folds):
            t_end  = min_tr + f * fold_size
            t_test = min(t_end + fold_size, n)
            if t_end >= n:
                continue
            r  = _fit_ranks_single(O[:t_end], y[:t_end], lam)
            pv = O[t_end:t_test] @ r['weights']
            mse_mat[li, f] = np.mean((y[t_end:t_test] - pv) ** 2)

    return np.nanmean(mse_mat, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#  OUT-OF-SAMPLE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def rolling_oos(X: np.ndarray, y: np.ndarray,
                w_prior: np.ndarray, dates: pd.DatetimeIndex,
                min_train: int = MIN_TRAIN,
                step: int = OOS_STEP,
                lambdas: np.ndarray = LAMBDA_GRID,
                window: int = None) -> tuple:
    """
    Rolling OOS for Albacorecomps.

    Lambda fixed from first min_train months — no look-ahead.
    window=None → expanding (all history); window=240 → rolling 20-year.
    Returns (oos_df, oos_lambda).
    """
    win_str = f'rolling {window}m' if window else 'expanding'
    print(f'\n  Selecting OOS lambda from first {min_train} months [{win_str}]...')
    cv_mse     = _cv_mses(X[:min_train], y[:min_train], lambdas, w_prior, n_folds=5)
    oos_lambda = lambdas[int(np.argmin(cv_mse))]
    print(f'  OOS λ = {oos_lambda:.5f}')

    steps   = range(min_train, len(X), step)
    records = []
    for i, t in enumerate(steps):
        t_start = max(0, t - window) if window else 0
        r       = _fit_single(X[t_start:t], y[t_start:t], oos_lambda, w_prior)
        y_pred  = float(X[t] @ r['weights'])
        records.append({'date': dates[t], 'actual': y[t], 'predicted': y_pred})
        if (i + 1) % 60 == 0:
            print(f'    {i+1}/{len(steps)} ...')

    df_oos           = pd.DataFrame(records).set_index('date')
    df_oos['error']  = df_oos['predicted'] - df_oos['actual']
    print(f'  OOS complete: {len(df_oos)} predictions')
    return df_oos, oos_lambda


def rolling_oos_ranks(X: np.ndarray, y: np.ndarray,
                      dates: pd.DatetimeIndex,
                      min_train: int = MIN_TRAIN,
                      step: int = OOS_STEP,
                      lambdas: np.ndarray = LAMBDA_GRID_RANKS,
                      window: int = ROLLING_WINDOW) -> tuple:
    """
    Rolling OOS for Albacoreranks (rank-space assemblage).

    Builds order-statistic matrix O from X, then runs rank-space model
    with fused-ridge penalty and rolling window.
    Lambda fixed from first min_train months — no look-ahead.
    Returns (oos_df, oos_lambda).
    """
    O = build_rank_matrix(X)
    win_str = f'rolling {window}m' if window else 'expanding'
    print(f'\n  [Ranks] Selecting OOS lambda from first {min_train} months [{win_str}]...')
    cv_mse     = _cv_mses_ranks(O[:min_train], y[:min_train], lambdas, n_folds=5)
    oos_lambda = lambdas[int(np.argmin(cv_mse))]
    print(f'  [Ranks] OOS λ = {oos_lambda:.5f}')

    steps   = range(min_train, len(O), step)
    records = []
    for i, t in enumerate(steps):
        t_start = max(0, t - window) if window else 0
        r       = _fit_ranks_single(O[t_start:t], y[t_start:t], oos_lambda)
        y_pred  = float(O[t] @ r['weights'])
        records.append({'date': dates[t], 'actual': y[t], 'predicted': y_pred})
        if (i + 1) % 60 == 0:
            print(f'    {i+1}/{len(steps)} ...')

    df_oos           = pd.DataFrame(records).set_index('date')
    df_oos['error']  = df_oos['predicted'] - df_oos['actual']
    print(f'  [Ranks] OOS complete: {len(df_oos)} predictions')
    return df_oos, oos_lambda


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def compute_benchmarks(growth: pd.DataFrame,
                       oos_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Naive benchmarks aligned to OOS dates.
    Each benchmark uses only information available at time t to predict y_t
    (the 12-month forward average of headline inflation).

    Three predictors:
      Random walk         — current 3m/3m headline (most naive)
      Core ex fresh food  — Japan's standard published core
      Core ex food&energy — the 'supercore'
    """
    col_map = {
        'Random walk':           'All items',
        'Core (ex fresh food)':  'All items, less fresh food',
        'Core (ex food&energy)': 'All items, less food (less alcoholic beverages) and energy',
    }
    bm = {name: growth[col].reindex(oos_dates)
          for name, col in col_map.items()
          if col in growth.columns}
    return pd.DataFrame(bm)


def compute_mean_benchmark(y: np.ndarray, dates: pd.DatetimeIndex,
                           oos_dates: pd.DatetimeIndex,
                           min_train: int = MIN_TRAIN) -> pd.Series:
    """
    Unconditional mean benchmark: predict mean(y[0:t]) at each OOS step.
    This is the R²=0 floor — any model that can't beat this is useless.
    """
    records = []
    for t, date in enumerate(dates):
        if date not in oos_dates:
            continue
        idx = np.where(dates == date)[0][0]
        records.append({'date': date, 'predicted': float(y[:idx].mean())})
    return pd.DataFrame(records).set_index('date')['predicted']


def compute_ols_benchmark(growth: pd.DataFrame, y: np.ndarray,
                          dates: pd.DatetimeIndex,
                          oos_dates: pd.DatetimeIndex,
                          min_train: int = MIN_TRAIN) -> pd.Series:
    """
    OLS regression benchmark — the paper's Xbm_t.

    At each OOS step t, fit OLS on expanding window:
        y = a + b1*headline + b2*core_ex_ff + b3*core_ex_fe + eps
    Then predict y[t] using current-period rates. Intercept captures long-run mean.
    """
    col_map = {
        'headline': 'All items',
        'core_ff':  'All items, less fresh food',
        'core_fe':  'All items, less food (less alcoholic beverages) and energy',
    }
    # build feature matrix aligned to dates
    X_bm = np.column_stack([
        growth[col].reindex(dates).ffill().values
        for col in col_map.values()
    ])

    records = []
    for date in oos_dates:
        t = np.where(dates == date)[0][0]
        if t < min_train:
            continue
        X_tr, y_tr = X_bm[:t], y[:t]
        beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        records.append({'date': date, 'predicted': float(X_bm[t] @ beta)})
    return pd.DataFrame(records).set_index('date')['predicted']


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    ss_r = np.sum((a - p) ** 2)
    ss_t = np.sum((a - a.mean()) ** 2)
    return {
        'RMSE': float(np.sqrt(mean_squared_error(a, p))),
        'MAE':  float(mean_absolute_error(a, p)),
        'R2':   float(1 - ss_r / ss_t) if ss_t > 0 else np.nan,
        'N':    int(mask.sum()),
    }


def print_scorecard(insample: dict, oos_df: pd.DataFrame,
                    bm_df: pd.DataFrame,
                    extra_oos: dict = None) -> None:
    """
    extra_oos : dict of {label: oos_DataFrame} for additional models
                (e.g. rolling-window assemblage, Albacoreranks).
    """
    actual = oos_df['actual'].values

    print('\n' + '=' * 65)
    print('RESULTS SCORECARD')
    print('=' * 65)

    print(f'\n  In-sample (full data, Albacorecomps):')
    print(f'    RMSE:             {insample["rmse"]:.4f}')
    print(f'    MAE:              {insample["mae"]:.4f}')
    print(f'    R2:               {insample["r2"]:.4f}')
    print(f'    Non-zero weights: {insample["n_nonzero"]}/{len(FEATURES)}')
    print(f'    Lambda:           {insample["best_lambda"]:.5f}')

    print(f'\n  Out-of-sample (step={OOS_STEP}, n={len(oos_df)}):')
    rows = [('Comps (expanding)', _metrics(actual, oos_df['predicted'].values))]
    if extra_oos:
        for label, df in extra_oos.items():
            pred = df['predicted'].reindex(oos_df.index).values
            rows.append((label, _metrics(actual, pred)))
    for col in bm_df.columns:
        rows.append((col, _metrics(actual, bm_df[col].values)))

    our_models = {'Comps (expanding)', 'Comps (rolling 20y)', 'Ranks (rolling 20y)'}
    print(f'  {"Model":<30}  {"RMSE":>7}  {"MAE":>7}  {"R2":>7}')
    print('  ' + '-' * 57)
    for name, m in rows:
        marker = ' <--' if name in our_models else ''
        print(f'  {name:<30}  {m["RMSE"]:>7.4f}  {m["MAE"]:>7.4f}  {m["R2"]:>7.4f}{marker}')

    print('=' * 65 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    fig.savefig(PLOTS_DIR / name, bbox_inches='tight')
    print(f'  saved {name}')


def fig_weights(result: dict, w_prior: np.ndarray) -> plt.Figure:
    """
    Optimized weights vs. prior for each component.
    Sorted by optimized weight descending; prior shown as an orange dot.
    """
    w_opt   = result['weights']
    idx     = np.argsort(w_opt)[::-1]
    labels  = [FEATURES[i] for i in idx]
    opt_s   = w_opt[idx]
    prior_s = w_prior[idx]
    colors  = [
        GROUP_COLORS['Rent'] if l == 'Rent'
        else GROUP_COLORS.get(COL_TO_GROUP.get(l, 'Other'), '#7f8c8d')
        for l in labels
    ]

    fig, ax = plt.subplots(figsize=(9, 13))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, opt_s * 100, color=colors, edgecolor='white',
            linewidth=0.4, label='Optimized weight')
    ax.scatter(prior_s * 100, y_pos, color='#e67e22', zorder=5,
               s=25, label='Prior (basket share)', marker='D')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Weight (%)')
    ax.set_title(
        f'Assemblage Weights vs. Prior\n'
        f'λ = {result["best_lambda"]:.4f} | '
        f'{result["n_nonzero"]}/{len(FEATURES)} non-zero | '
        f'in-sample R2 = {result["r2"]:.3f}',
        fontsize=11, fontweight='bold',
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, 'reg_fig1_weights.png')
    return fig


def fig_lambda_cv(result: dict) -> plt.Figure:
    """CV RMSE curve — shows how model performance varies with regularization."""
    lambdas = result['lambda_grid']
    cv_rmse = np.sqrt(result['cv_mse'])
    best    = result['best_lambda']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(lambdas, cv_rmse, color='#2c3e50', lw=2)
    ax.axvline(best, color='#e74c3c', lw=1.5, linestyle='--',
               label=f'Best λ = {best:.4f}')
    ax.set_xlabel('λ  (log scale)')
    ax.set_ylabel('CV RMSE')
    ax.set_title('Lambda Selection – Time-Series Cross-Validation',
                 fontweight='bold')
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, 'reg_fig2_lambda_cv.png')
    return fig


def fig_insample(result: dict, dates: pd.DatetimeIndex,
                 y: np.ndarray) -> plt.Figure:
    """In-sample fit: actual 12m forward headline vs. assemblage fitted values."""
    fitted = result['fitted']

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(dates, y,      color='#2c3e50', lw=1.8, label='Actual (12m fwd)')
    ax.plot(dates, fitted, color='#e74c3c', lw=1.5,
            linestyle='--', label=f'Assemblage fit  (R2={result["r2"]:.3f})')
    ax.axhline(0, color='black', lw=0.6, linestyle='--', alpha=0.4)
    ax.set_ylabel('3m/3m annualized %')
    ax.set_title('In-Sample: Assemblage vs. Actual 12m Forward Headline Inflation',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, 'reg_fig3_insample.png')
    return fig


def fig_ranks_weights(weights: np.ndarray, lam: float) -> plt.Figure:
    """Weight assigned to each rank (rank 1 = lowest 3m/3m, rank K = highest)."""
    K     = len(weights)
    ranks = np.arange(1, K + 1)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(ranks, weights * 100, color='#3498db', edgecolor='white', linewidth=0.4)
    ax.axhline(100 / K, color='#e74c3c', lw=1.5, linestyle='--',
               label=f'Uniform 1/K = {100/K:.1f}%')
    ax.set_xlabel('Rank (1 = lowest 3m/3m component, K = highest)')
    ax.set_ylabel('Weight (%)')
    ax.set_title(f'Albacoreranks: Weight by Rank Position  |  λ = {lam:.4f}',
                 fontweight='bold')
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, 'reg_fig5_ranks_weights.png')
    return fig


def fig_oos(oos_df: pd.DataFrame, bm_df: pd.DataFrame,
            extra_oos: dict = None) -> plt.Figure:
    """OOS predictions vs. benchmarks vs. actual."""
    actual = oos_df['actual']

    bm_styles = {
        'Random walk':           ('#95a5a6', ':',  1.2),
        'Core (ex fresh food)':  ('#27ae60', '--', 1.5),
        'Core (ex food&energy)': ('#3498db', '-.', 1.5),
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    extra_styles = {
        'Comps (rolling 20y)': ('#e74c3c', '--', 1.5),
        'Ranks (rolling 20y)': ('#8e44ad', '-',  1.8),
    }

    # top panel: time series
    ax = axes[0]
    ax.plot(actual.index, actual.values,
            color='#2c3e50', lw=2.2, label='Actual (12m fwd)')
    ax.plot(oos_df.index, oos_df['predicted'].values,
            color='#e74c3c', lw=1.5, linestyle=':', label='Comps (expanding)')
    if extra_oos:
        for name, df in extra_oos.items():
            c, ls, lw = extra_styles.get(name, ('#7f8c8d', '--', 1.2))
            ax.plot(df.index, df['predicted'].values, color=c, linestyle=ls,
                    lw=lw, label=name)
    for name, (c, ls, lw) in bm_styles.items():
        if name in bm_df.columns:
            ax.plot(bm_df.index, bm_df[name].values,
                    color=c, linestyle=ls, lw=lw, label=name)
    ax.axhline(0, color='black', lw=0.6, linestyle='--', alpha=0.4)
    ax.set_ylabel('3m/3m annualized %')
    ax.set_title('Out-of-Sample Forecast vs. Benchmarks (12m Forward Headline)',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=9, ncol=2)

    # bottom panel: rolling 36-month RMSE
    ax = axes[1]
    window = 36
    series_list = [
        ('Comps (expanding)', oos_df['predicted'], '#e74c3c', ':',  1.5),
    ]
    if extra_oos:
        for name, df in extra_oos.items():
            c, ls, lw = extra_styles.get(name, ('#7f8c8d', '--', 1.2))
            series_list.append((name, df['predicted'], c, ls, lw))
    for name, (c, ls, lw) in bm_styles.items():
        if name in bm_df.columns:
            series_list.append((name, bm_df[name], c, ls, lw))

    for name, pred, c, ls, lw in series_list:
        err2 = (actual - pred.reindex(actual.index)) ** 2
        roll_rmse = err2.rolling(window).mean().apply(np.sqrt)
        ax.plot(roll_rmse.index, roll_rmse.values,
                color=c, linestyle=ls, lw=lw, label=name)

    ax.set_ylabel(f'Rolling {window}m RMSE')
    ax.set_title('Rolling OOS RMSE – Assemblage vs. Benchmarks', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    _save(fig, 'reg_fig4_oos.png')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print('=' * 65)
    print('ASSEMBLAGE REGRESSION  –  Japan CPI Forward Inflation')
    print('=' * 65)

    # ── data ──────────────────────────────────────────────────────────────────
    print('\n[1/5] Loading and preparing data...')
    X, y, dates, growth = load_and_prepare()
    print(f'  Observations: {len(y)}  ({dates[0]:%Y-%m} – {dates[-1]:%Y-%m})')
    print(f'  Features:     {X.shape[1]} components')
    print(f'  Target mean:  {y.mean():.3f}%  std: {y.std():.3f}%')

    # ── prior weights ─────────────────────────────────────────────────────────
    print('\n[2/5] Loading prior weights...')
    w_prior = load_prior_weights()
    print(f'  Prior range: [{w_prior.min():.4f}, {w_prior.max():.4f}]  sum={w_prior.sum():.6f}')
    top3 = np.argsort(w_prior)[::-1][:3]
    for i in top3:
        print(f'  Top prior: {FEATURES[i]:<45} {w_prior[i]*100:.2f}%')

    # ── in-sample training ────────────────────────────────────────────────────
    print('\n[3/5] In-sample training (full data)...')
    insample = train(X, y, w_prior)
    print(f'  In-sample RMSE: {insample["rmse"]:.4f}  R2: {insample["r2"]:.4f}')
    print(f'  Non-zero weights: {insample["n_nonzero"]}/{len(FEATURES)}')

    top5_idx = np.argsort(insample['weights'])[::-1][:5]
    print('  Top 5 weights:')
    for i in top5_idx:
        print(f'    {FEATURES[i]:<45}  opt={insample["weights"][i]*100:.2f}%'
              f'  prior={w_prior[i]*100:.2f}%')

    # ── out-of-sample: comps expanding ───────────────────────────────────────
    print('\n[4/6] Albacorecomps – expanding window...')
    oos_exp_df, _ = rolling_oos(X, y, w_prior, dates, window=None)

    # ── out-of-sample: comps rolling ──────────────────────────────────────────
    print('\n[5/6] Albacorecomps – rolling 20-year window...')
    oos_roll_df, _ = rolling_oos(X, y, w_prior, dates, window=ROLLING_WINDOW)

    # ── out-of-sample: ranks rolling ──────────────────────────────────────────
    print('\n[6/6] Albacoreranks – rolling 20-year window...')
    oos_ranks_df, ranks_lam = rolling_oos_ranks(X, y, dates)

    # ── benchmarks ────────────────────────────────────────────────────────────
    print('\nComputing benchmarks...')
    bm_df = compute_benchmarks(growth, oos_exp_df.index)
    bm_df['Unconditional mean'] = compute_mean_benchmark(y, dates, oos_exp_df.index)
    bm_df['OLS (headline+core+supercore)'] = compute_ols_benchmark(
        growth, y, dates, oos_exp_df.index)

    extra_oos = {
        'Comps (rolling 20y)': oos_roll_df,
        'Ranks (rolling 20y)': oos_ranks_df,
    }

    # ── scorecard ─────────────────────────────────────────────────────────────
    print_scorecard(insample, oos_exp_df, bm_df, extra_oos=extra_oos)

    # ── in-sample rank weights (last full window) ─────────────────────────────
    O_full     = build_rank_matrix(X)
    t_start    = max(0, len(X) - ROLLING_WINDOW)
    r_ranks_is = _fit_ranks_single(O_full[t_start:], y[t_start:], ranks_lam)
    print(f'  Ranks in-sample R2 (last window): {r_ranks_is["r2"]:.4f}')

    # ── figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    fig_weights(insample, w_prior)
    fig_lambda_cv(insample)
    fig_insample(insample, dates, y)
    fig_oos(oos_exp_df, bm_df, extra_oos=extra_oos)
    fig_ranks_weights(r_ranks_is['weights'], ranks_lam)

    print(f'\nDone. Figures saved to {PLOTS_DIR.resolve()}')
    plt.show()


if __name__ == '__main__':
    main()
