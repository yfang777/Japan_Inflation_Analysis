"""
regression/evaluation.py  –  metrics and scorecard for model comparison
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OOS_STEP


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, R², and valid-observation count."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    ss_r = np.sum((a - p) ** 2)
    ss_t = np.sum((a - a.mean()) ** 2)
    return {
        'RMSE': float(np.sqrt(np.mean((a - p) ** 2))),
        'MAE':  float(np.mean(np.abs(a - p))),
        'R2':   float(1 - ss_r / ss_t) if ss_t > 0 else np.nan,
        'N':    int(mask.sum()),
    }


def print_scorecard(insample: dict, oos_df: pd.DataFrame,
                    bm_df: pd.DataFrame,
                    extra_oos: dict = None,
                    our_models: set = None,
                    features: list[str] = None,
                    primary_label: str = 'Comps (expanding)') -> None:
    """
    Print a comprehensive comparison table.

    Parameters
    ----------
    insample  : dict from train() with keys rmse, mae, r2, n_nonzero, best_lambda
    oos_df    : DataFrame with columns actual, predicted (primary OOS model)
    bm_df     : DataFrame of benchmark predictions aligned to oos_df.index
    extra_oos : dict of {label: oos_DataFrame} for additional models
    our_models: set of model labels to mark with '<--'
    features  : list of feature names (for reporting n_nonzero / n_total)
    """
    actual = oos_df['actual'].values

    if our_models is None:
        our_models = set()

    n_features = len(features) if features else insample.get('n_nonzero', '?')

    print('\n' + '=' * 65)
    print('RESULTS SCORECARD')
    print('=' * 65)

    print(f'\n  In-sample (full data):')
    print(f'    RMSE:             {insample["rmse"]:.4f}')
    print(f'    MAE:              {insample["mae"]:.4f}')
    print(f'    R2:               {insample["r2"]:.4f}')
    print(f'    Non-zero weights: {insample["n_nonzero"]}/{n_features}')
    print(f'    Lambda:           {insample["best_lambda"]:.5f}')

    print(f'\n  Out-of-sample (step={OOS_STEP}, n={len(oos_df)}):')

    rows = [(primary_label, metrics(actual, oos_df['predicted'].values))]

    # extra models
    if extra_oos:
        for label, df in extra_oos.items():
            pred = df['predicted'].reindex(oos_df.index).values
            rows.append((label, metrics(actual, pred)))

    # benchmarks
    for col in bm_df.columns:
        rows.append((col, metrics(actual, bm_df[col].values)))

    print(f'  {"Model":<30}  {"RMSE":>7}  {"MAE":>7}  {"R2":>7}')
    print('  ' + '-' * 57)
    for name, m in rows:
        marker = ' <--' if name in our_models else ''
        print(f'  {name:<30}  {m["RMSE"]:>7.4f}  {m["MAE"]:>7.4f}  {m["R2"]:>7.4f}{marker}')

    print('=' * 65 + '\n')
