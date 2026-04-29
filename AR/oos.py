"""AR/oos.py — expanding-window OOS driver for the univariate AR family.

At each decision date t, fit each model on data through t-h (so the realized
target y_{i+h} for any training row i has been observed by time t), then
forecast y_{t+h}. Returns a long DataFrame with one row per (model, horizon, t).

Indexing convention matches the rest of the repo: ``date`` is the decision
date; ``actual`` is y_{t+h} aligned to that decision date.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from AR.config_ar import HORIZONS, MIN_TRAIN, TRAIN_START
from AR.models import (
    fit_ar1_yoy, fit_ar_lags, fit_ar_lags_plus, fit_ar_ranks, predict,
)


def _design(ds: dict, h: int):
    """Return aligned (dates, yoy, L, R, target) for horizon h."""
    L      = ds['L']
    R      = ds['R']
    yoy    = ds['yoy']
    target = ds['targets'][h]

    common = L.dropna().index.intersection(yoy.dropna().index)
    return (
        common,
        yoy.reindex(common).values,
        L.reindex(common).values,
        R.reindex(common).values,
        target.reindex(common).values,        # may contain NaN at the tail
    )


def run_oos(ds: dict,
            train_start: str = TRAIN_START,
            min_train: int = MIN_TRAIN,
            horizons: list[int] | tuple[int, ...] = tuple(HORIZONS)) -> pd.DataFrame:
    """Long-format OOS frame: [date, model, horizon, forecast, actual]."""
    rows = []
    train_start_ts = pd.Timestamp(train_start)

    for h in horizons:
        dates, yoy_v, L_v, R_v, tgt = _design(ds, h)
        keep   = dates >= train_start_ts
        dates  = dates[keep]
        yoy_v  = yoy_v[keep]
        L_v    = L_v[keep]
        R_v    = R_v[keep]
        tgt    = tgt[keep]

        T = len(dates)
        for i in range(T):
            train_end = i - h                  # last training index (inclusive)
            n_train   = train_end + 1
            if n_train < min_train:
                continue

            ytr    = tgt[: train_end + 1]
            yoy_tr = yoy_v[: train_end + 1]
            L_tr   = L_v[: train_end + 1]
            R_tr   = R_v[: train_end + 1]

            valid_train = ~np.isnan(ytr)
            if valid_train.sum() < min_train:
                continue
            ytr    = ytr[valid_train]
            yoy_tr = yoy_tr[valid_train]
            L_tr   = L_tr[valid_train]
            R_tr   = R_tr[valid_train]

            t      = dates[i]
            actual = tgt[i]                     # may be NaN near the data tail
            yoy_i  = yoy_v[i]
            L_i    = L_v[i:i + 1]
            R_i    = R_v[i:i + 1]

            # Random walk
            rows.append((t, 'RW', h, float(yoy_i), float(actual)))

            # AR(1) on YoY
            a, b = fit_ar1_yoy(yoy_tr, ytr)
            rows.append((t, 'AR(1) on YoY', h,
                         float(predict(np.array([[yoy_i]]), a, b)[0]),
                         float(actual)))

            # AR_lags(12)
            a, b = fit_ar_lags(L_tr, ytr)
            rows.append((t, 'AR_lags', h,
                         float(predict(L_i, a, b)[0]), float(actual)))

            # AR+_lags(12)
            a, b = fit_ar_lags_plus(L_tr, ytr)
            rows.append((t, 'AR+_lags', h,
                         float(predict(L_i, a, b)[0]), float(actual)))

            # AR_ranks(12)
            a, b = fit_ar_ranks(R_tr, ytr)
            rows.append((t, 'AR_ranks', h,
                         float(predict(R_i, a, b)[0]), float(actual)))

    return pd.DataFrame(rows, columns=['date', 'model', 'horizon',
                                        'forecast', 'actual'])


if __name__ == '__main__':
    from AR.data_ar import build_ar_dataset
    ds  = build_ar_dataset()
    res = run_oos(ds)
    print(res.head(10))
    print(f'  rows={len(res)}, '
          f'dates={res["date"].min():%Y-%m}..{res["date"].max():%Y-%m}')
