# regression/regression_rank.py

Albacoreranks -- rank-space assemblage regression.

## Usage

```bash
python regression/regression_rank.py              # level 2 (default)
python regression/regression_rank.py --level 1    # level 1 (9 sectors)
```

## Model

```
min_w  mean(y - Ow)^2  +  lambda sum((w_{r+1} - w_r)^2)
s.t.   w >= 0,  mean(O)'w = mean(y)
```

## Functions

| Function | Description |
|---|---|
| `build_rank_matrix(X)` | Sort components low->high at each t |
| `_fit_ranks_single(O, y, lam)` | Fit rank model for a single lambda |
| `_cv_mses_ranks(O, y, lambdas, n_folds)` | Expanding-window CV |
| `rolling_oos_ranks(X, y, dates, ...)` | Rolling OOS with rank-space model |
| `main(level)` | Full pipeline for a given level |
