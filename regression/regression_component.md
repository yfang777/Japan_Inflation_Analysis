# regression/regression_component.py

Albacorecomps -- component-space assemblage regression.

## Usage

```bash
python regression/regression_component.py              # level 2 (default)
python regression/regression_component.py --level 1    # level 1 (9 sectors)
python regression/regression_component.py --level 3    # level 3 (744 items)
```

## Model

```
min_w  mean(y - Xw)^2  +  lambda ||w - w_prior||^2
s.t.   w >= 0,  sum(w) = 1
```

- **w_prior:** Basket weights from the CSV Weights row, normalised to sum=1
- **Lambda selection:** Expanding-window time-series CV (10-fold)

## Functions

| Function | Description |
|---|---|
| `_fit_single(X, y, lam, w_prior)` | Fit for a single lambda via SLSQP |
| `_cv_mses(X, y, lambdas, w_prior, n_folds)` | Expanding-window CV |
| `train(X, y, w_prior, lambdas, n_folds)` | Select best lambda, fit on full data |
| `rolling_oos(X, y, w_prior, dates, ...)` | Rolling OOS evaluation |
| `main(level)` | Full pipeline for a given level |
