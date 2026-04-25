# regression/regression_component.py

Albacorecomps -- component-space assemblage regression. Finds optimal component weights that are maximally predictive of 12-month forward headline inflation, shrinking toward official basket weights.

## Model

```
min_w  mean(y - Xw)²  +  λ ||w - w_prior||²
s.t.   w >= 0,  Σw = 1
```

- **X:** T x 47 matrix of component 3m/3m growth rates
- **y:** 12-month forward average of headline 3m/3m
- **w_prior:** Official CPI basket weights (renormalised to sum=1)
- **Penalty:** Flat (all components penalised equally for deviating from prior)
- **Lambda selection:** Expanding-window time-series CV (10-fold)

## Functions

| Function | Description |
|---|---|
| `_fit_single(X, y, lam, w_prior)` | Fit for a single lambda via SLSQP constrained optimisation |
| `_cv_mses(X, y, lambdas, w_prior, n_folds)` | Expanding-window CV; returns mean MSE per lambda |
| `train(X, y, w_prior, lambdas, n_folds)` | Select best lambda via CV, then fit on full data |
| `rolling_oos(X, y, w_prior, dates, ...)` | Rolling OOS evaluation (expanding or fixed-window) |
| `main()` | Full pipeline: data → train → OOS → benchmarks → scorecard → figures |

## Usage

```bash
python regression/regression_component.py
```

## Pipeline

1. Load and prepare data (47 features, 1990--2026)
2. Load prior weights from official basket
3. In-sample training with CV lambda selection
4. OOS expanding window (~298 predictions)
5. OOS rolling 20-year window
6. Compute benchmarks (random walk, core, supercore, mean, OLS)
7. Print scorecard
8. Generate 4 figures: weights, lambda CV, in-sample fit, OOS comparison
