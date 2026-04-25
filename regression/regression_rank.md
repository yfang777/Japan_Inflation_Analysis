# regression/regression_rank.py

Albacoreranks -- rank-space assemblage regression. Instead of weighting components by identity, sorts them low-to-high at each time step and learns which percentile of the cross-sectional distribution predicts future headline inflation.

## Model

```
min_w  mean(y - Ow)²  +  λ Σ(w_{r+1} - w_r)²   [fused ridge]
s.t.   w >= 0,  Ō'w = ȳ                          [mean constraint]
```

- **O:** T x K order-statistic matrix (X sorted low→high at each t)
- **Penalty:** Fused ridge -- encourages smooth weights across adjacent ranks
- **Constraint:** Mean-matching (not sum-to-1)

## Functions

| Function | Description |
|---|---|
| `build_rank_matrix(X)` | Sort component growth rates low→high at each t |
| `_fit_ranks_single(O, y, lam)` | Fit rank model for a single lambda |
| `_cv_mses_ranks(O, y, lambdas, n_folds)` | Expanding-window CV for rank model |
| `rolling_oos_ranks(X, y, dates, ...)` | Rolling OOS with rank-space model |
| `main()` | Full pipeline: data → OOS ranks → benchmarks → scorecard → figures |

## Usage

```bash
python regression/regression_rank.py
```

## Pipeline

1. Load and prepare data
2. OOS rolling 20-year window with rank-space model
3. In-sample fit on last 20-year window
4. Compute benchmarks
5. Print scorecard
6. Generate 2 figures: rank weights, OOS comparison
