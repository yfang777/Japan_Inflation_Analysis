# regression/evaluation.py

Metrics computation and scorecard printing for model comparison.

## Functions

| Function | Description |
|---|---|
| `metrics(actual, predicted)` | Compute RMSE, MAE, R², and valid-observation count. Handles NaN masking. |
| `print_scorecard(insample, oos_df, bm_df, extra_oos, our_models)` | Print a formatted comparison table: in-sample stats + all OOS models and benchmarks side by side. Models in `our_models` set are marked with `<--`. |

## Scorecard Output

```
=================================================================
RESULTS SCORECARD
=================================================================

  In-sample (full data):
    RMSE:             X.XXXX
    MAE:              X.XXXX
    R2:               X.XXXX
    Non-zero weights: NN/47
    Lambda:           X.XXXXX

  Out-of-sample (step=1, n=NNN):
  Model                           RMSE      MAE       R2
  ---------------------------------------------------------
  Comps (expanding)              X.XXXX  X.XXXX   X.XXXX <--
  ...
=================================================================
```
