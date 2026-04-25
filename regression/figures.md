# regression/figures.py

All regression visualisation functions. Produces 5 figure types saved to `plots/`.

## Figures

| Function | Output File | Description |
|---|---|---|
| `fig_weights(result, w_prior)` | `reg_fig1_weights.png` | Horizontal bar: optimised weights vs. prior, sorted descending, coloured by group |
| `fig_lambda_cv(result)` | `reg_fig2_lambda_cv.png` | CV RMSE curve vs. lambda (log scale), best lambda marked |
| `fig_insample(result, dates, y)` | `reg_fig3_insample.png` | Time series: actual 12m forward headline vs. assemblage fitted values |
| `fig_oos(oos_df, bm_df, extra_oos)` | `reg_fig4_oos.png` | 2-panel: (top) OOS predictions vs. benchmarks time series, (bottom) rolling 36-month RMSE |
| `fig_ranks_weights(weights, lam)` | `reg_fig5_ranks_weights.png` | Bar chart: weight by rank position (1=lowest, K=highest), uniform 1/K reference line |

## Helper

| Function | Description |
|---|---|
| `_save(fig, name)` | Save figure to `PLOTS_DIR` with `bbox_inches='tight'` |
