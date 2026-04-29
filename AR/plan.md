Plan: Reproducing AR_ranks Inflation Forecasting for Japan
Below is an end-to-end plan you can hand to Claude (e.g., in a Code Execution session). It is structured so each phase produces a verifiable artifact before moving on.
Phase 0 — Specify the target precisely
Before any code, lock down three choices, because they determine everything downstream:

Inflation series. Japan's standard analogue to US headline PCE is the headline CPI (総合) from e-Stat / Statistics Bureau. A common alternative is CPI excluding fresh food (コアCPI) because fresh-food prices are notoriously noisy in Japan. Pick one as primary; keep the other as a robustness check.
Frequency and transformation. Monthly, seasonally adjusted, converted to month-over-month log growth: π_t = 100 × (log CPI_t − log CPI_{t−1}). All 12 lags will be these monthly π's.
Forecast target at horizon h. The paper targets the year-over-year inflation rate observed h months in the future: y_{t+h} = (1/12) Σ_{j=1..12} π_{t+h−j+1}. Use the same definition so results are comparable.

Phase 1 — Data assembly
Load Japan CPI, build π_t, and construct the design matrix. For each date t with at least 12 prior observations, store:

The 12 lags vector L_t = (π_{t−1}, π_{t−2}, …, π_{t−12})
The 12 order statistics vector R_t = sort(L_t) ascending, so R1_t ≤ R2_t ≤ … ≤ R12_t
The targets y_{t+h} for h ∈ {1, 3, 6, 12, 24}

Sanity check: plot π_t and 12-month YoY. Japan should show the long deflationary stretch (1998–2012), Abenomics bump, and the 2022+ reflation. If the picture looks wrong, the data is wrong — stop and fix before modeling.
Phase 2 — Implement the four models
Each is a linear model of the form y_{t+h} = α + β'X_t + ε. They differ only in X and constraints on β:
ModelRegressors X_tConstraintRandom walk (numéraire)y_t (current YoY)β fixed at 1, no estimationAR(1) on YoYy_tunconstrainedAR_lags(12)L_t (12 time-ordered lags)unconstrainedAR⁺_lags(12)L_tβ ≥ 0 (NNLS)AR_ranks(12)R_t (12 order statistics)β ≥ 0 (NNLS)
For the non-negativity constraint, use scipy.optimize.nnls or cvxpy. Estimate one model per horizon h — coefficients are not shared across horizons.
Phase 3 — Out-of-sample evaluation protocol
Mirror the paper's two-window structure:

Window A: 2010m1–2019m12 (pre-COVID, low-inflation regime)
Window B: 2020m1–2024m12 or latest (post-COVID, reflation regime)

Use an expanding window: start training on data through, say, 1995–2009, produce a forecast for 2010m1, append the actual to the training set, refit, forecast 2010m2, and so on. Repeat through the end of Window B. This is computationally cheap because each fit is a small OLS/NNLS.
Store, for each (model, horizon, date), the forecast and the realized y. Then compute RMSE within each window and divide by the random walk RMSE to get the normalized numbers that go in your Table 3 analogue.
Phase 4 — The Table 3 analogue and the bar chart
Produce two outputs that exactly mirror the paper's Figure/Table 3:

Table. Rows = {AR_lags, AR⁺_lags, AR_ranks, AR(1) on YoY}; columns = {h=1, 3, 6, 12, 24} × {Window A, Window B}; cells = RMSE relative to random walk; bold the minimum in each column.
Coefficient bar chart. For AR_ranks fit on the full sample (or the latest expanding-window fit), plot β₁…β₁₂ against R1…R12. This is the diagnostic that tells you which ranks Japan's inflation listens to. The US result was "weight on the top six." Japan may differ — the long deflation era could plausibly elevate the importance of low ranks, or the result might still concentrate on the top end. This is the most interesting empirical question.

Phase 5 — Japan-specific robustness
Before drawing conclusions, run three checks the paper implicitly relies on:

Subsample stability. Refit AR_ranks separately on 1995–2012 (deflation era) and 2013–present (post-Abenomics). Compare bar charts. If they differ sharply, "the" coefficients are an average of two regimes and headline performance numbers should be reported per regime.
Consumption-tax hike dummies. Japan raised the consumption tax in April 1997, April 2014, and October 2019. Each created a one-off CPI jump. Either include dummies for those months or drop them from the training set; otherwise AR_ranks will assign suspicious weight to the highest rank simply because of these mechanical spikes.
Series choice. Re-run with core CPI excluding fresh food. If conclusions flip, headline noise is dominating and the core result is the more credible one.