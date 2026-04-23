# Changes from Original Notebook

Code lives in `config.py`, `cpi_analysis.py`, `regression.py`.  
Original notebook kept at `japan_inflation_regression.ipynb` for reference.

---

## Data fixes

### Column taxonomy (`config.py`)

| Type | Count | Treatment |
|---|---|---|
| Composite | 6 | Always exclude — derived from headline |
| Special / 別掲 | 4 | Exclude — cross-cutting aggregates that double-count components |
| True components | 46 | Regression features |
| Rent | 1 | See below |

Original notebook used 51 features, incorrectly including the 4 special columns (Energy, Expenses for education, Expenses for culture & recreation, Expenses for information & communication) alongside their constituent components.

### Basket weights

**Old:** normalised by sum of all CSV rows (~25B denominator), inflated by aggregate rows.  
**New:** denominator = 総合 raw weight (3,190,396,706) only. Renormalised to sum to 1.

---

## Regression fixes (`regression.py`)

### Penalty: shrinkw → flat

**Old:** `loss = MSE + λ Σ σᵢ (wᵢ − w̃ᵢ)²` — deviating from Rent's prior was 31× cheaper than Fruits', actively rewarding weight concentration on low-variance components.  
**New:** `loss = MSE + λ Σ (wᵢ − w̃ᵢ)²` — all components penalised equally.

### Lambda grid

**Old:** two-stage CV, narrow range [e⁻³, e²], best always at lower edge — grid anchored wrong.  
**New:** single-stage, 40 log-spaced values in [10⁻⁴, 10³]. Fixed from pre-OOS window only, no look-ahead.

### OOS evaluation

**Old:** step=12, 16 predictions — statistically meaningless.  
**New:** step=1, ~298 predictions.

### Benchmarks added

Random walk, Core ex fresh food, Core ex food & energy — none existed in the original.

---

## Experiments and findings

### Rent isolation (46 components, 3m/3m)

Removed Rent from features to avoid the zero-variance sink problem (σ=1.09, lowest of all components; ~85% is imputed rent which barely moves in Japan).

| Model | OOS RMSE | OOS R² |
|---|---|---|
| Assemblage | 1.67 | -0.48 |
| Core (ex fresh food) | 2.10 | -1.33 |
| Random walk | 2.17 | -1.50 |
| Core (ex food & energy) | 2.29 | -1.77 |

Beats benchmarks on RMSE. All R² negative — 12m forward 3m/3m is genuinely hard to predict.

### YoY instead of 3m/3m (46 components)

Switched growth rates and target to YoY (12m ahead headline YoY).  
Lambda collapsed to ceiling (10⁶) — weights became identical to prior. YoY components are too collinear and the official basket weights are already near-optimal for predicting future headline YoY. No benefit from reweighting.

| Model | OOS RMSE | OOS R² |
|---|---|---|
| Assemblage | 1.46 | -0.077 |
| Core (ex fresh food) | 1.40 | +0.005 |
| Random walk | 1.44 | -0.050 |

YoY RMSE improved but assemblage no longer beats core. Dropped.

### Rent re-included (47 components, 3m/3m) — current state

Added Rent back to observe the effect directly.

| Model | OOS RMSE | OOS R² |
|---|---|---|
| **Assemblage** | **1.30** | **+0.103** |
| Core (ex fresh food) | 2.10 | -1.33 |
| Random walk | 2.17 | -1.50 |
| Core (ex food & energy) | 2.29 | -1.77 |

Best OOS performance. But Rent absorbs 56.6% in-sample weight (prior: 18.33%) — the zero-variance sink mechanism is back. The OOS gain may be coming from the model learning to heavily discount volatile components, not from genuine forward-looking signal in Rent itself.

---

## Current problems

1. **Rent dominance** — 56.6% optimised weight vs 18.33% prior. Model reduces to "Rent + a few food items." Mechanically valid but economically degenerate: imputed rent is an administrative estimate, not a market signal.

2. **Lambda regime mismatch** — OOS lambda (~1.3) selected from 1990–2000 deflation era. Applied to post-2022 surge without updating. Could under- or over-regularise during structural breaks.

3. **Basket coverage gap** — Rent covers 18.33% of the basket. Whether it's included or excluded, the 46 components don't cleanly represent 100% of headline inflation. No market-rent series available to replace the imputed portion.

4. **Negative R² in 46-component run** — all models including benchmarks have negative R² under 3m/3m. This is a property of the target (mean of 12 future 3m/3m rates), not a model failure. Worth framing explicitly in the thesis.

---

## Open items

- [ ] Cap Rent weight at basket prior (18.33%) as a constrained variant — see if remaining components find a meaningful reweighting
- [ ] Add unconditional mean benchmark — likely beats all on R², which is itself a thesis finding
- [ ] Source market-rent series (MLIT) to replace imputed-dominated Rent column
- [ ] Sub-period evaluation: deflation era / Abenomics / post-2022 surge separately
- [ ] Rolling lambda re-estimation (update every N years) to address regime mismatch

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Column taxonomy, basket mappings, group colours |
| `cpi_analysis.py` | Data visualisation — core measures, basket weights, contributions, Rent deep-dive |
| `regression.py` | Assemblage regression — training, OOS, benchmarks, figures |
| `japan_inflation_regression.ipynb` | Original notebook — reference only |
| `data_eda.ipynb` | Original EDA — reference only |
