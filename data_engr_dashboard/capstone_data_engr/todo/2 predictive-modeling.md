# Plan 2 — Predictive modeling (predict alpha)

**Goal:** build deep-learning models to predict the fund's (or its holdings') forward returns/alpha,
using the insights from Plan 1, with architectures chosen so we can **mine interpretation** out of
them (not just a black-box score). Do Plan 1 first — its findings should guide feature/architecture
choices.

**Data:**
- `combined_panel.csv` (7,740 stock-months × 91) — for cross-sectional / sequence models. Label =
  `next_ret`. Features: per-stock fundamentals, factors, macro, country/sector dummies.
- `combined_monthly.csv` — for portfolio-level time-series models. Target = `alpha_vs_avg` / `fund_ret`.

## Directions (per roadmap; build on the existing `ds6050 project/models/`)
- The group repo already has CNN (with saliency) and a hybrid CNN+Transformer. **Re-run / extend
  those on the richer, corrected, survivorship-free datasets here** — that alone may change results.
- Candidate architectures: CNN/RNN over trade or monthly sequences; attention/Transformer over the
  stock-month panel; a cross-sectional MLP on characteristics (Gu/Kelly/Xiu style) predicting
  `next_ret`, aggregated to a portfolio forecast.
- **Interpretability is a requirement:** prefer attention weights, gradient saliency, or SHAP so the
  model tells us *which features/holdings/regimes* drive predictions — to verify or reject the
  Plan-1 theories.

## Methodology guardrails
- **Time-series splits only** (no shuffling); respect point-in-time. Hold out the most recent period.
- **Standardize / rank-normalize** features (units differ; cross-sectional rank-normalization per
  month is standard for characteristics models).
- Watch the small sample: 129 months / 57 fundamentals-months is tiny for DL — favor regularization,
  simple architectures, the panel (more rows), and the PCs to reduce dimensionality.
- Compare against the Plan-1 regularized baseline — DL must beat it to justify the complexity.

## Done when
One or more predictive models trained with proper time-series validation and an interpretability
analysis that confirms/rejects the Plan-1 drivers; results compared to the baseline; findings
written up. (Roadmap "Decide From There" follows.)
