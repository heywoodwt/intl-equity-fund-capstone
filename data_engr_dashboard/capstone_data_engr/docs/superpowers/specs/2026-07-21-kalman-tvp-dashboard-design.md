# Kalman TVP Factor Attribution in the Dashboard

**Date:** 2026-07-21
**Status:** Approved design — ready for implementation plan

## Summary

Bring the time-varying-parameter (TVP) factor-attribution model from branch
`heywood` (`kalman/`) into the Streamlit dashboard's **📈 Factor exposures**
tab. The model replaces the static 4-factor OLS with a Kalman filter where
**alpha and every factor beta drift as random walks**, letting a user see *when*
the manager generated alpha and *how* factor tilts shifted over time.

The model is added as a new option — **"Kalman (TVP)"** — on the existing model
radio (currently OLS / Ridge / ElasticNet). When selected, the results pane
dispatches to a purpose-built renderer instead of the penalized-regression
layout.

## Motivation

The Factor exposures tab already estimates time-varying betas via rolling-window
OLS/Ridge/ElasticNet. The Kalman filter is the principled state-space version of
the same idea: instead of a hard trailing window, every parameter follows a
random walk and is estimated with a proper measurement/process-noise trade-off,
yielding smoothed paths with honest confidence bands and a real-time (filtered)
vs. retrospective (smoothed) distinction the rolling window cannot express.

No new data is required: the model reads the same `combined_monthly.csv` the
dashboard already loads, which contains `fund_ret`, `RF`, and the Fama-French
factors `Mkt_RF`, `SMB`, `HML`, `Mom`.

## Model recap

- **State:** `[alpha, beta_1, …, beta_k]` for the `k` selected regressors.
- **Transition:** `x_t = x_{t-1} + eta_t` (identity transition — random walk).
- **Observation:** `y_t = alpha_t + sum_j beta_{j,t} * X_{j,t} + eps_t`, with a
  time-varying observation row `[1, X_{1,t}, …, X_{k,t}]`.
- **Initial state:** OLS on the first 12 months; covariance inflated ×10.
- **Noise covariances Q, R:** estimated by EM (diagonal Q, scalar R).
- **Outputs:** filtered means/covs (real-time), smoothed means/covs
  (retrospective), one-step-ahead residuals + prediction variance,
  log-likelihood.

## Architecture

Three small, independently-understandable units, mirroring the existing
`data.py` / `analytics.py` / `rolling_regression.py` split.

### 1. `dashboard/tvp_kalman_filter.py` (new — copied verbatim)

The dependency-free `KalmanFilter` class from `kalman/tvp_kalman_filter.py` on
branch `heywood`, copied unchanged. Provides `filter`, `smooth`, `em`,
`loglikelihood` for a one-dimensional-observation random-walk TVP regression.
No pykalman. Uses only numpy (plus scipy `minimize` inside EM, already an
indirect dependency via scikit-learn/statsmodels).

### 2. `dashboard/kalman_tvp.py` (new — the testable wrapper)

The dashboard-facing adapter. Single entry point:

```python
def fit_tvp(df, target, x_cols, n_iter=20, init_window=12) -> TVPResult
```

`TVPResult` (a dataclass) holds, indexed by `date`:
- `smoothed`, `filtered` — DataFrames of state means, columns `["alpha"] + x_cols`.
- `smoothed_se`, `filtered_se` — matching standard-error DataFrames
  (`sqrt` of the per-t covariance diagonal).
- `residuals`, `resid_std` — one-step-ahead prediction errors and their
  standardized form (`residual / sqrt(pred_var)`).
- `loglik` — filter log-likelihood.
- `ols_params` — the static full-sample OLS coefficients (for reference lines),
  a `pd.Series` indexed `["alpha"] + x_cols`.
- `n` — usable months.

Responsibilities:
- Drop rows with any NA in `[target] + x_cols`; require `n > k + 1`.
- Build the time-varying observation matrices `(n, 1, k+1)`.
- Initial state from numpy `lstsq` OLS on the first `init_window` months
  (statsmodels is available but `lstsq` keeps this unit lean; full-sample
  reference OLS may use whichever is simplest).
- Run EM for Q and R, then `filter` and `smooth`.
- Compute residuals and prediction variance exactly as the notebook (cell 17).

The core filter stays Streamlit-free and framework-free so it is unit-testable
in isolation. `fit_tvp` itself imports no Streamlit.

### 3. `dashboard/app.py` (changed)

- Add `"Kalman (TVP)"` to the model choices offered on the Factor tab's model
  radio. The radio currently reads `R.MODELS = ("OLS", "Ridge", "ElasticNet")`;
  the tab will offer `R.MODELS + ("Kalman (TVP)",)` without changing
  `rolling_regression.MODELS` itself. **Default stays OLS.**
- When `model == "Kalman (TVP)"`, call the new `render_kalman(...)` instead of
  `render_regression(...)`. The Advanced penalty/CV controls (α, L1, coefficient
  scale) are hidden for Kalman since they don't apply; the window slider is also
  irrelevant and hidden.

### 4. `tests/test_kalman_tvp.py` (new)

Following the repo convention (`sys.path.insert` the `dashboard/` dir, import the
module directly, no Streamlit, no network):
- Output shapes: `smoothed`/`filtered` are `(n, k+1)`; SE frames match.
- On a stationary synthetic factor set with constant true betas, the mean
  smoothed alpha/betas are close to the static OLS estimates.
- Standardized residuals are approximately mean-0, unit-variance on
  well-specified synthetic data.
- Guard: fewer months than states raises a clear `ValueError`.

## `render_kalman` — results body

Reuses the specification controls already in the left `controls` column (target,
regressor multiselect, sample period). Layout top-to-bottom:

1. **Regressor screening.** Reuse `R.screen_regressors` to drop zero-variance /
   exactly-collinear regressors, surfacing the same warnings as the OLS path.
   Fit `fit_tvp` on the kept regressors.

2. **Top metrics** (3 columns): full-sample OLS R² (reference), mean annualized
   Kalman alpha (via `R.annualize` on the smoothed alpha path), usable months.

3. **Filtered vs smoothed toggle** — a radio (`Smoothed (retrospective)` /
   `Filtered (real-time)`) that selects which path + SE band every chart below
   draws. Default: Smoothed.

4. **Time-varying alpha** — two stacked charts:
   - Monthly alpha with a 95% band (`± 2·SE`) and a dashed flat line at the
     static-OLS alpha.
   - Annualized alpha `((1 + alpha)^12 − 1)` with **event annotations** drawn
     only for events inside the selected period. Events (from notebook cell 22):
     `2018-10` Q4 2018 selloff, `2020-03` COVID crash, `2020-11` vaccine rally,
     `2022-01` rate-hike cycle.

5. **Time-varying betas** — one small-multiple per kept regressor: drifting path,
   95% band, dashed static-OLS reference. A "lines to plot" multiselect caps how
   many render at once (mirrors the existing `render_regression` beta control).

6. **Residual diagnostics** (inside an expander): standardized one-step errors
   over time (±2 reference lines), histogram vs. an N(0,1) overlay, ACF, and a
   Ljung-Box test at lags 6 and 12 (statsmodels `acorr_ljungbox`, already a
   dependency). All Plotly, matching the dashboard's chart style.

## Guardrails (reuse existing logic)

Random-walk TVP on highly persistent or level regressors (macro PCs, VIX,
yields) makes the drifting intercept extrapolate and explode — the identical
failure the OLS path already guards. So:
- Reuse `R.alpha_interpretable(df, x_fit)`. When it returns false, **hide the
  alpha panels** and show the existing explanatory note (the intercept isn't a
  meaningful alpha for off-zero / persistent regressors), while still rendering
  the beta paths and diagnostics.
- A caption notes the model was built for the Fama-French factor set and nudges
  the user toward that preset for a meaningful alpha.

## Data flow

```
combined_monthly.csv  (already loaded, with excess_ret = fund_ret - RF)
  → period filter (reg_period slider, existing)
  → R.screen_regressors  (drop degenerate/collinear)
  → kalman_tvp.fit_tvp(df, target, x_fit)   [cached]
  → render_kalman charts
```

`fit_tvp` results are cached with `@st.cache_data` keyed on target, the sorted
kept-regressor tuple, and the period bounds, so toggling filtered/smoothed or
changing which beta lines are plotted does not re-run EM.

## Dependencies

No new packages. `numpy`, `pandas`, `plotly`, `streamlit`, `statsmodels`, and
`scipy` (transitively) are all already declared in `dashboard/requirements.txt`.

## Out of scope (deferred)

- Model-comparison panel (OLS vs rolling vs Kalman AIC/BIC/RMSE) — considered,
  not included this pass.
- One-step-ahead forward forecast of next month's excess return.
- Regime-conditional analysis using `regime_index.csv` /
  `regime_conditional_results.csv` from the branch.
- The rolling-window OLS "middle ground" comparison (notebook cells 26–30);
  the dashboard already has its own rolling-regression view.

## Testing & verification

- `tests/test_kalman_tvp.py` covers `fit_tvp` deterministically.
- Manual verification: launch the app
  (`uv run streamlit run dashboard/app.py`), open Factor exposures, select the
  Fama-French factor preset, switch the model to **Kalman (TVP)**, and confirm
  alpha/beta paths render with bands, the filtered/smoothed toggle changes the
  lines, events annotate inside-range, and diagnostics populate. Then select a
  macro-PC preset and confirm the alpha panels hide with the guard note while
  betas still render.
