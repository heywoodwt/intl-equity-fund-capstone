# Rolling-Window OLS for the Kalman Factor Models — Design

**Date:** 2026-07-01
**Status:** Approved

## Motivation

The two Kalman notebooks each contrast two extremes of factor attribution for
FUND excess returns:

- **Static full-sample OLS** — constant alpha and betas over 2014-10 → 2025-06.
- **Kalman TVP** — alpha/betas drift every month as a random walk (state-space).

A **rolling-window OLS** sits between them: betas are re-estimated on a trailing
fixed-length window, giving time-varying exposures without state-space machinery.
It becomes a third comparison point in every existing comparison — beta-path
plots, the parameter summary table, one-step-ahead diagnostics, and model
comparison.

Data: `data/engineered_data/combined_monthly.csv`, 129 monthly observations,
`excess_ret = fund_ret - RF` (all decimal). Factors are the Developed-ex-US
Fama-French set `Mkt_RF, SMB, HML, Mom`.

## Scope

- **Windows:** both 24-month and 36-month.
- **Models:**
  - Model 1 (4-factor) in `kalman/tvp_kalman.ipynb`.
  - Model 3a (5-factor: 4 FF + `regime_index`) in `kalman/h_kalman.ipynb`.
- **Structure:** a reusable module `kalman/rolling_regression.py` plus comparison
  cells added to both notebooks. Mirrors the existing `kalman/tvp_kalman_filter.py`
  reusable-module pattern.

Out of scope: any change to the existing Kalman fits, the regime-index
construction, or the engineered data layer.

## Module — `kalman/rolling_regression.py`

Built on `statsmodels.regression.rolling.RollingOLS` (canonical, matches the
notebooks' existing `sm.OLS` usage). Three public functions:

### `rolling_ols(df_model, factor_cols, window, y_col="excess_ret")`
Rolling-window OLS of `y_col` on `[const] + factor_cols`.

- Returns a DataFrame indexed by the window **end-date** with columns:
  `alpha`, one column per factor beta, `<name>_se` for each coefficient,
  `r2`, `nobs`.
- The coefficient row at date *t* is estimated on the `window` months **ending
  at** *t* (contemporaneous / in-sample). This is the fair comparison to the
  Kalman **smoothed** betas.
- Rows before a full window are `NaN` (RollingOLS `min_nobs = window`); not
  back-filled — consistent with the repo's point-in-time rule.

### `rolling_one_step_pred(df_model, factor_cols, window, y_col="excess_ret")`
True out-of-sample one-step-ahead prediction: for each *t*, fit OLS on months
`[t-window, t-1]` and predict month *t*.

- Returns a DataFrame indexed by date with columns `fitted`, `residual`,
  `excess_ret`.
- This is the fair analogue of the Kalman **filtered** one-step-ahead
  predictions used in the existing diagnostics cells (no look-ahead).

### `rolling_summary(rolling_df, ols_params, factor_cols)`
Parameter-comparison table matching the shape of the existing Cell 20 table.

- Returns a DataFrame indexed by `["alpha"] + factor_cols` with columns
  `OLS_estimate, Rolling_mean, Rolling_std, Rolling_min, Rolling_max`.
- `ols_params` is the fitted static-OLS parameter vector (statsmodels order:
  const first).

## Notebook additions

A new section (`## Rolling-Window OLS`) is appended after the existing Kalman
sections in each notebook. Cells:

1. Import from `rolling_regression`; fit `rolling_ols` for window ∈ {24, 36}.
2. Plot rolling beta paths, overlaying the static-OLS horizontal line and the
   Kalman-smoothed path, reusing the existing figure style.
3. Print the `rolling_summary` table (OLS vs rolling-24 vs rolling-36).
4. One-step-ahead diagnostics: RMSE and R² of Static OLS vs Rolling-24 vs
   Rolling-36 vs Kalman, extending the existing model-comparison cell.
5. Export results CSV.

### Model 3a specifics (`h_kalman.ipynb`)
- `factor_cols = AUGMENTED_FACTORS` (`Mkt_RF, SMB, HML, Mom, regime_index`).
- Reuses the in-notebook `regime_index` variable produced upstream; no
  recomputation.

## Outputs

- `kalman/rolling_regression_results.csv` — Model 1 rolling betas + SE +
  fitted/residual for both windows.
- `kalman/rolling_regime_results.csv` — Model 3a equivalent.

CSV style follows the existing `tvp_kalman_results.csv` /
`regime_conditional_results.csv` exports (per-date rows, `_se` columns,
`fitted`, `residual`). Both windows are distinguished by a `window` column
(long format) so each file holds the 24- and 36-month paths.

## Correctness check

The repo has no test harness. The module carries a self-check under
`if __name__ == "__main__"`:

- Synthetic data generated from **known constant coefficients** plus small
  noise; assert `rolling_ols` recovers each beta within tolerance.
- Assert the first `window-1` rows are `NaN` and later rows are populated
  (edge/NaN behavior).
- Assert `rolling_one_step_pred` produces no NaN after its warm-up and that its
  residuals are larger than the in-sample fit (out-of-sample sanity).

Run once (`python kalman/rolling_regression.py`) to verify before wiring the
notebooks.

## Edge cases / notes

- 129 obs → rolling-24 ≈ 106 estimates, rolling-36 ≈ 94.
- Units already decimal; no unit conversion needed.
- No changes to existing cells' outputs; additions are purely appended sections.