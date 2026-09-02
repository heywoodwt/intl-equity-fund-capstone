# Plan 1 — Baseline rolling regression

**Goal:** explain how factors / macro regimes drove the fund's performance over time, with
interpretable, regime-aware regressions. Two stages: (a) classic rolling linear factor model,
(b) regularized non-linear extension (the advisor's explicit ask). Output: coefficient time-series,
plots, and a short findings note.

**Data:** `../../ds6015/data/engineered_data/combined_monthly.csv` (or `../data/processed/`).
129 months, 2014-10 → 2025-06. See its README for columns/units.

## Targets (decide up front; do both is fine)
- **Factor-model framing:** dependent = fund **excess return** `fund_ret - RF`; regress on factors;
  the intercept = alpha. Classic, directly interpretable.
- **Active-return framing:** dependent = `alpha_vs_avg` (fund − mean(EFA,SCZ,VSS)) or `alpha_vs_efa`;
  regress on factors/macro to see what drove *active* performance vs the benchmark.

## Stage A — rolling linear factor model
- Regressors: the Developed-ex-US FF factors `Mkt_RF, SMB, HML, RMW, CMA, Mom` (decimal).
- Rolling OLS, window 24 and 36 months (compare). For each window end: betas, alpha (annualize:
  `(1+a)^12 - 1`), R². This repeats `data/python_files.ipynb`'s `RollingOLS` idea **but with the
  corrected ex-US factors** — expect materially different betas/alpha than the old (US-contaminated)
  run; that contrast is itself a finding.
- Deliverables: plot each beta over time (regime shifts), rolling annualized alpha, rolling R².
  Interpret: which exposures explain returns, how they move across 2018 selloff / 2020 COVID /
  2022 drawdown.

## Stage B — add macro regime + non-linear basis + regularization
- **Why regularize:** 129 months vs many correlated regressors (factors + 17 macro). Plain OLS
  overfits. Options, in order of preference:
  1. **PCs as regressors** — use `pca_market_components.csv` (joined on `month`); the PCs are
     orthogonal, so coefficients are stable. Map findings back via `pca_market_loadings.csv`.
  2. **Ridge / ElasticNet** on standardized factors+macro (z-score first — units differ!). Tune the
     penalty by time-series CV.
- **Non-linear basis:** add polynomial/interaction terms (e.g., factor×regime, squared terms) or
  splines on key factors, then regularize. Goal is to capture state-dependent betas (e.g., market
  beta higher in high-VIX months) while staying tractable/interpretable.
- Rolling version: rolling Ridge/ElasticNet; plot coefficient paths. Compare linear vs non-linear
  fit (R², residuals) and report where non-linearity matters.

## Stage C — fundamentals-augmented (2018-2022 sub-window)
On the 57 holdings-months, regress `alpha_vs_avg` on the portfolio fundamentals (`*_whmean`/`*_wmean`)
and sector/country tilts (`sect_wt_*`, `ctry_wt_*`) to see which portfolio *characteristics* tracked
alpha. Small N → keep it tiny / regularized / descriptive; treat as exploratory.

## Practical notes
- **Standardize** features before Ridge/ElasticNet/non-linear (decimal factors vs percent macro).
- **No look-ahead:** features are already point-in-time; for predictive checks use forward returns.
- Don't discard the raw features for the PCs — use raw for interpretation, PCs for stability.
- Home: `dashboard/rolling_regression.py` (pure analysis fns — rolling/full-sample OLS, Ridge,
  ElasticNet, time-series-CV penalty selection); driven interactively by `dashboard/app.py`
  (Streamlit) and importable from a notebook (`sys.path.insert(0, "dashboard"); import
  rolling_regression`). Figures → group repo. See `dashboard/README.md`.

## Done when
Rolling betas + alpha + R² plotted and interpreted (linear); regularized non-linear version run and
compared; a short written findings note on what drove performance across regimes. Feeds Plan 2.
