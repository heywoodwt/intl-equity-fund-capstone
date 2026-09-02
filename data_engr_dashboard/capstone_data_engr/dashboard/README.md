# International Equity Fund dashboard

Interactive Streamlit app for the International Equity Fund
capstone — a quant-grade tear sheet. A sidebar **benchmark** selector
(default EFA, the iShares MSCI EAFE proxy; SCZ, VSS, and the equal-weight blend
are alternatives) and **period** slider drive the front tabs.

- **📊 Overview** — headline KPIs (CAGR, vol, Sharpe, Sortino, max DD, Calmar,
  alpha/beta, tracking error, info ratio, up/down capture), growth-of-$1 with an
  underwater drawdown panel, trailing- and calendar-year return tables, and a
  monthly-returns heatmap.
- **🛡️ Risk & drawdown** — full risk table (downside dev, skew/kurtosis,
  VaR/CVaR, hit rate), the worst-drawdowns episode table, a return histogram with
  VaR markers, up/down capture, and rolling volatility & Sharpe.
- **🎯 Benchmark & active** — cumulative active return, rolling beta / tracking
  error / information ratio / correlation, up/down capture, and a fund-vs-benchmark
  regression scatter.
- **📈 Factor exposures** — the rolling factor model (target, regressors, window);
  defaults to a clean Fama-French OLS read with the model/penalty/CV knobs behind
  an **Advanced** expander, plus a contribution decomposition.
- **🧭 Positioning (2018–2022)** — sector allocation vs the blended benchmark with
  active tilts, geographic allocation, value-weighted style/quality/growth
  characteristics, concentration, sector weights over time, and a comprehensive
  **performance-attribution** section (cumulative sector & stock contribution-to-return,
  interactive Brinson decomposition, full-period sector/country attribution plots,
  and dynamic monthly sector & country attribution plots from `brinson_fachler/`
  synchronized to the As-of month slider).
- **🔎 Data explorer** — distributions, returns over time, correlations.
- **🌐 Macro & regime** — fund performance conditioned on macro regimes (VIX,
  rates, curve, credit, inflation, momentum terciles), then an interactive
  principal-component explorer and a dictionary of every rolling-regression variable.

Performance/risk math lives in `analytics.py` (pure functions taking column
names — `ret_col`, `bench_col`, `rf_col` — never a hard-coded `fund_ret`, so an
uploaded fund flows through once coerced to the same schema).

Each PC is given an interpretive **name** (from its loadings) that appears
everywhere it's used, e.g. `Rates & inflation regime (PC1)`, `Size premium
(SMB) (PC7)`. Names/explanations live in `data.PC_INFO`.

This folder is **self-contained** — it reads only the finished CSVs in
`data/processed/` and imports nothing from the pipeline package or `.env`. That
keeps it liftable into a standalone deploy repo.

## Run locally (this repo)

```bash
uv sync --group dashboard                 # install dashboard deps
uv run streamlit run dashboard/app.py
```

Opens at http://localhost:8501.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (Overview, Risk & drawdown, Factor exposures, Data explorer, Glossary). |
| `analytics.py` | Pure performance/risk/benchmark-relative metrics — returns, vol, Sharpe/Sortino/Calmar, drawdowns, VaR/CVaR, beta/alpha, tracking error, info ratio, capture, trailing/calendar/rolling. No Streamlit; reusable from notebooks. |
| `rolling_regression.py` | Pure analysis functions — rolling/full-sample OLS, Ridge, ElasticNet, time-series-CV penalty selection. No Streamlit; reusable from notebooks. |
| `data.py` | CSV loading + column metadata (groups, labels, PC names/descriptions, glossary). |
| `requirements.txt` | Deploy-time pins (Streamlit Cloud reads this). |
| `.streamlit/config.toml` | Theme + server settings. |

## Reuse the analysis from a notebook

```python
import sys; sys.path.insert(0, "dashboard")
import data as D, rolling_regression as R

df = D.load_combined_with_pcs()
roll = R.rolling_fit(df, "excess_ret", D.FF_FACTORS, window=24, model="OLS")
fit  = R.full_sample_fit(df, "excess_ret", D.FF_FACTORS + D.MACRO,
                         model="Ridge", alpha=1.0)
```

## Data resolution

`data.py` looks for the CSVs in this order (first hit wins):

1. `$CAPSTONE_DATA_DIR`
2. `dashboard/data/` (bundled copy)
3. `../data/processed/` (this repo)

## Deploying (later)

The app needs the CSVs at runtime, which aren't in this folder by default. To
deploy `dashboard/` on its own (e.g. Streamlit Community Cloud):

1. Copy the required CSVs into `dashboard/data/`:
   `combined_monthly.csv`, `pca_market_components.csv`, `pca_market_loadings.csv`.
2. Point the deploy at `dashboard/app.py`; it installs from `requirements.txt`
   and finds the bundled `dashboard/data/`.

(These are the only inputs — no API keys or WRDS access required to *run* the app.)

## Notes / caveats

- **Targets:** `excess_ret` = `fund_ret − RF` (intercept = alpha);
  `alpha_vs_avg` / `alpha_vs_efa` are active returns vs the benchmarks.
- **Missing rows are dropped before rolling**, so a window counts observations,
  not the calendar. Factors/macro are complete; fundamentals & sector/country
  weights only span **2018-04 – 2022-12**, so selecting them restricts the run
  to that block.
- **Regularized coefficients are standardized within each window** (no
  look-ahead) and reported in per-within-window-SD units — comparable across
  regressors but not directly to OLS betas.
- Use **raw factors/macro for interpretation** and **PCs** (orthogonal) for
  stable penalized coefficients; map PCs back via `pca_market_loadings.csv`.

## Interpreting the coefficients (what drove returns)

Raw coefficients across variables on different scales (e.g. `vix` level ≈ 18 vs a
factor return ≈ 0.02) are **not comparable** — a tiny raw β can dominate. Handling:

- **Coefficient scale toggle (OLS):** *Standardized (Δret per 1σ)* = β × SD(x) —
  the change in monthly return per 1-SD move in the regressor, comparable across
  variables; this is the default for reading "what mattered." *Raw β* = native
  exposure/sensitivity. Standardizing only rescales slopes — **R², t-stats and
  alpha are unchanged** — so you lose nothing by reading the standardized view.
- **Don't standardize y, and keep alpha from the raw intercept.** Centering the
  regressors moves the intercept from alpha (return at factors = 0) to the mean
  return. The app always fits with `y` in return units and, for penalized models,
  **de-centers the intercept back to a true alpha**.
- **Alpha is the intercept, not the target — and it depends on the regressors.**
  `excess_ret = alpha + Σ βⱼ·factorⱼ + residual`. Alpha is the return *left
  unexplained* by the chosen factors, so it changes as you add/remove regressors
  (CAPM alpha ≠ Fama-French alpha ≠ macro-model alpha). What doesn't change is the
  dependent variable (the fund's return). This is by design — alpha is always
  "alpha *relative to* a specified factor model."
- **Alpha is only shown when the intercept is in-support.** The intercept = "return
  when every regressor = 0", which is a stable, meaningful alpha only when the
  regressors sit near 0 *both globally and within each rolling window*. The app
  (`alpha_interpretable`) requires each regressor to be **centered** (|mean| <
  0.5·sd) **and non-persistent** (|lag-1 autocorr| < 0.4):
  - **Factor returns** (Mkt_RF…Mom) pass → alpha shown.
  - **Level regressors** (VIX, yields, spreads) are off-zero → hidden.
  - **Macro-derived PCs** are centered globally but highly autocorrelated (PC1 ≈
    0.98), so within a 24-month window they drift > 1 sd from 0 — the intercept
    then *extrapolates* and explodes (e.g. a 21%/mo intercept → 942% annualized).
    These are hidden too; read the coefficients and contribution decomposition
    instead, and use factor returns if you want a clean alpha.
- **Full-sample table** shows all three side by side: `coef` (raw / per-1σ),
  `coef_per_sd` (Δret per 1σ — the comparable one), `beta_weight` (fully
  standardized, dimensionless).
- **Contribution decomposition** (the most defensible attribution):
  - *Variance share* = β · cov(x, y) / var(y); regressor shares **sum to R²** and
    are **scale-invariant** (standardization irrelevant). Read this first.
  - *Mean contribution* = β · mean(x); regressors + alpha **sum to mean return**.
    Intuitive but **level-sensitive** for non-return regressors (the split with
    alpha shifts if you re-center), so trust it for the factor-return regressors.
- **Caveat — standardization fixes scale, not collinearity.** Correlated macro
  variables (`y10`/`y2`/slopes/spreads) still yield unstable individual
  coefficients; for clean attribution prefer the orthogonal **PCs**.

### Numerical guards (why some selections are dropped or blocked)

Naively regressing every variable produces garbage; the app prevents the common
failure modes (`screen_regressors`, plus guards in `rolling_fit`):

- **Degenerate columns dropped.** Zero-variance columns (the trailing PCs `PC22`/
  `PC23` are ≈ 0 — excluded from the menu) and **exact linear combinations**
  (`slope_10y_2y = y10 − y2`, `slope_10y_3m = y10 − y3m`) make the design matrix
  singular. They're screened out (earlier-listed columns kept) and reported.
- **Rolling-OLS degrees of freedom.** OLS needs ≥ 2 observations per parameter per
  window, else it interpolates noise (this is what produced the 200-million-%
  alpha and R²→0). Below that the app blocks with a message → enlarge the window,
  drop regressors, or use **Ridge** (which is regularized and stable).
- **Collinearity warning.** If the surviving regressors are still highly collinear
  (condition number > 30, e.g. the macro block at ≈ 246), OLS is flagged as
  unstable and you're pointed to Ridge/PCs. Ridge standardizes within-window, so
  it stays well-behaved where OLS does not.

## ETF price/returns (`etf_prices.py`)

Loads daily and monthly total-return series for any ticker from Yahoo Finance.

- `load_etf(ticker)` -> `EtfPrices(meta, close_daily, ret_daily, ret_monthly, as_of, is_stale)`
- `validate_etf(ticker)` -> `EtfMeta`; raises `NotAnEtf` for anything whose
  `quoteType` is not `ETF`. Call this on user input only — `load_etf` deliberately
  skips the gate so index/non-ETF series (`^GSPC`, FUND) still load.

Cached on disk (prices 1 day, metadata 30 days) under `$CAPSTONE_ETF_CACHE_DIR`,
else `data/interim/etf_prices/`, else `dashboard/.cache/etf_prices/`. When Yahoo is
unreachable but a cached copy exists, the cached data is returned with
`is_stale=True` — surface that in the UI rather than presenting it as current.

## ETF factor analytics (`etf_analytics.py`)

Builds the monthly frame the Factor-exposures and Macro-&-regime tabs analyze,
for any ticker.

- `load_analytics_frame(ticker)` -> monthly frame with the ticker's returns in
  `fund_ret`. **FUND** short-circuits to the existing `combined_monthly` frame
  (123 months, holdings regressors intact); any other ticker is inner-joined onto
  the market frame.
- `load_market_frame()` -> the fund-independent join of Fama-French factors, macro,
  PCA components, and benchmark ETF returns (414 months, 1990-07 – 2024-12). A test
  guards that it reproduces `combined_monthly.csv` exactly on the overlap.
- `regressor_catalog(frame)` -> `data.regressor_catalog` with empty families dropped
  and, for ETF frames, the holdings-based preset swapped for `ETF_ALPHA_PRESET`.
- `price_status(ticker)` -> `(is_stale, as_of)`, so the UI can flag cached prices
  rather than presenting them as current.

**Coverage is deliberately ragged.** Factors reach back to 1990-07, macro to 2000-01,
benchmark ETFs to 2014-09, and the PCs only span 2014-10 – 2024-12. Everything is
clipped to the December-2024 data freeze (`data.LAST_MONTH`). The engines drop
incomplete rows, so selecting a PC regressor on a long-history ETF can quietly shrink
the fitted sample — a PC-bearing OLS fits on at most the 123 months the PCs cover. That
is why both tabs caption the effective window.

## Fund rating (`fund_rating.py`)

Module retained and tested, but no longer surfaced as a dashboard tab.

0–5 stars from risk-adjusted absolute performance plus holdings diversification.

- `rate(monthly_ret, rf, daily_close, div_inputs)` -> `Rating(stars, final,
  performance, diversification, windows, …)`. `stars is None` means **not rated**
  (under 12 months of history) — which is not the same as zero stars.
- Pure: no I/O, no network, no Streamlit. Inputs come from
  `etf_analytics.rating_returns()` and `etf_analytics.diversification_inputs()`.

**Scoring** (absolute thresholds, not peer-relative). Windows blend
0.20 · 1y + 0.40 · 3y + 0.40 · 5y, with missing windows redistributing their
weight. Within a window: Sharpe 55% (0 at ≤0.0, 100 at ≥1.5), annualized return
20% (0 at ≤0%, 100 at ≥15%), max drawdown 25% (0 at ≤−45%, 100 at ≥−5%).
Volatility is shown but not scored. Diversification is top-10 concentration 50%
(0 at ≥60%, 100 at ≤20%) plus normalized sector entropy 50% (0 at ≤0.50, 100 at
≥0.90). `final = 0.75·performance + 0.25·diversification`, or performance alone
when holdings are unavailable.

**Three deliberate choices worth knowing:**

- **Returns come from yfinance for every ticker, FUND included**, so all stars
  use identical methodology and the drawdown gets a real daily price path
  (`combined_monthly.csv` is monthly only). All price series are clipped to the
  December-2024 data freeze (`etf_prices.LAST_DATE`), so the rating never sees
  post-2024 prices.
- **Drawdown uses daily prices, Sharpe and return use monthly.** Monthly returns
  hide intra-month troughs and would systematically flatter a fund that crashed
  and recovered inside a month.
- **Sharpe is NaN'd below a volatility floor** (`_MIN_STD`). `analytics.sharpe`
  only guards `sd > 0`, so a constant series slips through with sd ≈ 1e-18 and
  yields a Sharpe of ≈1e16 — which would clamp to a perfect 100 on the
  heaviest-weighted metric. A NaN metric drops out and its weight redistributes.

No ETF has holdings data while the iShares endpoint is gated, so every ETF is
currently rated on performance only. FUND's diversification comes from the
engineered layer and is a 2022-12 snapshot — the tab labels the as-of date.
