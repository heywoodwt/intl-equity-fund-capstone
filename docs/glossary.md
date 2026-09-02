# Glossary

Minimal finance/ML terms used across this repo. For full treatment of the engineered features see
`data/engineered_data/README.md`.

- **The fund** — an anonymized developed-markets **ex-US** small/mid-cap equity
  fund. Our subject. Referred to as `FUND` where code needs a ticker placeholder.
- **ex-US** — excludes United States securities. Critical: the market factor must also be ex-US, which
  is why `data/ff_factors.csv` (Developed *incl.* US) is wrong here.
- **Benchmarks** — **EFA** (MSCI EAFE, developed large-cap), **SCZ** (MSCI EAFE small-cap), **VSS**
  (FTSE ex-US small-cap). The fund's performance is judged relative to these.
- **Return** — fractional price change over a period (decimal in engineered data; percent strings in
  `2014_2025_*_Monthly.csv`).
- **Alpha** — return not explained by factor exposure; the regression **intercept**. Proxy for
  manager skill. `alpha_vs_avg` / `alpha_vs_efa` in `combined_monthly.csv` are vs. benchmark, not
  factor-model alpha.
- **Beta** — sensitivity (regression slope) of the fund to a factor (e.g. the market).
- **Fama-French factors** — `Mkt_RF` (market minus risk-free), `SMB` (size), `HML` (value), `RMW`
  (profitability), `CMA` (investment), plus `Mom`/`WML` (momentum) and `RF` (risk-free rate).
  Regressors for explaining returns.
- **Point-in-time / look-ahead** — a value is point-in-time if it only uses information available on
  that date. Avoiding look-ahead (e.g. lagging filings) is required for honest backtests.
- **Survivorship bias** — dropping delisted/acquired names inflates results. The engineered returns
  deliberately retain them (e.g. Wirecard, GW Pharma).
- **Harmonic mean (multiples)** — correct way to aggregate `price/X` ratios across a portfolio
  (= aggregate price / aggregate fundamental); arithmetic mean overstates. Used for the `_whmean`
  columns.
- **Winsorize** — clip extreme values (here cross-sectionally at [5%, 95%]) before aggregating.
- **PCA components** — standardized, orthogonal linear combinations of features; additive,
  full-rank (no info lost). Join on `month`.
- **Grad-CAM** — gradient-based saliency that attributes a CNN's prediction back to inputs; used in
  `cnnV1.ipynb` to score which trades drove the fund.
- **next_ret** — the prediction **label**: next month's return for a stock (`combined_panel.csv`,
  `characteristics_panel.csv`).
