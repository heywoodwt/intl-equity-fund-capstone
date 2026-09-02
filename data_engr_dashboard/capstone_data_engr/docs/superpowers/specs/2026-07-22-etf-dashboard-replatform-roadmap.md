# ETF Dashboard Re-platform — Roadmap / Decomposition

**Date:** 2026-07-22
**Status:** Decomposition approved. Sub-projects specced/built individually.

## Vision

Re-platform the existing single-fund Streamlit dashboard ("International Equity Mutual
Fund") so it operates on **arbitrary ETFs**, driven by **full issuer-published holdings**,
and surfaces a **0–5 star performance rating** per ETF. Every analytical tab (Overview,
Risk, Benchmark, Factor exposures, Positioning, Attribution, Data explorer, Macro/regime)
should run on a user-selected ETF.

## Why ETFs

ETFs publish their **complete holdings daily** via the issuer (iShares, Vanguard, State
Street, …). That transparency is what makes the holdings-driven views and the
diversification component of the rating possible. Mutual funds disclose holdings only
quarterly with lag. (The originally-referenced api-ninjas Mutual Fund API is an
identity-only lookup — no performance data — so it is not used.)

## Data sources

- **Prices / returns:** yfinance (free, keyless, already in `capstone_data/yahoo.py`).
- **Holdings:** issuer daily CSVs (full constituent lists). **Not** yfinance top-10.
- **Factors / macro / risk-free:** existing repo pipeline (Ken French factors, FRED).

## Key constraint (surfaced up front)

Issuer CSVs give **current** holdings and daily snapshots **going forward** — not deep
history. Any view needing *point-in-time historical* holdings (notably the **Attribution**
tab's time-series sector/security contribution) starts from "today forward," not
backfilled. Historical attribution is out of scope until a point-in-time holdings archive
accumulates.

## Sub-projects

Each gets its own `spec → plan → implementation` cycle. Build order: **1 & 2 → 3 & 4 → 5**.

### 1. ETF holdings ingestion  *(foundational — spec first)*
Pipeline job in `capstone_data/` emitting tidy holdings tables to `data/processed/`.
Ticker→issuer resolution; download & parse full issuer CSVs; normalize to
`(etf_ticker, as_of_date, constituent, weight, sector, …)`. Feeds #4's diversification
and the Positioning/Attribution tabs.

### 2. ETF price/returns + universe layer
yfinance ETF history → returns series; a registry of supported ETFs. Feeds every
performance and factor computation.

### 3. ETF factor analytics recompute
Wire ETF returns into the existing factor-exposure / rolling-regression / Kalman-TVP /
regime machinery so those tabs run per selected ETF. Depends on #2 + existing factors.

### 4. 5-star rating
See "Rating design (sub-project #4)" below. Depends on #1 + #2.

### 5. Dashboard re-platform
Global ETF selector; re-point Overview / Risk / Benchmark / Positioning / Attribution /
Data-explorer tabs to the selected ETF; integrate the rating tab. Depends on #1–#4.

---

## Rating design (sub-project #4) — preserved from earlier brainstorm

Locked decisions to carry into #4's own spec:

- **Style:** risk-adjusted, **absolute** thresholds (no peer/benchmark ranking).
- **Final blend:** `final = 0.75·performance_composite + 0.25·diversification_composite`.

### Performance composite (from yfinance prices, #2)
Windows blended **0.20·1yr + 0.40·3yr + 0.40·5yr**; missing-window weight redistributed;
under ~1yr history → not rated. Per window, three metrics scored 0–100 (piecewise-linear,
clamped):

| Metric | Weight | 0 pts | 100 pts |
|---|---|---|---|
| Sharpe ratio (primary risk-adjusted) | 55% | ≤ 0.0 | ≥ 1.5 |
| Annualized return | 20% | ≤ 0% | ≥ 15% |
| Max drawdown (tail risk) | 25% | ≤ −45% | ≥ −5% |

Volatility is displayed but not weighted (Sharpe already prices it in). Risk-free rate is
a parameter (FRED 3-mo T-bill default, config-constant fallback offline).

### Diversification composite (from full issuer holdings, #1)

| Metric | Weight | 0 pts | 100 pts |
|---|---|---|---|
| Top-10 concentration (Σ top-10 weights; lower = better) | 50% | ≥ 60% | ≤ 20% |
| Sector spread (normalized Shannon entropy of sector weights; even = better) | 50% | ≤ 0.50 | ≥ 0.90 |

With full issuer holdings, concentration/entropy use the **complete** constituent list
(and an exact holdings count becomes available for display). If holdings are unavailable
for a ticker, the diversification weight is redistributed to performance and the tab notes
"rated on performance only."

### Star mapping
`stars = clamp( round(final / 100 * 5 * 2) / 2, 0.5, 5.0 )` — linear, half-star, floor 0.5.

### Structure / testing
Pure, testable scoring in `dashboard/fund_rating.py` (no network in the scoring path);
network touches (`fetch_history`, holdings load) cached in the UI layer. Deterministic
no-network unit tests over synthetic series, per repo convention.