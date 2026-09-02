# ETF Factor Analytics Recompute — Design (sub-project #3)

**Date:** 2026-07-30
**Status:** Approved. Ready for implementation planning.
**Depends on:** #2 (`dashboard/etf_prices.py`) — DONE.
**Roadmap:** `docs/superpowers/specs/2026-07-22-etf-dashboard-replatform-roadmap.md`

## Goal

Run the existing factor-exposure, rolling-regression, Kalman-TVP, and regime machinery
against a **user-selected ETF** instead of only FUND, without changing the analytics
engines themselves.

## Why this is small

The engines are already ticker-agnostic. `analytics.py`, `rolling_regression.py`, and
`kalman_tvp.py` never reference `fund_ret`; they take a frame plus column names
(`ret_col` / `y_col` / `target`, `x_cols`). The fund-specific coupling lives in exactly
two places:

1. `data.py:load_combined()` — reads `combined_monthly.csv`, whose `fund_ret` *is* FUND.
2. `app.py` — the literal string `"fund_ret"` in ~20 call sites.

So the work is to **produce a frame** with the same shape whose `fund_ret` holds the
selected ticker's returns. Nothing in the engines changes.

## Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Target column | Keep the name `fund_ret`, swap its contents | Zero changes to the ~20 `app.py` call sites and zero risk to the engines. The name is cosmetic — `D.label()` already controls display. #5 will rewrite those lines anyway. |
| History window | Extend to each ETF's own history | Factors reach back to 1990-07 and macro to 2000-01. Clipping to the current 129-month grid would discard a decade of history for older ETFs and weaken rolling/Kalman fits. |
| FUND-only holdings columns | Hide for non-FUND; swap the preset | For an ETF, `ff_*`/`sect_wt_*`/`ctry_wt_*` are meaningless. Showing ~90 all-NaN columns produces confusing empty fits. |
| FUND | Pinned default, bypasses the ETF gate | FUND is the DS-6015 capstone subject and is a `MUTUALFUND`, which `validate()` rejects by design. It keeps its full catalog and current preset. |
| UI scope | Selector on the two analytics tabs only | #5 owns the global selector. Two tabs make #3 usable and demoable without pre-empting the ETF semantics of Positioning/Attribution, which are not designed yet. |

## Verified facts

These were checked against the real data before the design was settled. They are the
load-bearing assumptions; re-check them if the pipeline changes.

- **Units already agree.** Every return in `combined_monthly.csv` is a **decimal**
  (`fund_ret` mean ≈ 0.008/mo), and `etf_prices.monthly_returns()` emits decimals.
  No conversion is needed anywhere.
- **`ff_factors_dev_ex_us.csv` is decimal** and its `Mkt_RF`/`SMB`/`RF` match
  `combined_monthly` exactly on the overlap.
- **Macro passes through verbatim.** All 17 macro columns are identical between
  `macro_monthly.csv` and `combined_monthly.csv`.
  ⚠️ `sp500_ret` is in **percent** while `fund_ret`/`Mkt_RF` are decimal. This is a
  pre-existing quirk of the engineered layer, reproduced as-is — do not "fix" it here.
- **The identities are exact:** `bench_avg_ret` = mean(`efa_ret`, `scz_ret`, `vss_ret`),
  `alpha_vs_avg` = `fund_ret − bench_avg_ret`, `alpha_vs_efa` = `fund_ret − efa_ret`.
- **The market frame reproduces `combined_monthly`.** Joining the four component tables
  yields all **27** shared fund-independent columns with max abs difference `< 1e-9`
  over the 129-month overlap. This is what lets the ETF path and the FUND path agree
  by construction.

### Source coverage

| Table | Range | Months |
|---|---|---|
| `ff_factors_dev_ex_us.csv` | 1990-07 → 2026-04 | 430 |
| `macro_monthly.csv` | 2000-01 → 2026-05 | 317 |
| `benchmark_etf_returns_monthly.csv` | 2014-09 → 2026-06 | 142 |
| `pca_market_components.csv` | 2014-10 → 2025-06 | 129 |
| **Joined market frame** | **1990-07 → 2026-06** | **432** |

The PCA components are the narrowest band; see "Ragged coverage" below.

## Architecture

One new module, `dashboard/etf_analytics.py`, between `etf_prices.py` and the engines.
It imports nothing from `capstone_data` and reads only processed CSVs, so `dashboard/`
stays liftable into a deploy repo.

```
ff_factors_dev_ex_us.csv ─┐
macro_monthly.csv         ├─ outer-join on `month` ─→ market_frame (432 mo)
pca_market_components.csv │                                  │
benchmark_etf_returns.csv ┘                                  │
                                                             ▼
etf_prices.load_etf(t).ret_monthly ──→ `fund_ret` ──→ load_analytics_frame(t)
                                                             │
                                          ┌──────────────────┴──────────────────┐
                                    FUND │                                     │ any ETF
                                          ▼                                     ▼
                            data.load_combined_with_pcs()          market_frame ⋈ returns
                            (129 mo, full holdings catalog)        (extended, no holdings)
```

### Components

| Function | Responsibility |
|---|---|
| `load_market_frame()` | The four-way outer join on `month`, `lru_cache`d. Fund-independent; knows nothing about tickers. |
| `etf_monthly_returns(ticker)` | Adapter over `etf_prices.load_etf()`. Maps the month-end `DatetimeIndex` to the `YYYY-MM` `month` key; names the column `fund_ret`. |
| `load_analytics_frame(ticker)` | Orchestrator. FUND → the existing combined frame untouched. Otherwise → market frame joined to the ticker's returns, plus derived columns. |
| `regressor_catalog(frame)` | Wraps `data.regressor_catalog`, drops empty families, swaps the preset. |

### `load_analytics_frame(ticker)` contract

- **`ticker == "FUND"`** (case-insensitive): return `data.load_combined_with_pcs()`
  unchanged. Preserves the exact capstone numbers and the full holdings catalog.
- **Otherwise:** inner-join the market frame to the ticker's monthly returns on `month`
  (inner, so rows exist only where both sides do), then derive:
  - `excess_ret = fund_ret − RF`
  - `bench_avg_ret = mean(efa_ret, scz_ret, vss_ret)` — computed here, since
    `benchmark_etf_returns_monthly.csv` carries only the three legs. Use
    **`skipna=False`**: a month missing any leg must yield NaN, not an average of
    whichever legs happen to exist. Pandas' default (`skipna=True`) would silently
    produce a two-ETF "blended benchmark" for such months and break the verified
    identity. Before 2014-09 all three are NaN, so `bench_avg_ret` and
    `alpha_vs_avg` are NaN there — expected, and dropped by the engines.
  - `alpha_vs_avg = fund_ret − bench_avg_ret`
  - `alpha_vs_efa = fund_ret − efa_ret`
  - `date` — `month` parsed to a month-start `Timestamp`, matching `data.load_combined()`
- Holdings columns (`ff_*`, `sect_wt_*`, `ctry_wt_*`, `n_holdings`, `sector_hhi`,
  `top_sector`, …) are absent by construction on the ETF path. Nothing filters them out;
  they were never joined in.
- Raises whatever `etf_prices.load()` raises (`TickerNotFound`, `NoPriceHistory`,
  `EtfDataError`). The caller renders the message.

### Regressor catalog

`data.regressor_catalog(df)` already builds each group as
`[c for c in FAMILY if c in df.columns]`, and `fundamentals_cols`/`sector_cols`/
`country_cols` are `df`-driven. On an ETF frame those groups therefore come back **empty
on their own** — no filtering logic is needed. The wrapper only has to:

1. Drop groups whose list is empty.
2. Swap the preset. The starred `ALPHA_DRIVERS_PRESET` depends on
   `ff_earn_yld_median` and `ff_roce_median`, which no ETF frame has. Non-FUND frames
   get `ETF_ALPHA_PRESET = ["Mkt_RF", "SMB", "HML", "PC1", "PC2"]` — factor and macro
   axes only, five regressors, which keeps the rolling OLS estimable and the coefficient
   plot legible (the same reasoning behind the six-regressor FUND preset).

### Ragged coverage

Coverage is deliberately uneven, matching the existing convention (fundamentals are
already 2018-2022 only):

- An ETF with history before 2014-10 has **NaN PCs** there, and before 2000-01 has NaN
  macro. The engines' existing `.dropna()` — `rolling_regression._clean` and the
  `dropna()` inside `fit_tvp` — handles this correctly.
- **The hazard is silent sample shrinkage:** selecting PC regressors on a long-history
  ETF can quietly cut the fitted window from ~300 months to 129. The tabs therefore show
  a caption stating the **effective fitted window and row count** after screening. This
  is a required part of the UI, not a nicety.
- Too-short history: `fit_tvp` already raises `ValueError` with an actionable message
  ("Need more months … than states"), and `rolling_fit` has its own degrees-of-freedom
  guard. The tabs catch these and render the message instead of a traceback.

## UI

A ticker selector on **📈 Factor exposures** and **🌐 Macro & regime** only. The other
five tabs stay on FUND until #5. The widget is written so #5 can promote it to global
without rework.

- FUND is the pinned default and does **not** go through the ETF gate.
- Any other ticker goes through `etf_prices.validate_etf()`; `NotAnEtf` and
  `TickerNotFound` render via `st.error` with the exception message.
- `is_stale=True` renders a warning that cached data is being shown, with the `as_of`
  date — never presented as current.
- When the selected ticker is not FUND, the holdings families are absent from the
  regressor menu and a caption notes that holdings-derived regressors are FUND-only.

## Testing

Deterministic and no-network, per repo convention: tests put `dashboard/` on `sys.path`
and monkeypatch `etf_prices.load_etf` / `etf_prices.load`. Reading a local processed CSV
is still deterministic and network-free, so the regression guard below is allowed to do
it. Cases:

- **Regression guard:** the market frame reproduces `combined_monthly.csv`'s shared
  fund-independent columns (max abs diff `< 1e-9`). This is the test that stops the two
  paths silently diverging; it reads the real processed CSV and is skipped if absent.
- Market frame: expected month range, join produces one row per month, no duplicates.
- FUND path returns the combined frame — holdings columns present, row count 129.
- ETF path: no `ff_*`/`sect_wt_*`/`ctry_wt_*` columns.
- Derived identities: `excess_ret`, `bench_avg_ret`, `alpha_vs_avg`, `alpha_vs_efa`.
- `bench_avg_ret` is NaN when any benchmark leg is missing (the `skipna=False`
  requirement above), not an average of the surviving legs.
- Month-key alignment: a month-end `DatetimeIndex` from `ret_monthly` maps to the right
  `YYYY-MM` strings and joins without loss.
- Catalog: empty families dropped; ETF frames get `ETF_ALPHA_PRESET`, FUND keeps
  `ALPHA_DRIVERS_PRESET`.
- Ragged coverage: an ETF frame spanning pre-2014 months has NaN PCs there and the
  engines still fit on the reduced sample.

## Files

| File | Status | Responsibility |
|---|---|---|
| `dashboard/etf_analytics.py` | Create | Market frame, ETF returns adapter, `load_analytics_frame`, catalog wrapper |
| `tests/test_etf_analytics.py` | Create | Deterministic no-network tests |
| `dashboard/app.py` | Modify | Ticker selector; wire the two analytics tabs; error/stale handling |
| `dashboard/README.md` | Modify | Document the module |
| `docs/superpowers/etf-replatform-STATUS.md` | Modify | Mark sub-project #3 done |

## Out of scope

- The global ETF selector and the other five tabs (#5).
- The 5-star rating (#4).
- Refitting the PCA over a longer window — it would change what each PC means and
  invalidate the hand-written `PC_INFO` descriptions and the Glossary tab.
- Point-in-time historical holdings for arbitrary ETFs (roadmap constraint: issuer CSVs
  give current holdings and forward snapshots only).

## Repo conventions

- `uv run pytest`, `uv run python`.
- `dashboard/` must not import `capstone_data`; intra-dashboard imports are bare
  (`import etf_prices`, not `from dashboard import etf_prices`).
- Stage only files you changed — never `git add -A`; the `.idea/` entries stay unstaged.
- Never put AI/Claude/Anthropic attribution in a commit message.
- ⚠️ Local `uv` is 0.7.3, older than whatever wrote `uv.lock` (`revision = 3`). Running
  `uv` rewrites the lock down to `revision = 2`. Do not commit that churn; revert
  `uv.lock` if it appears in `git status`.
