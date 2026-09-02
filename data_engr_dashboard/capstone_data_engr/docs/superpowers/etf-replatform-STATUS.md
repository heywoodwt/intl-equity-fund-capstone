# ETF Dashboard Re-platform — STATUS / Resume Here

**Last updated:** 2026-07-30
**Branch:** `heywood_s_version`

This is the living handoff doc for the multi-part effort to re-platform the dashboard
around ETFs and add a 5-star rating. Read this first to resume. Detailed design lives in
the specs/plans linked below.

## The big picture

Original ask: "develop a 5-star rating for whether a mutual fund's performance is good,"
using the api-ninjas mutual-fund API. Two findings reshaped it:

1. The **api-ninjas Mutual Fund API is identity-only** (ticker, name, ISIN, CUSIP, country,
   price, AUM) — no returns/risk — so it cannot drive a performance rating. Data source
   switched to **yfinance** (free, keyless, already in the repo) for prices.
2. The user chose to target **ETFs, not mutual funds**, because ETFs publish full holdings,
   and to **re-platform the whole dashboard** around arbitrary ETFs (every tab), plus a
   holdings-driven diversification dimension in the rating.

Because that scope spans several subsystems, it was **decomposed into 5 sub-projects**,
each with its own spec → plan → build cycle. See the roadmap for the full rationale.

## Key documents

- **Roadmap / decomposition:** `docs/superpowers/specs/2026-07-22-etf-dashboard-replatform-roadmap.md`
  (5 sub-projects, build order, the preserved rating design, the point-in-time caveat)
- **Sub-project #1 spec:** `docs/superpowers/specs/2026-07-22-etf-holdings-ingestion-design.md`
- **Sub-project #1 plan:** `docs/superpowers/plans/2026-07-22-etf-holdings-ingestion.md`
- Original rating brainstorm is folded into the roadmap under "Rating design (sub-project #4)".

## Sub-projects & status

Build order: **1 & 2 → 3 & 4 → 5**.

| # | Sub-project | Status |
|---|---|---|
| 1 | ETF holdings ingestion (iShares → `data/processed/etf_holdings.csv`) | ✅ **DONE** (see below) |
| 2 | ETF price/returns + universe layer (yfinance → returns; supported-ETF registry) | ✅ **DONE** |
| 3 | ETF factor analytics recompute (wire ETF returns into factor/rolling/Kalman/regime) | ✅ **DONE** |
| 4 | 5-star rating (perf composite from #2 + diversification from #1 → stars) | ✅ **DONE** |
| 5 | Dashboard re-platform (global ETF selector; re-point every tab) | ⬜ not started |

### ⚠️ Two open data issues (surfaced while building #4)

1. **`combined_monthly.csv`'s FUND returns are corrupted.** `fund_ret` is exactly
   `0.0375` for all six months **2025-01 → 2025-06** — a placeholder, not returns
   (benchmarks vary normally over the same months; yfinance shows ≈0.000). It
   inflates FUND's trailing-1y performance score to a bogus 100. The **rating
   routes around it** by sourcing returns from yfinance, but the **factor/regime
   tabs still use it**. Repairing this in the `capstone_data` pipeline is
   recommended follow-up work.
2. **No ETF holdings data exists.** `data/processed/etf_holdings.csv` has never
   been produced because the iShares endpoint is gated (see #1's gotcha below),
   so every ETF is rated on **performance only**. The diversification path is
   built and tested and lights up the moment that CSV appears — or use #1's
   documented `--from-files` workaround.

## Sub-project #1 — DONE (details)

**What was built** (all committed on `heywood_s_version`):

- `capstone_data/config.py` — `ISHARES_ETFS` dict: EFA, IEFA, ACWX, SCZ, IEMG, URTH
  (ticker → iShares holdings-CSV URL). Extend by adding a line.
- `capstone_data/etf_holdings.py`:
  - `parse(raw, etf_ticker)` — pure. Skips iShares preamble, extracts `as_of_date`, finds
    the `Ticker,`-prefixed header, coerces `weight/market_value/shares` to float, keeps
    cash/derivative rows (blank ticker), filters to rows with a numeric weight (so trailing
    disclaimer/footer lines can't crash or pollute), returns the `COLUMNS` schema.
  - `build(etfs=None)` — network fetch+parse each ETF; per-ETF skip-on-failure; returns
    `(frame, failures)`.
  - `build_from_files(src_dir=None)` — parse manually-downloaded CSVs from a directory
    (default `data/raw/etf_holdings/`); ticker inferred from filename; returns
    `(frame, failures)`.
  - `fetch`, `_get` (requests + retry), `_to_float`, `_ticker_from_filename`, `write`.
- `scripts/build_etf_holdings.py` — entrypoint. Network mode **and** `--from-files DIR` mode.
- `tests/test_etf_holdings.py` — 12 deterministic no-network tests (parse happy path,
  as-of date, numeric coercion, cash row, missing-header raise, trailing-footer safety,
  build skip-on-failure, empty frame, write roundtrip, ticker inference, build_from_files).
- `README.md` — build-order line + the iShares-access note.

**Output schema** (`data/processed/etf_holdings.csv`, latest snapshot, one row per
(ETF, constituent)):
`etf_ticker, as_of_date, constituent_ticker, name, sector, asset_class, weight, market_value, shares`

**Verification:** full repo test suite = **36 passed**. `--from-files` exercised end-to-end
(local sample CSVs → correct tidy CSV, exit 0).

### ⚠️ Important operational gotcha — iShares access is gated

The default **network fetch returns zero rows**. iShares gates its holdings-CSV endpoint
behind a sign-on/terms interstitial (Akamai-style) and serves a 1.4 MB HTML product page
(with `content-type: text/csv`) to any plain HTTP client. Verified that **none** of these
bypass it: browser User-Agent, `Accept: text/csv`, `X-Requested-With: XMLHttpRequest`,
`Referer`, cookie-priming via the product page. The URL itself is correct (it's the exact
link embedded in iShares' own page). `parse()` correctly detects the missing header and
fails loudly per-ETF.

**Workaround (the supported path to get real data):** download each ETF's holdings CSV by
hand from its iShares product page into `data/raw/etf_holdings/` (filename starting with the
ticker, e.g. `EFA_holdings.csv`), then run:

```
uv run python scripts/build_etf_holdings.py --from-files data/raw/etf_holdings
```

`data/raw/` is git-ignored, so downloaded CSVs won't be committed. If the network gate is
ever solved (different network, a session/OAuth flow, or a non-gated mirror), the plain
`build()` path is ready as-is.

### Commit trail (sub-project #1)

```
11bdf0b feat(pipeline): add --from-files manual iShares holdings ingestion
f4dd03b docs: note etf_holdings build step
120daf0 feat(pipeline): add build_etf_holdings entrypoint
a462e3d feat(pipeline): write etf_holdings.csv to data/processed
dfe3f2d feat(pipeline): add build() with per-ETF skip-on-failure
edab131 fix(pipeline): stop iShares parse at end of holdings block
89a7d4d feat(pipeline): parse iShares holdings CSV into tidy schema
fe386fe feat(pipeline): add curated ISHARES_ETFS universe to config
```
Design/doc commits: `92839b7` (rating spec), `3a22926` (recast to roadmap),
`5fc6360` (#1 spec), `325b4bb` (#1 plan).

## How to resume (next session)

1. Read this file + the roadmap.
2. **Next sub-project: #5 (dashboard re-platform)** — the last one. It promotes the per-tab
   ticker selector built in #3/#4 (`app.py:ticker_selector`) to a global one and re-points
   the remaining five tabs (Overview, Risk, Benchmark, Positioning, Data explorer). Note
   that Positioning and Attribution are holdings-driven, so their ETF semantics depend on
   the holdings gate in the data-issues note above.
   Also worth scheduling: the `combined_monthly.csv` repair (data issue #1).
3. Run the brainstorming → writing-plans → subagent-driven-development flow for the chosen
   sub-project (same process used for #1).
4. Rating design decisions already locked (in roadmap, for #4 when you get there):
   - Final = `0.75·performance + 0.25·diversification`.
   - Performance: risk-adjusted absolute; windows 0.20·1y + 0.40·3y + 0.40·5y; per window
     Sharpe 55% / annualized return 20% / max drawdown 25% (volatility shown, not scored);
     risk-free rate is a parameter (FRED 3-mo default).
   - Diversification (from #1 holdings): top-10 concentration 50% (≤20%→100, ≥60%→0) +
     sector-entropy 50% (≥0.90→100, ≤0.50→0). Missing holdings → weight redistributes to
     performance.
   - Stars: `clamp(round(final/100*5*2)/2, 0.5, 5.0)` — linear half-star, floor 0.5.
   - Pure scoring in `dashboard/fund_rating.py` (no network in scoring path); UI in `app.py`.

## Repo conventions (for any sub-project)

- `uv run python ...` / `uv run pytest`. Tests are deterministic & no-network; put repo root
  on `sys.path` and `from capstone_data import <module>`.
- Source logic in `capstone_data/<module>.py`; one entrypoint per build step in `scripts/`;
  source definitions as dicts in `config.py`; processed tables (CSV) in `data/processed/`.
- Point-in-time / no look-ahead is a core principle (see README).
- **Commits: never add Claude/Anthropic/AI attribution.** Stage only the files you changed
  (the `.idea/` files in the working tree are unrelated — never `git add -A`).
