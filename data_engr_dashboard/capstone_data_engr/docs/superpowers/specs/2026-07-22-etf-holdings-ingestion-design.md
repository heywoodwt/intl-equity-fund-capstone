# ETF Holdings Ingestion — Design (sub-project #1)

**Date:** 2026-07-22
**Status:** Approved, ready for planning
**Roadmap:** `2026-07-22-etf-dashboard-replatform-roadmap.md` (sub-project #1)

## Goal

Ingest **full published holdings** for a curated list of **iShares** ETFs and emit a
tidy, latest-snapshot holdings table to `data/processed/`. This is the foundational data
layer for the ETF re-platform: it feeds the diversification component of the 5-star rating
(sub-project #4) and the Positioning/Attribution dashboard views (sub-project #5).

Fits the repo contract: *external source → tidy table in `data/processed/`.*

## Scope decisions (locked)

- **Issuer:** iShares (BlackRock) **only**, via a curated list. iShares publishes clean
  per-product holdings CSVs at stable URLs — the most reliable source to parse.
- **Snapshots:** **latest only** — each run overwrites with current holdings. No
  point-in-time archive in this sub-project (deferred; see roadmap). This fully covers the
  #4 rating, which needs only current holdings.
- **Format:** CSV, to match the repo's existing tracked `data/processed/` tables.

## Components

Follows the repo convention: source logic in a `capstone_data/` module, one script
entrypoint per build step, source definitions in `config.py`.

### `capstone_data/config.py` — `ISHARES_ETFS`
A dict mapping ETF ticker → iShares product holdings-CSV URL (the stable
`...ajax?fileType=csv&fileName={ticker}_holdings&dataType=fund` endpoint).

Starter universe (international-equity aligned; extend by adding a line):
`EFA, IEFA, ACWX, SCZ, IEMG, URTH`.

### `capstone_data/etf_holdings.py`
- `fetch(ticker, url) -> bytes` — the **only** network touch. Downloads the raw CSV.
- `parse(raw: bytes, etf_ticker: str) -> pd.DataFrame` — pure. Skips the iShares preamble,
  extracts `as_of_date`, normalizes to the output schema, coerces types.
- `build(etfs=config.ISHARES_ETFS) -> pd.DataFrame` — fetch+parse each ETF, concatenate,
  skipping failures (with a summary of successes/failures).
- `write(df) -> Path` — writes `data/processed/etf_holdings.csv`.

### `scripts/build_etf_holdings.py`
Build-step entrypoint: calls `etf_holdings.build()` then `write()`, prints a per-ETF
success/row-count summary and any skipped ETFs.

## Parsing detail

iShares holdings CSVs prepend ~9 preamble metadata lines before the tabular header. The
parser:

1. Decodes bytes and scans lines for the header row (the line beginning with
   `Ticker,Name,Sector,` — the first real column header). Everything above it is preamble.
2. Extracts the as-of date from the preamble line `Fund Holdings as of,"<date>"` →
   `as_of_date` (ISO `YYYY-MM-DD`).
3. Reads the table from the header row down; stops at trailing disclaimer lines if present
   (rows past the constituent block).
4. Coerces `Weight (%)` strings (e.g. `"3.45"`, possibly with commas) → float;
   `Market Value` / `Shares` strings with thousands separators → float.

If the header row is **not found**, raise an explicit error naming the ETF — a silent
iShares layout change must fail loudly, not corrupt the table.

## Output schema — `data/processed/etf_holdings.csv`

One row per (ETF, constituent), latest snapshot:

| Column | Source | Notes |
|---|---|---|
| `etf_ticker` | config key | e.g. `EFA` |
| `as_of_date` | preamble | ISO date of the snapshot |
| `constituent_ticker` | `Ticker` | may be blank for cash/derivatives |
| `name` | `Name` | |
| `sector` | `Sector` | blank/"Cash and/or Derivatives" for non-equity |
| `asset_class` | `Asset Class` | lets downstream filter to `Equity` |
| `weight` | `Weight (%)` | float, percent (sums to ~100 per ETF) |
| `market_value` | `Market Value` | float |
| `shares` | `Shares` | float |

**Cash / derivative rows are kept** (not dropped) and identified via `asset_class`, so
weights still sum to ~100%. Downstream (#4 sector entropy) filters to `asset_class ==
"Equity"` itself.

## Error handling

- **Network / HTTP failure** for an ETF → log, **skip that ETF**, continue with the rest;
  the run exits with a non-zero summary count but writes what succeeded.
- **Header not found / layout changed** → explicit parse error naming the ETF.
- **Empty or all-cash result** → warn (kept in output, but flagged in the summary).

## Testing (no-network, repo convention)

A saved **sample iShares holdings CSV fixture** (real preamble + a handful of constituent
rows incl. a cash row) lives under `tests/`. Deterministic `parse` tests assert:

- preamble is skipped and the correct header row is found;
- `as_of_date` is extracted correctly;
- `Weight (%)` / `Market Value` / `Shares` string → float coercion (incl. thousands
  separators);
- output columns exactly match the schema;
- cash/derivative row is retained with its `asset_class`;
- a fixture with no recognizable header raises the explicit parse error.

The network `fetch` is **not** exercised in tests.

## Out of scope (YAGNI / deferred to roadmap)

- Non-iShares issuers (SPDR, Vanguard).
- Point-in-time / historical snapshot archive.
- The rating and dashboard wiring (sub-projects #4 and #5).