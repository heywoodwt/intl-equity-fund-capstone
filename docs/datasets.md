# Datasets

Catalog of the data in this repo. **The CSV files are the source of truth** — column lists and row
counts here are a guide, not a contract; `head -1 <file>` to confirm.

For the **engineered / modeling-ready layer** (`data/engineered_data/`) this doc only points you in;
its own `README.md` (column groups, units, windows) and `REPORT.md` (how/why it was built) are
authoritative. Don't duplicate them here.

## Critical gotchas (repeat of CLAUDE.md, with detail)

- **Don't use `data/ff_factors.csv`.** Fingerprinting showed it is Ken French **Developed (incl.
  US)** (corr 1.000), not Developed-ex-US (corr 0.84). Wrong for an ex-US fund. Use
  `engineered_data/ff_factors_dev_ex_us.csv`. The intent in `data/python_files.ipynb` was correct;
  only the saved artifact is mislabeled.
- **Mixed units.** Fund/benchmark returns, alpha, FF factors = **decimals** (0.012 = 1.2%); macro
  rates/inflation = **percent**; sector/country weights = **fractions** (0–1); FactSet fundamentals
  = **native units**. The `2014_2025_*_Monthly.csv` files store performance as **percent strings**
  (`"-1.42%"`).
- **`Action`** = `+1` buy / `-1` sell; values may carry trailing spaces — `.str.strip()` before
  casting.

## Raw fund data — custody statements (`data/*.PDF`)

`annual_custody_statement_YYYY.pdf`, one per year **2018–2022**. The original
source — annual custody/transaction statements from US Bank. Everything in the trade pipeline is
extracted from these. See `docs/data-pipeline.md`.

## Transaction tables (`data/`)

Lineage: PDF → `purchase_and_sale_transactions_full_desc_*.csv` → `split_transactions_*.csv` →
(cleaning) → `clean_transactions.csv` → `clean_transactions_base_usd.csv` → `monthly_holdings.csv`.

| File | Grain / rows | Key columns | Notes |
|------|------|-------------|-------|
| `purchase_and_sale_transactions.csv` | 1 trade | `Date Posted, Activity, Description, Principal Cash` | First-pass extract; `Description` is raw statement text. |
| `purchase_and_sale_transactions_full_desc[_YYYY].csv` | 1 trade | same | Per-year and combined; descriptions reassembled across wrapped PDF lines. |
| `split_transactions_YYYY.csv` | 1 trade | `Date Posted, Activity, Quantity, Unit Type, Asset Name, Trade Date, Broker, Price Per Unit, Principal Cash` | Regex-parsed fields out of `Description`. Per-year. |
| `clean_transactions.csv` | 1,184 trades | `Trade_Date, Clean_Name, Action, Qty, Price, Currency, Ticker` | Deduped/normalized, tickers resolved, **local-currency** prices. `Action` ±1. |
| `clean_transactions_base_usd.csv` | 1,184 trades | + `FX_Rate_to_USD, Price_USD, Transaction_Value_USD` | Above, converted to **USD**. The trade table models should use. Currencies: EUR/USD/GBP/JPY/SEK/NOK/CHF/DKK. |
| `monthly_holdings.csv` | (month, ticker) | `Month (YYYY-MM), Ticker, Shares` | Share balances reconstructed from cumulative trades. 130 tickers, **2018-04 → 2022-12**. |

## Fund & benchmark performance (`data/2014_2025_*_Monthly.csv`)

Four files, 128 months **2014-10 → 2025-06**, two columns each: `Date` (e.g. `"Oct 31, 2014"`),
`Performance` (percent string). Externally sourced monthly total returns.

- `2014_2025_dataset_Monthly.csv` — **the fund** ("Portfolio"). | `_EFA_` `_SCZ_` `_VSS_` — the three
  benchmark ETFs.

## Security-level panels (`data/`)

| File | Grain | Coverage | Notes |
|------|-------|----------|-------|
| `monthly_prices.csv` | wide: Date × ~693 tickers | 2006-05 → 2026-04 | Month-end prices, local currency. |
| `monthly_returns.csv` | wide: Date × ~693 tickers | 2006-06 → 2026-04 | Month-over-month returns. **Has survivorship gaps** — the engineered panel's FactSet returns are the fix. |
| `characteristics_panel.csv` | long: (date, ticker), ~127k rows | 2009-07 → 2026-04, 683 tickers | Per-stock predictive characteristics: momentum (`mom_1m/3m/12m/36m`), `volatility`, `max_ret`, `min_ret`, `skewness`, `vol_trend`, `beta`, `mkt_rf/smb/hml`, and **`next_ret` (the label)**. Gu-Kelly-Xiu / Jegadeesh-Titman / Ang style features. |

## Reference / universe (`data/`)

- `full_universe_tickers.txt` — ~1,421 candidate tickers (the searched universe).
- `msci_world_equities.csv` — MSCI World constituents (`Ticker, Name, Sector, Location, Exchange,
  Currency, Weight (%), YF_Ticker`). ⚠️ **Not a usable security master** for our fund — only 8–25 of
  130 holdings matched. Reference only.
- `ff_factors.csv` — see gotcha above; **do not use**.

## Engineered / modeling-ready (`data/engineered_data/`)

Built by the sibling `capstone_data_engr` repo. **See that folder's `README.md` and `REPORT.md`.**
Quick index:

| File | What |
|------|------|
| `combined_monthly.csv` | One row per month (129 × 132). Returns/alpha targets + FF factors + macro + portfolio fundamentals + sector/country. The **time-series / regression** dataset. |
| `combined_panel.csv` | One row per (holding, month) (7,740 × 91). Stock fundamentals + returns + `next_ret` + broadcast factors/macro + one-hots. The **cross-sectional / DL** dataset. |
| `ff_factors_dev_ex_us.csv` | **Correct** Developed-ex-US 5-factor + Mom, decimal. Replaces `ff_factors.csv`. |
| `holding_profile.csv` | 129 securities: ticker ↔ FactSet ids, name, country, sector. Dimension table. |
| `pca_market_*.csv`, `pca_full_*.csv` | PCA components + loadings (additive, full-rank). Join on `month`. |

## `ML implementation/data/`

A working copy of selected files above (notebooks load by bare filename). Treat `data/` as canonical;
this copy may lag. Adds one output: `portfolio_decomposition.csv` (hybrid-model OOS results — see
`docs/models.md`).
