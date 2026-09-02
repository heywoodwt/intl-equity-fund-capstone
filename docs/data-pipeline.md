# Data pipeline

How raw custody-statement PDFs become the cleaned trade/holdings tables, and where the rest of the
data comes from. **The scripts and notebooks are the source of truth** for exact logic.

## Trade extraction (in this repo)

```
annual_custody_statement_YYYY.pdf         (annual custody statements, 2018–2022)
        │  data/extract transactions.py    (pdfplumber + regex)
        ▼
purchase_and_sale_transactions_full_desc_YYYY.csv
        │  data/split_columns.py           (regex parse of Description)
        ▼
split_transactions_YYYY.csv
        │  cleaning + ticker resolution  (code not committed; see note below)
        ▼
clean_transactions.csv                     (normalized, tickers resolved, local currency)
        │  FX conversion  (code not committed)
        ▼
clean_transactions_base_usd.csv            (USD prices + Transaction_Value_USD)
        │  cumulate shares by month  (code not committed)
        ▼
monthly_holdings.csv                       (Month, Ticker, Shares)
```

### `data/extract transactions.py`
Opens one PDF, scans every line, and uses `transaction_start_pattern`
(`date  Purchase|Sale  desc  amount`) to detect Purchase/Sale rows, appending wrapped continuation
lines to the description until the next dated line. Writes `Date Posted, Activity, Description,
Principal Cash`. **Edit the `pdf_filename` / `csv_filename` constants at the top and rerun per year.**
Skips Dividend/Fee rows by design.

### `data/split_columns.py`
Reads one `..._full_desc_YYYY.csv`, regex-extracts `Quantity, Unit Type, Asset Name, Trade Date,
Broker, Price Per Unit` from the free-text `Description`, drops the description, writes
`split_transactions_YYYY.csv`. **Filenames are hard-coded** (currently 2022) — change them per year.

> These two scripts are per-year and manual. There is no orchestration script; you run them once per
> statement. **The downstream merge/clean/FX/holdings transformation code is not committed** — only
> its outputs (`clean_transactions*.csv`, `monthly_holdings.csv`, and the intermediate
> `cleanDataV1.csv` / `split_descV1.csv`). The two notebooks `ML implementation/pandas_cleanDataV1.ipynb`
> and `pandas_split_descV1.ipynb` only **load and inspect** those CSVs (`df.info()`), they don't
> produce them.

## Externally sourced data (not produced by repo scripts)

- **Fund & benchmark monthly performance** (`2014_2025_*_Monthly.csv`) — pulled externally.
- **`data/python_files.ipynb`** — pulls FUND returns via `yfinance` and Ken French
  **Developed-ex-US** factors via `pandas_datareader`, for the factor regressions (see
  `docs/models.md`). Note: this notebook's intent is correct ex-US factors; the saved
  `ff_factors.csv` artifact is the wrong (incl-US) set — don't use it.
- **Security panels** (`monthly_prices.csv`, `monthly_returns.csv`, `characteristics_panel.csv`) —
  precomputed price/return/characteristic panels.

## Engineered layer (sibling repo)

Everything in `data/engineered_data/` is built by **`capstone_data_engr`** (a separate `uv` project,
kept out of this repo to isolate pipeline code + credentials). It enriches holdings/trades with
macro (FRED), corrected FF factors, FactSet fundamentals (via WRDS), survivorship-free returns, and
sector/country, then assembles `combined_monthly.csv` / `combined_panel.csv` + PCA.

Rebuild order (needs WRDS + FRED credentials):
`build_macro.py → build_factors.py → build_fundamentals.py → build_combined.py → build_pca.py`

Full rationale, point-in-time lag policy, and known limitations are in
`data/engineered_data/REPORT.md`. **Do not regenerate the engineered CSVs from this repo** — it
doesn't contain the pipeline.
