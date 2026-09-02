# 4. Macro regime indicators

_Date: 2026-06-15 · Status: complete (free layer)_

## Goal
Build the macro regime layer from the roadmap: VIX, inflation, money supply, credit, yield curve,
US equity index — all from free sources.

## What we did
- **Sources:** FRED (`fred.py` — keyed JSON API primary, keyless `fredgraph` CSV fallback) for 14
  series, and Yahoo (`yahoo.py` — `^GSPC`, no key) for the S&P 500.
- **Builder (`macro.py`):** resample everything to month-end, derive features, write
  `data/processed/macro_monthly.csv`.
- **Features:** `vix`/`vix_chg`; inflation YoY (`cpi/core_cpi/pce/core_pce`); `m2_yoy`; rates &
  curve (`y10/y2/y3m`, `slope_10y_2y`, `slope_10y_3m`); credit (`baa_spread`, `ig_oas`, `hy_oas`);
  equity (`sp500`, `sp500_ret`, `sp500_mom_12m`).
- **Point-in-time safety:** economic *releases* (CPI/PCE/M2) are lagged one month (published late),
  market series are not; the current incomplete month is dropped. Unit tests assert both.

## Status
Complete and validated (feature unit tests + a live Yahoo pull). Running the full build needs a free
FRED API key (`https://fredaccount.stlouisfed.org/apikeys`) in `.env`.

## Next
Pursue WRDS for the fundamentals layer (report 5).
