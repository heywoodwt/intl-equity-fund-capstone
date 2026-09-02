# 8. WRDS query fix & FactSet crosswalk

_Date: 2026-06-16 · Status: complete — holdings 97% mapped to FactSet_

## Goal
Get the FactSet discovery run working, then validate we can actually map the fund's holdings to
FactSet ids (the make-or-break step for the fundamentals layer).

## Blocker found & fixed
`wrds.raw_sql` is broken under our stack (pandas 2.x + SQLAlchemy 1.4): it hands pandas a
SQLAlchemy-1.4 `Connection`, which pandas 2.x no longer recognizes, then falls back to calling
`.cursor()` on it → `'Connection' object has no attribute 'cursor'`. We can't upgrade SQLAlchemy to
2.0 (breaks wrds's own `execute`) or drop pandas below 2.2 (our `"ME"` resample needs it). Fix:
`capstone_data/wrds_io.py` — `query()` runs through the engine's **raw psycopg2 connection**, which
pandas reads fine. Use it for all data pulls (`describe_table`/`list_tables` are unaffected).

## Discovery (factset_ff_int)
Full schema dump in `data/interim/factset_schema_dump.txt`. Highlights:
- **`ff_advanced_der_af_{eu,ap}`** — derived annual ratios keyed by `fsym_id` (`-R`) + `date`
  (fiscal period end). Has valuation (`ff_pe_dil`, `ff_pbk_tang`, `ff_div_yld`, `ff_entrpr_val_*`),
  quality (`ff_roe`/`ff_roea`, `ff_roic`, `ff_ebit_oper_mgn`, `ff_debt_eq`, `ff_zscore`,
  `ff_fscore`), and growth (`ff_*_gr`). No publication-date column → point-in-time needs a
  conservative reporting lag.
- **`cs3_monthly_prices_final_int`** — month-end price, shares out, dividends, return per `fsym_id`.
- **Symbology** (`factset` lib): `sym_ticker_region`, `sym_isin`, `sym_sedol`, `sym_coverage`.

## Crosswalk built & validated
Join path: Yahoo ticker → FactSet `ticker_region` (suffix→country map) → `fsym_id`.
**126 / 130 holdings (97%) matched** — 120 via current symbology, **6 via the historical table
(recovers delisted names → addresses the survivorship gap)**. The 4 misses are Nordic class-B
tickers (`AF-B.ST`, `ALK-B.CO`, `RAY.ST`, `VITB.ST`); hand-map via SEDOL.

## Artifacts
`capstone_data/wrds_io.py`, `holdings.py`, `factset.py` (crosswalk); `scripts/factset_explore.py`,
`scripts/build_crosswalk.py`; `data/interim/holdings_fsym_map.csv`.

## Next
Extraction: pull valuation/quality/growth fields from `ff_advanced_der_af_{eu,ap}` (+ `_usc` for
US-listed ADRs) for the mapped `fsym_id`s over 2018–2022, apply a reporting lag for point-in-time,
then portfolio-weight into the monthly dataset. Resolve the 4 unmatched via SEDOL.
