# 9. FactSet fundamentals extraction

_Date: 2026-06-16 · Status: complete_

## Goal
Extract point-in-time valuation/quality/growth fundamentals for the holdings and roll them up into
a value-weighted monthly portfolio dataset.

## Crosswalk → 100%
Added 4 hand overrides for Nordic class-B tickers (FactSet uses `<ROOT>.B-<REGION>`):
`ALK-B.CO→ALK.B-DK`, `RAY.ST→RAY.B-SE`, `VITB.ST→VIT.B-SE`, `AF-B.ST→AFRY-SE`. **130/130 holdings now
map to a FactSet `fsym_id`.**

## Extraction
- `factset.fetch_fundamentals` pulls the 25 derived ratios from
  `ff_advanced_der_af_{eu,ap}` (int) + `ff_advanced_der_af_am` (usc, for US-listed ADRs);
  `fetch_prices` pulls month-end price/shares/return from the `cs3_monthly_prices_final_*` tables.
- Two FactSet fields are empty in these tables — swapped **`ff_roea`→`ff_roce`** and
  **`ff_pfcf_dil`→`ff_pfcf`**. Pulled 989 annual fundamental rows + 7,740 monthly price rows.

## Layer 1 — point-in-time stock-month panel  (`data/processed/fundamentals_panel.csv`)
Annual figures are lagged 4 months (`FUND_REPORTING_LAG_MONTHS`, since the tables have no
publication-date column), then as-of joined onto a monthly grid. **129 tickers × 60 months;
any feature populated in 99% of stock-months.** Values are raw FactSet units (mostly %).

## Layer 2 — value-weighted portfolio  (`data/processed/portfolio_fundamentals_monthly.csv`)
Weights = `shares × FactSet price × FX(USD per ccy)`, FX from Yahoo `<CCY>USD=X`; each ratio is
winsorized cross-sectionally per month at [5%, 95%] before weighting. For every feature we report the
value-weighted **mean** and **median**; for the 6 price multiples (P/E, P/B, P/S, P/FCF, EV/EBITDA,
EV/Sales) we also report the value-weighted **harmonic mean** — the economically correct aggregation
(= aggregate price / aggregate fundamental, over positive values), since the arithmetic mean
overstates multiples. **57 months × 60 columns.** Sanity check: portfolio P/E ≈ 31 (harmonic) ≈ 32
(median) vs ~52 (arithmetic) — the harmonic/median aren't distorted by low-earnings names.

## Caveats / knobs to tune
- Ratios kept in FactSet native units (mostly %); the panel is raw — standardize/rank downstream as
  needed. The portfolio table gives weighted mean + median (+ harmonic for price multiples) so you
  can pick the right central tendency per use.
- Reporting lag is a uniform 4 months (no per-filing publication date available).
- 1 of 130 holdings has no fundamentals in the FF tables (129 in the panel).

## New code
`capstone_data/factset.py` (extraction), `fx.py`, `fundamentals.py`,
`scripts/build_fundamentals.py`; config features/lag/winsor; holdings overrides.

## Next
Combine the layers (stock-month panel + portfolio fundamentals + macro + ex-US factors) into the
modeling dataset; add sector/country aggregates; PCA.
