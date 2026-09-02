# 12. Macro fix & sector/country enrichment

_Date: 2026-06-16 · Status: complete — data validated "good to go"_

## A. Macro credit-spread fix
Dropped the ICE BofA OAS series (`ig_oas`/`hy_oas`/`hy_oas_chg`) — FRED truncated them to 2023-06+
(licensing). Added a **Moody's Baa−Aaa quality spread** (`baa_aaa_spread`, from FRED `AAA`/`BAA`
yields) alongside the existing `baa_spread` (Baa−10y). Both are complete **2000-2026**. The macro
layer now has **17 features, no empty columns**.

## B. Sector & country enrichment (FactSet)
- **Country** = entity domicile (`factset_common.sym_entity.iso_country`); **Sector** = RBICS L1
  (`sym_entity_sector_rbics` → `rbics_structure_l2_curr.l1_name`, ~13 broad named sectors).
- `factset.fetch_profile` + `profile.py` (`build_holding_profile`, `build_breakdowns`).
- **`holding_profile.csv`** — 130 holdings, **100% country & sector known**: 24 countries
  (JP 28, GB 21, DE 11, FR/SE 10…), 10 sectors (Technology 35, Consumer Cyclicals/Industrials 17,
  Healthcare 15…).
- **`portfolio_sector_country_monthly.csv`** — value-weighted % per sector (10 cols) and country
  (24 cols), + concentration (`top_sector`, `top_sector_wt`, `sector_hhi`, `n_sectors`,
  `n_countries`, `wt_classified_*`). Weights sum to 1.0.
- **`position_values_monthly.csv`** — per-holding USD market value (shares × price × FX), persisted
  for transparency; shared by the fundamentals rollup and the breakdowns.

## C. Integration & PCA
- **`combined_monthly.csv`** (129 × 132): added the sector/country weights + concentration meta.
- **`combined_panel.csv`** (7,740 × 91): added per-stock `country`/`sector` + one-hot dummies
  (24 `ctry_*` + 10 `sect_*`).
- PCA re-run on the updated blocks (market 23 features; full 48); originals untouched, PCs joinable.

## Validation ("good to go")
- No fully-empty columns in either combined dataset.
- Sector & country weights sum to 1.0 every month; country/sector 100% known per holding.
- Per-stock returns 93% / next_ret 95% (rest are genuine non-trading months).
- Tests 3/3.

## Next
Data is solid. Ready for the baseline rolling regressions (not started, per request). Optional later:
security-master refinements, FactSet total-return (`ret_factset`) variant.
