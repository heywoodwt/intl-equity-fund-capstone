# 10. Combining the layers

_Date: 2026-06-16 · Status: complete_

## Goal
Join the enrichment layers into the modeling datasets the next phases need. Offline — local CSVs
only, joined on `month` ('YYYY-MM').

## Outputs

### `data/processed/combined_monthly.csv` — for the baseline rolling regressions
**129 months (2014-10 → 2025-06) × 93 columns.** Backbone is the fund's monthly performance, with
everything left-joined:
- **Returns/targets (decimal):** `fund_ret`, `efa_ret`, `scz_ret`, `vss_ret`, `bench_avg_ret`,
  `alpha_vs_avg` (= fund − mean(EFA,SCZ,VSS)), `alpha_vs_efa`.
- **Ex-US FF factors:** `Mkt_RF, SMB, HML, RMW, CMA, RF, Mom` (all 129 months).
- **Macro regime:** the 19 `macro_monthly` features (all 129 months).
- **Portfolio fundamentals:** the 56 wmean/whmean/median columns (the 57 holdings-months,
  2018-2022; NaN elsewhere).

So factor/macro regressions can run over the full 2014-2025 window; fundamentals-augmented analysis
over 2018-2022.

### `data/processed/combined_panel.csv` — for the predictive (DL) stage
**7,740 rows (129 tickers × 60 months) × 57 columns:** the point-in-time stock fundamentals +
per-stock monthly `ret` + `next_ret` label + macro and factors broadcast by month.

## Caveats
- **Panel return coverage is 78%** — it uses the modeling repo's `monthly_returns.csv`, which has
  the survivorship gap (the ~25 delisted/missing holdings). FactSet's price table carries
  `one_month_return` for *all* mapped securities (incl. delisted); swapping that in would lift
  coverage toward 100% and stay consistent with the fundamentals source.
- Fundamentals exist only for 2018-2022 (the holdings window); macro/factors/returns span longer.

## Code
`capstone_data/combine.py` (`build_monthly` / `build_panel`), `scripts/build_combined.py`; config
performance/returns paths.

## Next
PCA / feature engineering for linear independence (roadmap), then the baseline rolling regressions.
Optional: sector/country aggregates; FactSet-returns swap for the panel.
