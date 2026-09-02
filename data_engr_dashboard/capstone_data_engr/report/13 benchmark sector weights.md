# 13. Benchmark sector weights & active tilts

_Date: 2026-06-24 · Status: complete — point-in-time, validated_

## Why
We had the fund's own value-weighted sector weights (`sect_wt_*`, RBICS L1) but
**nothing for the benchmarks**. EFA / SCZ / VSS were in the data only as returns
(`efa_ret`, `bench_avg_ret`, `alpha_vs_*`). So we could measure *how much* the
fund beat the benchmark, but not *what* it was structurally doing differently —
e.g. the sector bets behind the active return. This layer adds the benchmark
sector composition and the fund-vs-benchmark **active sector tilts**.

## A. Source — point-in-time, not a snapshot
A snapshot of today's ETF sector mix (e.g. yfinance `funds_data.sector_weightings`)
would violate the repo's point-in-time rule: backfilling 2026 weights onto 2018
rows is look-ahead. Instead we reconstruct the weights **as of each month-end**
from FactSet Ownership holdings on WRDS.

- **Holdings:** `factset_own.own_fund_detail_eq` — each ETF's month-end positions
  (`fsym_id`, `adj_mv`). Resolved fund ids via `own_ent_fund_identifiers`:
  **EFA=04BZJ5-E**, **SCZ=04D200-E**, **VSS=04DVTB-E**. Full monthly coverage
  **2014-01 → 2025-12** (EFA ~862, SCZ ~1,926, VSS ~3,817 holdings/month).
- **Sector:** holding `fsym_id` → entity (`own_sec_entity_eq`) → **RBICS L1**
  (`factset.sym_entity_sector_rbics` + `rbics_structure_l2_curr`, `focus_flag`
  primary). This is the **same taxonomy and code path** as the fund's own
  holdings (`profile.py`), so weights are directly comparable.
- **Weighting:** value-weight `adj_mv` by sector per month. MV-classified
  **100.0%** every month for all three ETFs.

> Rejected alternatives: WRDS **ETF Global** (`etfg_samp`) has a clean
> `industry.sector_exposure` field, but UVA only has the *sample* — one month
> (2021-11), 13 tickers, and missing SCZ/VSS. WRDS **MSCI** libraries are
> ESG/climate only (no index constituents). FactSet Ownership was the one source
> with full point-in-time history for all three.

## B. Outputs
- **`benchmark_sector_weights.csv`** (new) — tidy long, one row per
  (`benchmark`, `month`): `n_holdings`, `wt_classified_sector`, and 13 RBICS L1
  `sect_wt_*`. The benchmarks span all 13 L1 sectors — the fund's 10 plus
  **Utilities, Telecommunications, Non-Corporate**, which the fund holds ~0 of.
- **`combined_monthly.csv`** (129 × 197, was ×132): added
  - `efa_sect_wt_*`, `scz_sect_wt_*`, `vss_sect_wt_*` — per-ETF (13 each);
  - `bench_sect_wt_*` — equal-weight blend (mirrors `bench_avg_ret = mean(efa,scz,vss)`);
  - `active_sect_wt_*` — **fund − blended benchmark**, per sector. Defined only in
    the fund's holdings window (2018-04 → 2022-12); NaN elsewhere so a tilt never
    implies holdings we don't have. (Sectors the fund never holds count as 0
    inside the window.)

Per-ETF weights populate the full 129 months; tilts populate the 57 fund-holdings
months. (Not added to `combined_panel.csv` — that's the per-stock grain; these are
month-level, like the fund's own `sect_wt_*`.)

## C. What it shows
Average active tilt vs the blended benchmark, 2018-2022 (the fund's structural bets):

| Overweight | | Underweight | |
|---|---:|---|---:|
| Healthcare | +24.8 pp | Finance | −20.0 pp |
| Technology | +12.2 pp | Non-Energy Materials | −8.0 pp |
| Consumer Cyclicals | +4.9 pp | Consumer Non-Cyclicals | −3.7 pp |

A classic developed-ex-US small/mid-cap **growth/quality** posture: heavily
overweight Healthcare and Technology, structurally underweight Financials and
Materials. This is the sector backdrop for the active-return models.

## Build
```bash
uv run python scripts/build_benchmark_sectors.py   # benchmark_sector_weights.csv (WRDS)
uv run python scripts/build_combined.py            # folds weights + tilts into combined_monthly
```

## Validation
- Holdings reported at month-end → point-in-time, no look-ahead, no snapshot backfill.
- Sector weights sum to 1.0 each month (MV-classified 100%); same RBICS L1 labels
  as the fund.
- `bench_sect_wt_*` verified = mean(efa, scz, vss); `active_sect_wt_*` NaN outside
  2018-2022, 0-fund-weight handled for sectors the fund never holds.
- Tests 5/5 (`test_benchmark_weights.py` added: weighting + tilt-masking logic).

## Next
Optional: per-benchmark tilts (vs EFA alone) are derivable from the raw
`*_sect_wt_*` columns if a model wants them. Country tilts could be built the same
way (entity domicile is already in the holdings path).
