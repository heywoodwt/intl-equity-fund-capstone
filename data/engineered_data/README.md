# Engineered data

Finished, modeling-ready datasets produced by the data-engineering pipeline (the sibling
`capstone_data_engr` repo). These integrate the fund's holdings/trades with external data
(returns, Fama–French factors, macro regime indicators, FactSet fundamentals, sector/country) into
two analysis-ready tables plus engineered (PCA) features. See **`REPORT.md`** in this folder for the
full process and rationale.

> **Source of truth:** these CSVs. Regenerate via `capstone_data_engr` (`uv run python
> scripts/build_*.py`). Intermediary layer files (per-source macro/fundamentals/returns/etc.) are
> *not* copied here — they're rolled into the combined tables below.

## ⚠️ Read first
- **Do not use the group repo's `data/ff_factors.csv`** — it's the Ken French **`Developed` (incl.
  US)** set, wrong for this ex-US fund. Use **`ff_factors_dev_ex_us.csv`** here (and it's already
  merged into `combined_monthly.csv`). See `../../docs/datasets.md`.
- **Units differ by group:** fund/benchmark returns, alpha, and FF factors are **decimals**
  (0.012 = 1.2%); macro rate/inflation/return features are in **percent** (e.g. `sp500_ret`,
  `cpi_yoy`, `y10`); sector/country weights are **fractions** (0–1); FactSet fundamentals are in
  **FactSet native units** (multiples are ratios; margins/growth/yields are %).
- **Point-in-time (no look-ahead):** macro releases (CPI/PCE/M2) lagged 1 month; fundamentals lagged
  4 months (reporting delay). Market data (VIX, yields, S&P 500, factors, prices) is contemporaneous.
- **Windows:** fundamentals & sector/country exist only for the **holdings window 2018-04 → 2022-12**;
  returns/alpha, factors, and macro span **2014-10 → 2025-06** (monthly).
- **Join key:** `month` (`YYYY-MM`); the panel also keys on `ticker`.

## Primary modeling datasets

### `combined_monthly.csv` — 129 months × 197 cols (2014-10 → 2025-06)
One row per month. The dataset for **rolling regressions / time-series models**. Column groups:
- **Returns & alpha targets** (decimal): `fund_ret`, `efa_ret`, `scz_ret`, `vss_ret`,
  `bench_avg_ret`, `alpha_vs_avg` (= fund − mean(EFA,SCZ,VSS)), `alpha_vs_efa`.
- **Fama–French Developed-ex-US factors** (decimal): `Mkt_RF, SMB, HML, RMW, CMA, RF, Mom`.
- **Macro regime** (17, percent/levels): `vix`, `vix_chg`, `cpi_yoy`, `core_cpi_yoy`, `pce_yoy`,
  `core_pce_yoy`, `m2_yoy`, `y10`, `y2`, `y3m`, `slope_10y_2y`, `slope_10y_3m`, `baa_spread`,
  `baa_aaa_spread`, `sp500`, `sp500_ret`, `sp500_mom_12m`.
- **Portfolio fundamentals** (*2018-2022 only*): for each of 25 ratios, `<ratio>_wmean` (value-
  weighted) and `<ratio>_median`; for the 6 price multiples (`ff_pe_dil`, `ff_pbk_tang`,
  `ff_psales_dil`, `ff_pfcf`, `ff_entrpr_val_ebitda_oper`, `ff_entrpr_val_sales`) also
  `<ratio>_whmean` (value-weighted **harmonic** mean — the correct way to aggregate multiples). Plus
  `n_holdings`, `n_valued`, `wt_with_fundamentals`.
- **Sector/country breakdown** (*2018-2022 only*): `sect_wt_*` (10 RBICS sectors), `ctry_wt_*`
  (24 countries) — value-weighted fractions; plus `top_sector`, `top_sector_wt`, `sector_hhi`,
  `n_sectors`, `n_countries`, `wt_classified_{sector,country}`.
- **Benchmark sector weights** (point-in-time, RBICS L1, *full window*): `efa_sect_wt_*`,
  `scz_sect_wt_*`, `vss_sect_wt_*` (the three benchmark ETFs, 13 sectors each) and the equal-weight
  blend `bench_sect_wt_*` (mirrors `bench_avg_ret`). Reconstructed from each ETF's **month-end FactSet
  Ownership holdings** mapped to the same RBICS L1 taxonomy as the fund — so they're directly
  comparable to `sect_wt_*` and carry no look-ahead (not a current snapshot backfilled onto history).
- **Active sector tilts** (*2018-2022 only*): `active_sect_wt_*` = fund weight − blended benchmark
  weight, per RBICS L1 sector — the fund's structural sector bets vs the benchmark. (Defined only where
  the fund's own holdings exist; sectors the fund never holds, e.g. Utilities/Telecommunications, are
  treated as 0 within that window.)

### `combined_panel.csv` — 7,740 rows (129 tickers × 60 months) × 91 cols
One row per (holding, month). The dataset for **cross-sectional / deep-learning models**.
- **Keys:** `ticker`, `month` (+ `fsym_id`, `fiscal_date`).
- **Stock fundamentals** (25, raw per-stock, point-in-time): the `ff_*` ratios.
- **Returns:** `ret` (decimal month-over-month, FactSet price return), `next_ret` (next month's
  return — the prediction label).
- **Factors & macro** (broadcast by month): the same FF factors and 17 macro features.
- **Profile:** `country`, `sector` (strings) + one-hot dummies (`ctry_*` ×24, `sect_*` ×10).
- Coverage: `ret` present 93%, `next_ret` 95% (gaps = months a stock wasn't trading).

## Engineered features — PCA (additive; originals above are retained)
Standardized, orthogonal principal components. **All components kept** (full-rank rotation — no
information lost); use `*_loadings.csv` to interpret/truncate. Join `*_components.csv` to the tables
above on `month`.
- `pca_market_components.csv` — `month` + PC1..PC23 (129 months). From macro + FF factors (full
  window). For the rolling regression.
- `pca_full_components.csv` — `month` + PC1..PC48 (57 months). From macro + factors + fundamentals.
- `pca_market_loadings.csv` / `pca_full_loadings.csv` — feature loadings per PC, plus
  `_explained_var_ratio` and `_cumulative` rows. (Market: 10 PCs reach 95% var; full: 12 PCs.)

## Supporting datasets
- **`ff_factors_dev_ex_us.csv`** — 430 months (1990-07 → 2026-04). Ken French **Developed-ex-US**
  5 factors + Momentum (`Mkt_RF, SMB, HML, RMW, CMA, RF, Mom`), decimal. The correct factor set for
  this fund (replaces the mislabeled `data/ff_factors.csv` in the group repo).
- **`holding_profile.csv`** — 129 securities (the 130 holdings less 1 with no FactSet coverage):
  `ticker, fsym_id, factset_entity_id, name, country, sector`. Reference/dimension table — what each
  held security is, and the FactSet id used to source its fundamentals.
- **`benchmark_constituents_monthly.parquet`** — 951,059 rows (2014-01 → 2025-12), the **security-grain**
  source the benchmark `*_sect_wt_*` columns in `combined_monthly.csv` roll up. One row per
  (`benchmark` [efa/scz/vss], `month`, holding): `fsym_id`, `ticker_region` (99% matched), `name`,
  `country` (entity **domicile**, not listing venue), `sector` (RBICS L1, same taxonomy as the fund),
  `adj_mv` (USD), `weight` (share of the ETF's month-end MV — sums to 1.0 per benchmark-month).
  Point-in-time month-end FactSet Ownership holdings (no snapshot backfill); ~862/1926/3817 holdings
  per month for EFA/SCZ/VSS. Parquet (19 MB) — use `pd.read_parquet`. For constituent-level work
  (country tilts, single-name overlap with the fund, concentration) the rolled-up sector weights
  discard.

## Example
```python
import pandas as pd
m   = pd.read_csv("combined_monthly.csv")
pcs = pd.read_csv("pca_market_components.csv")
# explainable features + abstracted PCs together:
df  = m.merge(pcs, on="month", how="left")
```
