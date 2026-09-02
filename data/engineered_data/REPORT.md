# Data engineering & enrichment — process report

How the datasets in this folder were built, what we did to the data, and why. The work was done in
the sibling **`capstone_data_engr`** repo (kept separate to keep pipeline code and intermediate data
out of the group repo). That repo's `report/` folder has the step-by-step log (reports 1–13); this
document consolidates it for the team.

---

## 1. Objective & starting point

The fund (developed-markets **ex-US** small/mid-cap,
benchmarked vs EFA / SCZ / VSS) gave us **holdings and trades only** (~130 securities, ~1,184 trades,
2018–2022). To explain and predict its performance we needed to enrich that with returns, risk
factors, macro regime indicators, and company fundamentals, then assemble a single modeling dataset.

**Inventory & gaps (report 1).** We already had equity/fund/benchmark returns, Fama–French factors,
and momentum/risk characteristics; we were missing macro indicators, valuation/quality fundamentals,
and sector/country aggregates. A coverage check found ~105/130 holdings in the existing return
universe, with the missing ~25 skewing toward **delisted/acquired names** (Wirecard, GW Pharma,
Farfetch, Kahoot) — a survivorship risk — and that `msci_world_equities.csv` was **not** a usable
security master (only 8–25/130 tickers matched).

**Two guiding principles** (report 2), enforced in code throughout:
1. **Point-in-time / no look-ahead** — every value reflects what was knowable on that date.
2. **Survivorship/delisting** — include securities that later disappeared.

## 2. Architecture

- **Two layers:** a stock-month panel (per holding) and a monthly portfolio rollup (the aggregate),
  serving the cross-sectional/DL and the time-series/regression use-cases respectively.
- **Separate `uv`-managed repo** with one module per source and a contract: *external sources → tidy,
  point-in-time, month-keyed tables*. Join key is `month` (`YYYY-MM`).

## 3. What we built, layer by layer

### Macro regime indicators (reports 4, 12)
FRED (keyed API) for VIX, inflation (CPI/PCE + cores), M2, the Treasury curve, and credit spreads;
Yahoo for the S&P 500. Resampled to month-end; **economic releases (CPI/PCE/M2) lagged one month**
(published in arrears) while market series are contemporaneous; the current incomplete month is
dropped.
- **Credit-spread fix:** the ICE BofA OAS series we first used (`BAMLC0A0CM`/`BAMLH0A0HYM2`) turn out
  to return data only from **2023-06** on FRED (an ICE licensing change truncated the histories). We
  replaced them with Moody's **`baa_spread`** (Baa−10y) and a **`baa_aaa_spread`** (Baa−Aaa quality
  spread), both complete 2000–2026.

### Fama–French Developed-ex-US factors (report 6) — and a bug we found
We pulled Ken French's **Developed-ex-US** 5-factor + Momentum set (`ff_factors_dev_ex_us.csv`). To
validate, we fingerprinted the modeling repo's existing `ff_factors.csv` against seven Ken French
regions: it matches **`Developed` (which *includes* the US)** exactly (corr 1.000) and clearly
differs from `Developed_ex_US` (corr 0.84). **For an ex-US fund that's a real error** — those factors
carry ~60% US market exposure the fund never held, contaminating any alpha/beta work. The intent in
`data/python_files.ipynb` was correct (`Developed_ex_US`); only the saved artifact was wrong. We did
not modify the group repo's data (per request) but documented it in `docs/datasets.md` and ship the
corrected file here.

### WRDS / FactSet access (reports 5, 8)
UVA's WRDS does **not** include Compustat Global, Datastream, or Worldscope (so Compustat NA/CRSP
only reach US-listed names). It **does** include **FactSet Fundamentals International**
(`factset_ff_int`) and Capital IQ — confirmed queryable — which cover the international holdings. The
advisor cleared use of any accessible WRDS data.
- *Engineering note:* `wrds.raw_sql` is broken under our stack (pandas 2.x + SQLAlchemy 1.4), so all
  data pulls go through a small helper that queries the raw psycopg2 connection.

### FactSet fundamentals (reports 7, 9)
- **Identifier crosswalk:** holdings are Yahoo tickers (`NESN.SW`); FactSet keys on `fsym_id`. We map
  Yahoo ticker → FactSet `ticker_region` → `fsym_id`. **100% of holdings mapped** (4 Nordic class-B
  tickers hand-resolved; 6 delisted names recovered via FactSet's *historical* symbology — directly
  closing the survivorship gap). yfinance ISINs were tried and rejected (missing/wrong for
  international names).
- **Extraction:** 25 valuation/quality/growth ratios from the derived annual tables
  (`ff_advanced_der_af_{eu,ap,am}`). Two FactSet fields are empty in these tables, so we use `ff_roce`
  (not `ff_roea`) and `ff_pfcf` (not `ff_pfcf_dil`).
- **Point-in-time:** annual figures are lagged **4 months** (filing delay; the tables carry no
  publication date) and as-of joined onto a monthly grid → the stock-month panel (99% populated).
- **Portfolio aggregation:** value weights = shares × FactSet price × FX (USD per currency, from
  Yahoo `<CCY>USD=X`); each ratio winsorized cross-sectionally at [5%, 95%] per month. We report the
  value-weighted **mean** and **median** for all features, and the value-weighted **harmonic mean**
  for the 6 price multiples — harmonic is the correct aggregation for `price/X` multiples (= aggregate
  price / aggregate fundamental); arithmetic overstates them. Sanity check: portfolio P/E ≈ 31
  (harmonic) ≈ 32 (median) vs an inflated ~52 (arithmetic).

### Survivorship-free returns (report 11)
Per-stock monthly returns come from the **FactSet price table** (covers delisted names), not the
original return file. `ret` is a decimal month-over-month price return. Coverage rose **78% → 93%**
(next_ret 95%); the residual is months a security genuinely wasn't trading (pre-listing/post-
delisting), correctly left missing.

### Sector & country (report 12)
From FactSet: **country = entity domicile** (`sym_entity.iso_country`); **sector = RBICS Level-1**
(~13 broad named sectors via `sym_entity_sector_rbics` → `rbics_structure_l2_curr`). All 129
FactSet-covered holdings classified (24 countries — JP 28, GB 21, DE 11…; 10 sectors — Technology 35,
Consumer Cyclicals/Industrials 17…). We computed **value-weighted monthly sector and country breakdowns** (weights sum
to 1.0) plus concentration metrics (HHI, top weights).

### Benchmark sector weights & active tilts (report 13)
The fund's three benchmark ETFs (EFA / SCZ / VSS) were in the data only as *returns*, so we could
measure how much the fund beat them but not the sector bets behind that active return. We added the
benchmarks' **point-in-time sector composition**. A current snapshot (e.g. yfinance) would be
look-ahead — backfilling today's mix onto 2018 — so instead we reconstruct each ETF's weights **as of
each month-end** from **FactSet Ownership holdings** (`factset_own.own_fund_detail_eq`), mapping every
holding to the **same RBICS L1 sector** the fund uses and value-weighting by market value. Coverage is
100% of market value every month, 2014–2025. `combined_monthly.csv` gains per-ETF weights
(`efa/scz/vss_sect_wt_*`), the equal-weight blend (`bench_sect_wt_*`), and **active tilts**
(`active_sect_wt_*` = fund − blended benchmark, in the 2018–2022 holdings window). The tilts confirm a
textbook small/mid-cap growth posture: ~+25pp Healthcare and +12pp Technology, ~−20pp Financials and
−8pp Materials vs the benchmark. WRDS ETF Global (a more direct sector-exposure source) was rejected —
UVA only has the one-month, 13-ticker *sample*, missing SCZ/VSS.

### Combine (report 10)
Joined everything on `month` into `combined_monthly.csv` (fund performance as the backbone; factors,
macro, portfolio fundamentals, sector/country breakdowns, benchmark sector weights + active tilts
left-joined) and `combined_panel.csv`
(stock fundamentals + returns + `next_ret` + broadcast factors/macro + per-stock country/sector and
one-hot dummies).

### PCA feature engineering (report 11)
Standardized SVD-based PCA as an **additive, interpretable orthogonalization** — features are
z-scored and rotated into linearly-independent components, and **all components are kept** (full-rank
rotation, no information lost), with explained-variance ratios for optional truncation. The original
explainable datasets are untouched; PCs are written separately and join on `month`. Two blocks:
market (macro + factors, full window) and full (+ fundamentals, 2018–2022).

## 4. Key decisions & rationale (summary)
- **Harmonic mean for price multiples** — the only correct portfolio aggregation for `price/X` ratios.
- **Point-in-time lags** (macro 1 mo, fundamentals 4 mo) — prevents look-ahead bias.
- **Winsorization [5,95]** before weighting — ratios explode for near-zero-denominator names.
- **FactSet for returns & fundamentals** — survivorship-free and internally consistent.
- **PCA additive, not destructive** — keep the "extremely explainable" raw features *and* the
  abstracted PCs, so models can use either or both.
- **Corrected ex-US factors** — the fund is ex-US; the market factor must be too.

## 5. Validation
No fully-empty columns in either combined dataset; sector & country weights sum to 1.0 each month;
sector/country known for 100% of holdings; per-stock returns 93% / next_ret 95%; unit tests pass for
the macro and factor feature logic. Aggregate fundamentals and breakdowns spot-checked for sanity.

## 6. Known limitations
- **Fundamentals & sector/country exist only for 2018-04 → 2022-12** (the holdings window); returns,
  factors, and macro extend to 2025-06.
- **Reporting lag is a uniform 4 months** (no per-filing publication date in the FactSet tables).
- **Returns are price returns** (decimal, local currency; dividends excluded). FactSet's total-return
  figure is available in the pipeline (`ret_factset`, in percent) if a total-return basis is wanted.
- **Sector = current RBICS classification** (treated as static; sectors rarely change).
- **Mixed units across column groups** — see `README.md`.

## 7. Reproduce
All code is in `capstone_data_engr` (uv project). Rebuild order:
`build_macro.py` → `build_factors.py` → `build_fundamentals.py` (WRDS) → `build_benchmark_sectors.py`
(WRDS) → `build_combined.py` → `build_pca.py`. FactSet/WRDS steps need WRDS credentials; FRED needs a
free API key. The per-step reports (`capstone_data_engr/report/1..13`) document each stage in detail.
