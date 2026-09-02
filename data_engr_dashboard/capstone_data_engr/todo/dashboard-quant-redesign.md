# Dashboard redesign — from "data-scientist exploratory" to "quant-fund tear sheet"

**Goal.** Turn `dashboard/` into a utility a fund manager opens to evaluate the fund the way an institutional allocator / quant PM does — a
standardized performance, risk, attribution, and positioning tear sheet — instead of a
methodology-forward modeling sandbox. Eventually accepts an uploaded fund; for now it builds on the
engineered data we already have, via pure functions that take the engineered frame(s) as input.

## 1. What we have today (and why it reads "data-scientist")

Three tabs, all research-forward:
- **Rolling regression** — model picker (OLS/Ridge/ElasticNet), penalty slider, time-series CV,
  standardization toggles, condition-number warnings, contribution decomposition.
- **Data explorer** — variable distributions, returns-over-time, correlation heatmap.
- **Glossary** — PCA component explorer (loadings) + variable dictionary.

It's rigorous and correct, but it leads with *methodology and knobs*. A fund manager opens a fund
dashboard to answer: *How has it performed? How risky is it? Is it beating its benchmark, and
why? What does it own and how is it positioned? What is it exposed to?* Those standard "boxes"
aren't surfaced; they're implied inside a regression tool.

## 1a. Decisions (locked)

- **Audience = both, one surface.** Lead with an allocator/due-diligence-grade tear sheet
  (Overview → Risk → Benchmark); keep PM diagnostics (Factor exposures → Positioning → Macro) one
  click deeper as a second tab group. Standardized KPIs are the front door; depth is behind it.
- **Default benchmark = blended avg of EFA/SCZ/VSS** (`bench_avg_ret`, matches the existing
  `alpha_vs_avg`). All four (EFA, SCZ, VSS, blended) stay selectable; surface a note that EFA is
  large-cap and SCZ/VSS are the size-appropriate small-cap references.
- **Research tooling = "Advanced".** Tab 4 defaults to a clean FF 6-factor OLS; Ridge/ElasticNet,
  penalty/CV, and the PCA explorer collapse into an Advanced expander / appendix. Rigor preserved,
  not led with.

## 2. The institutional checklist (the "boxes") vs. what our data supports

Legend: ✅ fully supported now · 🟡 partial / caveated · 🔲 needs data we don't have yet.

| Box a fund manager expects | Supported? | Notes / source columns |
|---|---|---|
| Headline KPIs (CAGR, vol, Sharpe, Sortino, Calmar, max DD, IR) | ✅ | `fund_ret`, `RF`, benchmark rets — monthly, 2014-10→2025-06 (129 mo) |
| Growth of $1, fund vs benchmarks (log toggle) | ✅ | already partly present |
| Underwater / drawdown curve + top-drawdown table | ✅ | derived from `fund_ret` |
| Trailing-period returns table (1M/3M/1Y/3Y/5Y/ITD) | ✅ | fund vs bench vs active |
| Calendar-year returns + monthly returns heatmap | ✅ | classic tear-sheet views |
| Full risk table (downside dev, skew, kurt, VaR/CVaR, hit rate) | ✅ | monthly frequency (state it) |
| Benchmark-relative: beta, tracking error, info ratio, correlation | ✅ | vs EFA / SCZ / VSS / blended |
| Up / down capture ratios | ✅ | standard capture math |
| Batting average (% months beating benchmark) | ✅ | `alpha_vs_*` > 0 |
| Rolling risk (vol, Sharpe, beta, TE, IR, correlation) | ✅ | window-selectable |
| Factor exposures & factor alpha (FF dev-ex-US 6-factor) | ✅ | `Mkt_RF…Mom` — keep the regression, reframe as *risk* |
| Return/risk attribution to factors | ✅ | existing contribution decomposition, relabeled |
| Macro-regime sensitivity (perf by VIX / rate regime) | 🟡 | conditional buckets from macro cols; PCA = appendix |
| Sector allocation vs benchmark + active tilts | 🟡 | `sect_wt_*`, `bench_sect_wt_*`, `active_sect_wt_*` — **2018-04→2022-12 only** |
| Geographic / country allocation | 🟡 | `ctry_wt_*` — same 57-month window |
| Style & quality characteristics (valuation/quality/growth) | 🟡 | `ff_*` value-weighted — fund's own profile, **no benchmark comparison** |
| Concentration (HHI, top-sector, #holdings/sectors/countries) | 🟡 | `sector_hhi`, `top_sector*`, `n_*` — 57-mo window |
| Brinson attribution (allocation vs selection effect) | 🟡 | feasible — fund side local, benchmark side via WRDS (now **Step 1**, see §6a) |
| Security-level contribution to return | 🟡 | from `position_values_monthly` + `stock_returns_monthly` (now **Step 1A**, §6a) |
| Fund-vs-benchmark valuation premium/discount | 🔲 | no benchmark fundamentals (only benchmark sector weights) |
| AUM / flows / fees / expense ratio | 🔲 | not in our data |

**Hard constraints to honor honestly in the UI:**
- **Two coverage windows.** Returns/factors/macro: full 129 months. Holdings-derived
  fundamentals, sector & country weights: **2018-04 → 2022-12 (57 months) only.** Positioning
  views must banner this and not imply current holdings.
- **Monthly frequency.** VaR/Sharpe/vol are monthly-based (annualized properly: ×12 / ×√12). No
  daily tail metrics.
- **Benchmark choice matters.** `bench_avg_ret` = simple mean of EFA/SCZ/VSS. EFA is *large-cap*;
  SCZ (MSCI EAFE Small) and VSS (FTSE ex-US Small) are the size-appropriate benchmarks for a
  dev-ex-US small/mid fund. Make the benchmark selectable and note the size-segment mismatch of EFA.
- **No benchmark fundamentals / sector returns** → style and sector views are descriptive
  (what the fund holds), not return-attribution, until we source those (§6).

## 3. Proposed information architecture (tabs)

Reorder so the manager-facing answers come first; push research tooling to the back as "Advanced."

1. **Overview (tear sheet).** KPI strip (CAGR, ann. vol, Sharpe, Sortino, max DD, Calmar, ITD
   alpha & beta vs selected benchmark, tracking error, info ratio). Growth-of-$1 (fund vs
   EFA/SCZ/VSS/blended, log toggle) with an underwater drawdown panel beneath. Trailing-returns
   table (1M/3M/6M/YTD/1Y/3Y/5Y/ITD, fund vs bench vs active). Calendar-year table. Monthly-returns
   heatmap (year × month).
2. **Risk & drawdown.** Full risk-metric table; top-5 drawdown table (peak/trough/recovery dates,
   depth, length, recovery time); rolling volatility & rolling Sharpe (window control); return
   histogram with VaR/CVaR markers; up/down-month stats.
3. **Benchmark & active management.** Benchmark selector. Cumulative active return; rolling
   tracking error, info ratio, beta, correlation; up/down capture chart; batting average. CAPM
   alpha/beta vs selected benchmark.
4. **Factor exposures.** Reframed rolling regression: current full-sample factor betas as a
   labeled, significance-annotated bar (R², annualized factor alpha as headline); rolling beta
   paths ("how exposures shifted across 2018/2020/2022"); return/risk contribution decomposition.
   Model knobs (Ridge/ElasticNet/CV/PCA) collapse into an **Advanced** expander; default is a clean
   FF 6-factor OLS.
5. **Positioning (2018–2022).** Banner the window. Sector allocation fund-vs-benchmark + active
   tilts (snapshot bar + stacked-area over time); country allocation; concentration metrics. Style
   & quality characteristics tables (valuation / quality / growth) with key-metric time series.
6. **Macro & regime (Advanced/appendix).** Performance conditioned on macro regimes (VIX terciles,
   rate-up vs rate-down, curve regime); the existing PCA explorer + variable glossary live here as
   methodology/appendix.

## 4. The quant metric battery (definitions to implement)

All from monthly series, annualized with ×12 (returns) / ×√12 (vol).
- **Return:** cumulative ∏(1+r)−1; CAGR (1+cum)^(12/n)−1; arithmetic ann. mean×12.
- **Vol/risk:** ann. std×√12; downside deviation (MAR=0 and =RF); skew; excess kurtosis;
  historical VaR & CVaR at 95/99; best/worst month; % positive months.
- **Risk-adjusted:** Sharpe (excess mean / excess std, annualized); Sortino; Calmar (CAGR/|maxDD|).
- **Drawdown:** running peak → underwater series; max/avg DD; longest DD & recovery; current DD;
  episode table (peak, trough, recovery, depth, length, recovery months).
- **Benchmark-relative:** beta (cov/var on excess); CAPM alpha (annualized intercept); tracking
  error std(active)×√12; information ratio mean(active)×12/TE; correlation; up-capture
  (Σ fund in up-bench months / Σ bench up) and down-capture; batting average (% active>0).
- **Rolling versions** of vol, Sharpe, beta, TE, IR, correlation, capture (window slider).
- **Tables:** trailing-period (geometric) returns; calendar-year returns; month×year matrix.

## 5. Implementation architecture (pure functions in, render out)

Mirror the existing `rolling_regression.py` pattern (pure, no Streamlit, notebook-importable):
- **`dashboard/analytics.py` (new).** Performance/risk/attribution math. Suggested API:
  `growth_of_dollar(df, cols)`, `drawdown_series(returns)`, `drawdown_table(returns, top=5)`,
  `performance_summary(df, ret, bench, rf)` → dict of KPIs, `trailing_returns(df, ret, bench)`,
  `calendar_returns(df, ret)`, `monthly_return_matrix(df, ret)`, `rolling_metrics(df, ret, bench,
  rf, window)`, `capture_ratios(df, ret, bench)`, `regime_buckets(df, ret, regime_col, q=3)`,
  `positioning_snapshot(df, asof)` / `sector_active_tilts(df)` / `characteristics_table(df)`.
  Each takes the engineered frame + column names — so an uploaded fund flows through unchanged once
  it's coerced to the same schema.
- **`dashboard/rolling_regression.py`** — keep as the factor engine; Tab 4 calls it.
- **`dashboard/data.py`** — extend: benchmark registry (EFA/SCZ/VSS/blended + display names + the
  large-vs-small note), labels for the new metrics, a `benchmark_series(df, key)` helper, and the
  "selected benchmark" plumbing.
- **`dashboard/app.py`** — re-tabbed per §3; thin rendering that calls `analytics.py`. A shared
  benchmark selector + date range in the sidebar.

Design for the upload future now: every analytic takes `(df, ret_col, bench_col, rf_col, ...)`, no
hard-coded `fund_ret`/`FUND`, so the eventual upload module only has to produce a frame in the
standard schema (the WRDS-enrichment backend is a later, separate piece).

## 6. Gaps & what would unlock the missing boxes (future data asks)

- **Return-based attribution (Brinson allocation + selection).** Needs **sector-level returns**
  for fund and benchmark. Near-term partial: we already have `position_values_monthly.csv` +
  `stock_returns_monthly.csv` + `holding_profile.csv` (sector per security) → can build
  **security- and sector-level contribution-to-return** for 2018–2022 (Phase 2), which is most of
  what a manager wants from attribution.
- **Fund-vs-benchmark style/valuation gap.** Needs benchmark constituent fundamentals (FactSet
  Ownership holdings × FactSet Fundamentals) — same pipeline as `benchmark.py`, extended.
- **Current positioning.** Needs holdings past 2022-12 (refresh the FactSet pull).
- **AUM / flows / fees.** Not in scope of the data engineering repo; would come from the fund.

## 6a. Step 1 (NEW, do first): sector returns & contribution/attribution

Promoted to the first build step at the manager's request — sector return attribution is a headline
quant box. Two halves, different cost; **portfolio half is local and immediate, benchmark half needs
a WRDS run.**

### Findings from a feasibility test (drove the design below)
- Holdings cover **2018-04 → 2022-12 (57 mo, ~130 names)** — attribution is limited to this window.
- A holdings-based USD return reconstruction tracks the official `fund_ret` at **corr ≈ 0.88–0.90**
  but with a **~2–3%/month residual** (cash, fees, intra-month trades, small-cap pricing staleness).
  → Present as **holdings-based gross attribution reconciled to NAV with an explicit `Residual /
  unexplained` line.** The *relative* sector ranking is reliable; it will not tie to NAV exactly.
- Return field nuance: `ret_factset` is corporate-action-adjusted but **local-currency**;
  `value_usd/Shares` carries FX but has **corporate-action glitches** (e.g. a spurious −68% name in
  2018-05). **Use `ret_factset` × implied FX move**, where FX (local→USD) is recoverable per
  security-month as `value_usd / (Shares × price_m)`. Winsorize residual outliers.

### 1A — Portfolio sector returns + contribution (local, no WRDS) — build now
- New builder `capstone_data/attribution.py` (pure, mirrors `profile.build_breakdowns`). Inputs:
  `position_values_monthly.csv`, `stock_returns_monthly.csv`, `holding_profile.csv`.
- Per security-month: USD total return `r_i,t = (1+ret_factset/100)·(fx_t/fx_{t-1}) − 1` (winsorized);
  begin-of-month weight `w_i,t = value_usd_{i,t-1}/Σ value_usd_{·,t-1}`; contribution `c_i,t = w·r`.
- Roll up to sector: begin weight `W_sec`, portfolio sector return `r_p,sec = Σc/ΣW`, sector
  contribution `C_sec`. Reconstructed gross `r̂_t = Σ C_sec`; `Residual_t = fund_ret − r̂_t`.
- Outputs: `portfolio_sector_returns_monthly.csv` (month × sector: begin_wt, ret, contrib) and
  `security_contrib_monthly.csv` (month × ticker: wt, ret, contrib, sector) for top movers.
- Tests (mirror `tests/`): weights sum to 1; `Σ contrib == r̂`; FX/winsor edge cases.

### 1B — Benchmark sector returns (WRDS run) — implement now, you run it
- Extend `benchmark.py`: reuse the per-ETF constituent holdings (`fsym_id`, `adj_mv`) it already
  fetches; pull `one_month_return` for those `fsym_id`s via `factset.fetch_prices` (cs3 tables);
  value-weight by RBICS sector & month. **Scope the pull to 2018–2022** (Brinson needs only that).
  Chunk the `fsym_id IN (...)` list (thousands of constituents). Report MV coverage per
  benchmark-month (share of adj_mv with both sector and return).
- Output: `benchmark_sector_returns_monthly.csv` (benchmark × month × sector: begin_wt, ret) +
  equal-weight blend `bench` (consistent with `bench_avg_ret`). Reconcile Σ wt·ret vs the actual
  ETF return (`efa_ret`/…); report the gap.

### 1C — Brinson-Fachler attribution (local, once 1A+1B exist)
- Per sector-month vs the selected benchmark (default blended): `Allocation = (w_p−w_b)(r_b,sec−r_b)`,
  `Selection = w_b(r_p,sec−r_b,sec)`, `Interaction = (w_p−w_b)(r_p,sec−r_b,sec)`; effects sum to
  reconstructed active return. Note the benchmark-only `non_corporate` sector (ETF cash/non-equity)
  surfaces as a pure allocation term — label it.
- Compute in the dashboard analytics layer (deterministic from the two return tables + weights);
  optionally persist `attribution_monthly.csv`.

### ⚠️ Data bug found by 1B (must decide a fix)
Building the benchmark sector returns independently from FactSet holdings surfaced a real
upstream bug: **the stored `scz_ret` and `vss_ret` series are the *negative* of the true return.**
- Evidence: corr(independent reconstruction, stored) = **−0.97** for SCZ/VSS but **+0.99** for EFA;
  corr vs **−**stored = +0.97 (MAD 1.2% vs 9%). Mar-2020 COVID: SCZ stored **+21.0%**, recon
  **−17.6%**; VSS stored **+23.0%**, recon **−19.3%** (small-caps fell ~−18/−23% that month).
- Source: the modeling-repo files `2014_2025_SCZ_Monthly.csv` / `..._VSS_Monthly.csv`
  (`config.PERF_FILES`). EFA and the fund file are correct.
- Blast radius: `scz_ret`, `vss_ret` (negated) → `bench_avg_ret` (mean of 3, 2 wrong) →
  `alpha_vs_avg` all wrong in `combined_monthly.csv` **and** its copy in
  `../ds6015/data/engineered_data/` (the contract) **and** the current dashboard. `alpha_vs_efa` OK.
- **Correction was subtler than a flip:** the corruption is sign-flipped for **2014-2022** but
  correct for **2023-2025** (verified vs Yahoo total returns), so a single negation is wrong. **Fix
  applied:** `combine.load_benchmark_etf_returns()` sources EFA/SCZ/VSS from **Yahoo** (auto-adjusted
  total return, cached to `benchmark_etf_returns_monthly.csv`); `fund_ret` stays from the perf file.
  Every benchmark year now matches the real ETF; fund active return ~+2.3%/yr (was an implausible
  +7.4%); Brinson reconciliation residual −0.21%. Flag the team to fix the source files.
- Note: 1B/1C are unaffected by the bug (benchmark sector returns are reconstructed independently,
  not from the perf files), so Brinson uses correct benchmark returns regardless.

### ⚠️ Data bug #2 found by 1A/1C (position weights corrupted — OPEN)
Building contribution attribution surfaced a second, deeper issue: **the fund's position
weights are corrupted for corporate-action names**, because `fundamentals.position_values`
computes value as `Shares × price_m`, but FactSet cs3 `price_m` is an **adjusted/indexed level**,
not the actual tradeable price.
- Evidence: ORP.PA (Orpea) shows `price_m` ≈ €5,326 (Nov-2021) when the stock actually traded ~€67
  (~80×); `one_month_return` is correct (−56% Jan-2022 = the real collapse). Result: Orpea is valued
  at **~60% of the portfolio**, NAS.OL ~38% — and the **median top-holding weight is 37%**, with
  21/57 months having a single name >40%. A registered mutual fund (FUND) legally cannot hold 60%
  in one issuer, so this is definitively a data artifact.
- Blast radius: corrupts `position_values_monthly.csv` → the **existing** `sect_wt_*`, `ctry_wt_*`,
  `top_sector*`, `sector_hhi`, value-weighted `ff_*` fundamentals (all in `combined_monthly` and the
  contract) and the new 1A/1C fund-side attribution. The current dashboard already shows ~63%
  healthcare from this. Benchmark side (1B) is unaffected (uses ownership `adj_mv` = real USD MV).
- Root cause: adjusted/indexed price × as-reported shares. Only ~2 names are egregious (those with
  huge cumulative adjustment factors); most names are ~OK because adjusted ≈ actual without big
  corporate actions.
- **RESOLVED (option A done):** `factset.fetch_unadj_prices` now pulls the actual price from
  `factset_own.own_sec_prices_eq.unadj_price` (mapping holdings `-R`→`-S` via `sym_coverage`),
  and `position_values` values positions on it (carrying real `usd_per_ccy` for downstream FX).
  A second `fundamentals.correct_split_basis` guard divides out split/consolidation **share-basis**
  breaks (per-share move ≥3× off the security's actual return) — fixed 6 (incl. AML.L ÷20.9).
  Result: median top-holding weight **37% → 8%**, no month >25%, and the holdings-based fund return
  reconstruction tightened from **corr 0.885 / 3.0% MAD → 0.975 / 1.27% MAD**. Rebuilt fundamentals
  + combined + attribution; re-copied the `ds6015` contract.
- **Residual caveat:** AML.L (Aston Martin) still peaks ~20% in ~10 months — its 2020 had a
  consolidation + multiple capital raises + near-bankruptcy crash that a shares-only holdings file
  can't fully untangle. Flag it in the Positioning tab; everything else is clean.

### Dashboard surfaces this unlocks
- **Sector contribution-to-return** (bar / waterfall, single month or cumulative) — fund only (1A).
- **Top contributors / detractors** (security level) — from `security_contrib_monthly` (1A).
- **Brinson allocation vs selection** vs benchmark, per sector + total, over time (1C).
These live in Tab 5 (Positioning → add an **Attribution** sub-section) and feed Tab 3 (active mgmt).

## 7. Suggested build order

1. **Step 1A — portfolio sector returns + contribution** (`attribution.py` + tests; local). Unblocks
   contribution-to-return and top-mover visuals immediately.
2. `analytics.py` + tests (Sharpe/DD/capture); **Tab 1 Overview + Tab 2 Risk** (full-history, no caveats).
3. **Step 1B — benchmark sector returns** (extend `benchmark.py`; you run the WRDS build) → **1C Brinson**.
4. Tab 3 Benchmark/active (add capture + Brinson summary).
5. Tab 4: reframe existing regression; move knobs to Advanced.
6. Tab 5 Positioning (2018–2022) + the Attribution sub-section.
7. Tab 6: regime buckets; relocate PCA/glossary as appendix.
