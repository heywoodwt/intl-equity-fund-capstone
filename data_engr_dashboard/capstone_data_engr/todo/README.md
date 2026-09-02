# Plan & orientation (start here)

Forward plan for the **analysis/modeling** phase. The data-engineering phase is complete; the
datasets are built and documented. This folder lays out what's next.

## Where things are
- **Finished datasets** → `../data/processed/` (this repo) and copied to
  `../../ds6015/data/engineered_data/` with a full data dictionary (`README.md`) + process
  write-up (`REPORT.md`). **Read that README before modeling.**
- **What we did & why** → `../report/1..13` (step-by-step) and the group repo's
  `data/engineered_data/REPORT.md` (consolidated).
- **Build/refresh the data** → `../README.md` ("Build order").
- **Overall project roadmap & data asks** → `roadmap.md` (whole-project plan, data-eng → modeling)
  and `data-access-requests.md` (what to request from the school/sponsor) — both in this folder.

## The plan (in order)
1. **[Baseline rolling regression](1%20baseline-rolling-regression.md)** — explain the fund's
   performance: rolling factor/macro regressions, coefficient paths, then a regularized non-linear
   extension. (The advisor specifically asked for the non-linear + regularized version.)
2. **[Predictive modeling](2%20predictive-modeling.md)** — predict alpha with DL (CNN/RNN/attention),
   exploiting the baseline insights, formulated for interpretability.

Both consume the engineered datasets. **Suggested convention:** do the modeling **here**
(`capstone_data_engr`, e.g. a new `analysis/` package + `scripts/`), reading the engineered CSVs, and
copy final figures/findings to the group repo — same rationale as the data work (keep churn out of
the shared repo). Adjust if the team prefers the modeling to live in `ds6015`.

## ⚠️ Must-know gotchas (a fresh agent will get these wrong otherwise)
- **Factors:** use `ff_factors_dev_ex_us.csv` (already merged into `combined_monthly.csv`). The group
  repo's `data/ff_factors.csv` is the wrong region (`Developed` incl. US).
- **Units differ by column group:** fund/benchmark returns, alpha, and FF factors are **decimals**;
  macro rate/inflation/return features are **percent**; sector/country weights are **fractions**;
  FactSet fundamentals are FactSet-native (multiples are ratios; margins/growth/yields %).
  **Standardize features before any regularized or non-linear model.**
- **Point-in-time already enforced:** macro releases lagged 1 mo, fundamentals lagged 4 mo. Don't
  re-lag. But when building rolling targets, remember `next_ret` is the forward label.
- **Windows:** fundamentals & sector/country exist only **2018-04 → 2022-12**; returns/alpha/factors/
  macro span **2014-10 → 2025-06**. Plan windows accordingly.
- **Benchmark sector weights** (`efa/scz/vss_sect_wt_*`, blend `bench_sect_wt_*`) are point-in-time and
  span the **full window**; the **active tilts** (`active_sect_wt_*` = fund − benchmark) exist only in
  the **2018-2022** holdings window (NaN elsewhere — they need the fund's own `sect_wt_*`).
- **Returns** are decimal price returns (local ccy); ~93% panel coverage (gaps = non-trading months).
- **Env:** `uv` project, pandas pinned `<3` (wrds compatibility). Re-fetching data needs a FRED key
  and WRDS creds (`.env` + `~/.pgpass` + Duo) — but modeling only needs the CSVs, no network.
- **PCs are additive:** `combined_monthly.csv` keeps the raw explainable features; PCs are separate,
  joinable on `month`. Use raw features for interpretability, PCs to tame collinearity.
