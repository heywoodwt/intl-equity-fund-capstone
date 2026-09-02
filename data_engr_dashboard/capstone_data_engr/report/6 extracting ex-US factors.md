# 6. Extracting ex-US Fama-French factors

_Date: 2026-06-15 · Status: complete — found & fixed a region bug_

## Goal
Produce the **Developed-ex-US** Fama–French factors for the enriched dataset, and verify what region
the modeling repo's existing `ff_factors.csv` actually is.

## What we did
- `french.py` — fetch & parse Ken French zipped CSVs (text preamble → monthly table).
- `factors.py` — build Developed-ex-US 5 factors + momentum, plus `identify_region()` that
  fingerprints any factor file against 7 Ken French regions (correlation + max abs diff).
- `scripts/build_factors.py` + a parser unit test.
- **Output:** `data/processed/ff_factors_dev_ex_us.csv` — 430 months (1990-07 → 2026-04), decimal,
  columns `Mkt_RF, SMB, HML, RMW, CMA, RF, Mom`.

## Key finding (⚠️ region bug)
The modeling repo's `ff_factors.csv` is the Ken French **`Developed` set (includes the US)**, **not
`Developed_ex_US`**:

| region | mean corr | max abs diff |
|---|---|---|
| **Developed** | **1.0000** | **0.00000** |
| Developed_ex_US | 0.8378 | 0.07780 |

For an ex-US fund, those factors carry ~60% US market exposure the fund never held — contaminating
any alpha/beta decomposition or factor regression. The notebook *intended* ex-US
(`python_files.ipynb` requests `Developed_ex_US_3_Factors`); only the saved artifact is wrong.

## Action taken
Per instruction, **did not modify the modeling repo's data** — only fixed it here (the corrected
file lives in this repo) and documented the issue in `../capstone/docs/datasets.md`. Anything
derived from `ff_factors.csv` (e.g. `characteristics_panel.csv`'s `mkt_rf/smb/hml`) may inherit the
mismatch and should be re-checked.

## Next
Build the FactSet International fundamentals layer; then sector/country aggregates; then combine.
