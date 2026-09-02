# 11. Survivorship-free returns & PCA feature engineering

_Date: 2026-06-16 · Status: complete (one macro data issue flagged)_

## A. Survivorship-free per-stock returns
Switched the panel's returns from the modeling repo's `monthly_returns.csv` (78% coverage, survivorship
gap) to the **FactSet price table**, which covers delisted names too.
- `build_fundamentals` now persists `data/processed/stock_returns_monthly.csv` (extended the price
  pull to 2023-03 so `next_ret` is defined at the last panel month).
- Columns: `ret_price` = decimal month-over-month price return (what the panel uses); `ret_factset`
  = FactSet's published return, **in percent and total-return** (incl. dividends; ÷100 to use).
  Sanity check: `ret_price` mean 1.00%/mo vs `ret_factset` 1.16% — the gap ≈ dividends.
- **Panel return coverage 78% → 93%** (`next_ret` 95%). The remaining ~7% are months a security
  wasn't trading (pre-listing / post-delisting) — correctly NaN. The survivorship hole is filled.

## B. PCA feature engineering (`pca.py`, `scripts/build_pca.py`)
Additive, interpretable orthogonalization: features are z-scored and rotated into linearly-independent
PCs via SVD. **We keep ALL components** (a full-rank rotation — no information lost) and report
explained-variance ratios so the team can truncate to top-k. Scattered gaps are mean-imputed in
standardized space.

**The original dataset is retained** — `combined_monthly.csv` is untouched; PCs are written as
separate files, joinable on `month` (verified: join keeps all 129 months). So models can use the
explainable originals, the PCs, or both.

Two feature blocks:
| block | features | months | PCs | PCs to 95% var |
|---|---|---|---|---|
| **market** (macro + FF factor returns) | 25 | 129 (2014-2025) | 25 | 11 |
| **full** (+ portfolio fundamentals) | 47 | 57 (2018-2022) | 47 | 12 |

Outputs: `pca_{market,full}_components.csv` (month × PCs) and `pca_{market,full}_loadings.csv`
(feature loadings + `_explained_var_ratio` + `_cumulative` rows). Market PC1 explains 34% of variance.

## ⚠️ Flagged: macro credit-spread gap (not our bug)
The PCA surfaced that `ig_oas` / `hy_oas` / `hy_oas_chg` are empty before 2023. FRED's ICE BofA OAS
series (`BAMLC0A0CM`, `BAMLH0A0HYM2`) now return data **only from 2023-06** — an ICE licensing change
truncated the long histories. `baa_spread` (Moody's Baa−10Y) is complete (2000-2026) and covers the
credit dimension. **Recommend:** drop the two ICE OAS series from the macro config (optionally add a
Moody's Baa−Aaa quality spread), then re-run `build_macro → build_combined → build_pca`.

## Next
Apply the credit-spread fix; baseline rolling regressions; sector/country aggregates.
