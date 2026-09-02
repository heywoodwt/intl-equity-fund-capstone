# 1. Project scoping & data inventory

_Date: 2026-06-15 · Status: complete_

## Goal
Understand the capstone (explain *and* predict the fund's performance), inventory what
data already exists, and decide how to organize the enrichment work.

## What we did
- Read the group repo (`../capstone`) and documented it (`CLAUDE.md` + `docs/`).
- Mapped the roadmap's "Data Engineering & Enrichment" checklist to actual files.
- Ran a coverage check of the fund's holdings against the existing equity datasets.

## Key findings
- **Fund:** the Fund — developed-markets **ex-US** small/mid-cap;
  benchmarked vs EFA / SCZ / VSS. ~130 holdings, 1,184 trades (2018–2022).
- **Already have:** equity & fund/benchmark returns, FF factors, momentum/risk characteristics
  (`characteristics_panel.csv`), and a sector/country *reference* (`msci_world_equities.csv`).
- **Missing:** macro regime indicators, valuation/quality fundamentals, engineered sector/country
  aggregates, and the combined dataset (the roadmap deliverable).
- **Coverage:** 105 of 130 holdings already have returns + characteristics (~80%). The 25 missing
  skew toward delisted/acquired names (Wirecard, GW Pharma, Farfetch, Kahoot) → a
  **survivorship/delisting gap**.
- **`msci_world_equities.csv` is not a usable security master** (only 8–25 / 130 tickers match) —
  sector/country enrichment needs a real identifier mapping.

## Decision
Do the enrichment in a **separate repo** (`capstone_data_engr`) to keep large intermediate data and
pipeline code out of the group repo. Contract: *external sources → one enriched dataset the
modeling repo consumes.*

## Next
Decide where each enrichment input comes from (report 2).
