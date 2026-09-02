# 3. Data-engineering repo setup

_Date: 2026-06-15 · Status: complete_

## Goal
Stand up `capstone_data_engr` cleanly so each source is a small, testable module feeding tidy
outputs.

## What we did
- **Structure:** `capstone_data/` package (`config.py` + one module per source) · `scripts/`
  entrypoints · `tests/` · `data/{raw,interim,processed}/`.
- **Tooling: uv.** `pyproject.toml` + `uv.lock` are the single source of truth (`requirements.txt`
  removed). Marked `package = false` (application-style; scripts/tests put the repo root on
  `sys.path`). Run with `uv run ...`; add deps with `uv add`.
- **Git hygiene:** `data/raw/` and `data/interim/` are git-ignored (reproducible caches);
  `data/processed/` (the small curated deliverables) is tracked. `.venv/` and `.env` ignored.
- **Contract:** external sources → tidy, **point-in-time**, **month-keyed** tables in
  `data/processed/` (keys: `date` month-end + `month` `YYYY-MM`).

## Artifacts
Repo scaffold + `README.md` documenting setup, the contract, and the roadmap.

## Next
First data layer: macro regime indicators (report 4).
