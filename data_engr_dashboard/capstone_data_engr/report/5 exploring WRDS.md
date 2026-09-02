# 5. Exploring WRDS

_Date: 2026-06-15 · Status: complete — access mapped & cleared_

## Goal
Determine what WRDS data UVA actually has for **international** fundamentals (this is an ex-US fund).

## What we did
- Added `wrds` to the project and wrote `scripts/wrds_explore.py` to list accessible libraries.
- **Access mechanics:** WRDS has **no API key** — it uses your WRDS username + password via the
  `wrds` package over PostgreSQL, caching credentials in `~/.pgpass`; Duo MFA required.
- Ran the probe (266 libraries) and inspected the ones relevant to fundamentals.

## Key findings
- **Not available:** Compustat **Global**, Refinitiv **Datastream**, **Worldscope** → Compustat NA
  and CRSP only reach US-listed holdings (a minority of the book).
- **Available:** Compustat North America, CRSP, IBES (incl. international files), Execucomp, Bank,
  Snapshot, Historical Segments.
- **🎯 The win:** **`factset_ff_int` — FactSet Fundamentals International — is accessible and
  queryable** (96 tables: the `ff_advanced_*` files for Asia-Pacific & Europe). **`ciq`** (Capital
  IQ, 258 tables) is a second global-fundamentals option. So the international-fundamentals gap is
  **solvable within existing access** — no Compustat Global purchase needed.
- **Cleared:** the capstone advisor approved use of any WRDS item we have access to.

## Next
Build the fundamentals layer from `factset_ff_int`. (First, the quick factors fix — report 6.)
