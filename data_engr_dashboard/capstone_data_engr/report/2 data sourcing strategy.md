# 2. Data sourcing strategy & access plan

_Date: 2026-06-15 · Status: complete_

## Goal
Decide where each enrichment input comes from, and what (if anything) we need to request access to.

## Two principles (now enforced in code)
1. **Point-in-time / no look-ahead.** Use each value *as it was known on that date*. Free tools
   (yfinance/OpenBB) return *today's* fundamentals — using them for a 2019 trade is look-ahead bias.
2. **Survivorship / delisting.** Sources must include delisted & acquired names, or the analysis
   quietly drops exactly the trades most likely to explain alpha.

## What we did
- Catalogued each need as free vs. paid/sponsor and wrote `../capstone/todo/data-access-requests.md`
  (what to ask the school vs. the fund, and what's already free).

## Key findings
- **Free, no ask needed:** macro (FRED), FX, Fama–French factors (Ken French), and prices/returns
  for currently-listed names.
- **Needs paid/institutional or sponsor:** international point-in-time fundamentals, returns/data
  for delisted names, and a security master / identifier mapping.
- **WRDS** is the key resource to pursue for fundamentals; the fund (sponsor) is the only source for
  authoritative point-in-time holdings.

## Next
Stand up the repo and build the free layers first (report 3); pursue WRDS in parallel (report 5).
