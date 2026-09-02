# 5-Star Fund Rating — Design (sub-project #4)

**Date:** 2026-07-30
**Status:** Approved. Ready for implementation planning.
**Depends on:** #1 (holdings ingestion, built but unfed), #2 (`dashboard/etf_prices.py`), #3 (`dashboard/etf_analytics.py`) — all DONE.
**Roadmap:** `docs/superpowers/specs/2026-07-22-etf-dashboard-replatform-roadmap.md` ("Rating design (sub-project #4)")

## Goal

Score any fund 0–5 stars on risk-adjusted absolute performance plus holdings
diversification, and surface it in the dashboard.

The scoring math was locked during the roadmap brainstorm and is carried here
unchanged. This spec settles the questions the roadmap left open: where the
numbers come from, and what happens when they are missing.

## Findings that shaped this design

Both were verified against the real data before any design was settled.

### 1. There is no ETF holdings data

`data/processed/etf_holdings.csv` does not exist and `data/raw/` is empty.
Sub-project #1 built the ingestion machinery, but the iShares endpoint is gated
behind a sign-on interstitial (see the STATUS doc's operational gotcha), so
nothing has ever been ingested.

The diversification composite is **25% of the locked formula** and has **no
source for any ETF**. The roadmap's fallback — redistribute the weight to
performance and note "rated on performance only" — is therefore not an edge
case today. It is the only path for every ETF.

FUND is the exception: its diversification *is* computable from the engineered
layer, and doing so is what keeps the 25% component exercised on real data
rather than shipping as untested code.

### 2. `combined_monthly.csv`'s FUND returns are corrupted

`fund_ret` is exactly `0.0375` for all six months **2025-01 → 2025-06**, while
the benchmark columns vary normally across the same months. No other value
repeats more than twice in 129 rows, and yfinance reports ≈0.000 for those
months. It is a placeholder, not a return series.

The effect on the rating is decisive:

| FUND return source | 1y score | Composite | Stars |
|---|---|---|---|
| CSV as-is (with placeholders) | 100.0 | 42.6 | 2.5 |
| CSV trimmed to 2024-12 | 23 | 7.5 | 1.0 |
| yfinance (chosen) | 23.9 | 15.5 | **1.5** |

A perfect 100.0 one-year score built on six identical fabricated +3.75% months
is not defensible in a published rating.

⚠️ **This design routes around the bug; it does not fix it.** The placeholder
months still affect FUND's factor/regime tabs from #3. Repairing
`combined_monthly.csv` in the `capstone_data` pipeline is recommended
follow-up work, tracked separately.

## Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Return source | **yfinance for every ticker**, FUND included | Identical methodology for every star, and the corrupted CSV months never enter. `combined_monthly.csv` remains authoritative for the factor/regime tabs; only the rating changes source. |
| Return frequency | **Hybrid** — monthly Sharpe/return, daily max drawdown | Monthly returns badly understate drawdown (EFA 2020-03: −14% monthly vs ≈−33% true peak-to-trough), and drawdown carries 25% weight. Sharpe stays monthly so it aligns with the monthly `RF` column with no resampling approximation on the heaviest-weighted metric. |
| Diversification | Build the scorer; wire FUND from the engineered layer; ETFs take the fallback | Keeps 25% of the rating exercised on real data. ETFs have no holdings source until the iShares gate is solved. |
| Risk-free rate | Ken French `RF` from the market frame | The roadmap said "FRED 3-mo default", but it *also* requires no network in the scoring path. `RF` is already joined into every frame by #3, is point-in-time, and is monthly decimal — matching the return series. Adding FRED would violate the roadmap's own constraint. |
| UI | New **⭐ Rating** tab, reusing #3's `ticker_selector` | Consistent with #3; #5 later promotes the selector to global. |

## Scoring (carried from the roadmap, unchanged)

All thresholds are piecewise-linear and clamped to [0, 100].

### Performance composite

Windows, blended by weight; a window counts only if it has its full observation
count. Missing windows drop out and surviving weights renormalize.

| Window | Months | Weight |
|---|---|---|
| 1 year | 12 | 0.20 |
| 3 years | 36 | 0.40 |
| 5 years | 60 | 0.40 |

Per window, three metrics:

| Metric | Weight | 0 pts | 100 pts | Frequency |
|---|---|---|---|---|
| Sharpe ratio | 55% | ≤ 0.0 | ≥ 1.5 | monthly |
| Annualized return | 20% | ≤ 0% | ≥ 15% | monthly |
| Max drawdown | 25% | ≤ −45% | ≥ −5% | **daily** |

Volatility is computed and displayed but **not scored** — Sharpe already prices
it in.

### Diversification composite

| Metric | Weight | 0 pts | 100 pts |
|---|---|---|---|
| Top-10 concentration (Σ top-10 weights) | 50% | ≥ 60% | ≤ 20% |
| Sector spread (normalized Shannon entropy) | 50% | ≤ 0.50 | ≥ 0.90 |

Normalized entropy is `-Σ w·ln(w) / ln(n_sectors)` over positive sector weights
renormalized to sum to 1. With a single sector `ln(1) = 0`, so the normalizer is
undefined; that case returns entropy **0.0** (a one-sector fund is maximally
concentrated, which scores 0) rather than dividing by zero.

### Blend and stars

```
final  = 0.75 · performance + 0.25 · diversification
       = performance                      (when diversification is unavailable)
stars  = clamp(round(final / 100 * 5 * 2) / 2, 0.5, 5.0)
```

Under 12 months of history → **not rated** (`stars = None`), which is distinct
from zero stars: an unrated fund is not a bad fund.

## Architecture

```
etf_prices.load(ticker) ──┐
                          ├─→ etf_analytics.rating_returns(t) ──┐   (I/O)
market frame `RF` ────────┘                                     │
                                                                ▼
position_values_monthly.csv ──┐                        fund_rating.rate(...)   (PURE)
portfolio_sector_country.csv  ├─→ diversification_inputs(t) ─┘        │
etf_holdings.csv (absent)  ───┘                                       ▼
                                                              Rating(stars, …)
```

`fund_rating.py` is **pure** — every function takes series or dataclasses and
returns numbers, with no I/O and no Streamlit. All loading lives in
`etf_analytics.py`, which already owns "what data exists for this ticker".
`app.py` wires the two. `etf_analytics` imports `fund_rating` for the
dataclasses; never the reverse, so there is no import cycle.

### `dashboard/fund_rating.py` (pure)

| Name | Responsibility |
|---|---|
| `linear_score(x, lo, hi)` | Clamped piecewise-linear 0–100. Supports `hi < lo` for the inverted drawdown and concentration scales. `NaN` in → `NaN` out. |
| `WindowMetrics` | `sharpe`, `ann_return`, `max_drawdown`, `ann_vol`, `n_months`. |
| `window_metrics(monthly_ret, rf, daily_close=None)` | Computes the four metrics. Sharpe/return/vol from the monthly series; drawdown from `daily_close` clipped to the window, falling back to the monthly series when `daily_close is None`. |
| `score_window(m)` | The 55/20/25 blend for one window. |
| `performance_composite(windows)` | Weight-renormalizing blend across available windows. |
| `DiversificationInputs` | `top10_weight`, `sector_entropy`, `n_holdings`, `as_of`. |
| `diversification_composite(d)` | The 50/50 blend. |
| `final_score(perf, div)` | 0.75/0.25, or `perf` alone when `div is None`. |
| `stars(final)` | Half-star mapping with a 0.5 floor; `None` when `final` is `NaN`. |
| `Rating` | `stars`, `final`, `performance`, `diversification`, `windows`, `performance_only`, `as_of`, `not_rated_reason`. |
| `rate(monthly, daily_close, div_inputs)` | Orchestrates the above into a `Rating`. |

### `dashboard/etf_analytics.py` (additions, I/O)

| Name | Responsibility |
|---|---|
| `rating_returns(ticker)` | Returns a **2-tuple** `(monthly, daily_close)`: a `(month, fund_ret, RF)` DataFrame from `etf_prices.load(ticker).ret_monthly` inner-joined to the market frame's `RF`, and the `close_daily` Series. Uses the **ungated `load()`**, not `load_etf`/`validate`, so FUND (a MUTUALFUND) works. |
| `diversification_inputs(ticker)` | `DiversificationInputs` or `None`. FUND resolves from `position_values_monthly.csv` (top-10 weight by `value_usd`) and `portfolio_sector_country_monthly.csv` (sector entropy), at the latest month present in both. Other tickers read `etf_holdings.csv` when it exists, else return `None`. |

## Error handling and degradation

- `rate()` **never raises** on thin or degenerate data. It returns a `Rating`
  with `stars=None` and a `not_rated_reason` string.
- Zero-variance returns make Sharpe undefined (`NaN`); that window is skipped
  rather than scored 0, so a flat series does not masquerade as a bad one.
- Network and lookup failures surface at the `etf_analytics` boundary as the
  existing `EtfDataError` family, rendered by the tab the same way #3 does.
- `is_stale` from `etf_prices` is surfaced as a warning, per #2/#3 convention.
- **Diversification staleness must be labeled.** FUND's holdings end 2022-12
  while its performance windows end at the latest traded month (2025-09 at time
  of writing) — roughly a three-year gap. The UI shows the diversification
  as-of date beside the score. Presenting a 2022 diversification number as
  current would misrepresent the fund.

## Expected output

Verified by prototype against live data, using the locked thresholds:

| Ticker | 1y | 3y | 5y | Performance | Diversification | Final | Stars |
|---|---|---|---|---|---|---|---|
| FUND | 23.9 | 26.9 | 0.0 | 15.5 | 63.7 (as of 2022-12) | 27.6 | **1.5** |
| EFA | 95.5 | 69.1 | 36.6 | 61.4 | — | 61.4 | **3.0** |
| SCZ | 96.0 | 64.6 | 19.4 | 52.8 | — | 52.8 | **2.5** |
| VSS | 95.9 | 67.8 | 25.5 | 56.5 | — | 56.5 | **3.0** |

FUND's diversification inputs, verified: 64 positions, top-10 = 47.9%
(score 30.2); 10 sectors, normalized entropy = 0.889 (score 97.2).

⚠️ **The capstone fund rates below all three of its benchmarks.** That is a
data-supported result consistent with the fund's 3–5 year underperformance, but
it is a prominent number in a capstone deliverable and should not be a surprise
at the end.

## Testing

Deterministic and no-network, per repo convention: tests put `dashboard/` on
`sys.path` and monkeypatch `etf_prices`. Beyond the happy path:

- **Threshold boundaries** — exactly 0.0 and 1.5 Sharpe, exactly −45% and −5%
  drawdown, exactly 20% and 60% concentration, exactly 0.50 and 0.90 entropy.
- **Clamping** past both ends of every scale.
- **Inverted scales** (`hi < lo`) score in the right direction.
- **Weight renormalization** when a window is missing; a fund with only a 1y
  window scores on that window alone.
- **Star mapping** — half-star rounding at boundaries, the 0.5 floor, the 5.0
  ceiling, and `None` for `NaN`.
- **Not rated** under 12 months, with a reason string.
- **Performance-only fallback** when `div_inputs is None`, and that
  `final == performance` in that case.
- **Hybrid frequency** — a series with a sharp intra-month trough scores a
  worse drawdown from daily closes than from monthly returns.
- **Zero-variance** returns skip the window instead of scoring 0.
- **FUND diversification loader** reproduces top-10 = 47.9% and entropy = 0.889
  at 2022-12 (reads local CSVs; deterministic and network-free).
- **`diversification_inputs` returns `None`** for a ticker with no holdings.
- **Single-sector entropy** returns 0.0 rather than raising on `ln(1) = 0`.

## Files

| File | Status | Responsibility |
|---|---|---|
| `dashboard/fund_rating.py` | Create | Pure scoring: scorers, window metrics, composites, star mapping |
| `tests/test_fund_rating.py` | Create | Deterministic no-network tests |
| `dashboard/etf_analytics.py` | Modify | `rating_returns`, `diversification_inputs` |
| `dashboard/app.py` | Modify | ⭐ Rating tab |
| `dashboard/README.md` | Modify | Document the module |
| `docs/superpowers/etf-replatform-STATUS.md` | Modify | Mark sub-project #4 done |

## Out of scope

- **Repairing `combined_monthly.csv`'s placeholder months.** Recommended
  follow-up in the `capstone_data` pipeline; the rating routes around it.
- **Solving the iShares holdings gate.** Until it is solved, or CSVs are
  downloaded by hand per #1's documented workaround, every ETF is rated on
  performance only. The diversification path is built and tested, and lights up
  the moment `etf_holdings.csv` appears.
- Peer/category-relative ranking — the rating is deliberately absolute.
- The global ETF selector and re-pointing the other tabs (#5).

## Repo conventions

- `uv run pytest`, `uv run python`.
- `dashboard/` must not import `capstone_data`; intra-dashboard imports are bare
  (`import fund_rating`, not `from dashboard import fund_rating`).
- Stage only files you changed — never `git add -A`; `.idea/` entries stay unstaged.
- Never put AI/Claude/Anthropic attribution in a commit message.
- ⚠️ Local `uv` is 0.7.3, older than whatever wrote `uv.lock` (`revision = 3`).
  Running `uv` rewrites the lock down to `revision = 2`. Do not commit that
  churn; run `git checkout -- uv.lock` if it appears.
