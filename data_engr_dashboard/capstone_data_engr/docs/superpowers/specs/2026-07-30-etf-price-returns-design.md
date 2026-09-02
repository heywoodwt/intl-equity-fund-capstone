# ETF Price/Returns + Universe Layer — Design (sub-project #2)

**Date:** 2026-07-30
**Branch:** `heywood_s_version`
**Roadmap:** `docs/superpowers/specs/2026-07-22-etf-dashboard-replatform-roadmap.md`
**Status:** Design approved. Ready for plan.

Sub-project #2 of the ETF dashboard re-platform. Supplies price history and return
series for an arbitrary user-selected ETF, and decides what counts as a supported
ticker. Feeds #3 (factor analytics recompute) and #4 (the 5-star rating).

## Scope

**In:** yfinance price fetch, ETF validation/metadata, daily + monthly return
computation, a two-tier cache, and the error contract consumers rely on.

**Out:** the rating itself and `dashboard/fund_rating.py` (#4); wiring returns into
the factor/rolling/Kalman/regime machinery (#3); the global ETF selector and tab
re-pointing (#5); holdings, which #1 already delivers via
`data/processed/etf_holdings.csv`.

## Decisions

Five design decisions were settled during brainstorming. Each has consequences that
the implementation must honour.

### 1. Fully open universe — no registry

Any yfinance ticker is selectable. There is **no** curated list and **no**
precomputed returns table in `data/processed/`. This departs from the repo's
batch-CSV convention deliberately: the roadmap's goal is arbitrary ETFs, and a
registry would mean editing `config.py` to add one.

Consequences the implementation must handle:

- Every selection is a live network call, so caching is load-bearing rather than an
  optimization.
- There is no reproducible on-disk snapshot of the universe. Results may shift
  between sessions when Yahoo revises history. Acceptable; not worked around.
- `config.ISHARES_ETFS` (from #1) stays what it is — the holdings universe, not the
  price universe. The two are intentionally decoupled: a ticker can have prices
  without holdings, which is exactly the case #4's rating already handles by
  redistributing the diversification weight to performance.

### 2. Both daily and monthly grain

One fetch of daily bars; monthly derived from it.

- **Daily** feeds #4's Sharpe and max drawdown. Monthly data materially understates
  drawdown — a trough within a month is invisible to month-end sampling — and #4
  weights max drawdown at 25%.
- **Monthly** feeds #3, because Ken French factors exist only monthly, and matches
  the pipeline's `month` (`YYYY-MM`) join key.

Monthly uses the same month-end convention already in `capstone_data/yahoo.py`:
resample daily closes to `"ME"` and take the last observation, rather than
yfinance's own `1mo` bars, which are stamped month-start.

### 3. Two-tier cache

In-process memo (Streamlit) over a per-ticker disk cache.

| Layer | Location | TTL | Rationale |
|---|---|---|---|
| Memory | `st.cache_data` on `load_etf()` | 12 h | hot path; slider/tab interactions |
| Disk — prices | `<cache_dir>/{TICKER}.parquet` | 1 day | survives restart; offline fallback |
| Disk — metadata | `<cache_dir>/{TICKER}_info.json` | 30 days | expense ratio/name barely move |

`cache_dir()` resolves in this order, mirroring `data.py:data_dir()`:

1. `$CAPSTONE_ETF_CACHE_DIR` — explicit override
2. `<repo>/data/interim/etf_prices/` when the repo layout is present (local dev)
3. `dashboard/.cache/etf_prices/` — the lifted-deploy fallback

`data/interim/` is already git-ignored, so nothing downloaded gets committed;
`dashboard/.cache/` must be added to `.gitignore`. Freshness is judged by file
mtime, so no timestamp is embedded in the payload.

The cache is self-filling from whatever tickers users actually request — it is a
cache, not a registry, and never constrains what is selectable.

### 4. Placement: entirely in `dashboard/etf_prices.py`

**Revised 2026-07-30.** The original decision was a split — logic in
`capstone_data/etf_prices.py`, a thin Streamlit wrapper in `dashboard/`. That was
made without knowledge of an invariant documented in `dashboard/data.py`:

> Self-contained: imports nothing from the pipeline package … This is what makes the
> `dashboard/` folder liftable into a deploy repo on its own.

The invariant holds today — no file in `dashboard/` imports `capstone_data`. The
split would have been the first to break it, so the module moves wholly into
`dashboard/`.

- `dashboard/etf_prices.py` — everything: fetch, cache, resample, returns, plus a
  small `load_etf()` wrapper applying `st.cache_data`. Streamlit is imported
  **lazily inside that wrapper**, so importing the module in a test never requires
  Streamlit.
- Intra-dashboard imports are bare (`import etf_prices`), matching
  `dashboard/kalman_tvp.py`'s `from tvp_kalman_filter import KalmanFilter`.

This also matches the closest existing analogues — `analytics.py` and
`kalman_tvp.py` are dashboard-local computation modules with deterministic tests.

Because the module can no longer import `capstone_data.config`, it defines its own
`START_DATE = "2000-01-01"` (matching `config.START_DATE`) and resolves its cache
directory locally via `cache_dir()`, mirroring the existing `data_dir()` pattern.

### 5. Strict ETF validation via `.info`

A ticker is accepted only if yfinance `.info` reports `quoteType == "ETF"`.
Metadata (`longName`, expense ratio, AUM, category) is pulled in the same call for
display.

`.info` is the least reliable part of yfinance — slow, sometimes incomplete,
sometimes rate-limited. Two mitigations are required, not optional:

1. Metadata is cached on disk for 30 days, so a given ticker is gated once per month
   rather than once per session.
2. When `.info` fails but a **cached, previously-validated** metadata record exists,
   the cached record is used. A ticker that validated last week does not become
   unusable because Yahoo rate-limited this request.

A ticker with no cached record and a failing `.info` is rejected with a clear error.

**Consequence — the gate applies to user-selected tickers only.** `quoteType == "ETF"`
excludes **FUND** (`MUTUALFUND`) and `^GSPC` (`INDEX`), both of which the current
dashboard depends on. Internal benchmark and index series bypass `validate()` and
call the fetch/return functions directly. `validate()` is invoked by the selector
path in #5, not by `load()`'s internals, so this separation is structural rather
than a convention to remember.

## Module surface

`dashboard/etf_prices.py`:

| Function | Kind | Contract |
|---|---|---|
| `fetch_daily_close(ticker, start=None)` | network | `auto_adjust=True` daily close as a `pd.Series`; total return, dividend-adjusted. Raises `NoPriceHistory` on an empty frame. |
| `fetch_info(ticker)` | network | Normalized metadata subset. Raises `TickerNotFound` when yfinance yields nothing. |
| `validate(ticker)` | network | Returns `EtfMeta`; raises `NotAnEtf` when `quoteType != "ETF"`. |
| `daily_returns(close)` | **pure** | Simple returns from the close series. |
| `monthly_returns(close)` | **pure** | Month-end resample (`"ME"`, last) then simple returns. |
| `load(ticker, start=None)` | orchestrator | Cache-aware. Returns `EtfPrices(meta, close_daily, ret_daily, ret_monthly, as_of, is_stale)`. |

`start` defaults to the module's `START_DATE` (`2000-01-01`), giving ample history
for #4's 5-year window.

**`load()` does not enforce the ETF gate.** It fetches metadata on a best-effort
basis — on `.info` failure with no cached record, `meta` is `None` and the load
still succeeds on price data alone. Only `validate()` enforces `quoteType`, and only
the #5 selector path calls it. This is what lets `^GSPC` and FUND flow through
`load()` for internal use while user-entered tickers are still gated. `load()` raises
only when *price* data is unavailable.

Returns are decimals (0.0123 = 1.23%), consistent with the rest of the pipeline.

## Error contract

One hierarchy, so consumers can distinguish "bad input" from "Yahoo is down":

```
EtfDataError
├── TickerNotFound   — yfinance knows nothing about this symbol
├── NotAnEtf         — resolves, but quoteType != "ETF"
└── NoPriceHistory   — resolves, but returned no usable bars
```

`TickerNotFound` and `NotAnEtf` come from the metadata path, so in practice they
surface from `validate()`. `load()` can raise only `NoPriceHistory` (or return stale
cached data), since it treats metadata as best-effort.

**Degradation:** when the network fails and a disk cache entry exists, the cached
data is returned with `is_stale=True` and `as_of` set to the cache timestamp, rather
than raising. The caller decides how to surface staleness; #5's UI is expected to
show a notice. A failure with no cache entry raises.

## Testing

`tests/test_etf_prices.py` — deterministic and no-network, following the
**dashboard** test convention already used by `test_analytics.py:6` and
`test_kalman_tvp.py:6`: `sys.path.insert(0, <repo>/dashboard)` then a bare
`import etf_prices as P`. Streamlit is never imported, because `load_etf()` imports
it lazily.

Pure functions, tested against synthetic series:

- `monthly_returns` picks the true month-end close, including a month whose last
  calendar day is not a trading day.
- `monthly_returns` over a daily series spanning a month boundary equals the
  month-end-to-month-end return.
- `daily_returns` drops the leading NaN and matches hand-computed values.
- Both handle a single-observation and an empty series without raising.

Orchestration and errors, with `fetch_*` monkeypatched:

- `load` returns a fresh result and writes both cache files.
- `load` reads from disk within TTL without calling the network.
- `load` re-fetches once the price TTL has lapsed.
- Network failure with a warm cache returns `is_stale=True` rather than raising.
- Network failure with a cold cache raises `EtfDataError`.
- `validate` raises `NotAnEtf` for `quoteType` of `MUTUALFUND` and `INDEX`
  (covering the FUND and `^GSPC` cases explicitly).
- `.info` failure with a cached validated record uses the cached record.
- `load` succeeds with `meta=None` when `.info` fails and no metadata is cached,
  provided price data is available.
- Empty price frame raises `NoPriceHistory`.

## Conventions

- `uv run python ...` / `uv run pytest`.
- `dashboard/` stays liftable: no imports from `capstone_data`, and Streamlit
  imported lazily so the module is testable without it.
- Point-in-time / no look-ahead remains a core principle. Note that this layer is
  explicitly *not* point-in-time — it fetches current adjusted history — which is
  correct for performance measurement but must not be fed into any historical
  attribution path.
- Commits: stage only the files changed; never `git add -A` (the `.idea/` files in
  the working tree are unrelated). No AI attribution in commit messages.