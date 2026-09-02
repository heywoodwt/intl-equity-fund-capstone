# ETF Price/Returns + Universe Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the dashboard daily and monthly total-return series for any user-entered ETF ticker, with strict ETF validation and a cache that keeps working when Yahoo does not.

**Architecture:** One self-contained module, `dashboard/etf_prices.py`. Pure functions (return computation) are separated from network functions (`fetch_*`) and from a two-tier cache (in-process Streamlit memo over per-ticker files on disk). `load()` orchestrates and degrades to stale cache on network failure; `validate()` separately enforces `quoteType == "ETF"` and is called only by the ticker-selector path. Streamlit is imported lazily so tests never need it.

**Tech Stack:** Python 3.13, pandas 2.3, yfinance 1.4, pyarrow (parquet), pytest 8, `uv` for running everything.

**Spec:** `docs/superpowers/specs/2026-07-30-etf-price-returns-design.md`

---

## Critical context for the implementer

Read this before Task 1. Three repo rules cause silent breakage if violated:

1. **`dashboard/` must not import `capstone_data`.** `dashboard/data.py`'s docstring states the folder is "liftable into a deploy repo on its own." No dashboard file imports the pipeline package. Do not add the first one. This is why constants like `START_DATE` are redefined locally instead of imported.
2. **Intra-dashboard imports are bare.** Write `import etf_prices`, not `from dashboard import etf_prices`. Precedent: `dashboard/kalman_tvp.py:20` does `from tvp_kalman_filter import KalmanFilter`.
3. **Tests for dashboard modules put `dashboard/` on `sys.path`.** Precedent: `tests/test_analytics.py:6`. Tests are deterministic and never touch the network.

**Commit rules for this repo:** stage only files you changed — never `git add -A`. The `.idea/` files in the working tree are unrelated and must stay unstaged. Never put AI/Claude/Anthropic attribution in a commit message.

**Run everything with `uv`:** `uv run pytest`, `uv run python`.

## File structure

| File | Status | Responsibility |
|---|---|---|
| `dashboard/etf_prices.py` | Create | Everything: errors, dataclasses, pure return math, yfinance fetch, disk cache, `load`, `validate`, `load_etf` |
| `tests/test_etf_prices.py` | Create | Deterministic no-network tests |
| `.gitignore` | Modify | Ignore `dashboard/.cache/` |
| `pyproject.toml` | Modify | Declare `pyarrow` in the `dashboard` dependency group |
| `dashboard/README.md` | Modify | Document the module and its cache |

The module is one file because its parts are small and change together (a new cached field touches the dataclass, the writer, and the reader at once). Expected final size ~230 lines.

---

### Task 1: Scaffolding — dependency, gitignore, module skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `dashboard/etf_prices.py`
- Create: `tests/test_etf_prices.py`

- [ ] **Step 1: Declare pyarrow in the dashboard dependency group**

`pyarrow` is currently only a transitive dependency of Streamlit. The parquet cache depends on it directly, so declare it. In `pyproject.toml`, change the `dashboard` group (lines 17-22) to:

```toml
dashboard = [
    "plotly>=6.8.0",
    "pyarrow>=15.0",
    "scikit-learn>=1.7.2",
    "statsmodels>=0.14.6",
    "streamlit>=1.58.0",
]
```

- [ ] **Step 2: Ignore the deploy-mode cache directory**

Append to `.gitignore`:

```
dashboard/.cache/
```

- [ ] **Step 3: Create the module with errors, constants, and cache-dir resolution**

Create `dashboard/etf_prices.py`:

```python
"""ETF price history and return series from Yahoo Finance.

Self-contained by design: imports nothing from the pipeline package, so
``dashboard/`` stays liftable into a deploy repo on its own. Streamlit is
imported lazily inside :func:`load_etf` so this module is importable (and
testable) without it.

Two-tier cache. An in-process Streamlit memo sits over per-ticker files on
disk, so a cold start or a Yahoo outage reads from disk instead of failing.
The cache fills itself from whatever tickers users request -- it is a cache,
not a registry, and never constrains what is selectable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent

# Matches capstone_data.config.START_DATE. Redefined rather than imported to
# keep dashboard/ free of pipeline imports (see module docstring).
START_DATE = "2000-01-01"

PRICE_TTL = timedelta(days=1)
INFO_TTL = timedelta(days=30)


class EtfDataError(Exception):
    """Base for anything that stops us returning usable ETF data."""


class TickerNotFound(EtfDataError):
    """Yahoo knows nothing about this symbol."""


class NotAnEtf(EtfDataError):
    """Symbol resolves, but is not an ETF."""


class NoPriceHistory(EtfDataError):
    """Symbol resolves, but returned no usable price bars."""


def cache_dir() -> Path:
    """Locate the writable cache directory, creating it if needed.

    Resolution order mirrors ``data.py:data_dir()``:
      1. ``$CAPSTONE_ETF_CACHE_DIR``            (explicit override)
      2. ``<repo>/data/interim/etf_prices/``    (local dev; git-ignored)
      3. ``dashboard/.cache/etf_prices/``       (lifted deploy)
    """
    env = os.environ.get("CAPSTONE_ETF_CACHE_DIR")
    if env:
        path = Path(env)
    else:
        interim = _HERE.parent / "data" / "interim"
        path = (interim / "etf_prices" if interim.exists()
                else _HERE / ".cache" / "etf_prices")
    path.mkdir(parents=True, exist_ok=True)
    return path
```

- [ ] **Step 4: Create the test file with the dashboard sys.path preamble**

Create `tests/test_etf_prices.py`:

```python
"""Deterministic checks for ETF price/return loading (no network, no Streamlit)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import pandas as pd
import pytest

import etf_prices as P


def test_cache_dir_respects_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_ETF_CACHE_DIR", str(tmp_path / "c"))
    assert P.cache_dir() == tmp_path / "c"
    assert P.cache_dir().exists()


def test_error_hierarchy():
    for exc in (P.TickerNotFound, P.NotAnEtf, P.NoPriceHistory):
        assert issubclass(exc, P.EtfDataError)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: PASS, 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore dashboard/etf_prices.py tests/test_etf_prices.py
git commit -m "feat(dashboard): scaffold etf_prices module with cache dir and errors"
```

---

### Task 2: Pure return computation

The only functions with no I/O. Everything downstream depends on them, so they are built and tested first.

**Files:**
- Modify: `dashboard/etf_prices.py`
- Test: `tests/test_etf_prices.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_prices.py`:

```python
def _closes(values, start="2020-01-01"):
    """Daily close series on consecutive calendar days."""
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_daily_returns_values_and_no_leading_nan():
    out = P.daily_returns(_closes([100.0, 110.0, 99.0]))
    assert len(out) == 2
    assert out.iloc[0] == pytest.approx(0.10)
    assert out.iloc[1] == pytest.approx(-0.10)


def test_monthly_returns_uses_true_month_end_close():
    # Jan has 31 daily points, Feb 29 (2020 is a leap year).
    idx = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    close = pd.Series(range(1, len(idx) + 1), index=idx, dtype=float)
    out = P.monthly_returns(close)
    # One return: Jan month-end (31.0) -> Feb month-end (60.0).
    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(60.0 / 31.0 - 1.0)


def test_monthly_returns_when_month_end_is_not_a_trading_day():
    # Business days only: January 2021 ends on Friday the 29th.
    idx = pd.date_range("2021-01-01", "2021-02-26", freq="B")
    close = pd.Series(100.0, index=idx)
    close.loc["2021-01-29"] = 123.0   # last trading day of January
    close.loc["2021-02-26"] = 150.0   # last trading day of February
    out = P.monthly_returns(close)
    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(150.0 / 123.0 - 1.0)


def test_returns_on_single_observation_are_empty():
    one = _closes([100.0])
    assert P.daily_returns(one).empty
    assert P.monthly_returns(one).empty


def test_returns_on_empty_series_are_empty():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert P.daily_returns(empty).empty
    assert P.monthly_returns(empty).empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: FAIL with `AttributeError: module 'etf_prices' has no attribute 'daily_returns'`

- [ ] **Step 3: Implement the pure functions**

Append to `dashboard/etf_prices.py`:

```python
# --- pure return math ------------------------------------------------------

def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns from an adjusted close series (leading NaN dropped)."""
    if close.empty:
        return close.astype(float)
    return close.pct_change().dropna()


def monthly_returns(close: pd.Series) -> pd.Series:
    """Simple month-end-to-month-end returns from a daily close series.

    Resamples to calendar month-end and takes the last observation, so a month
    whose final calendar day is not a trading day still uses that month's last
    traded price. This matches ``capstone_data/yahoo.py``; yfinance's own
    ``1mo`` bars are stamped month-start and are not used.
    """
    if close.empty:
        return close.astype(float)
    return close.resample("ME").last().pct_change().dropna()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: PASS, 7 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_prices.py tests/test_etf_prices.py
git commit -m "feat(dashboard): add pure daily/monthly return computation"
```

---

### Task 3: Disk cache primitives

Read/write/freshness for the two cached artifacts. Still no network.

**Files:**
- Modify: `dashboard/etf_prices.py`
- Test: `tests/test_etf_prices.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_prices.py`:

```python
import os
import time


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the module's cache at a temp dir for the duration of a test."""
    monkeypatch.setenv("CAPSTONE_ETF_CACHE_DIR", str(tmp_path))
    return tmp_path


def _age_file(path, days):
    """Backdate a file's mtime by `days` so TTL checks see it as stale."""
    old = time.time() - days * 86400
    os.utime(path, (old, old))


def test_price_cache_roundtrip(cache):
    close = _closes([100.0, 101.0, 102.0])
    P.write_price_cache("efa", close)
    back = P.read_price_cache("EFA")
    assert back is not None
    pd.testing.assert_series_equal(back, close, check_names=False)


def test_price_cache_missing_returns_none(cache):
    assert P.read_price_cache("NOPE") is None


def test_info_cache_roundtrip(cache):
    meta = P.EtfMeta(ticker="EFA", long_name="iShares MSCI EAFE",
                     quote_type="ETF", expense_ratio=0.0033,
                     aum=5.0e10, category="Foreign Large Blend")
    P.write_info_cache("EFA", meta)
    assert P.read_info_cache("EFA") == meta


def test_is_fresh_true_for_new_file_false_when_aged(cache):
    close = _closes([100.0, 101.0])
    P.write_price_cache("EFA", close)
    path = P.price_path("EFA")
    assert P.is_fresh(path, P.PRICE_TTL)
    _age_file(path, days=3)
    assert not P.is_fresh(path, P.PRICE_TTL)


def test_is_fresh_false_for_missing_file(cache):
    assert not P.is_fresh(P.price_path("GHOST"), P.PRICE_TTL)


def test_cache_paths_are_ticker_case_insensitive(cache):
    assert P.price_path("efa") == P.price_path("EFA")
    assert P.info_path("efa") == P.info_path("EFA")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: FAIL with `AttributeError: module 'etf_prices' has no attribute 'write_price_cache'`

- [ ] **Step 3: Implement the dataclass and cache primitives**

Append to `dashboard/etf_prices.py`:

```python
# --- cached artifacts ------------------------------------------------------

@dataclass(frozen=True)
class EtfMeta:
    """Identity/description fields pulled from yfinance ``.info``."""
    ticker: str
    long_name: str | None = None
    quote_type: str | None = None
    expense_ratio: float | None = None
    aum: float | None = None
    category: str | None = None


def price_path(ticker: str) -> Path:
    return cache_dir() / f"{ticker.upper()}.parquet"


def info_path(ticker: str) -> Path:
    return cache_dir() / f"{ticker.upper()}_info.json"


def is_fresh(path: Path, ttl: timedelta, now: datetime | None = None) -> bool:
    """True when `path` exists and its mtime is within `ttl`."""
    if not path.exists():
        return False
    now = now or datetime.now()
    return (now - datetime.fromtimestamp(path.stat().st_mtime)) < ttl


def write_price_cache(ticker: str, close: pd.Series) -> Path:
    path = price_path(ticker)
    close.rename("close").to_frame().to_parquet(path)
    return path


def read_price_cache(ticker: str) -> pd.Series | None:
    """Cached close series, or None when absent/unreadable."""
    path = price_path(ticker)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)["close"]
    except Exception:       # corrupt/partial file -> treat as a cache miss
        return None


def write_info_cache(ticker: str, meta: EtfMeta) -> Path:
    path = info_path(ticker)
    path.write_text(json.dumps(asdict(meta)))
    return path


def read_info_cache(ticker: str) -> EtfMeta | None:
    """Cached metadata, or None when absent/unreadable."""
    path = info_path(ticker)
    if not path.exists():
        return None
    try:
        return EtfMeta(**json.loads(path.read_text()))
    except Exception:       # corrupt file or schema drift -> cache miss
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: PASS, 13 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_prices.py tests/test_etf_prices.py
git commit -m "feat(dashboard): add on-disk price/metadata cache primitives"
```

---

### Task 4: Network fetch functions

Thin wrappers over yfinance, isolated so every other test can monkeypatch them.

**Files:**
- Modify: `dashboard/etf_prices.py`
- Test: `tests/test_etf_prices.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_prices.py`:

```python
class _FakeYF:
    """Stand-in for the yfinance module surface we use."""

    def __init__(self, frame=None, info=None):
        self._frame = frame
        self._info = info

    def download(self, *a, **k):
        return self._frame

    def Ticker(self, ticker):                     # noqa: N802 - mirrors yfinance
        outer = self

        class _T:
            @property
            def info(self):
                return outer._info
        return _T()


def _price_frame(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=idx, dtype=float)


def test_fetch_daily_close_returns_named_series(monkeypatch):
    monkeypatch.setattr(P, "_yf", _FakeYF(frame=_price_frame([1.0, 2.0])))
    out = P.fetch_daily_close("efa")
    assert out.name == "EFA"
    assert list(out) == [1.0, 2.0]


def test_fetch_daily_close_empty_frame_raises(monkeypatch):
    monkeypatch.setattr(P, "_yf", _FakeYF(frame=pd.DataFrame()))
    with pytest.raises(P.NoPriceHistory, match="EFA"):
        P.fetch_daily_close("EFA")


def test_fetch_daily_close_all_nan_raises(monkeypatch):
    monkeypatch.setattr(P, "_yf", _FakeYF(frame=_price_frame([float("nan")] * 3)))
    with pytest.raises(P.NoPriceHistory):
        P.fetch_daily_close("EFA")


def test_fetch_info_normalizes_fields(monkeypatch):
    monkeypatch.setattr(P, "_yf", _FakeYF(info={
        "longName": "iShares MSCI EAFE ETF",
        "quoteType": "ETF",
        "netExpenseRatio": 0.0033,
        "totalAssets": 5.0e10,
        "category": "Foreign Large Blend",
        "irrelevant": "ignored",
    }))
    meta = P.fetch_info("efa")
    assert meta == P.EtfMeta(ticker="EFA", long_name="iShares MSCI EAFE ETF",
                             quote_type="ETF", expense_ratio=0.0033,
                             aum=5.0e10, category="Foreign Large Blend")


def test_fetch_info_falls_back_to_annual_report_expense_ratio(monkeypatch):
    monkeypatch.setattr(P, "_yf", _FakeYF(info={
        "quoteType": "ETF", "annualReportExpenseRatio": 0.0007,
    }))
    assert P.fetch_info("VTI").expense_ratio == pytest.approx(0.0007)


def test_fetch_info_empty_raises_ticker_not_found(monkeypatch):
    monkeypatch.setattr(P, "_yf", _FakeYF(info={}))
    with pytest.raises(P.TickerNotFound, match="ZZZZ"):
        P.fetch_info("ZZZZ")


def test_fetch_info_quote_type_none_raises_ticker_not_found(monkeypatch):
    monkeypatch.setattr(P, "_yf", _FakeYF(info={"quoteType": "NONE"}))
    with pytest.raises(P.TickerNotFound):
        P.fetch_info("ZZZZ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: FAIL with `AttributeError: module 'etf_prices' has no attribute '_yf'`

- [ ] **Step 3: Implement the fetch functions**

Add the yfinance import near the top of `dashboard/etf_prices.py`, immediately after `import pandas as pd`:

```python
import yfinance as _yf
```

Then append:

```python
# --- network ---------------------------------------------------------------

# yfinance spells the expense ratio differently across quote types; take the
# first key that carries a value.
_EXPENSE_KEYS = ("netExpenseRatio", "annualReportExpenseRatio", "expenseRatio")


def fetch_daily_close(ticker: str, start: str | None = None) -> pd.Series:
    """Adjusted daily close for `ticker` (network).

    ``auto_adjust=True`` yields a total-return series -- dividends and splits
    are already reflected, which is what the performance metrics require.
    """
    ticker = ticker.upper()
    raw = _yf.download(ticker, start=start or START_DATE, interval="1d",
                       auto_adjust=True, progress=False, multi_level_index=False)
    if raw is None or raw.empty or "Close" not in raw:
        raise NoPriceHistory(f"no price history for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):     # defensive: collapse stray 2-D frame
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise NoPriceHistory(f"no usable price bars for {ticker}")
    close.name = ticker
    return close


def fetch_info(ticker: str) -> EtfMeta:
    """Normalized yfinance ``.info`` for `ticker` (network)."""
    ticker = ticker.upper()
    info = _yf.Ticker(ticker).info or {}
    if not info or info.get("quoteType") in (None, "NONE"):
        raise TickerNotFound(f"Yahoo returned no identity for {ticker}")
    expense = next((info[k] for k in _EXPENSE_KEYS if info.get(k) is not None), None)
    return EtfMeta(
        ticker=ticker,
        long_name=info.get("longName"),
        quote_type=info.get("quoteType"),
        expense_ratio=expense,
        aum=info.get("totalAssets"),
        category=info.get("category"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: PASS, 20 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_prices.py tests/test_etf_prices.py
git commit -m "feat(dashboard): add yfinance price and metadata fetch"
```

---

### Task 5: Cached metadata loading and `validate`

`load_meta` is best-effort (never raises); `validate` is strict. This split is what lets `^GSPC` and FUND flow through `load()` while user-entered tickers are gated.

**Files:**
- Modify: `dashboard/etf_prices.py`
- Test: `tests/test_etf_prices.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_prices.py`:

```python
def _boom(*a, **k):
    raise RuntimeError("yahoo is down")


ETF_META = P.EtfMeta(ticker="EFA", long_name="iShares MSCI EAFE",
                     quote_type="ETF", expense_ratio=0.0033)


def test_load_meta_fetches_and_caches(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(P, "fetch_info", lambda t: calls.append(t) or ETF_META)
    assert P.load_meta("EFA") == ETF_META
    assert P.info_path("EFA").exists()
    P.load_meta("EFA")                       # second call served from disk
    assert calls == ["EFA"]


def test_load_meta_returns_none_when_uncached_and_network_fails(cache, monkeypatch):
    monkeypatch.setattr(P, "fetch_info", _boom)
    assert P.load_meta("EFA") is None


def test_load_meta_uses_stale_cache_when_network_fails(cache, monkeypatch):
    P.write_info_cache("EFA", ETF_META)
    _age_file(P.info_path("EFA"), days=90)   # well past INFO_TTL
    monkeypatch.setattr(P, "fetch_info", _boom)
    assert P.load_meta("EFA") == ETF_META


def test_validate_accepts_etf(cache, monkeypatch):
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)
    assert P.validate("EFA") == ETF_META


def test_validate_rejects_mutual_fund(cache, monkeypatch):
    meta = P.EtfMeta(ticker="FUND", quote_type="MUTUALFUND")
    monkeypatch.setattr(P, "fetch_info", lambda t: meta)
    with pytest.raises(P.NotAnEtf, match="MUTUALFUND"):
        P.validate("FUND")


def test_validate_rejects_index(cache, monkeypatch):
    meta = P.EtfMeta(ticker="^GSPC", quote_type="INDEX")
    monkeypatch.setattr(P, "fetch_info", lambda t: meta)
    with pytest.raises(P.NotAnEtf, match="INDEX"):
        P.validate("^GSPC")


def test_validate_raises_when_metadata_unavailable(cache, monkeypatch):
    monkeypatch.setattr(P, "fetch_info", _boom)
    with pytest.raises(P.TickerNotFound):
        P.validate("EFA")


def test_validate_uses_cached_record_when_info_fails(cache, monkeypatch):
    P.write_info_cache("EFA", ETF_META)
    _age_file(P.info_path("EFA"), days=90)
    monkeypatch.setattr(P, "fetch_info", _boom)
    assert P.validate("EFA") == ETF_META     # rate limit must not un-validate
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: FAIL with `AttributeError: module 'etf_prices' has no attribute 'load_meta'`

- [ ] **Step 3: Implement `load_meta` and `validate`**

Append to `dashboard/etf_prices.py`:

```python
# --- metadata + validation -------------------------------------------------

def load_meta(ticker: str) -> EtfMeta | None:
    """Cached metadata for `ticker`, best-effort.

    Never raises: metadata is a display nicety, so a flaky ``.info`` must not
    take down a ticker whose prices fetched cleanly. Falls back to a stale
    cached record, then to None.
    """
    ticker = ticker.upper()
    if is_fresh(info_path(ticker), INFO_TTL):
        cached = read_info_cache(ticker)
        if cached is not None:
            return cached
    try:
        meta = fetch_info(ticker)
    except Exception:
        return read_info_cache(ticker)      # stale record, or None
    write_info_cache(ticker, meta)
    return meta


def validate(ticker: str) -> EtfMeta:
    """Strict ETF gate for user-entered tickers.

    Only the ticker-selector path calls this. ``load`` deliberately does not,
    so internal benchmark/index series (``^GSPC``, FUND) stay loadable.
    """
    meta = load_meta(ticker)
    if meta is None:
        raise TickerNotFound(f"could not resolve {ticker.upper()}")
    if meta.quote_type != "ETF":
        raise NotAnEtf(f"{meta.ticker} is a {meta.quote_type}, not an ETF")
    return meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: PASS, 28 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_prices.py tests/test_etf_prices.py
git commit -m "feat(dashboard): add cached metadata loading and strict ETF validation"
```

---

### Task 6: The `load` orchestrator

Ties prices, cache, and returns together. Raises only on missing price data; degrades to stale cache otherwise.

**Files:**
- Modify: `dashboard/etf_prices.py`
- Test: `tests/test_etf_prices.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_prices.py`:

```python
def test_load_returns_prices_and_both_return_grains(cache, monkeypatch):
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    close = pd.Series(range(1, len(idx) + 1), index=idx, dtype=float)
    monkeypatch.setattr(P, "fetch_daily_close", lambda t, start=None: close)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)

    out = P.load("EFA")
    assert out.meta == ETF_META
    assert not out.is_stale
    assert len(out.ret_daily) == len(close) - 1
    assert len(out.ret_monthly) == 2          # Jan->Feb, Feb->Mar
    assert out.as_of == close.index[-1]


def test_load_writes_price_cache_and_reuses_it(cache, monkeypatch):
    close = _closes([100.0, 101.0, 102.0])
    calls = []
    monkeypatch.setattr(P, "fetch_daily_close",
                        lambda t, start=None: calls.append(t) or close)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)

    P.load("EFA")
    assert P.price_path("EFA").exists()
    P.load("EFA")                             # within PRICE_TTL -> no refetch
    assert calls == ["EFA"]


def test_load_refetches_once_price_ttl_lapses(cache, monkeypatch):
    close = _closes([100.0, 101.0])
    calls = []
    monkeypatch.setattr(P, "fetch_daily_close",
                        lambda t, start=None: calls.append(t) or close)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)

    P.load("EFA")
    _age_file(P.price_path("EFA"), days=3)
    P.load("EFA")
    assert calls == ["EFA", "EFA"]


def test_load_falls_back_to_stale_cache_on_network_failure(cache, monkeypatch):
    close = _closes([100.0, 101.0, 102.0])
    P.write_price_cache("EFA", close)
    _age_file(P.price_path("EFA"), days=30)
    monkeypatch.setattr(P, "fetch_daily_close", _boom)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)

    out = P.load("EFA")
    assert out.is_stale
    assert len(out.close_daily) == 3


def test_load_raises_when_network_fails_and_cache_is_cold(cache, monkeypatch):
    monkeypatch.setattr(P, "fetch_daily_close", _boom)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)
    with pytest.raises(P.EtfDataError, match="EFA"):
        P.load("EFA")


def test_load_propagates_no_price_history(cache, monkeypatch):
    def no_history(t, start=None):
        raise P.NoPriceHistory("no price history for ZZZZ")
    monkeypatch.setattr(P, "fetch_daily_close", no_history)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)
    with pytest.raises(P.NoPriceHistory):
        P.load("ZZZZ")


def test_load_succeeds_with_meta_none_when_info_fails(cache, monkeypatch):
    close = _closes([100.0, 101.0])
    monkeypatch.setattr(P, "fetch_daily_close", lambda t, start=None: close)
    monkeypatch.setattr(P, "fetch_info", _boom)

    out = P.load("EFA")
    assert out.meta is None                   # metadata is best-effort
    assert len(out.ret_daily) == 1


def test_load_does_not_apply_the_etf_gate(cache, monkeypatch):
    """^GSPC is an INDEX; load must still work (only validate gates)."""
    close = _closes([100.0, 101.0])
    monkeypatch.setattr(P, "fetch_daily_close", lambda t, start=None: close)
    monkeypatch.setattr(P, "fetch_info",
                        lambda t: P.EtfMeta(ticker="^GSPC", quote_type="INDEX"))
    out = P.load("^GSPC")
    assert out.meta.quote_type == "INDEX"
    assert len(out.close_daily) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: FAIL with `AttributeError: module 'etf_prices' has no attribute 'load'`

- [ ] **Step 3: Implement `EtfPrices` and `load`**

Append to `dashboard/etf_prices.py`:

```python
# --- orchestration ---------------------------------------------------------

@dataclass(frozen=True)
class EtfPrices:
    """Everything a consumer needs for one ticker.

    ``is_stale`` is True when the network failed and cached data was served
    instead; ``as_of`` is then the last date in that cached series.
    """
    meta: EtfMeta | None
    close_daily: pd.Series
    ret_daily: pd.Series
    ret_monthly: pd.Series
    as_of: pd.Timestamp
    is_stale: bool


def _load_close(ticker: str, start: str | None) -> tuple[pd.Series, bool]:
    """Cached-or-fetched close series, plus whether it is stale."""
    if is_fresh(price_path(ticker), PRICE_TTL):
        cached = read_price_cache(ticker)
        if cached is not None:
            return cached, False
    try:
        close = fetch_daily_close(ticker, start=start)
    except Exception as exc:
        stale = read_price_cache(ticker)
        if stale is not None:
            return stale, True
        if isinstance(exc, EtfDataError):
            raise
        raise EtfDataError(f"could not load prices for {ticker}: {exc}") from exc
    write_price_cache(ticker, close)
    return close, False


def load(ticker: str, start: str | None = None) -> EtfPrices:
    """Price history and return series for `ticker`.

    Does **not** enforce the ETF gate -- see :func:`validate`. Metadata is
    best-effort, so this raises only when price data is unavailable.
    """
    ticker = ticker.upper()
    close, is_stale = _load_close(ticker, start)
    return EtfPrices(
        meta=load_meta(ticker),
        close_daily=close,
        ret_daily=daily_returns(close),
        ret_monthly=monthly_returns(close),
        as_of=close.index[-1],
        is_stale=is_stale,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: PASS, 36 passed.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, 72 passed (36 pre-existing + 36 new).

- [ ] **Step 6: Commit**

```bash
git add dashboard/etf_prices.py tests/test_etf_prices.py
git commit -m "feat(dashboard): add load orchestrator with stale-cache degradation"
```

---

### Task 7: Streamlit memo wrapper

The only Streamlit-aware code. Imported lazily so the module stays testable without Streamlit installed.

**Files:**
- Modify: `dashboard/etf_prices.py`
- Test: `tests/test_etf_prices.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_etf_prices.py`:

```python
def test_module_import_does_not_require_streamlit():
    """dashboard/etf_prices.py must be importable without Streamlit."""
    src = (Path(__file__).resolve().parents[1] / "dashboard" / "etf_prices.py").read_text()
    top_level = [ln for ln in src.splitlines()
                 if ln.startswith("import streamlit") or ln.startswith("from streamlit")]
    assert top_level == [], f"streamlit imported at module scope: {top_level}"


def test_load_etf_delegates_to_load(cache, monkeypatch):
    close = _closes([100.0, 101.0])
    monkeypatch.setattr(P, "fetch_daily_close", lambda t, start=None: close)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)
    out = P.load_etf("EFA")
    assert out.meta == ETF_META
    assert len(out.ret_daily) == 1


def test_validate_etf_delegates_to_validate(cache, monkeypatch):
    monkeypatch.setattr(P, "fetch_info",
                        lambda t: P.EtfMeta(ticker="FUND", quote_type="MUTUALFUND"))
    with pytest.raises(P.NotAnEtf):
        P.validate_etf("FUND")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: FAIL with `AttributeError: module 'etf_prices' has no attribute 'load_etf'`

- [ ] **Step 3: Implement the wrapper**

Append to `dashboard/etf_prices.py`:

```python
# --- Streamlit surface -----------------------------------------------------
# Streamlit is imported inside the function, not at module scope, so this file
# stays importable by tests (and by any non-Streamlit consumer).

def load_etf(ticker: str, start: str | None = None) -> EtfPrices:
    """``load`` behind Streamlit's 12-hour in-process memo.

    Falls back to an uncached call when Streamlit is unavailable, so the
    module behaves identically outside a Streamlit runtime.
    """
    try:
        import streamlit as st
    except ImportError:
        return load(ticker, start=start)

    @st.cache_data(ttl=12 * 3600, show_spinner=False)
    def _cached(t: str, s: str | None) -> EtfPrices:
        return load(t, start=s)

    return _cached(ticker, start)


def validate_etf(ticker: str) -> EtfMeta:
    """``validate`` behind the same memo."""
    try:
        import streamlit as st
    except ImportError:
        return validate(ticker)

    @st.cache_data(ttl=12 * 3600, show_spinner=False)
    def _cached(t: str) -> EtfMeta:
        return validate(t)

    return _cached(ticker)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_prices.py -v`
Expected: PASS, 39 passed.

**If `load_etf` fails here**, it is because Streamlit *is* installed in this repo's dashboard group, so the real `st.cache_data` path runs. Two known failure modes and their fixes:

- `CachedStFunctionWarning` / missing `ScriptRunContext` noise: harmless outside a Streamlit runtime; the test still passes. Leave it.
- `UnhashableParamError` on the `start: str | None` argument: pass it positionally as already written (`_cached(ticker, start)`) — do not rename the inner params, and do not reach for `hash_funcs`. Both arguments are plain `str`/`None` and hash fine.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_prices.py tests/test_etf_prices.py
git commit -m "feat(dashboard): add Streamlit-memoized load_etf/validate_etf"
```

---

### Task 8: Documentation

**Files:**
- Modify: `dashboard/README.md`
- Modify: `docs/superpowers/etf-replatform-STATUS.md`

- [ ] **Step 1: Document the module in the dashboard README**

Append a section to `dashboard/README.md`:

```markdown
## ETF price/returns (`etf_prices.py`)

Loads daily and monthly total-return series for any ticker from Yahoo Finance.

- `load_etf(ticker)` -> `EtfPrices(meta, close_daily, ret_daily, ret_monthly, as_of, is_stale)`
- `validate_etf(ticker)` -> `EtfMeta`; raises `NotAnEtf` for anything whose
  `quoteType` is not `ETF`. Call this on user input only — `load_etf` deliberately
  skips the gate so index/mutual-fund series (`^GSPC`, FUND) still load.

Cached on disk (prices 1 day, metadata 30 days) under `$CAPSTONE_ETF_CACHE_DIR`,
else `data/interim/etf_prices/`, else `dashboard/.cache/etf_prices/`. When Yahoo is
unreachable but a cached copy exists, the cached data is returned with
`is_stale=True` — surface that in the UI rather than presenting it as current.
```

- [ ] **Step 2: Mark sub-project #2 done in the STATUS doc**

In `docs/superpowers/etf-replatform-STATUS.md`, change the sub-project #2 row of the status table from `⬜ not started` to `✅ **DONE**`, and update **Last updated** to `2026-07-30`. In the "How to resume" section, change the recommendation from #2 to **#3 or #4** (both are unblocked once #2 lands).

- [ ] **Step 3: Run the full suite one final time**

Run: `uv run pytest -q`
Expected: PASS, 75 passed (36 pre-existing + 39 new).

- [ ] **Step 4: Commit**

```bash
git add dashboard/README.md docs/superpowers/etf-replatform-STATUS.md
git commit -m "docs: document etf_prices module and mark sub-project #2 done"
```

---

## Done when

- `uv run pytest -q` reports 75 passed.
- `dashboard/etf_prices.py` contains no module-scope `import streamlit`.
- No file in `dashboard/` imports `capstone_data` (verify: `grep -rn "capstone_data" dashboard/` returns nothing).
- `git status --short` shows only the `.idea/` entries that were already there.
