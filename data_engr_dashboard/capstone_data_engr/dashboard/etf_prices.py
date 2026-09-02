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
import yfinance as _yf

_HERE = Path(__file__).resolve().parent

# Matches capstone_data.config.START_DATE. Redefined rather than imported to
# keep dashboard/ free of pipeline imports (see module docstring).
START_DATE = "2000-01-01"

# The dashboard is frozen to end at December 2024 (mirrors ``data.LAST_MONTH``).
# Redefined here, not imported, for the same reason as START_DATE. Bars dated
# later are dropped in :func:`load`, so the live Yahoo pull ends at the same
# month as the bundled CSVs and no view ever shows post-2024 prices.
LAST_DATE = "2024-12-31"

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
    # Enforce the December-2024 data freeze (see LAST_DATE). Everything below
    # derives from `close`, so clipping here bounds close_daily, both return
    # grains, and as_of in one place.
    close = close[close.index <= pd.Timestamp(LAST_DATE)]
    if close.empty:
        raise NoPriceHistory(
            f"no price history for {ticker} on or before {LAST_DATE}")
    return EtfPrices(
        meta=load_meta(ticker),
        close_daily=close,
        ret_daily=daily_returns(close),
        ret_monthly=monthly_returns(close),
        as_of=close.index[-1],
        is_stale=is_stale,
    )


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
