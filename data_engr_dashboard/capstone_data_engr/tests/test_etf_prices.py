"""Deterministic checks for ETF price/return loading (no network, no Streamlit)."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import pandas as pd
import pytest

import etf_prices as P


def test_cache_dir_respects_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_ETF_CACHE_DIR", str(tmp_path / "c"))
    assert P.cache_dir() == tmp_path / "c"
    assert P.cache_dir().exists()


def test_cache_dir_uses_repo_interim_when_present(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPSTONE_ETF_CACHE_DIR", raising=False)
    fake_here = tmp_path / "repo" / "dashboard"
    fake_here.mkdir(parents=True)
    (tmp_path / "repo" / "data" / "interim").mkdir(parents=True)
    monkeypatch.setattr(P, "_HERE", fake_here)

    result = P.cache_dir()

    assert result == tmp_path / "repo" / "data" / "interim" / "etf_prices"
    assert result.exists()


def test_cache_dir_falls_back_to_deploy_cache_when_no_repo_interim(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPSTONE_ETF_CACHE_DIR", raising=False)
    fake_here = tmp_path / "repo" / "dashboard"
    fake_here.mkdir(parents=True)
    monkeypatch.setattr(P, "_HERE", fake_here)

    result = P.cache_dir()

    assert result == fake_here / ".cache" / "etf_prices"
    assert result.exists()


def test_error_hierarchy():
    for exc in (P.TickerNotFound, P.NotAnEtf, P.NoPriceHistory):
        assert issubclass(exc, P.EtfDataError)


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
    # check_freq=False: parquet does not round-trip DatetimeIndex.freq, and no
    # consumer depends on it (real yfinance data carries no freq either).
    pd.testing.assert_series_equal(back, close, check_names=False, check_freq=False)


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


def test_load_clips_prices_to_the_december_2024_freeze(cache, monkeypatch):
    """A series running into 2025 is truncated at LAST_DATE, so close_daily,
    as_of, and the monthly grain all stop at December 2024."""
    idx = pd.date_range("2024-11-01", "2025-03-31", freq="D")
    close = pd.Series(range(1, len(idx) + 1), index=idx, dtype=float)
    monkeypatch.setattr(P, "fetch_daily_close", lambda t, start=None: close)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)

    out = P.load("EFA")
    assert out.close_daily.index.max() <= pd.Timestamp(P.LAST_DATE)
    assert out.as_of == pd.Timestamp("2024-12-31")
    assert out.ret_monthly.index.max() == pd.Timestamp("2024-12-31")


def test_load_raises_when_all_prices_are_after_the_freeze(cache, monkeypatch):
    """A ticker that only traded in 2025 has no in-window data to serve."""
    idx = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    close = pd.Series(range(1, len(idx) + 1), index=idx, dtype=float)
    monkeypatch.setattr(P, "fetch_daily_close", lambda t, start=None: close)
    monkeypatch.setattr(P, "fetch_info", lambda t: ETF_META)
    with pytest.raises(P.NoPriceHistory, match="2024-12-31"):
        P.load("EFA")


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
