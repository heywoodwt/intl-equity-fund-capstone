"""Deterministic checks for the ETF analytics frame (no network, no Streamlit)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import pandas as pd
import pytest

import data as D
import etf_analytics as EA
import etf_prices

D_DIR = D.data_dir()


def test_market_frame_month_range_and_uniqueness():
    mf = EA.load_market_frame()
    assert mf["month"].is_unique
    assert mf["month"].min() == "1990-07"
    assert mf["month"].max() == "2024-12"        # December-2024 data freeze
    assert len(mf) == 414


def test_market_frame_has_all_four_source_families():
    mf = EA.load_market_frame()
    for col in ("Mkt_RF", "SMB", "RF", "Mom"):        # Fama-French
        assert col in mf.columns
    for col in ("vix", "y10", "cpi_yoy", "sp500_ret"):  # macro
        assert col in mf.columns
    for col in ("PC1", "PC2"):                         # PCA
        assert col in mf.columns
    for col in ("efa_ret", "scz_ret", "vss_ret"):      # benchmark ETFs
        assert col in mf.columns


def test_market_frame_has_no_date_column():
    """`date` is derived per-frame by load_analytics_frame, not carried here."""
    assert "date" not in EA.load_market_frame().columns


def test_market_frame_coverage_matches_sources():
    mf = EA.load_market_frame()
    assert mf["Mkt_RF"].notna().sum() == 414
    assert mf["PC1"].notna().sum() == 123
    assert mf["efa_ret"].notna().sum() == 124


def test_market_frame_matches_combined_monthly_exactly():
    """Every column the market frame shares with combined_monthly.csv must be
    identical over their overlap. If this fails, the ETF path and the FUND
    path have diverged and every cross-ticker comparison is suspect."""
    mf = EA.load_market_frame()
    combined = pd.read_csv(D_DIR / "combined_monthly.csv")

    shared = [c for c in mf.columns if c in combined.columns and c != "month"]
    assert len(shared) == 27, f"expected 27 shared columns, got {len(shared)}"

    joined = combined[["month"] + shared].merge(
        mf[["month"] + shared], on="month", suffixes=("_c", "_m"))
    assert len(joined) == 123

    mismatched = []
    for col in shared:
        diff = (joined[f"{col}_c"] - joined[f"{col}_m"]).abs().max()
        if pd.notna(diff) and diff >= 1e-9:
            mismatched.append((col, diff))
    assert mismatched == [], f"columns diverged: {mismatched}"


def _monthly(values, start="2020-01-31"):
    """Month-end-stamped monthly returns, the shape etf_prices returns."""
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


class _FakePrices:
    """Stand-in for etf_prices.EtfPrices carrying only what we consume."""

    def __init__(self, ret_monthly, is_stale=False, as_of=None):
        self.ret_monthly = ret_monthly
        self.is_stale = is_stale
        # Real EtfPrices stamps as_of from the daily close; for these tests the
        # last monthly point is close enough and keeps the fixtures short.
        self.as_of = as_of if as_of is not None else (
            ret_monthly.index[-1] if len(ret_monthly) else None)


def test_etf_monthly_returns_maps_month_end_to_month_key(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01, -0.02, 0.03])))
    out = EA.etf_monthly_returns("efa")
    assert list(out.columns) == ["month", "fund_ret"]
    assert list(out["month"]) == ["2020-01", "2020-02", "2020-03"]
    assert out["fund_ret"].tolist() == [0.01, -0.02, 0.03]


def test_etf_monthly_returns_uppercases_ticker(monkeypatch):
    seen = []
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: seen.append(t) or _FakePrices(_monthly([0.01])))
    EA.etf_monthly_returns("efa")
    assert seen == ["EFA"]


def test_etf_monthly_returns_on_empty_series(monkeypatch):
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(empty))
    out = EA.etf_monthly_returns("EFA")
    assert out.empty
    assert list(out.columns) == ["month", "fund_ret"]


def _frame(**cols):
    """Minimal frame with a `month` key plus whatever columns a test needs."""
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"month": [f"2020-{i + 1:02d}" for i in range(n)], **cols})


def test_derive_returns_computes_all_four_identities():
    df = _frame(fund_ret=[0.05], RF=[0.01],
                efa_ret=[0.02], scz_ret=[0.03], vss_ret=[0.04])
    out = EA.derive_returns(df)
    assert out["excess_ret"].iloc[0] == pytest.approx(0.04)
    assert out["bench_avg_ret"].iloc[0] == pytest.approx(0.03)
    assert out["alpha_vs_avg"].iloc[0] == pytest.approx(0.02)
    assert out["alpha_vs_efa"].iloc[0] == pytest.approx(0.03)


def test_derive_returns_bench_avg_is_nan_when_any_leg_missing():
    """skipna=False: a partial month must not yield a 2-ETF blended benchmark."""
    df = _frame(fund_ret=[0.05], RF=[0.01],
                efa_ret=[0.02], scz_ret=[float("nan")], vss_ret=[0.04])
    out = EA.derive_returns(df)
    assert pd.isna(out["bench_avg_ret"].iloc[0])
    assert pd.isna(out["alpha_vs_avg"].iloc[0])
    assert out["alpha_vs_efa"].iloc[0] == pytest.approx(0.03)   # still computable


def test_derive_returns_adds_month_start_date():
    out = EA.derive_returns(_frame(fund_ret=[0.05, 0.01], RF=[0.0, 0.0],
                                   efa_ret=[0.0, 0.0], scz_ret=[0.0, 0.0],
                                   vss_ret=[0.0, 0.0]))
    assert out["date"].tolist() == [pd.Timestamp("2020-01-01"),
                                    pd.Timestamp("2020-02-01")]


def test_derive_returns_does_not_mutate_input():
    df = _frame(fund_ret=[0.05], RF=[0.01],
                efa_ret=[0.02], scz_ret=[0.03], vss_ret=[0.04])
    EA.derive_returns(df)
    assert "excess_ret" not in df.columns


def test_load_analytics_frame_fund_returns_the_combined_frame():
    """The capstone fund must keep its exact numbers and holdings regressors."""
    out = EA.load_analytics_frame("FUND")
    assert len(out) == 123
    assert any(c.startswith("ff_") for c in out.columns)
    assert any(c.startswith("sect_wt_") for c in out.columns)
    assert "PC1" in out.columns


def test_load_analytics_frame_fund_is_case_insensitive():
    assert len(EA.load_analytics_frame("fund")) == 123


def test_load_analytics_frame_etf_has_no_holdings_columns(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    out = EA.load_analytics_frame("EFA")
    assert not [c for c in out.columns if c.startswith("ff_")]
    assert not [c for c in out.columns if c.startswith("sect_wt_")]
    assert not [c for c in out.columns if c.startswith("ctry_wt_")]


def test_load_analytics_frame_etf_joins_onto_market_data(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    out = EA.load_analytics_frame("EFA")
    assert len(out) == 24
    assert out["month"].iloc[0] == "2020-01"
    for col in ("Mkt_RF", "vix", "PC1", "efa_ret", "date",
                "excess_ret", "bench_avg_ret", "alpha_vs_avg", "alpha_vs_efa"):
        assert col in out.columns
    assert out["fund_ret"].notna().all()


def test_load_analytics_frame_etf_keeps_only_months_with_returns(monkeypatch):
    """Inner join: no rows for months the ticker did not trade."""
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01, 0.02])))
    out = EA.load_analytics_frame("EFA")
    assert len(out) == 2


def test_load_analytics_frame_etf_pre_pca_months_have_nan_pcs(monkeypatch):
    """Ragged coverage: PCs start 2014-10, factors far earlier."""
    monkeypatch.setattr(
        EA.etf_prices, "load_etf",
        lambda t, start=None: _FakePrices(_monthly([0.01] * 6, start="2005-01-31")))
    out = EA.load_analytics_frame("EFA")
    assert out["PC1"].isna().all()
    assert out["Mkt_RF"].notna().all()   # factors reach back to 1990


def test_load_analytics_frame_propagates_price_errors(monkeypatch):
    def boom(t, start=None):
        raise etf_prices.NoPriceHistory("no price history for ZZZZ")
    monkeypatch.setattr(EA.etf_prices, "load_etf", boom)
    with pytest.raises(etf_prices.NoPriceHistory):
        EA.load_analytics_frame("ZZZZ")


def test_price_status_reports_staleness(monkeypatch):
    stale = _FakePrices(_monthly([0.01, 0.02]), is_stale=True)
    monkeypatch.setattr(EA.etf_prices, "load_etf", lambda t, start=None: stale)
    is_stale, as_of = EA.price_status("EFA")
    assert is_stale is True
    assert as_of == pd.Timestamp("2020-02-29")


def test_price_status_is_never_stale_for_the_capstone_fund():
    """FUND comes from a local CSV, so the yfinance cache never applies."""
    assert EA.price_status("FUND") == (False, None)


def test_regressor_catalog_fund_keeps_holdings_families_and_preset():
    frame = EA.load_analytics_frame("FUND")
    cat = EA.regressor_catalog(frame)
    assert D.ALPHA_DRIVERS_PRESET_NAME in cat
    assert "Portfolio fundamentals" in cat
    assert "Sector weights" in cat


def test_regressor_catalog_etf_drops_holdings_and_swaps_preset(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    frame = EA.load_analytics_frame("EFA")
    cat = EA.regressor_catalog(frame)

    assert "Portfolio fundamentals" not in cat
    assert "Sector weights" not in cat
    assert "Country weights" not in cat
    assert D.ALPHA_DRIVERS_PRESET_NAME not in cat
    assert EA.ETF_ALPHA_PRESET_NAME in cat
    assert cat[EA.ETF_ALPHA_PRESET_NAME] == ["Mkt_RF", "SMB", "HML", "PC1", "PC2"]


def test_regressor_catalog_never_returns_empty_groups(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    cat = EA.regressor_catalog(EA.load_analytics_frame("EFA"))
    assert all(cols for cols in cat.values())


def test_regressor_catalog_etf_preset_columns_all_exist(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    frame = EA.load_analytics_frame("EFA")
    cat = EA.regressor_catalog(frame)
    for col in cat[EA.ETF_ALPHA_PRESET_NAME]:
        assert col in frame.columns


def test_rating_returns_gives_monthly_with_rf_and_daily(monkeypatch):
    monthly = _monthly([0.01] * 24)
    daily = pd.Series(range(1, 101), dtype=float,
                      index=pd.date_range("2020-01-01", periods=100, freq="D"))

    class _P:
        ret_monthly = monthly
        close_daily = daily
        is_stale = False
        as_of = daily.index[-1]

    monkeypatch.setattr(EA.etf_prices, "load", lambda t, start=None: _P())
    m, d = EA.rating_returns("EFA")
    assert list(m.columns) == ["month", "fund_ret", "RF"]
    assert len(m) == 24
    assert m["RF"].notna().all()
    assert d is daily


def test_rating_returns_uses_the_ungated_load(monkeypatch):
    """FUND is a MUTUALFUND; validate_etf would reject it."""
    called = []

    class _P:
        ret_monthly = _monthly([0.01] * 12)
        close_daily = pd.Series([1.0, 2.0],
                                index=pd.date_range("2020-01-01", periods=2))
        is_stale = False
        as_of = None

    monkeypatch.setattr(EA.etf_prices, "load",
                        lambda t, start=None: called.append(t) or _P())
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: pytest.fail("must not use load_etf"))
    EA.rating_returns("FUND")
    assert called == ["FUND"]


def test_diversification_inputs_fund_matches_verified_values():
    d = EA.diversification_inputs("FUND")
    assert d is not None
    assert d.as_of == "2022-12"
    assert d.n_holdings == 64
    assert d.top10_weight == pytest.approx(0.479, abs=0.005)
    assert d.sector_entropy == pytest.approx(0.889, abs=0.005)


def test_diversification_inputs_is_none_without_holdings():
    """No etf_holdings.csv exists, so every ETF takes the fallback."""
    assert EA.diversification_inputs("EFA") is None


def test_sector_entropy_is_zero_for_a_single_sector():
    assert EA.sector_entropy(pd.Series([1.0])) == pytest.approx(0.0)


def test_sector_entropy_is_one_for_perfectly_even_weights():
    assert EA.sector_entropy(pd.Series([0.25] * 4)) == pytest.approx(1.0)


def test_sector_entropy_ignores_zero_and_negative_weights():
    even = EA.sector_entropy(pd.Series([0.5, 0.5]))
    padded = EA.sector_entropy(pd.Series([0.5, 0.5, 0.0, -0.1]))
    assert padded == pytest.approx(even)
