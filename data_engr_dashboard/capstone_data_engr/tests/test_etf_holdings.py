"""Deterministic checks for iShares holdings ingestion (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from capstone_data import etf_holdings as H

# Real-shaped iShares export: text preamble, blank line, header, 2 equity rows,
# 1 cash/derivative row. Market values are quoted with thousands separators;
# the cash row has "-" for ticker.
SAMPLE = b'''iShares MSCI EAFE ETF
Fund Holdings as of,"Jul 18, 2026"
Inception Date,"Aug 14, 2001"
Shares Outstanding,"1,000,000"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,CUSIP,ISIN,SEDOL,Price,Location,Exchange,Currency
NESN,NESTLE SA,Consumer Staples,Equity,"1,234,567.89","2.15","1,234,567.89","12,345",-,CH0038863350,-,85.20,Switzerland,SIX Swiss Exchange,CHF
ASML,ASML HOLDING NV,Information Technology,Equity,"2,000,000.00","3.50","2,000,000.00","5,000",-,NL0010273215,-,700.00,Netherlands,Euronext,EUR
-,BLK CSH FND TREASURY,Cash and/or Derivatives,Money Market,"500,000.00","0.30","500,000.00","500,000",-,-,-,1.00,United States,-,USD
'''


def test_parse_schema_and_row_count():
    df = H.parse(SAMPLE, "EFA")
    assert list(df.columns) == H.COLUMNS
    assert len(df) == 3
    assert (df["etf_ticker"] == "EFA").all()


def test_parse_as_of_date():
    df = H.parse(SAMPLE, "EFA")
    assert (df["as_of_date"] == "2026-07-18").all()


def test_parse_numeric_coercion():
    df = H.parse(SAMPLE, "EFA").set_index("name")
    row = df.loc["NESTLE SA"]
    assert row["market_value"] == pytest.approx(1234567.89)
    assert row["weight"] == pytest.approx(2.15)
    assert row["shares"] == pytest.approx(12345.0)


def test_parse_cash_row_kept_and_ticker_blanked():
    df = H.parse(SAMPLE, "EFA").set_index("name")
    cash = df.loc["BLK CSH FND TREASURY"]
    assert cash["asset_class"] == "Money Market"
    assert cash["constituent_ticker"] == ""


SAMPLE_WITH_FOOTER = SAMPLE + (
    b"\n"
    b"The information is provided for illustrative purposes, not investment advice.\n"
    b"BlackRock Inc,all,rights,reserved,with,many,extra,commas,far,beyond,the,header,width,indeed,truly\n"
)


def test_parse_ignores_trailing_disclaimer_footer():
    df = H.parse(SAMPLE_WITH_FOOTER, "EFA")
    assert len(df) == 3
    assert (df["etf_ticker"] == "EFA").all()
    assert not df["name"].str.contains("illustrative", case=False).any()
    assert not df["name"].str.contains("BlackRock", case=False).any()


def test_parse_missing_header_raises():
    with pytest.raises(ValueError, match="header row not found"):
        H.parse(b"just some text\nno,constituent,table\n", "EFA")


def test_build_concats_and_skips_failures(monkeypatch):
    def fake_fetch(ticker, url):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return SAMPLE
    monkeypatch.setattr(H, "fetch", fake_fetch)

    df, failures = H.build({"EFA": "u1", "BAD": "u2"})
    assert list(df["etf_ticker"].unique()) == ["EFA"]
    assert len(df) == 3
    assert "BAD" in failures and "boom" in failures["BAD"]


def test_build_all_fail_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(H, "fetch",
                        lambda t, u: (_ for _ in ()).throw(RuntimeError("x")))
    df, failures = H.build({"EFA": "u1"})
    assert df.empty
    assert list(df.columns) == H.COLUMNS
    assert set(failures) == {"EFA"}


def test_write_roundtrip(tmp_path):
    df = H.parse(SAMPLE, "EFA")
    out = H.write(df, tmp_path / "etf_holdings.csv")
    assert out.exists()
    back = pd.read_csv(out)
    assert list(back.columns) == H.COLUMNS
    assert len(back) == 3


def test_ticker_from_filename():
    assert H._ticker_from_filename("EFA_holdings.csv") == "EFA"
    assert H._ticker_from_filename("/downloads/iemg.csv") == "IEMG"
    assert H._ticker_from_filename("ACWX-holdings-20260718.csv") == "ACWX"


def test_build_from_files_parses_directory(tmp_path):
    (tmp_path / "EFA_holdings.csv").write_bytes(SAMPLE)
    df, failures = H.build_from_files(tmp_path)
    assert failures == {}
    assert list(df["etf_ticker"].unique()) == ["EFA"]
    assert len(df) == 3


def test_build_from_files_records_parse_failures(tmp_path):
    (tmp_path / "BAD_holdings.csv").write_bytes(b"not a holdings file\n")
    df, failures = H.build_from_files(tmp_path)
    assert df.empty
    assert list(df.columns) == H.COLUMNS
    assert "BAD" in failures