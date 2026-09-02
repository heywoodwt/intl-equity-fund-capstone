"""Deterministic checks for holdings-based attribution (no network).

Run with pytest (`python -m pytest`) or directly
(`python tests/test_attribution.py`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from capstone_data import attribution as A


def _positions():
    """Two names over three months: A in EUR (FX moves), B in USD (FX = 1).

    A: 10 sh @ EUR 100 (notional 1000 EUR); usd_per_ccy 1.1 -> 1.2 -> 1.05.
    B: 5 sh @ USD 200 (value 1000 USD), constant.
    """
    rows = [
        # ticker, fsym_id, month, ccy, price_m, shares, value_usd
        ("A", "A", "2020-01", "EUR", 100.0, 10, 1100.0),
        ("A", "A", "2020-02", "EUR", 100.0, 10, 1200.0),
        ("A", "A", "2020-03", "EUR", 100.0, 10, 1050.0),
        ("B", "B", "2020-01", "USD", 200.0, 5, 1000.0),
        ("B", "B", "2020-02", "USD", 200.0, 5, 1000.0),
        ("B", "B", "2020-03", "USD", 200.0, 5, 1000.0),
    ]
    return pd.DataFrame(rows, columns=["ticker", "fsym_id", "month", "iso_currency",
                                       "price_m", "Shares", "value_usd"])


def _stock_returns():
    rows = [  # ret_factset is PERCENT local; ret_price decimal local (unused here)
        ("A", "2020-02", "EUR", 5.0, 0.05),
        ("A", "2020-03", "EUR", -10.0, -0.10),
        ("B", "2020-02", "USD", 2.0, 0.02),
        ("B", "2020-03", "USD", 0.0, 0.0),
    ]
    return pd.DataFrame(rows, columns=["fsym_id", "month", "iso_currency",
                                       "ret_factset", "ret_price"])


def _profile():
    return pd.DataFrame({"fsym_id": ["A", "B"], "sector": ["Tech", "Finance"],
                         "country": ["DE", "US"]})


def test_implied_fx_move():
    fx = A.implied_fx_move(_positions()).set_index(["iso_currency", "month"])["fx_move"]
    assert np.isclose(fx[("EUR", "2020-02")], 1.2 / 1.1)
    assert np.isclose(fx[("EUR", "2020-03")], 1.05 / 1.2)
    assert np.isclose(fx[("USD", "2020-02")], 1.0)         # USD never moves


def test_security_usd_returns_combine_local_and_fx():
    r = A.security_usd_returns(_positions(), _stock_returns()).set_index(["fsym_id", "month"])
    assert np.isclose(r.loc[("A", "2020-02"), "ret_usd"], (1.05) * (1.2 / 1.1) - 1)
    assert np.isclose(r.loc[("A", "2020-03"), "ret_usd"], (0.90) * (1.05 / 1.2) - 1)
    assert np.isclose(r.loc[("B", "2020-02"), "ret_usd"], 0.02)


def test_begin_weights_sum_to_one_and_use_prior_month():
    w = A.begin_weights(_positions())
    # No weights in the first month (no prior).
    assert (w["month"] == "2020-01").sum() == 0
    by_month = w.groupby("month")["w_begin"].sum()
    assert np.allclose(by_month.values, 1.0)
    # 2020-02 weights come from 2020-01 values: A=1100, B=1000 -> 1100/2100.
    wa = w[(w.month == "2020-02") & (w.fsym_id == "A")]["w_begin"].iloc[0]
    assert np.isclose(wa, 1100.0 / 2100.0)


def test_contributions_sum_to_reconstructed_return():
    sec = A.security_contributions(_positions(), _stock_returns(), _profile())
    sect = A.sector_returns(sec)
    # Hand-computed 2020-02 reconstructed gross.
    wa, wb = 1100 / 2100, 1000 / 2100
    ra = (1.05) * (1.2 / 1.1) - 1
    rb = 0.02
    expected = wa * ra + wb * rb
    got = sect[sect.month == "2020-02"]["contrib"].sum()
    assert np.isclose(got, expected)
    # Single-name sector return equals that name's USD return.
    tech = sect[(sect.month == "2020-02") & (sect.sector == "Tech")].iloc[0]
    assert np.isclose(tech["sector_ret"], ra)
    assert np.isclose(tech["begin_wt"], wa)


def test_reconcile_residual_ties_to_fund_ret():
    sec = A.security_contributions(_positions(), _stock_returns(), _profile())
    sect = A.sector_returns(sec)
    fund = pd.Series({"2020-02": 0.10, "2020-03": -0.20})
    rec = A.reconcile(sect, fund).set_index("month")
    for mth in ("2020-02", "2020-03"):
        assert np.isclose(rec.loc[mth, "recon_gross"] + rec.loc[mth, "residual"],
                          rec.loc[mth, "fund_ret"])


def test_brinson_effects_sum_to_active():
    port = pd.DataFrame({
        "month": ["M", "M"], "sector": ["S1", "S2"],
        "begin_wt": [0.6, 0.4], "sector_ret": [0.10, 0.00]})
    bench = pd.DataFrame({
        "month": ["M", "M"], "sector": ["S1", "S2"],
        "begin_wt": [0.5, 0.5], "sector_ret": [0.08, 0.02]})
    attr = A.brinson(port, bench).set_index("sector")
    # Hand values: r_b = 0.05.
    assert np.isclose(attr.loc["S1", "allocation"], 0.1 * 0.03)
    assert np.isclose(attr.loc["S1", "selection"], 0.5 * 0.02)
    assert np.isclose(attr.loc["S1", "interaction"], 0.1 * 0.02)
    # Effects sum to active = r_p - r_b = 0.06 - 0.05 = 0.01.
    assert np.isclose(attr["active"].sum(), 0.01)


def test_blend_benchmark_sectors():
    bench = pd.DataFrame({
        "benchmark": ["efa", "scz"], "month": ["M", "M"], "sector": ["S1", "S1"],
        "begin_wt": [0.4, 0.6], "sector_ret": [0.10, 0.20]})
    blend = A.blend_benchmark_sectors(bench, benches=("efa", "scz")).iloc[0]
    assert np.isclose(blend["begin_wt"], (0.4 + 0.6) / 2)          # mean ETF weight
    assert np.isclose(blend["sector_ret"], (0.4 * 0.10 + 0.6 * 0.20) / 1.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("ok: all attribution tests passed")
