"""Deterministic checks for the dashboard analytics (no network, no Streamlit)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import numpy as np
import pandas as pd

import analytics as A


def _df(returns, bench=None, rf=None, start="2018-01"):
    n = len(returns)
    dates = pd.period_range(start, periods=n, freq="M").to_timestamp()
    d = {"date": dates, "fund_ret": returns}
    if bench is not None:
        d["bench_ret"] = bench
    if rf is not None:
        d["RF"] = rf
    return pd.DataFrame(d)


def test_total_return_and_cagr_constant():
    df = _df([0.01] * 24)
    r = A.series(df, "fund_ret")
    assert np.isclose(A.total_return(r), 1.01 ** 24 - 1)
    assert np.isclose(A.cagr(r), 1.01 ** 12 - 1)       # 24 months -> annualized
    assert np.isclose(A.ann_vol(r), 0.0)                # constant -> zero vol


def test_max_drawdown_and_calmar():
    df = _df([0.10, -0.50, 1.20])
    r = A.series(df, "fund_ret")
    assert np.isclose(A.max_drawdown(r), 0.55 / 1.10 - 1.0)   # -0.5
    # wealth ends at 1.1*0.5*2.2 = 1.21 over 3 months
    assert np.isclose(A.cagr(r), 1.21 ** (12 / 3) - 1.0)


def test_drawdown_table_episode():
    df = _df([0.10, -0.50, 1.20, 0.0])
    tbl = A.drawdown_table(df, "fund_ret", top=5)
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert np.isclose(row["depth"], 0.55 / 1.10 - 1.0)
    assert row["months_to_trough"] == 1 and row["recovery_months"] == 1
    assert row["peak"].to_period("M") == pd.Period("2018-01", "M")
    assert row["recovery"].to_period("M") == pd.Period("2018-03", "M")


def test_sharpe_and_sign():
    df = _df([0.02, 0.01, 0.03, 0.00, 0.02], rf=[0.001] * 5)
    r, _, rf = A._aligned(df, "fund_ret", rf_col="RF")
    assert A.sharpe(r, rf) > 0
    assert np.isclose(A.sharpe(r, rf),
                      (r - rf).mean() / (r - rf).std(ddof=1) * np.sqrt(12))


def test_beta_alpha_exact():
    rng = np.random.default_rng(0)
    b = rng.normal(0.01, 0.04, 60)
    df = _df(list(2.0 * b), bench=list(b), rf=[0.0] * 60)
    r, bb, rf = A._aligned(df, "fund_ret", "bench_ret", "RF")
    bt, a_m, a_ann = A.capm_alpha_beta(r, bb, rf)
    assert np.isclose(bt, 2.0)                # fund = 2x benchmark
    assert abs(a_m) < 1e-12                   # no alpha
    assert np.isclose(A.tracking_error(r, bb), (r - bb).std(ddof=1) * np.sqrt(12))


def test_capture_ratios():
    bench = [0.10, -0.10, 0.05, -0.05]
    fund = [0.05, -0.20, 0.10, 0.00]    # half upside, double the first downside
    df = _df(fund, bench=bench)
    r, b, _ = A._aligned(df, "fund_ret", "bench_ret")
    up = (1.05 * 1.10 - 1) / (1.10 * 1.05 - 1)        # equal up legs here
    assert np.isclose(A.capture_ratio(r, b, up=True), up)
    assert A.capture_ratio(r, b, up=False) > 1.0       # captured more downside
    assert np.isclose(A.batting_average(r, b), 0.5)    # beats bench in 2 of 4


def test_trailing_and_calendar():
    # 2 calendar years of +1%/mo
    df = _df([0.01] * 24, start="2018-01")
    cal = A.calendar_returns(df, ["fund_ret"]).set_index("Year")["fund_ret"]
    assert np.isclose(cal.loc[2018], 1.01 ** 12 - 1)
    tr = A.trailing_returns(df, ["fund_ret"]).set_index("Period")["fund_ret"]
    assert np.isclose(tr.loc["1Y"], 1.01 ** 12 - 1)              # cumulative
    assert np.isclose(tr.loc["ITD"], 1.01 ** 12 - 1)            # 24mo annualized
    assert np.isclose(tr.loc["3Y"], np.nan, equal_nan=True)    # not enough months


def test_summary_keys_and_var():
    rng = np.random.default_rng(1)
    df = _df(list(rng.normal(0.008, 0.04, 80)),
             bench=list(rng.normal(0.005, 0.03, 80)), rf=[0.001] * 80)
    s = A.performance_summary(df, "fund_ret", "bench_ret")
    for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
              "beta", "alpha_ann", "tracking_error", "information_ratio",
              "up_capture", "down_capture", "batting_avg", "var95", "cvar95"):
        assert k in s
    assert s["cvar95"] <= s["var95"]            # tail mean ≤ the quantile


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("ok: all analytics tests passed")
