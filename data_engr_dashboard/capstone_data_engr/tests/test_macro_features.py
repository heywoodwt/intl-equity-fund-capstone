"""Deterministic checks for macro feature logic (no network).

Run with pytest (`python -m pytest`) or directly (`python tests/test_macro_features.py`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from capstone_data import macro


def _synthetic_raw(n=30):
    idx = pd.date_range("2018-01-31", periods=n, freq="ME")
    raw = pd.DataFrame(index=idx)
    raw.index.name = "date"
    cpi = 100 * (1.005 ** np.arange(n))          # +0.5% / month
    for c in ("cpi", "core_cpi", "pce", "core_pce"):
        raw[c] = cpi
    raw["m2"] = 1000 * (1.004 ** np.arange(n))
    raw["vix"] = 20.0
    raw["y10"], raw["y2"], raw["y3m"] = 3.0, 2.0, 1.5
    raw["baa_spread"] = 2.0
    raw["aaa_yield"], raw["baa_yield"] = 4.0, 5.0
    raw["sp500"] = 2000 * (1.01 ** np.arange(n))  # +1% / month
    return raw


def test_features_no_lag():
    f = macro.build_features(_synthetic_raw(), apply_release_lag=False)
    expected_yoy = (1.005 ** 12 - 1) * 100
    assert abs(f["cpi_yoy"].iloc[12] - expected_yoy) < 1e-6
    assert abs(f["slope_10y_2y"].iloc[0] - 1.0) < 1e-9
    assert abs(f["slope_10y_3m"].iloc[0] - 1.5) < 1e-9
    assert abs(f["sp500_ret"].iloc[1] - 1.0) < 1e-6


def test_release_lag_shifts_only_economic_series():
    raw = _synthetic_raw()
    f0 = macro.build_features(raw, apply_release_lag=False)
    f1 = macro.build_features(raw, apply_release_lag=True)
    # CPI YoY known at month t (lagged) == value computed for t-1 (unlagged).
    assert f1["cpi_yoy"].iloc[13] == f0["cpi_yoy"].iloc[12]
    assert pd.isna(f1["cpi_yoy"].iloc[12])
    # Market features are observable in real time -> untouched by the lag.
    pd.testing.assert_series_equal(f0["sp500_ret"], f1["sp500_ret"])
    pd.testing.assert_series_equal(f0["vix"], f1["vix"])


if __name__ == "__main__":
    test_features_no_lag()
    test_release_lag_shifts_only_economic_series()
    print("ok: all macro feature tests passed")
