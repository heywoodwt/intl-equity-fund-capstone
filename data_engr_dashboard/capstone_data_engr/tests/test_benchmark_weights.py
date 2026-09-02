"""Deterministic checks for benchmark sector-weight logic (no network).

Run with pytest (`python -m pytest`) or directly
(`python tests/test_benchmark_weights.py`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from capstone_data import benchmark, combine


def _holdings():
    # Two month-ends; market values chosen so weights are exact fractions.
    return pd.DataFrame({
        "report_date": ["2020-01-31"] * 3 + ["2020-02-29"] * 2,
        "fsym_id":     ["AAA", "BBB", "CCC", "AAA", "BBB"],
        "adj_mv":      [60.0, 30.0, 10.0, 50.0, 50.0],
    })


def _sector_map():
    # CCC has no sector -> contributes to the denominator but not to wt_classified.
    return pd.DataFrame({
        "fsym_id": ["AAA", "BBB", "CCC"],
        "sector":  ["Finance", "Technology", None],
    })


def test_weight_by_sector_values_and_classification():
    w = benchmark.weight_by_sector(_holdings(), _sector_map()).set_index("month")
    # Jan: Finance 60/100, Technology 30/100, 10 unclassified.
    assert abs(w.loc["2020-01", "sect_wt_finance"] - 0.60) < 1e-9
    assert abs(w.loc["2020-01", "sect_wt_technology"] - 0.30) < 1e-9
    assert abs(w.loc["2020-01", "wt_classified_sector"] - 0.90) < 1e-9
    assert w.loc["2020-01", "n_holdings"] == 3
    # Feb: only AAA/BBB (both classified) -> 50/50, fully classified.
    assert abs(w.loc["2020-02", "sect_wt_finance"] - 0.50) < 1e-9
    assert abs(w.loc["2020-02", "wt_classified_sector"] - 1.0) < 1e-9
    # A sector absent in a month is 0, not NaN.
    assert w["sect_wt_finance"].notna().all()


def test_active_tilts_only_in_fund_window_and_handle_missing_fund_sectors():
    # Two months: one with fund holdings (sect_wt_* present), one without (NaN).
    out = pd.DataFrame({
        "month":               ["2020-01", "2025-01"],
        "sect_wt_finance":     [0.40, np.nan],   # fund holds finance in the window
        "bench_sect_wt_finance":   [0.25, 0.25],
        "bench_sect_wt_utilities": [0.05, 0.05],  # a sector the fund never holds
    })
    out = combine.add_active_sector_tilts(out)
    # In the fund window: tilt = fund - benchmark; missing fund sector treated as 0.
    assert abs(out.loc[0, "active_sect_wt_finance"] - 0.15) < 1e-9
    assert abs(out.loc[0, "active_sect_wt_utilities"] - (-0.05)) < 1e-9
    # Outside the fund window: tilts are NaN (we don't know the fund's holdings).
    assert pd.isna(out.loc[1, "active_sect_wt_finance"])
    assert pd.isna(out.loc[1, "active_sect_wt_utilities"])


if __name__ == "__main__":
    test_weight_by_sector_values_and_classification()
    test_active_tilts_only_in_fund_window_and_handle_missing_fund_sectors()
    print("ok: all benchmark weight tests passed")
