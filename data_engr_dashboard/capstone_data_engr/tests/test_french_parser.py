"""Deterministic checks for the Ken French CSV parser (no network).

Run: `uv run pytest` or `uv run python tests/test_french_parser.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from capstone_data import french

# Mimics the real file: text preamble, monthly table, then an annual table.
SAMPLE = """This file was created using the 202604 Bloomberg database.

Missing data are indicated by -99.99.



,Mkt-RF,SMB,HML,RMW,CMA,RF
199007    ,1.99    ,2.63    ,0.38   ,-0.56    ,0.58    ,0.68
199008    ,-99.99  ,1.00    ,0.10   ,0.20     ,0.30    ,0.66

  Annual Factors: January-December

1990     ,-5.00   ,1.00    ,2.00   ,3.00     ,4.00    ,5.00
"""


def test_parse_monthly_stops_before_annual():
    df = french.parse_monthly(SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 2  # only the two monthly rows, not the annual row
    assert df.index[0] == pd.Timestamp("1990-07-31")
    assert df.index[1] == pd.Timestamp("1990-08-31")
    assert abs(df.loc["1990-07-31", "Mkt-RF"] - 1.99) < 1e-9
    # Sentinel left as-is at this stage (factors._to_decimal masks it later).
    assert abs(df.loc["1990-08-31", "Mkt-RF"] - (-99.99)) < 1e-9


if __name__ == "__main__":
    test_parse_monthly_stops_before_annual()
    print("ok: french parser test passed")
