#!/usr/bin/env python3
"""Build point-in-time benchmark (EFA / SCZ / VSS) sector RETURNS from FactSet
Ownership holdings + cs3 prices (WRDS).

    uv run python scripts/build_benchmark_returns.py

Writes data/processed/benchmark_sector_returns_monthly.csv — one row per
(benchmark, month, sector) with the beginning-of-month sector weight, the USD
sector return, and MV coverage. Same RBICS L1 taxonomy as the fund's own
holdings, so it feeds Brinson-Fachler attribution (build_attribution.py).
Scoped to the fund's 2018-2022 holdings window (all Brinson can use). Needs WRDS
credentials (see README); requires Duo MFA on first connect.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from capstone_data import benchmark, config, wrds_io  # noqa: E402


def main():
    db = wrds_io.connect()
    try:
        print("Building benchmark sector returns (2018-2022)…")
        sr = benchmark.build_sector_returns(db)
    finally:
        db.close()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / "benchmark_sector_returns_monthly.csv"
    sr.to_csv(path, index=False)

    print(f"\nbenchmark_sector_returns_monthly: {len(sr)} rows "
          f"({sr['benchmark'].nunique()} ETFs x months x sectors) -> {path}")
    for b, g in sr.groupby("benchmark"):
        print(f"  {b.upper()}: {g['month'].min()} .. {g['month'].max()} | "
              f"avg MV-coverage {g['mv_coverage'].mean():.1%}")


if __name__ == "__main__":
    main()
