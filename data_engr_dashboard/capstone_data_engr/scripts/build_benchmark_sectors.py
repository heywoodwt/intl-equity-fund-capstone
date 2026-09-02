#!/usr/bin/env python3
"""Build point-in-time benchmark (EFA / SCZ / VSS) sector weights from FactSet
Ownership (WRDS).

    uv run python scripts/build_benchmark_sectors.py

Writes data/processed/benchmark_sector_weights.csv — one row per (benchmark,
month) with RBICS L1 ``sect_wt_*`` weights, ``wt_classified_sector`` and
``n_holdings``. Same RBICS taxonomy as the fund's own holdings, so fund-vs-
benchmark sector tilts (added in combine.py) are apples-to-apples. Needs WRDS
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
        weights = benchmark.build_sector_weights(db)
        print("Building benchmark constituents (security-grain holdings):")
        constituents = benchmark.build_constituents(db)
    finally:
        db.close()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / "benchmark_sector_weights.csv"
    weights.to_csv(path, index=False)

    cpath = config.PROCESSED_DIR / "benchmark_constituents_monthly.csv"
    constituents.to_csv(cpath, index=False)
    # Per-(benchmark, month) weights should sum to ~1.0 — quick integrity check.
    wsum = (constituents.groupby(["benchmark", "month"])["weight"].sum())
    print(f"\nbenchmark_constituents_monthly: {len(constituents):,} rows x "
          f"{constituents.shape[1]} cols -> {cpath}")
    print(f"  weight sums per benchmark-month in [{wsum.min():.4f}, {wsum.max():.4f}]; "
          f"sector-classified {constituents['sector'].notna().mean():.1%}, "
          f"ticker-matched {constituents['ticker_region'].notna().mean():.1%}\n")

    sect_cols = [c for c in weights.columns if c.startswith("sect_wt_")]
    print(f"benchmark_sector_weights: {len(weights)} rows "
          f"({weights['benchmark'].nunique()} ETFs x months) x {weights.shape[1]} cols "
          f"-> {path}")
    for b, g in weights.groupby("benchmark"):
        print(f"  {b.upper()}: {len(g)} months ({g['month'].min()} -> {g['month'].max()}) "
              f"| MV-classified {g['wt_classified_sector'].mean():.1%} "
              f"| avg {g['n_holdings'].mean():.0f} holdings")
    # Spot-check: latest-month EFA sector mix.
    efa = weights[weights["benchmark"] == "efa"].iloc[-1]
    top = efa[sect_cols].sort_values(ascending=False).head(5)
    print(f"\nEFA {efa['month']} top sectors:")
    for k, v in top.items():
        print(f"  {k[len('sect_wt_'):]:<24} {v:.3f}")


if __name__ == "__main__":
    main()
