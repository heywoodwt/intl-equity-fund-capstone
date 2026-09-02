#!/usr/bin/env python3
"""Build the Developed-ex-US Fama-French factor table and verify which region
the modeling repo's ff_factors.csv actually is.

Usage:
    uv run python scripts/build_factors.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capstone_data import config, factors  # noqa: E402


def main():
    # 1) Build the canonical Developed-ex-US factor file for our dataset.
    df = factors.fetch_factors()
    path = factors.save(df)
    print(f"Wrote {len(df)} months ({df.index.min().date()} -> {df.index.max().date()}) "
          f"to {path}")

    # 2) Identify which Ken French region the modeling repo's file matches.
    if not config.MODELING_FF_FACTORS.exists():
        print(f"\n[verify] modeling file not found at {config.MODELING_FF_FACTORS}; "
              f"skipping region check.")
        return

    print(f"\n[verify] Identifying region of {config.MODELING_FF_FACTORS} ...")
    res = factors.identify_region()
    print(f"{'region':<24}{'overlap':>9}{'mean_corr':>12}{'max_abs_diff':>14}")
    for region, r in sorted(res["per_region"].items(),
                            key=lambda kv: kv[1]['max_abs_diff']):
        print(f"{region:<24}{r['overlap']:>9}{r['mean_corr']:>12.4f}{r['max_abs_diff']:>14.5f}")

    print(f"\nBest match: {res['best_match']} "
          f"(matched within tolerance: {res['matched_within_tol']})")
    if res["is_developed_ex_us"]:
        print("VERDICT: modeling repo's ff_factors.csv IS Developed-ex-US. ✅")
    else:
        print("VERDICT: modeling repo's ff_factors.csv is NOT Developed-ex-US. ⚠️  "
              f"It looks like: {res['best_match']}")


if __name__ == "__main__":
    main()
