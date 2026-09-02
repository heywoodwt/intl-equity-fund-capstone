#!/usr/bin/env python3
"""Combine the enrichment layers into the modeling datasets (offline).

    uv run python scripts/build_combined.py

Writes data/processed/combined_monthly.csv and data/processed/combined_panel.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capstone_data import combine, config  # noqa: E402


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    monthly = combine.build_monthly()
    panel = combine.build_panel()

    p_m = config.PROCESSED_DIR / "combined_monthly.csv"
    p_p = config.PROCESSED_DIR / "combined_panel.csv"
    monthly.to_csv(p_m, index=False)
    panel.to_csv(p_p, index=False)

    n_fund = monthly["ff_pe_dil_whmean"].notna().sum()
    print(f"combined_monthly: {monthly.shape[0]} months x {monthly.shape[1]} cols "
          f"({monthly['month'].min()} -> {monthly['month'].max()}) -> {p_m}")
    print(f"  fund_ret present: {monthly['fund_ret'].notna().sum()} | "
          f"factors: {monthly['Mkt_RF'].notna().sum()} | "
          f"macro(vix): {monthly['vix'].notna().sum()} | "
          f"portfolio fundamentals: {n_fund} months")
    print(monthly[["month", "fund_ret", "alpha_vs_avg", "Mkt_RF", "vix",
                   "ff_pe_dil_whmean"]].dropna().head(4).to_string(index=False))

    ret_cov = panel["ret"].notna().mean()
    print(f"\ncombined_panel: {panel.shape[0]} rows x {panel.shape[1]} cols "
          f"| {panel['ticker'].nunique()} tickers x {panel['month'].nunique()} months -> {p_p}")
    print(f"  per-stock return present: {ret_cov:.0%} of rows | "
          f"next_ret present: {panel['next_ret'].notna().mean():.0%}")


if __name__ == "__main__":
    main()
