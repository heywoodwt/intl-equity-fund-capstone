#!/usr/bin/env python3
"""Build holdings-based return attribution for the fund (local, no WRDS).

    uv run python scripts/build_attribution.py

Reads the processed position values, survivorship-free stock returns, and holding
profile, and writes:

  * security_contrib_monthly.csv       — month x security: begin weight, USD
                                         return, contribution, sector/country.
  * portfolio_sector_returns_monthly.csv — month x sector: begin weight, covered
                                         weight, sector USD return, contribution.
  * attribution_monthly.csv            — Brinson-Fachler allocation/selection/
                                         interaction vs the blended benchmark
                                         (only if benchmark_sector_returns_monthly.csv
                                         exists; built by build_benchmark_returns.py).

Returns are reconstructed from holdings (gross) and reconciled to the official
fund return with an explicit residual (see capstone_data/attribution.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from capstone_data import attribution, combine, config  # noqa: E402


def main():
    proc = config.PROCESSED_DIR
    positions = pd.read_csv(proc / "position_values_monthly.csv")
    stock_rets = pd.read_csv(proc / "stock_returns_monthly.csv")
    hold_profile = pd.read_csv(proc / "holding_profile.csv")

    sec = attribution.security_contributions(positions, stock_rets, hold_profile)
    sect = attribution.sector_returns(sec)

    perf = combine.load_performance().set_index("month")["fund_ret"]
    recon = attribution.reconcile(sect, perf)

    proc.mkdir(parents=True, exist_ok=True)
    sec.to_csv(proc / "security_contrib_monthly.csv", index=False)
    sect.to_csv(proc / "portfolio_sector_returns_monthly.csv", index=False)

    months = sect["month"].nunique()
    print(f"security_contrib_monthly:        {len(sec)} rows "
          f"({sec['ticker'].nunique()} names x {months} months)")
    print(f"portfolio_sector_returns_monthly: {len(sect)} rows "
          f"({sect['sector'].nunique()} sectors x {months} months) "
          f"-> {sect['month'].min()} .. {sect['month'].max()}")
    rc = recon.dropna(subset=["fund_ret", "recon_gross"])
    print(f"\nReconciliation (reconstructed gross vs official fund_ret), n={len(rc)}:")
    print(f"  corr {rc['recon_gross'].corr(rc['fund_ret']):.3f} | "
          f"mean |residual| {rc['residual'].abs().mean():.4f} | "
          f"median |residual| {rc['residual'].abs().median():.4f}")

    # Latest-month sector contribution snapshot.
    last = sect[sect["month"] == sect["month"].max()].sort_values("contrib")
    print(f"\n{sect['month'].max()} sector contribution (USD, gross):")
    for _, r in last.iterrows():
        print(f"  {r['sector']:<26} wt {r['begin_wt']:6.1%}  ret {r['sector_ret']:+7.2%}"
              f"  contrib {r['contrib']:+.3%}")

    # Brinson, if the benchmark sector returns have been built.
    bpath = proc / "benchmark_sector_returns_monthly.csv"
    if bpath.exists():
        bench = pd.read_csv(bpath)
        bench["sector"] = bench["sector"].astype(str)
        blended = attribution.blend_benchmark_sectors(bench)
        # Align the fund's RBICS sector names to the benchmark's (both RBICS L1).
        attr = attribution.brinson(sect, blended)
        attr.to_csv(proc / "attribution_monthly.csv", index=False)

        # Reconcile to the TRUE active return (fund_ret - corrected bench_avg_ret).
        # Allocation is benchmark-return-driven (robust); selection/interaction
        # inherit the noisy holdings-reconstructed fund sector returns; a residual
        # ties the holdings-based decomposition to the official active return.
        comb = pd.read_csv(proc / "combined_monthly.csv")[["month", "fund_ret", "bench_avg_ret"]]
        tot = attr.groupby("month")[["allocation", "selection", "interaction",
                                     "active"]].sum().reset_index()
        tot = tot.merge(comb, on="month", how="left")
        tot["true_active"] = tot["fund_ret"] - tot["bench_avg_ret"]
        tot["residual"] = tot["true_active"] - tot["active"]
        tot.to_csv(proc / "attribution_summary_monthly.csv", index=False)

        print(f"\nattribution_monthly: {len(attr)} rows -> {proc/'attribution_monthly.csv'}")
        print(f"attribution_summary_monthly: {len(tot)} months "
              f"-> {proc/'attribution_summary_monthly.csv'}")
        print("  cumulative over window (holdings-reconstructed):")
        print(f"    allocation (robust)        {tot['allocation'].sum():+.2%}")
        print(f"    selection  (indicative)    {tot['selection'].sum():+.2%}")
        print(f"    interaction(indicative)    {tot['interaction'].sum():+.2%}")
        print(f"    = reconstructed active     {tot['active'].sum():+.2%}")
        print(f"    true active (fund-bench)   {tot['true_active'].sum():+.2%}")
        print(f"    residual (recon error)     {tot['residual'].sum():+.2%}")
        print("  NB: fund-side sector returns are holdings-reconstructed (few names/"
              "sector); read allocation as robust, selection/interaction directionally.")
    else:
        print(f"\n[skip Brinson] {bpath.name} not found — run "
              "scripts/build_benchmark_returns.py (WRDS) first.")


if __name__ == "__main__":
    main()
