#!/usr/bin/env python3
"""Build the processed iShares ETF holdings snapshot.

    uv run python scripts/build_etf_holdings.py                 # fetch over network
    uv run python scripts/build_etf_holdings.py --from-files DIR  # parse local CSVs

Normalizes iShares holdings into a tidy per-(ETF, constituent) table and writes
data/processed/etf_holdings.csv (latest snapshot). ETFs that fail to fetch or
parse are skipped and reported; the rest still write.

iShares gates its automated CSV endpoint behind a sign-on/terms interstitial and
serves an HTML page to plain HTTP clients, so the default network mode may return
nothing. In that case, download each product's holdings CSV by hand from its
iShares page into data/raw/etf_holdings/ (filename starting with the ticker, e.g.
EFA_holdings.csv) and run with --from-files data/raw/etf_holdings.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capstone_data import config, etf_holdings  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-files", metavar="DIR", default=None,
                    help="Parse manually-downloaded iShares CSVs from DIR "
                         "instead of fetching over the network.")
    args = ap.parse_args()

    if args.from_files:
        df, failures = etf_holdings.build_from_files(args.from_files)
    else:
        df, failures = etf_holdings.build(config.ISHARES_ETFS)
    out = etf_holdings.write(df)

    n_etfs = df["etf_ticker"].nunique() if not df.empty else 0
    print(f"etf_holdings: {len(df)} rows across {n_etfs} ETFs -> {out}")
    if not df.empty:
        by_etf = df.groupby("etf_ticker").size()
        for ticker, n in by_etf.items():
            asof = df.loc[df["etf_ticker"] == ticker, "as_of_date"].iloc[0]
            print(f"  {ticker:<6} {n:>4} holdings  as of {asof}")
    if failures:
        print("\nSkipped (fetch/parse failed):")
        for ticker, msg in failures.items():
            print(f"  {ticker:<6} {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
