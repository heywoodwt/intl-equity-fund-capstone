#!/usr/bin/env python3
"""Map the fund's holdings to FactSet `fsym_id` and report coverage.

    uv run python scripts/build_crosswalk.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from capstone_data import config, factset, holdings, wrds_io  # noqa: E402


def main():
    tickers = holdings.read_holding_tickers()
    db = wrds_io.connect()
    try:
        out = factset.crosswalk(db, tickers)
    finally:
        db.close()

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = config.INTERIM_DIR / "holdings_fsym_map.csv"
    out.to_csv(path, index=False)

    n, matched = len(out), out["fsym_id"].notna().sum()
    print(f"Holdings: {n} | matched to fsym_id: {matched} ({matched / n:.0%})")
    print(out["source"].value_counts().to_string())
    miss = out.loc[out["fsym_id"].isna(), "yahoo_ticker"].tolist()
    if miss:
        print("Unmatched:", ", ".join(miss))
    print("Wrote", path)


if __name__ == "__main__":
    main()
