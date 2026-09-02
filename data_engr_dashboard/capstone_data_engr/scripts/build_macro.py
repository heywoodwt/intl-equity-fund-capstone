#!/usr/bin/env python3
"""Build data/processed/macro_monthly.csv from free sources (FRED + Yahoo).

Usage:
    cp .env.example .env        # then put your free FRED key in it
    pip install -r requirements.txt
    python scripts/build_macro.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from capstone_data import macro  # noqa: E402


def main():
    df = macro.build()
    path = macro.save(df)
    print(f"Wrote {len(df)} months ({df['month'].iloc[0]} -> {df['month'].iloc[-1]}) "
          f"to {path}")
    print("\nLast 3 rows:")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
