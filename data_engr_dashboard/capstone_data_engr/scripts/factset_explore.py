#!/usr/bin/env python3
"""One-shot schema dump of WRDS FactSet Fundamentals International + the
symbology needed to map the fund's holdings to FactSet ids.

WRDS auth: username/password (no API key) + ~/.pgpass + Duo MFA.
    export WRDS_USERNAME=yourusername      # or put it in .env
    uv run python scripts/factset_explore.py

Output: prints to stdout and writes data/interim/factset_schema_dump.txt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from capstone_data import config, wrds_io  # noqa: E402

# FactSet FF International tables for valuation/quality features — columns only.
FF_TABLES = [
    "ff_advanced_af_eu", "ff_advanced_af_ap",            # annual fundamentals
    "ff_advanced_der_af_eu", "ff_advanced_der_af_ap",    # derived annual ratios
    "ff_advanced_ltm_eu", "ff_advanced_ltm_ap",          # last-twelve-months
    "ff_advanced_der_ltm_eu", "ff_advanced_der_ltm_ap",
]
# Tables where we also want a 3-row sample (to see id / ticker / date formats).
SAMPLE_FF = ["cs3_monthly_prices_final_int"]
# Libraries that should hold ticker <-> ISIN/SEDOL <-> fsym_id symbology.
SYM_LIBS = ["factset_ff_int", "factset", "factset_common"]
SYM_KEYWORDS = ("sym", "isin", "sedol", "cusip", "ticker", "coverage", "entity")
KNOWN_SYM = [
    "sym_coverage", "sym_ticker_region", "sym_ticker_exchange",
    "sym_isin", "sym_sedol", "sym_cusip", "sym_security", "sym_entity",
]


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(str(s))

    db = wrds_io.connect()
    try:
        # 1) Table lists for the libraries of interest.
        for lib in SYM_LIBS:
            try:
                tbls = sorted(db.list_tables(library=lib))
                emit(f"\n===== library {lib}: {len(tbls)} tables =====")
                emit(", ".join(tbls))
            except Exception as exc:  # noqa: BLE001
                emit(f"\n[{lib}] list_tables failed: {exc}")

        # 2) Column schemas for the FF fundamentals/ratio tables (no samples —
        #    these have 100s of columns).
        for t in FF_TABLES:
            emit(f"\n----- factset_ff_int.{t} : columns -----")
            try:
                emit(db.describe_table("factset_ff_int", t).to_string())
            except Exception as exc:  # noqa: BLE001
                emit(f"  describe failed: {exc}")

        # 3) Price/id tables + symbology: describe + 3-row sample.
        sample_targets = [("factset_ff_int", t) for t in SAMPLE_FF]
        for lib in SYM_LIBS:
            try:
                tbls = set(db.list_tables(library=lib))
            except Exception:  # noqa: BLE001
                continue
            candidates = sorted(t for t in tbls
                                if any(k in t.lower() for k in SYM_KEYWORDS))
            if candidates:
                emit(f"\n[{lib}] symbology-like tables: {', '.join(candidates)}")
            sample_targets += [(lib, t) for t in KNOWN_SYM if t in tbls]

        for lib, t in sample_targets:
            emit(f"\n----- {lib}.{t} : columns + sample -----")
            try:
                emit(db.describe_table(lib, t).to_string())
            except Exception as exc:  # noqa: BLE001
                emit(f"  describe failed: {exc}")
            try:
                emit("  sample:")
                emit(wrds_io.query(db, f"select * from {lib}.{t} limit 3").to_string())
            except Exception as exc:  # noqa: BLE001
                emit(f"  sample failed: {exc}")
    finally:
        db.close()

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.INTERIM_DIR / "factset_schema_dump.txt"
    out_path.write_text("\n".join(lines))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
