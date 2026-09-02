#!/usr/bin/env python3
"""Probe what WRDS data this account can actually access.

WRDS has NO API key: it authenticates with your WRDS *username + password*
(the same as the website login). The first connection offers to save a
``~/.pgpass`` file so future runs are non-interactive. WRDS requires Duo MFA,
so be ready to approve a push on first connect.

Run:
    export WRDS_USERNAME=yourusername     # or put it in .env (username isn't secret)
    uv run python scripts/wrds_explore.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import wrds  # noqa: E402

# Libraries that matter for an international (developed ex-US) equity fund.
# We need to confirm which of these UVA actually subscribes to.
CANDIDATES = {
    # International fundamentals (what this ex-US fund needs). UVA lacks Compustat
    # Global, so these FactSet / Capital IQ libraries are the likely substitutes.
    "factset_ff_int": "FactSet Fundamentals INTERNATIONAL — primary ex-US candidate",
    "ciq": "Capital IQ — global fundamentals (alt international source)",
    "factset_ff_usc": "FactSet Fundamentals US/Canada",
    # US/Canada-only coverage (a minority of holdings: US-listed ADRs + US names).
    "comp": "Compustat NA (US/Canada only)",
    "comp_na_daily_all": "Compustat NA daily (US/Canada only)",
    "crsp": "CRSP (US)",
    # Estimates / EPS — IBES has international files (act_epsint, actpsum_epsint).
    "ibes": "I/B/E/S estimates (incl. international: act_epsint, actpsum_epsint)",
    # Confirmed NOT available at UVA:
    "comp_global_daily": "Compustat Global daily (NOT at UVA)",
    "tr_ds": "Refinitiv Datastream (NOT at UVA)",
    "trws": "Refinitiv Worldscope (NOT at UVA)",
}


def main():
    user = os.environ.get("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=user) if user else wrds.Connection()
    try:
        libs = sorted(db.list_libraries())
        print(f"\nAccessible libraries ({len(libs)}):\n" + ", ".join(libs))

        print("\n--- Libraries we care about ---")
        for lib, note in CANDIDATES.items():
            if lib in libs:
                try:
                    tables = sorted(db.list_tables(library=lib))
                    print(f"\n[{lib}] ACCESSIBLE — {note}\n  {len(tables)} tables; "
                          f"sample: {', '.join(tables[:20])}")
                except Exception as exc:  # noqa: BLE001
                    print(f"\n[{lib}] accessible but couldn't list tables: {exc}")
            else:
                print(f"\n[{lib}] not in your library list — {note}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
