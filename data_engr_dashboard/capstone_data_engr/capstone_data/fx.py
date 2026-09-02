"""Monthly FX rates as USD per 1 unit of currency, via Yahoo `<CCY>USD=X`.

Yahoo's `<CCY>USD=X` pairs are uniformly quoted as USD per 1 CCY, which avoids
the per-currency orientation guesswork of FRED's DEX series.
"""

import pandas as pd

from . import yahoo


def fetch_usd_per_ccy(currencies, start):
    """DataFrame indexed by month-end date, one column per currency = USD per 1
    unit. USD is omitted (rate is 1.0 by definition)."""
    cols = {}
    for ccy in sorted({str(c).upper() for c in currencies if c and str(c) != "nan"}):
        if ccy == "USD":
            continue
        try:
            cols[ccy] = yahoo.fetch_monthly_close(f"{ccy}USD=X", start)
        except Exception:  # noqa: BLE001 — skip currencies Yahoo can't resolve
            pass
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df
