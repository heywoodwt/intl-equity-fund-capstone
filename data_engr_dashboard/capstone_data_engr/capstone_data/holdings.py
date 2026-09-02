"""Read the fund's holdings and translate Yahoo tickers to FactSet ticker-region.

FactSet's `sym_ticker_region` keys are `<root>-<REGION>`, where REGION is a
two-letter listing-country code. We map Yahoo exchange suffixes to those codes;
a ticker with no suffix is treated as US (NYSE/Nasdaq — US names + ADRs).
"""

import pandas as pd

from . import config

# Yahoo exchange suffix -> FactSet ticker-region country code.
YAHOO_SUFFIX_TO_REGION = {
    "T": "JP", "HK": "HK", "L": "GB", "PA": "FR", "DE": "DE", "F": "DE",
    "AS": "NL", "MC": "ES", "MI": "IT", "OL": "NO", "ST": "SE", "CO": "DK",
    "HE": "FI", "SI": "SG", "AX": "AU", "TA": "IL", "BR": "BE", "VI": "AT",
    "SW": "CH", "NZ": "NZ", "IR": "IE", "LS": "PT",
}

# Hand-resolved FactSet ticker_region for holdings whose Yahoo ticker doesn't map
# by rule (Nordic class-B shares: FactSet uses "<ROOT>.B-<REGION>"). Confirmed
# via factset.sym_coverage.
MANUAL_TICKER_REGION = {
    "ALK-B.CO": "ALK.B-DK",
    "RAY.ST": "RAY.B-SE",
    "VITB.ST": "VIT.B-SE",
    "AF-B.ST": "AFRY-SE",
}


def read_holding_tickers(path=None):
    """Unique Yahoo tickers the fund has held."""
    path = path or config.HOLDINGS_FILE
    df = pd.read_csv(path)
    return sorted(df["Ticker"].dropna().astype(str).str.strip().unique())


def to_ticker_region(yahoo_ticker):
    """Yahoo ticker -> FactSet `ticker_region` (or None if the suffix is unknown)."""
    t = str(yahoo_ticker).strip()
    if t in MANUAL_TICKER_REGION:
        return MANUAL_TICKER_REGION[t]
    if "." in t:
        root, suffix = t.rsplit(".", 1)
        region = YAHOO_SUFFIX_TO_REGION.get(suffix.upper())
        if region is None:
            return None
    else:
        root, region = t, "US"
    return f"{root}-{region}"
