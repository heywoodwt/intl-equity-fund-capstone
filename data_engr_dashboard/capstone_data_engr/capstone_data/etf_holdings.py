"""Ingest full published holdings for a curated list of iShares ETFs.

iShares publishes each product's holdings as CSV at a stable ajax URL. The file
has a short text preamble (fund name, "Fund Holdings as of", share buckets) and
a blank line before the real header row, which starts with "Ticker,Name,Sector,".
We locate that header, read the constituent table, and normalize to one tidy row
per (ETF, constituent). Cash/derivative rows are kept (identified by asset_class)
so weights still sum to ~100%.
"""

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

from capstone_data import config

_HEADERS = {"User-Agent": "capstone-data-engr/0.1"}
# Capture the whole date (it contains a comma, e.g. "Jul 18, 2026"), quotes optional.
_ASOF_RE = re.compile(r'Fund Holdings as of,\s*"?(.+?)"?\s*$', re.IGNORECASE)

# Output columns, in order.
COLUMNS = [
    "etf_ticker", "as_of_date", "constituent_ticker", "name",
    "sector", "asset_class", "weight", "market_value", "shares",
]

# iShares source column -> our column.
_COLMAP = {
    "Ticker": "constituent_ticker",
    "Name": "name",
    "Sector": "sector",
    "Asset Class": "asset_class",
    "Weight (%)": "weight",
    "Market Value": "market_value",
    "Shares": "shares",
}


def _get(url, timeout=30, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise last


def fetch(ticker, url):
    """Download the raw iShares holdings CSV bytes for one ETF (network)."""
    return _get(url).content


def _to_float(s):
    """Coerce an iShares numeric column (quoted strings, thousands commas) to float."""
    cleaned = (s.astype(str).str.replace(",", "", regex=False).str.strip()
               .replace({"-": None, "": None, "nan": None}))
    return pd.to_numeric(cleaned, errors="coerce")


def parse(raw, etf_ticker):
    """Parse raw iShares holdings CSV bytes into the tidy schema (pure)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = text.splitlines()

    m = next((_ASOF_RE.search(ln) for ln in lines if _ASOF_RE.search(ln)), None)
    as_of = pd.to_datetime(m.group(1).strip()).date().isoformat() if m else None

    hdr = next((i for i, ln in enumerate(lines)
                if ln.lstrip().startswith("Ticker,")), None)
    if hdr is None:
        raise ValueError(f"{etf_ticker}: iShares holdings header row not found")

    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), on_bad_lines="skip")
    df = df[[c for c in _COLMAP if c in df.columns]].rename(columns=_COLMAP)

    for col in ("weight", "market_value", "shares"):
        df[col] = _to_float(df[col])
    df = df.loc[df["weight"].notna()].copy()
    df["constituent_ticker"] = (df["constituent_ticker"].astype(str).str.strip()
                                .replace({"-": "", "nan": ""}))
    df.insert(0, "as_of_date", as_of)
    df.insert(0, "etf_ticker", etf_ticker)
    return df[COLUMNS].reset_index(drop=True)


def build(etfs=None):
    """Fetch+parse each ETF, concatenating results. Returns (frame, failures).

    A network or parse failure for one ETF is recorded in ``failures`` and skipped;
    the rest still build.
    """
    etfs = etfs if etfs is not None else config.ISHARES_ETFS
    frames, failures = [], {}
    for ticker, url in etfs.items():
        try:
            frames.append(parse(fetch(ticker, url), ticker))
        except Exception as exc:  # network (requests) or parse (ValueError)
            failures[ticker] = str(exc)
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=COLUMNS))
    return out, failures


def _ticker_from_filename(path):
    """Infer the ETF ticker from a downloaded holdings filename.

    e.g. ``EFA_holdings.csv`` -> ``EFA``, ``acwx-holdings-20260718.csv`` -> ``ACWX``.
    """
    return re.split(r"[_\-. ]", Path(path).stem, maxsplit=1)[0].upper()


def build_from_files(src_dir=None):
    """Parse manually-downloaded iShares holdings CSVs from a directory.

    iShares gates its automated CSV endpoint behind a sign-on/terms interstitial,
    so holdings are downloaded by hand from each product page into ``src_dir``
    (default data/raw/etf_holdings/). Each ``*.csv`` file's ETF ticker is inferred
    from its filename. Returns ``(frame, failures)``, mirroring ``build``.
    """
    src_dir = Path(src_dir) if src_dir is not None else (config.RAW_DIR / "etf_holdings")
    frames, failures = [], {}
    for path in sorted(Path(src_dir).glob("*.csv")):
        ticker = _ticker_from_filename(path)
        try:
            frames.append(parse(path.read_bytes(), ticker))
        except Exception as exc:  # parse (ValueError) or read error
            failures[ticker] = str(exc)
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=COLUMNS))
    return out, failures


def write(df, path=None):
    """Write the holdings frame to CSV (default data/processed/etf_holdings.csv)."""
    path = path if path is not None else (config.PROCESSED_DIR / "etf_holdings.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path