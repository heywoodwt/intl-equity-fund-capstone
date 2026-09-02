"""Fetch Fama-French factor tables from Ken French's Dartmouth data library.

The published files are zipped CSVs with a text preamble, a monthly table, then
an annual table. We extract just the monthly block. Values are in percent and
missing data is coded -99.99 / -999; this module returns them as published (the
builder in ``factors.py`` masks sentinels and converts to decimal).
"""

import io
import re
import time
import zipfile

import pandas as pd
import requests

_HEADERS = {"User-Agent": "capstone-data-engr/0.1"}
_MONTH_RE = re.compile(r"^\s*\d{6}\s*,")


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


def fetch_zip_csv(url):
    """Download a French CSV-zip and return the inner CSV as text."""
    zf = zipfile.ZipFile(io.BytesIO(_get(url).content))
    return zf.read(zf.namelist()[0]).decode("latin-1")


def parse_monthly(text):
    """Extract the first monthly table from a French CSV.

    Returns a DataFrame indexed by month-end date, values still in percent.
    Stops at the blank line / "Annual Factors" section that follows the
    monthly block.
    """
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if _MONTH_RE.match(ln))
    header = lines[start - 1]
    rows = []
    for ln in lines[start:]:
        if _MONTH_RE.match(ln):
            rows.append(ln)
        else:
            break
    df = pd.read_csv(io.StringIO(header + "\n" + "\n".join(rows)))
    df = df.rename(columns={df.columns[0]: "yyyymm"})
    period = pd.PeriodIndex(df["yyyymm"].astype(int).astype(str), freq="M")
    df = df.drop(columns=["yyyymm"])
    df.index = period.to_timestamp(how="end").normalize()
    df.index.name = "date"
    df.columns = [c.strip() for c in df.columns]
    return df
