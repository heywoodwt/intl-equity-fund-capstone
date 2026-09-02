"""Fetch FRED economic series.

Primary path: the FRED JSON API (https://api.stlouisfed.org), which needs a
free API key in env ``FRED_API_KEY`` (https://fredaccount.stlouisfed.org/apikeys).

Fallback: the keyless ``fredgraph`` CSV endpoint, used automatically when no key
is set. (Note: some restricted networks block the fredgraph host; the keyed API
host is more widely reachable.)
"""

import io
import os
import time

import pandas as pd
import requests

API_URL = "https://api.stlouisfed.org/fred/series/observations"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {"User-Agent": "capstone-data-engr/0.1"}


def _get(url, params, timeout=30, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise last


def fetch_series(series_id, start, end=None, api_key=None):
    """Return a float ``pd.Series`` (DatetimeIndex) for one FRED series.

    Missing observations (FRED encodes them as ``.``) become NaN.
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    series = _fetch_api(series_id, start, end, api_key) if api_key \
        else _fetch_csv(series_id, start, end)
    series.name = series_id
    return series


def _fetch_api(series_id, start, end, api_key):
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    if end:
        params["observation_end"] = end
    obs = _get(API_URL, params).json()["observations"]
    dates = pd.to_datetime([o["date"] for o in obs])
    vals = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce")
    return pd.Series(vals.to_numpy(), index=dates)


def _fetch_csv(series_id, start, end):
    params = {"id": series_id, "cosd": start}
    if end:
        params["coed"] = end
    text = _get(CSV_URL, params).text
    df = pd.read_csv(io.StringIO(text))
    dates = pd.to_datetime(df.iloc[:, 0])
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(dates))
