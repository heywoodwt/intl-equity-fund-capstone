"""Build the monthly macro-regime table from free sources (FRED + Yahoo).

Output: one row per calendar month-end, point-in-time safe (economic releases
are lagged by their publication delay so a row never contains data that wasn't
public by that month-end). Columns: ``date``, ``month`` + raw levels' derived
features. This is the macro layer of the enriched dataset (see roadmap).
"""

import pandas as pd

from . import config, fred, yahoo

# Each lagged raw series (config.RELEASE_LAG_MONTHS) yields a "<name>_yoy"
# feature; shift that feature forward by the same publication lag.
_LAGGED_FEATURES = {f"{k}_yoy": v for k, v in config.RELEASE_LAG_MONTHS.items()}


def _to_month_end(series):
    return series.resample("ME").last()


def fetch_raw(api_key=None, cache=True):
    """Fetch every source series, resampled to month-end.

    Returns a DataFrame of raw month-end levels (columns = friendly names).
    """
    cols = {}
    if cache:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    for series_id, name in config.FRED_SERIES.items():
        series = fred.fetch_series(series_id, config.START_DATE, api_key=api_key)
        if cache:
            series.to_csv(config.RAW_DIR / f"fred_{series_id}.csv")
        cols[name] = _to_month_end(series)

    for ticker, name in config.YAHOO_SERIES.items():
        series = yahoo.fetch_monthly_close(ticker, config.START_DATE)
        if cache:
            series.to_csv(config.RAW_DIR / f"yahoo_{name}.csv")
        cols[name] = series

    raw = pd.DataFrame(cols).sort_index()
    raw.index.name = "date"
    return raw


def build_features(raw, apply_release_lag=True):
    """Derive point-in-time monthly macro features from raw month-end levels."""
    f = pd.DataFrame(index=raw.index)

    # Risk sentiment
    f["vix"] = raw["vix"]
    f["vix_chg"] = raw["vix"].diff()

    # Inflation: year-over-year % change of the price indexes
    for col in ("cpi", "core_cpi", "pce", "core_pce"):
        f[f"{col}_yoy"] = raw[col].pct_change(12, fill_method=None) * 100

    # Money supply growth
    f["m2_yoy"] = raw["m2"].pct_change(12, fill_method=None) * 100

    # Rates & yield-curve shape
    f["y10"] = raw["y10"]
    f["y2"] = raw["y2"]
    f["y3m"] = raw["y3m"]
    f["slope_10y_2y"] = raw["y10"] - raw["y2"]
    f["slope_10y_3m"] = raw["y10"] - raw["y3m"]

    # Credit spreads (Moody's — full history, unaffected by ICE licensing)
    f["baa_spread"] = raw["baa_spread"]                        # Baa - 10y Treasury
    f["baa_aaa_spread"] = raw["baa_yield"] - raw["aaa_yield"]  # Baa - Aaa quality spread

    # US equity market
    f["sp500"] = raw["sp500"]
    f["sp500_ret"] = raw["sp500"].pct_change(fill_method=None) * 100
    f["sp500_mom_12m"] = raw["sp500"].pct_change(12, fill_method=None) * 100

    if apply_release_lag:
        for col, lag in _LAGGED_FEATURES.items():
            if col in f:
                f[col] = f[col].shift(lag)
    return f


def build(api_key=None, apply_release_lag=True, cache=True):
    """Full pipeline: fetch -> features -> tidy frame with date/month keys."""
    raw = fetch_raw(api_key=api_key, cache=cache)
    feats = build_features(raw, apply_release_lag=apply_release_lag)
    out = feats.reset_index()
    period = out["date"].dt.to_period("M")
    out.insert(1, "month", period.astype(str))
    # Drop the current, still-incomplete calendar month (partial market data).
    out = out[period < pd.Timestamp.today().to_period("M")].reset_index(drop=True)
    return out


def save(df, path=None):
    path = path or (config.PROCESSED_DIR / "macro_monthly.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
