"""Fetch index levels from Yahoo Finance via yfinance (no API key)."""

import pandas as pd
import yfinance as yf


def fetch_monthly_close(ticker, start):
    """Return a month-end close ``pd.Series`` for a Yahoo ticker.

    Pulls daily bars and resamples to month-end so the index is true
    calendar month-ends (yfinance's own ``1mo`` bars are stamped month-start).
    """
    raw = yf.download(
        ticker, start=start, interval="1d",
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # defensive: collapse stray 2-D frame
        close = close.iloc[:, 0]
    monthly = close.resample("ME").last()
    monthly.name = ticker
    return monthly
