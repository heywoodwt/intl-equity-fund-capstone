"""Per-ETF analytics frames for the factor / regime tabs.

Sits between :mod:`etf_prices` (which fetches prices) and the analytics engines
(``analytics``, ``rolling_regression``, ``kalman_tvp``), which are already
column-generic. This module's job is to hand them a frame whose ``fund_ret``
column holds the *selected* ticker's monthly returns.

Self-contained by design: imports nothing from the pipeline package, so
``dashboard/`` stays liftable into a deploy repo on its own.

Two paths. FUND -- the capstone fund -- short-circuits to the existing
``combined_monthly`` frame so its numbers and its holdings-derived regressors
are preserved exactly. Any other ticker is joined onto a fund-independent
"market frame" that spans far more history than ``combined_monthly`` does.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import data as D
import etf_prices
import fund_rating

# The capstone fund. It is a non-ETF, so `etf_prices.validate` rejects it by
# design; it is served from the existing combined frame instead of yfinance.
CAPSTONE_TICKER = "FUND"


@lru_cache(maxsize=1)
def load_market_frame() -> pd.DataFrame:
    """Fund-independent monthly frame: factors, macro, PCs, benchmark returns.

    Joined on ``month`` (``YYYY-MM``) with an outer join, so each source keeps
    its own natural coverage: factors reach back to 1990-07, macro to 2000-01,
    benchmark ETFs to 2014-09, and the PCA components only span
    2014-10 - 2024-12. Every source is clipped to the December-2024 data freeze
    (:data:`data.LAST_MONTH`). Callers must tolerate NaN outside a column's
    range -- the engines already drop incomplete rows.

    Verified to reproduce every shared column of ``combined_monthly.csv``
    exactly over their overlap; see ``test_market_frame_matches_combined``.
    """
    d = D.data_dir()

    ff = pd.read_csv(d / "ff_factors_dev_ex_us.csv")
    # This is the only source keyed by a month-end date rather than `month`.
    ff["month"] = pd.PeriodIndex(pd.to_datetime(ff["date"]), freq="M").astype(str)
    ff = ff.drop(columns=["date"])

    macro = pd.read_csv(d / "macro_monthly.csv").drop(columns=["date"])
    pcs = pd.read_csv(d / "pca_market_components.csv")
    bench = pd.read_csv(d / "benchmark_etf_returns_monthly.csv")

    out = (ff.merge(macro, on="month", how="outer")
             .merge(pcs, on="month", how="outer")
             .merge(bench, on="month", how="outer"))
    out = out[out["month"] <= D.LAST_MONTH]   # December-2024 data freeze
    return out.sort_values("month").reset_index(drop=True)


# --- per-ticker returns ----------------------------------------------------

def etf_monthly_returns(ticker: str) -> pd.DataFrame:
    """Monthly total returns for `ticker` as ``(month, fund_ret)`` (network).

    ``etf_prices`` stamps monthly returns on a month-end ``DatetimeIndex``;
    every processed table here is keyed by a ``YYYY-MM`` string, so convert.
    The column is named ``fund_ret`` because the analytics engines and the
    ``app.py`` call sites already use that name -- see the module docstring.
    """
    ticker = ticker.upper()
    ret = etf_prices.load_etf(ticker).ret_monthly
    if ret.empty:
        return pd.DataFrame({"month": pd.Series(dtype=str),
                             "fund_ret": pd.Series(dtype=float)})
    return pd.DataFrame({
        "month": pd.PeriodIndex(ret.index, freq="M").astype(str),
        "fund_ret": ret.to_numpy(dtype=float),
    })


# --- derived columns -------------------------------------------------------

BENCH_LEGS = ["efa_ret", "scz_ret", "vss_ret"]


def derive_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``excess_ret``, ``bench_avg_ret``, ``alpha_vs_*`` and ``date``.

    Mirrors the identities baked into ``combined_monthly.csv``, all verified
    exact: the blended benchmark is the simple mean of the three benchmark
    ETFs, and each alpha is the fund return minus its benchmark.

    ``skipna=False`` on the blend is deliberate. Pandas would otherwise average
    whichever legs happen to be present, silently reporting a two-ETF blend as
    the three-ETF benchmark for any month with a gap.
    """
    out = frame.copy()
    out["date"] = pd.PeriodIndex(out["month"], freq="M").to_timestamp()

    if "RF" in out.columns:
        out["excess_ret"] = out["fund_ret"] - out["RF"]
    if all(c in out.columns for c in BENCH_LEGS):
        out["bench_avg_ret"] = out[BENCH_LEGS].mean(axis=1, skipna=False)
        out["alpha_vs_avg"] = out["fund_ret"] - out["bench_avg_ret"]
    if "efa_ret" in out.columns:
        out["alpha_vs_efa"] = out["fund_ret"] - out["efa_ret"]
    return out


# --- orchestration ---------------------------------------------------------

def is_capstone(ticker: str) -> bool:
    """True for the capstone fund, which takes the combined-frame path."""
    return ticker.upper() == CAPSTONE_TICKER


def load_analytics_frame(ticker: str) -> pd.DataFrame:
    """Monthly analytics frame for `ticker`, with returns in ``fund_ret``.

    FUND returns the existing combined frame untouched -- same 123 months,
    same numbers, and the holdings-derived regressors (``ff_*``, ``sect_wt_*``,
    ``ctry_wt_*``) the capstone tabs depend on.

    Any other ticker is inner-joined onto the market frame, so rows exist only
    for months where both the ticker traded and market data exists. Holdings
    columns are absent by construction -- they are never joined in, because
    they describe FUND's portfolio and mean nothing for another fund.

    Raises whatever :func:`etf_prices.load_etf` raises (``TickerNotFound``,
    ``NoPriceHistory``, ``EtfDataError``); the caller renders the message.
    """
    if is_capstone(ticker):
        return D.load_combined_with_pcs()

    returns = etf_monthly_returns(ticker)
    merged = returns.merge(load_market_frame(), on="month", how="inner")
    return derive_returns(merged).sort_values("month").reset_index(drop=True)


def price_status(ticker: str) -> tuple[bool, pd.Timestamp | None]:
    """``(is_stale, as_of)`` for `ticker`'s price data.

    ``load_analytics_frame`` returns a bare frame, which drops the staleness
    flag that :class:`etf_prices.EtfPrices` carries -- but the UI must never
    present cached data as current, so expose it separately. The underlying
    ``load_etf`` is memoized, so this does not refetch.

    The capstone fund is read from a local CSV, so the yfinance cache never
    applies to it.
    """
    if is_capstone(ticker):
        return False, None
    prices = etf_prices.load_etf(ticker.upper())
    return prices.is_stale, prices.as_of


# --- regressor catalog -----------------------------------------------------

# The FUND preset mixes in portfolio fundamentals (`ff_earn_yld_median`,
# `ff_roce_median`) that exist only for that fund. For an ETF the equivalent
# question -- systematic exposure vs. everything else -- is answered with
# factor and macro axes alone. Five regressors keeps the rolling OLS estimable
# and the coefficient plot legible, the same reasoning behind the FUND preset.
ETF_ALPHA_PRESET_NAME = "⭐ Factor & macro exposure"
ETF_ALPHA_PRESET = ["Mkt_RF", "SMB", "HML", "PC1", "PC2"]


def regressor_catalog(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Named regressor groups for `frame`, with empty families removed.

    ``data.regressor_catalog`` filters every group against the frame's columns,
    so an ETF frame already yields empty holdings groups; they are dropped here
    rather than shown as unusable menu entries. The starred preset is swapped
    for :data:`ETF_ALPHA_PRESET` when the frame has no portfolio fundamentals.
    """
    catalog = D.regressor_catalog(frame)
    has_holdings = bool(D.fundamentals_cols(frame))

    if not has_holdings:
        catalog.pop(D.ALPHA_DRIVERS_PRESET_NAME, None)
        preset = [c for c in ETF_ALPHA_PRESET if c in frame.columns]
        catalog = {ETF_ALPHA_PRESET_NAME: preset, **catalog}

    return {name: cols for name, cols in catalog.items() if cols}


# --- rating inputs ---------------------------------------------------------

def rating_returns(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """``(monthly, daily_close)`` for the rating (network).

    Returns come from yfinance for **every** ticker, FUND included, so all
    stars use identical methodology and the drawdown gets a genuine daily price
    path (``combined_monthly.csv`` is monthly only). The series is clipped to
    the December-2024 data freeze inside :func:`etf_prices.load`, so the rating
    never sees post-2024 prices.

    Uses the ungated :func:`etf_prices.load`, not ``load_etf``/``validate``,
    because FUND is a non-ETF that the ETF gate rejects by design.
    """
    ticker = ticker.upper()
    prices = etf_prices.load(ticker)
    ret = prices.ret_monthly
    monthly = pd.DataFrame({
        "month": pd.PeriodIndex(ret.index, freq="M").astype(str),
        "fund_ret": ret.to_numpy(dtype=float),
    })
    rf = load_market_frame()[["month", "RF"]]
    return monthly.merge(rf, on="month", how="inner"), prices.close_daily


def sector_entropy(weights: pd.Series) -> float:
    """Shannon entropy of sector weights, normalized to [0, 1].

    Zero/negative weights are dropped and the rest renormalized. A single
    sector returns 0.0 rather than dividing by ``ln(1) == 0`` -- one sector is
    maximal concentration, which is exactly what a 0 means here.
    """
    w = weights.dropna()
    w = w[w > 0]
    if len(w) <= 1:
        return 0.0
    w = w / w.sum()
    return float(-(w * np.log(w)).sum() / np.log(len(w)))


def diversification_inputs(ticker: str):
    """Holdings-derived rating inputs, or ``None`` when unavailable.

    FUND resolves from the engineered layer (position values + sector
    weights). Other tickers need ``etf_holdings.csv``, which does not exist
    while the iShares endpoint is gated -- so they return None and the rating
    redistributes that weight to performance.
    """
    if not is_capstone(ticker):
        path = D.data_dir() / "etf_holdings.csv"
        if not path.exists():
            return None
        holdings = pd.read_csv(path)
        rows = holdings[holdings["etf_ticker"].str.upper() == ticker.upper()]
        if rows.empty:
            return None
        w = rows["weight"].dropna()
        w = w[w > 0] / w[w > 0].sum()
        sectors = rows.groupby("sector")["weight"].sum()
        return fund_rating.DiversificationInputs(
            top10_weight=float(w.nlargest(10).sum()),
            sector_entropy=sector_entropy(sectors),
            n_holdings=int(len(rows)),
            as_of=str(rows["as_of_date"].iloc[0]),
        )

    d = D.data_dir()
    pos = pd.read_csv(d / "position_values_monthly.csv")
    sect = pd.read_csv(d / "portfolio_sector_country_monthly.csv")
    months = set(pos["month"]) & set(sect["month"])
    if not months:
        return None
    as_of = max(months)

    p = pos[pos["month"] == as_of]
    weights = p["value_usd"] / p["value_usd"].sum()
    sect_cols = [c for c in sect.columns if c.startswith("sect_wt_")]
    row = sect[sect["month"] == as_of][sect_cols].iloc[0]

    return fund_rating.DiversificationInputs(
        top10_weight=float(weights.nlargest(10).sum()),
        sector_entropy=sector_entropy(row),
        n_holdings=int(len(p)),
        as_of=as_of,
    )
