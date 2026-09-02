"""Build the point-in-time FactSet fundamentals panel and the value-weighted
monthly portfolio fundamentals for the fund.

Layer 1 (panel): one row per (holding, month) with point-in-time valuation /
quality / growth ratios — annual figures lagged by their reporting delay so a
row never contains data that wasn't public yet.
Layer 2 (portfolio): value-weighted average of each feature across the month's
holdings (weights = shares x price converted to USD).
"""

import numpy as np
import pandas as pd

from . import config, fx


def _month_end(values):
    """'YYYY-MM' (or dates) -> calendar month-end Timestamps."""
    return pd.PeriodIndex(pd.Index(values).astype(str), freq="M").to_timestamp(how="end").normalize()


def build_panel(fund_raw, fsym_to_ticker):
    """Point-in-time stock-month fundamentals panel over the fund window."""
    f = fund_raw.copy()
    f["date"] = pd.to_datetime(f["date"])
    # Annual figures only become public ~lag months after fiscal year-end.
    f["available"] = f["date"] + pd.DateOffset(months=config.FUND_REPORTING_LAG_MONTHS)
    f = f.dropna(subset=["available"]).sort_values("available")

    months = pd.date_range(config.PANEL_START, config.PANEL_END, freq="ME")
    grid = (pd.MultiIndex
            .from_product([sorted(f["fsym_id"].unique()), months],
                          names=["fsym_id", "month_end"])
            .to_frame(index=False)
            .sort_values("month_end"))

    # As-of join: for each (security, month) take the latest already-public annual record.
    panel = pd.merge_asof(grid, f, left_on="month_end", right_on="available",
                          by="fsym_id", direction="backward")
    panel["ticker"] = panel["fsym_id"].map(fsym_to_ticker)
    panel["month"] = panel["month_end"].dt.to_period("M").astype(str)
    keep = ["ticker", "fsym_id", "month", "date"] + config.FACTSET_FEATURES
    return (panel[keep].rename(columns={"date": "fiscal_date"})
            .sort_values(["ticker", "month"]).reset_index(drop=True))


def build_stock_returns(prices, fsym_to_ticker):
    """Per-stock monthly returns from the FactSet price table (covers delisted
    names → survivorship-free). `ret_price` = month-over-month price return
    (decimal, local currency); `ret_factset` = FactSet's published return for
    reference."""
    p = prices.copy()
    p["ticker"] = p["fsym_id"].map(fsym_to_ticker)
    p["month"] = pd.to_datetime(p["price_date"]).dt.to_period("M").astype(str)
    p = p.dropna(subset=["ticker"]).sort_values(["fsym_id", "month"])
    p["ret_price"] = p.groupby("fsym_id")["price_m"].pct_change(fill_method=None)
    p["ret_factset"] = pd.to_numeric(p["one_month_return"], errors="coerce")
    return p[["ticker", "fsym_id", "month", "iso_currency", "price_m",
              "ret_price", "ret_factset"]].reset_index(drop=True)


def position_values(prices, holdings_df, ticker_to_fsym, unadj_prices=None):
    """Per-(holding, month) USD market value: shares x price x FX(USD per ccy).
    Shared by the portfolio fundamentals rollup and the sector/country breakdowns.

    ``price`` here must be the **actual (unadjusted)** price, not the cs3 adjusted
    ``price_m`` (which over-/under-states the level for corporate-action names and
    blows up their weight). Pass ``unadj_prices`` from
    :func:`factset.fetch_unadj_prices`; if omitted, falls back to ``price_m`` with a
    warning (legacy/incorrect behaviour). ``iso_currency`` still comes from the cs3
    price table.
    """
    h = holdings_df.rename(columns={"Ticker": "ticker", "Month": "month"}).copy()
    h["month"] = h["month"].astype(str)
    h["fsym_id"] = h["ticker"].map(ticker_to_fsym)

    p = prices.copy()
    p["month"] = pd.to_datetime(p["price_date"]).dt.to_period("M").astype(str)
    pos = h.merge(p[["fsym_id", "month", "price_m", "iso_currency"]],
                  on=["fsym_id", "month"], how="left")
    pos["month_end"] = _month_end(pos["month"])

    if unadj_prices is not None:
        pos = pos.merge(unadj_prices[["fsym_id", "month", "unadj_price"]],
                        on=["fsym_id", "month"], how="left")
        n_missing = pos["unadj_price"].isna().sum()
        if n_missing:
            print(f"  [position_values] {n_missing} position-months lack an "
                  f"unadjusted price; falling back to adjusted price_m for those.")
        pos["price_for_value"] = pos["unadj_price"].fillna(pos["price_m"])
    else:
        print("  [position_values] WARNING: no unadjusted prices supplied — using "
              "adjusted price_m, which mis-states weights for corporate-action names.")
        pos["price_for_value"] = pos["price_m"]

    fxtab = fx.fetch_usd_per_ccy(pos["iso_currency"].dropna().unique(), config.PANEL_START)

    def usd_per(row):
        c = row["iso_currency"]
        if c == "USD":
            return 1.0
        if isinstance(c, str) and c in fxtab.columns:
            return fxtab[c].asof(row["month_end"])
        return np.nan

    pos["usd_per_ccy"] = pos.apply(usd_per, axis=1)
    pos["value_usd"] = pos["Shares"] * pos["price_for_value"] * pos["usd_per_ccy"]
    return pos


def correct_split_basis(pos, stock_returns, jump_factor=3.0):
    """Undo split/consolidation **share-basis** breaks in position value.

    The per-share USD price (``value_usd / Shares``) should move with the
    security's return. A split/consolidation resets the per-share price level, but
    the holdings file's ``Shares`` stay on the old basis, so the position's value
    jumps spuriously — e.g. Aston Martin's 2020 ~20:1 consolidation made its weight
    ~20x too large (≈50% of the book). We flag a month whose implied per-share move
    diverges from the security's actual return by ≥ ``jump_factor`` (a corporate
    action, since the per-share price shouldn't jump independently of the return)
    and divide ``Shares`` and ``value_usd`` by the cumulative factor, restoring a
    consistent basis (the per-share price is unchanged). Conservative: only clear
    splits (factor ≥ jump_factor or ≤ 1/jump_factor) are touched; ordinary returns
    and FX moves are well below the threshold.
    """
    df = pos.sort_values(["fsym_id", "month"]).copy()
    ret = stock_returns[["fsym_id", "month", "ret_factset"]].copy()
    ret["exp_ret"] = pd.to_numeric(ret["ret_factset"], errors="coerce") / 100.0
    df = df.merge(ret[["fsym_id", "month", "exp_ret"]], on=["fsym_id", "month"], how="left")
    df["price_usd"] = df["value_usd"] / df["Shares"]
    raw_ret = df.groupby("fsym_id")["price_usd"].pct_change(fill_method=None)
    jump = (1.0 + raw_ret) / (1.0 + df["exp_ret"].fillna(0.0))
    is_ca = jump.notna() & np.isfinite(jump) & ((jump >= jump_factor) | (jump <= 1.0 / jump_factor))
    df["_jump"] = np.where(is_ca, jump, 1.0)
    df["_cum"] = df.groupby("fsym_id")["_jump"].cumprod()
    flagged = df.loc[df["_jump"] != 1.0, ["ticker", "month", "_jump"]]
    if len(flagged):
        print(f"  [corporate-action guard] corrected {len(flagged)} split/consolidation "
              "basis break(s): "
              + ", ".join(f"{r.ticker}@{r.month}(÷{r._jump:.1f})" for _, r in flagged.iterrows()))
    df["value_usd"] = df["value_usd"] / df["_cum"]
    df["Shares"] = df["Shares"] / df["_cum"]
    return df.drop(columns=["exp_ret", "price_usd", "_jump", "_cum"])


def build_portfolio(panel, pos):
    """Value-weighted monthly portfolio fundamentals (weights = USD position value)."""
    pos = pos.merge(panel[["fsym_id", "month"] + config.FACTSET_FEATURES],
                    on=["fsym_id", "month"], how="left")

    lo_q, hi_q = config.WINSOR_QUANTILES
    rows = []
    for month, g in pos.groupby("month"):
        valued = g.dropna(subset=["value_usd"]).copy()
        raw = valued[config.FACTSET_FEATURES].copy()  # median uses raw (robust)
        # Winsorize each ratio cross-sectionally this month before weighting.
        for feat in config.FACTSET_FEATURES:
            lo, hi = valued[feat].quantile([lo_q, hi_q])
            valued[feat] = valued[feat].clip(lo, hi)

        w = valued["value_usd"]
        total = w.sum()
        rec = {"month": month, "n_holdings": len(g), "n_valued": int(len(valued))}
        for feat in config.FACTSET_FEATURES:
            rec[f"{feat}_wmean"] = _wmean(valued[feat], w)
            rec[f"{feat}_median"] = (float(raw[feat].median())
                                     if raw[feat].notna().any() else np.nan)
            # Harmonic mean only for price multiples (and only over positive values).
            if feat in config.FACTSET_PRICE_MULTIPLES:
                rec[f"{feat}_whmean"] = _whmean(valued[feat], w)
        covered = valued.dropna(subset=config.FACTSET_FEATURES, how="all")["value_usd"].sum()
        rec["wt_with_fundamentals"] = float(covered / total) if total > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _wmean(x, w):
    """Value-weighted arithmetic mean over non-null entries."""
    m = x.notna() & w.notna()
    return float((x[m] * w[m]).sum() / w[m].sum()) if w[m].sum() > 0 else np.nan


def _whmean(x, w):
    """Value-weighted harmonic mean over POSITIVE entries (= aggregate
    price / aggregate fundamental for a price multiple)."""
    m = x.notna() & w.notna() & (x > 0)
    denom = (w[m] / x[m]).sum()
    return float(w[m].sum() / denom) if denom > 0 else np.nan
