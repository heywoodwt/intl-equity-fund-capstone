"""Performance, risk, and benchmark-relative analytics for the fund tear sheet.

Pure functions over the tidy monthly combined frame — no Streamlit, no I/O. Each
takes column names (``ret_col``, ``bench_col``, ``rf_col``), never a hard-coded
``fund_ret``, so an uploaded fund flows through unchanged once coerced to the same
schema. Returns are **monthly decimals**; annualization uses 12 (returns) and
√12 (volatility). Risk-free defaults to the ``RF`` column (monthly decimal).

The dashboard imports these; a notebook can too::

    import sys; sys.path.insert(0, "dashboard")
    import data as D, analytics as A
    df = D.load_combined_with_pcs()
    A.performance_summary(df, "fund_ret", "bench_avg_ret")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PPY = 12  # periods per year (monthly data)


# --- helpers ---------------------------------------------------------------

def series(df: pd.DataFrame, col: str, date_col: str = "date") -> pd.Series:
    """A clean, date-indexed, sorted return series (NaNs dropped)."""
    s = df[[date_col, col]].dropna().sort_values(date_col)
    return s.set_index(date_col)[col]


def _aligned(df, ret_col, bench_col=None, rf_col=None, date_col="date"):
    """Return (r, b, rf) Series aligned on the dates where ``ret_col`` exists.
    ``b``/``rf`` are None / 0 when not requested or absent."""
    r = series(df, ret_col, date_col)
    b = None
    if bench_col:
        b = series(df, bench_col, date_col)
        idx = r.index.intersection(b.index)
        r, b = r.loc[idx], b.loc[idx]
    if rf_col and rf_col in df.columns:
        rf = series(df, rf_col, date_col).reindex(r.index).fillna(0.0)
    else:
        rf = pd.Series(0.0, index=r.index)
    return r, b, rf


# --- scalar metrics --------------------------------------------------------

def total_return(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0)


def cagr(r: pd.Series) -> float:
    n = len(r)
    return float((1.0 + r).prod() ** (PPY / n) - 1.0) if n else np.nan


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(PPY)) if len(r) > 1 else np.nan


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = r - rf
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(PPY)) if sd > 0 else np.nan


def downside_deviation(r: pd.Series, mar: float = 0.0) -> float:
    d = np.minimum(r - mar, 0.0)
    rms = np.sqrt((d ** 2).mean())
    return float(rms * np.sqrt(PPY))


def sortino(r: pd.Series, rf: pd.Series | float = 0.0, mar: float = 0.0) -> float:
    ex = (r - rf).mean()
    dd = downside_deviation(r, mar)
    return float(ex * PPY / dd) if dd > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    """Underwater curve: wealth / running peak − 1 (≤ 0)."""
    wealth = (1.0 + r).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min()) if len(r) else np.nan


def calmar(r: pd.Series) -> float:
    mdd = abs(max_drawdown(r))
    return float(cagr(r) / mdd) if mdd > 0 else np.nan


def var_historical(r: pd.Series, alpha: float = 0.05) -> float:
    """Historical Value-at-Risk: the ``alpha`` quantile of monthly returns
    (negative = a loss)."""
    return float(r.quantile(alpha)) if len(r) else np.nan


def cvar_historical(r: pd.Series, alpha: float = 0.05) -> float:
    """Conditional VaR / expected shortfall: mean return in the worst ``alpha`` tail."""
    v = r.quantile(alpha)
    tail = r[r <= v]
    return float(tail.mean()) if len(tail) else np.nan


def hit_rate(r: pd.Series) -> float:
    return float((r > 0).mean()) if len(r) else np.nan


# --- benchmark-relative ----------------------------------------------------

def beta(r: pd.Series, b: pd.Series) -> float:
    cov = np.cov(r, b, ddof=1)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan


def capm_alpha_beta(r: pd.Series, b: pd.Series, rf: pd.Series | float = 0.0):
    """CAPM regression of excess fund on excess benchmark.
    Returns (beta, monthly_alpha, annualized_alpha)."""
    er, eb = r - rf, b - rf
    bt = beta(er, eb)
    a_m = float(er.mean() - bt * eb.mean())
    return bt, a_m, float((1.0 + a_m) ** PPY - 1.0)


def tracking_error(r: pd.Series, b: pd.Series) -> float:
    return float((r - b).std(ddof=1) * np.sqrt(PPY)) if len(r) > 1 else np.nan


def information_ratio(r: pd.Series, b: pd.Series) -> float:
    act = r - b
    te = act.std(ddof=1)
    return float(act.mean() / te * np.sqrt(PPY)) if te > 0 else np.nan


def capture_ratio(r: pd.Series, b: pd.Series, up: bool = True) -> float:
    """Up/down capture: compounded fund return / compounded benchmark return over
    the months where the benchmark was up (``up=True``) or down (``up=False``)."""
    mask = b > 0 if up else b < 0
    if mask.sum() == 0:
        return np.nan
    fr = (1.0 + r[mask]).prod() - 1.0
    br = (1.0 + b[mask]).prod() - 1.0
    return float(fr / br) if br != 0 else np.nan


def batting_average(r: pd.Series, b: pd.Series) -> float:
    return float((r > b).mean()) if len(r) else np.nan


def performance_summary(df: pd.DataFrame, ret_col: str, bench_col: str | None = None,
                        rf_col: str = "RF", date_col: str = "date") -> dict:
    """One dict of headline tear-sheet metrics for ``ret_col`` (vs ``bench_col``)."""
    r, b, rf = _aligned(df, ret_col, bench_col, rf_col, date_col)
    out = {
        "n_months": len(r),
        "start": r.index.min(), "end": r.index.max(),
        "total_return": total_return(r), "cagr": cagr(r),
        "ann_vol": ann_vol(r), "sharpe": sharpe(r, rf), "sortino": sortino(r, rf),
        "max_drawdown": max_drawdown(r), "calmar": calmar(r),
        "best_month": float(r.max()), "worst_month": float(r.min()),
        "pct_positive": hit_rate(r),
        "skew": float(r.skew()), "kurtosis": float(r.kurt()),
        "var95": var_historical(r, 0.05), "cvar95": cvar_historical(r, 0.05),
    }
    if b is not None and len(b):
        bt, _, a_ann = capm_alpha_beta(r, b, rf)
        out.update({
            "beta": bt, "alpha_ann": a_ann,
            "tracking_error": tracking_error(r, b),
            "information_ratio": information_ratio(r, b),
            "correlation": float(r.corr(b)),
            "up_capture": capture_ratio(r, b, True),
            "down_capture": capture_ratio(r, b, False),
            "batting_avg": batting_average(r, b),
            "active_return_ann": cagr(r) - cagr(b),
            "bench_cagr": cagr(b),
        })
    return out


# --- time-series outputs ---------------------------------------------------

def growth_of_dollar(df: pd.DataFrame, ret_cols, date_col: str = "date",
                     start: float = 1.0) -> pd.DataFrame:
    """Cumulative growth of ``start`` for each return column (missing month = 0%)."""
    d = df[[date_col, *ret_cols]].sort_values(date_col)
    out = {date_col: d[date_col].to_numpy()}
    for c in ret_cols:
        out[c] = start * (1.0 + d[c].fillna(0.0)).cumprod().to_numpy()
    return pd.DataFrame(out)


def drawdown_table(df: pd.DataFrame, ret_col: str, date_col: str = "date",
                   top: int = 5) -> pd.DataFrame:
    """The ``top`` deepest drawdown episodes: peak, trough, recovery dates, depth,
    and length / recovery in months. An unrecovered drawdown has NaT recovery."""
    r = series(df, ret_col, date_col)
    dd = drawdown_series(r)
    episodes, in_dd, peak_date, trough_date, trough = [], False, None, None, 0.0
    prev_date = None
    for date, v in dd.items():
        if not in_dd and v < 0:
            in_dd, peak_date, trough_date, trough = True, prev_date or date, date, v
        elif in_dd:
            if v < trough:
                trough, trough_date = v, date
            if v >= -1e-12:  # recovered to the prior peak
                episodes.append((peak_date, trough_date, date, trough))
                in_dd = False
        prev_date = date
    if in_dd:  # still under water at the end
        episodes.append((peak_date, trough_date, pd.NaT, trough))

    rows = []
    for peak, tr, rec, depth in episodes:
        length = (None if pd.isna(rec)
                  else int(round((rec.to_period("M") - peak.to_period("M")).n)))
        to_trough = int(round((tr.to_period("M") - peak.to_period("M")).n))
        recov = (None if pd.isna(rec)
                 else int(round((rec.to_period("M") - tr.to_period("M")).n)))
        rows.append({"peak": peak, "trough": tr, "recovery": rec, "depth": depth,
                     "months_to_trough": to_trough, "recovery_months": recov,
                     "length_months": length})
    tbl = pd.DataFrame(rows).sort_values("depth")
    return tbl.head(top).reset_index(drop=True)


def trailing_returns(df: pd.DataFrame, ret_cols, date_col: str = "date") -> pd.DataFrame:
    """Trailing-period returns anchored to the latest date: 1M/3M/6M/YTD/1Y/3Y/5Y/ITD.
    Windows ≤ 1y are cumulative; longer windows are annualized (geometric). A row
    is NaN where the fixed window lacks enough months."""
    d = df.sort_values(date_col)
    asof = d[date_col].max()
    specs = [("1M", 1), ("3M", 3), ("6M", 6), ("YTD", None),
             ("1Y", 12), ("3Y", 36), ("5Y", 60), ("ITD", None)]
    rows = []
    for label, m in specs:
        row = {"Period": label}
        for c in ret_cols:
            s = d[[date_col, c]].dropna()
            if label == "YTD":
                w = s.loc[s[date_col].dt.year == asof.year, c]
            elif label == "ITD":
                w = s[c]
            else:
                w = s[c].tail(m)
                if len(w) < m:
                    w = w.iloc[0:0]
            n = len(w)
            if n == 0:
                row[c] = np.nan
                continue
            tr = (1.0 + w).prod() - 1.0
            row[c] = (1.0 + tr) ** (PPY / n) - 1.0 if n > PPY else tr
        rows.append(row)
    return pd.DataFrame(rows)


def calendar_returns(df: pd.DataFrame, ret_cols, date_col: str = "date") -> pd.DataFrame:
    """Calendar-year compounded returns, one row per year, a column per series."""
    d = df.copy()
    d["__year"] = d[date_col].dt.year
    rows = []
    for yr, g in d.groupby("__year"):
        row = {"Year": int(yr)}
        for c in ret_cols:
            w = g[c].dropna()
            row[c] = (1.0 + w).prod() - 1.0 if len(w) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_return_matrix(df: pd.DataFrame, ret_col: str,
                          date_col: str = "date") -> pd.DataFrame:
    """Year × month matrix of returns (+ a compounded ``Year`` column)."""
    d = df[[date_col, ret_col]].dropna().copy()
    d["__year"] = d[date_col].dt.year
    d["__month"] = d[date_col].dt.month
    mat = d.pivot_table(index="__year", columns="__month", values=ret_col)
    mat.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in mat.columns]
    year_tot = d.groupby("__year")[ret_col].apply(lambda s: (1.0 + s).prod() - 1.0)
    mat["Year"] = year_tot
    mat.index.name = "Year"
    return mat


def cumulative_active(df: pd.DataFrame, ret_col: str, bench_col: str,
                      date_col: str = "date") -> pd.DataFrame:
    """Cumulative *relative* return (fund wealth / benchmark wealth − 1) over time —
    the compounded active return vs the benchmark."""
    r, b, _ = _aligned(df, ret_col, bench_col, None, date_col)
    rel = (1.0 + r).cumprod() / (1.0 + b).cumprod() - 1.0
    return rel.rename("cum_active").reset_index()


def regime_buckets(df: pd.DataFrame, ret_col: str, regime_col: str, q: int = 3,
                   labels=None, date_col: str = "date") -> pd.DataFrame:
    """Fund performance conditioned on a macro regime: split ``regime_col`` into
    ``q`` quantile buckets and report, per bucket, the count, mean monthly &
    annualized return, hit rate, and annualized volatility."""
    d = df[[regime_col, ret_col]].dropna()
    if d.empty:
        return pd.DataFrame()
    try:
        d = d.assign(bucket=pd.qcut(d[regime_col], q, labels=labels, duplicates="drop"))
    except ValueError:
        d = d.assign(bucket=pd.cut(d[regime_col], q, labels=labels))
    g = d.groupby("bucket", observed=True)[ret_col]
    out = pd.DataFrame({
        "n": g.size(),
        "mean_monthly": g.mean(),
        "ann_return": g.mean() * PPY,
        "hit_rate": g.apply(lambda s: (s > 0).mean()),
        "ann_vol": g.std(ddof=1) * np.sqrt(PPY),
    }).reset_index()
    return out


def rolling_metrics(df: pd.DataFrame, ret_col: str, bench_col: str | None = None,
                    rf_col: str = "RF", window: int = 12,
                    date_col: str = "date") -> pd.DataFrame:
    """Rolling annualized vol & Sharpe (and, with a benchmark, beta / tracking
    error / information ratio / correlation), one row per window end."""
    cols = [date_col, ret_col] + ([bench_col] if bench_col else [])
    d = df[cols].dropna(subset=[ret_col]).sort_values(date_col).reset_index(drop=True)
    r = d[ret_col]
    rf = (series(df, rf_col).reindex(d[date_col].values).fillna(0.0).to_numpy()
          if rf_col in df.columns else np.zeros(len(d)))
    out = pd.DataFrame({date_col: d[date_col]})
    out["vol"] = r.rolling(window).std(ddof=1) * np.sqrt(PPY)
    ex = r - rf
    out["sharpe"] = (ex.rolling(window).mean() / ex.rolling(window).std(ddof=1)
                     * np.sqrt(PPY))
    if bench_col:
        b = d[bench_col]
        act = r - b
        out["te"] = act.rolling(window).std(ddof=1) * np.sqrt(PPY)
        out["ir"] = act.rolling(window).mean() / act.rolling(window).std(ddof=1) * np.sqrt(PPY)
        out["corr"] = r.rolling(window).corr(b)
        out["beta"] = r.rolling(window).cov(b) / b.rolling(window).var()
    return out.dropna(subset=["vol"]).reset_index(drop=True)
