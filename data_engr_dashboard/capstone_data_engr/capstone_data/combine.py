"""Combine the enrichment layers into modeling datasets.

- ``combined_monthly.csv`` — one row per month: fund/benchmark returns + alpha,
  portfolio fundamentals, macro regime indicators, ex-US FF factors, portfolio
  and benchmark (EFA/SCZ/VSS) sector weights + active sector tilts. The dataset
  for the baseline rolling regressions.
- ``combined_panel.csv`` — one row per (holding, month): stock fundamentals +
  monthly return + next_ret, with macro and factors broadcast by month. For the
  predictive (DL) stage.

All inputs are local CSVs (no network). Join key is ``month`` ('YYYY-MM').
"""

import pandas as pd

from . import config, profile


def _month(series):
    # Some performance files have "April 30,2023" (missing space after comma).
    s = pd.Series(series).astype(str).str.replace(r",(\d)", r", \1", regex=True)
    return pd.to_datetime(s, format="mixed").dt.to_period("M").astype(str)


def load_benchmark_etf_returns():
    """Monthly **total returns** for EFA / SCZ / VSS from Yahoo (auto-adjusted
    close = dividends reinvested), cached to ``benchmark_etf_returns_monthly.csv``.

    Why not the modeling-repo perf files: their SCZ/VSS ``Performance`` columns are
    corrupted — sign-flipped for 2014-2022 (verified vs an independent FactSet
    constituent reconstruction) but correct for 2023-2025, i.e. a *partial* flip
    that can't be undone with a single sign. Yahoo's adjusted ETF series is the
    definitive source (EFA stored already matches it at MAD 0.4%). ``fund_ret``
    still comes from the perf file (the fund's NAV return isn't on Yahoo).
    """
    path = config.PROCESSED_DIR / "benchmark_etf_returns_monthly.csv"
    if path.exists():
        return pd.read_csv(path)
    from . import yahoo
    rows = {}
    for tk, col in (("EFA", "efa_ret"), ("SCZ", "scz_ret"), ("VSS", "vss_ret")):
        px = yahoo.fetch_monthly_close(tk, "2014-08-01")
        rows[col] = px.pct_change()
    out = pd.DataFrame(rows)
    out.index = out.index.to_period("M").astype(str)
    out.index.name = "month"
    out = out.dropna(how="all").reset_index()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_performance():
    """Monthly fund/benchmark returns (decimal) + alpha targets, keyed by month.

    ``fund_ret`` is the fund's NAV total return from the modeling-repo perf file;
    EFA/SCZ/VSS are Yahoo total returns (see :func:`load_benchmark_etf_returns` for
    why the perf-file benchmark columns are not used)."""
    fund_raw = pd.read_csv(config.PERF_FILES["fund"])
    fund = (fund_raw["Performance"].astype(str).str.replace("%", "", regex=False)
            .astype(float) / 100.0)
    perf = pd.DataFrame({"fund_ret": fund.values}, index=_month(fund_raw["Date"]))
    perf.index.name = "month"
    perf = perf.reset_index().merge(load_benchmark_etf_returns(), on="month", how="left")
    perf["bench_avg_ret"] = perf[["efa_ret", "scz_ret", "vss_ret"]].mean(axis=1)
    perf["alpha_vs_avg"] = perf["fund_ret"] - perf["bench_avg_ret"]
    perf["alpha_vs_efa"] = perf["fund_ret"] - perf["efa_ret"]
    return perf


def load_factors():
    df = pd.read_csv(config.PROCESSED_DIR / "ff_factors_dev_ex_us.csv")
    df["month"] = _month(df["date"])
    return df.drop(columns=["date"])


def load_macro():
    df = pd.read_csv(config.PROCESSED_DIR / "macro_monthly.csv")
    return df.drop(columns=["date"])  # keep the 'month' key


def load_benchmark_sector_weights():
    """Per-ETF point-in-time RBICS sector weights (``benchmark_sector_weights.csv``),
    pivoted wide to ``efa_sect_wt_*`` / ``scz_sect_wt_*`` / ``vss_sect_wt_*`` plus an
    equal-weight blend ``bench_sect_wt_*`` (mirrors ``bench_avg_ret``). Returns
    ``None`` if the file hasn't been built yet (the layer is optional)."""
    path = config.PROCESSED_DIR / "benchmark_sector_weights.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    wtcols = [c for c in df.columns if c.startswith("sect_wt_")]
    benches = sorted(df["benchmark"].unique())
    wide = None
    for b in benches:
        g = (df[df["benchmark"] == b].set_index("month")[wtcols]
             .rename(columns=lambda c: f"{b}_{c}"))
        wide = g if wide is None else wide.join(g, how="outer")
    # Equal-weight blend across the three ETFs (each sector is 0-filled per ETF,
    # so the mean is over all benchmarks, not just those holding the sector).
    for c in wtcols:
        members = [f"{b}_{c}" for b in benches]
        wide[f"bench_{c}"] = wide[members].mean(axis=1)
    return wide.reset_index()


def add_active_sector_tilts(out):
    """Active sector tilts: fund weight - blended benchmark weight, per RBICS L1
    sector. Defined only where the fund's own holdings exist (the 2018-2022
    window); NaN elsewhere so the tilt never implies holdings we don't have."""
    fund_sect = [c for c in out.columns if c.startswith("sect_wt_")]
    if not fund_sect:
        return out
    has_fund = out[fund_sect].notna().any(axis=1)
    for c in [col for col in out.columns if col.startswith("bench_sect_wt_")]:
        sector = c[len("bench_sect_wt_"):]
        fund_col = f"sect_wt_{sector}"
        if fund_col in out.columns:
            fund_w = out[fund_col]                       # NaN outside fund window
        else:  # sector the fund never holds -> 0 within the window, NaN outside
            fund_w = pd.Series(0.0, index=out.index).where(has_fund)
        out[f"active_sect_wt_{sector}"] = fund_w - out[c]
    return out


def load_stock_returns():
    """Per-ticker monthly return (+ next_ret) from the survivorship-free FactSet
    price table (`stock_returns_monthly.csv`)."""
    df = pd.read_csv(config.PROCESSED_DIR / "stock_returns_monthly.csv")
    df = df.sort_values(["ticker", "month"])
    df["next_ret"] = df.groupby("ticker")["ret_price"].shift(-1)
    return (df.rename(columns={"ret_price": "ret"})[["ticker", "month", "ret", "next_ret"]])


def build_monthly():
    """Monthly aggregate dataset (performance backbone, everything left-joined)."""
    out = (load_performance()
           .merge(load_factors(), on="month", how="left")
           .merge(load_macro(), on="month", how="left")
           .merge(pd.read_csv(config.PROCESSED_DIR / "portfolio_fundamentals_monthly.csv"),
                  on="month", how="left")
           .merge(pd.read_csv(config.PROCESSED_DIR / "portfolio_sector_country_monthly.csv"),
                  on="month", how="left"))
    bench = load_benchmark_sector_weights()
    if bench is not None:
        out = add_active_sector_tilts(out.merge(bench, on="month", how="left"))
    return out.sort_values("month").reset_index(drop=True)


def build_panel():
    """Stock-month panel: fundamentals + return/next_ret + macro + factors +
    per-stock country/sector (with one-hot dummies)."""
    panel = pd.read_csv(config.PROCESSED_DIR / "fundamentals_panel.csv")
    prof = pd.read_csv(config.PROCESSED_DIR / "holding_profile.csv")[
        ["ticker", "country", "sector"]]
    out = (panel
           .merge(load_stock_returns(), on=["ticker", "month"], how="left")
           .merge(load_factors(), on="month", how="left")
           .merge(load_macro(), on="month", how="left")
           .merge(prof, on="ticker", how="left"))
    # Dummyified identifiers for country and sector.
    ctry = pd.get_dummies(out["country"], prefix="ctry").astype("int8")
    sect = pd.get_dummies(out["sector"].map(profile.slug), prefix="sect").astype("int8")
    out = pd.concat([out, ctry, sect], axis=1)
    return out.sort_values(["ticker", "month"]).reset_index(drop=True)
