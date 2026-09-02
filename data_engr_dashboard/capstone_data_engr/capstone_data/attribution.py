"""Holdings-based return attribution for the fund.

Security- and sector-level **contribution-to-return** from the monthly position
values and the survivorship-free FactSet returns, plus **Brinson-Fachler**
allocation / selection vs the benchmark.

Returns are *reconstructed from holdings* (gross): they track the official fund
return at ~0.9 correlation with a ~2%/mo residual (cash, fees, intra-month
trades, small-cap pricing staleness). Every decomposition therefore carries an
explicit ``residual`` term so the pieces tie back to the official fund return by
construction. Coverage is the holdings window (2018-04 .. 2022-12); attribution
is defined from the second month on (a return needs a prior-month weight).

Pure functions over tidy frames — no Streamlit, no I/O, no network. The implied
FX (local->USD) is recovered from the position file itself
(``value_usd / (Shares * price_m)``), so this module is fully offline and exactly
consistent with how ``value_usd`` was built in :mod:`capstone_data.fundamentals`.

Conventions
-----------
* ``positions``: ticker, fsym_id, month ('YYYY-MM'), iso_currency, price_m,
  Shares, value_usd (USD market value, month-end).
* ``stock_returns``: fsym_id, month, ret_factset (FactSet 1-month return, **percent,
  local currency**), ret_price (price return, **decimal, local currency**).
* ``profile``: fsym_id -> sector (RBICS L1), country.
* All returns emitted by this module are **USD, decimal**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import profile as _profile

# Sanity bounds for a single security's monthly USD total return (after FX).
RET_CLIP = (-0.80, 2.00)


# --- security-level returns & weights --------------------------------------

def implied_fx_move(positions: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month FX move (local->USD) per currency, implied by the
    position file. ``value_usd = Shares * price_m * usd_per_ccy``, so
    ``usd_per_ccy = value_usd / (Shares * price_m)``; we take the per-month median
    across a currency's holdings (robust to a stale price on one name) and the
    ratio to the prior month. USD is exactly 1.0.

    Returns long: ``iso_currency, month, fx_move`` (1.0 where no prior month).
    """
    p = positions.copy()
    if "usd_per_ccy" not in p.columns:
        # Legacy fallback: recover FX from value = Shares x price_m x FX. Only valid
        # when value_usd was built from price_m; the build now writes usd_per_ccy
        # directly (value uses the unadjusted price, so this ratio would be wrong).
        p["usd_per_ccy"] = np.where(
            p["iso_currency"].eq("USD"), 1.0,
            p["value_usd"] / (p["Shares"] * p["price_m"]))
    fx = (p.groupby(["iso_currency", "month"])["usd_per_ccy"].median()
          .reset_index().sort_values(["iso_currency", "month"]))
    fx["fx_move"] = fx.groupby("iso_currency")["usd_per_ccy"].pct_change().add(1.0)
    fx.loc[fx["iso_currency"].eq("USD"), "fx_move"] = 1.0
    return fx[["iso_currency", "month", "fx_move"]]


def security_usd_returns(positions: pd.DataFrame,
                         stock_returns: pd.DataFrame) -> pd.DataFrame:
    """Per (fsym_id, month) USD total return.

    ``ret_usd = (1 + local_ret) * fx_move - 1`` where ``local_ret`` is the FactSet
    adjusted 1-month return (``ret_factset/100``; falls back to ``ret_price`` where
    missing) and ``fx_move`` is the currency's month FX move. Winsorized to
    :data:`RET_CLIP`. Emits ``ret_usd`` plus the inputs for transparency.
    """
    r = stock_returns.copy()
    r["local_ret"] = pd.to_numeric(r["ret_factset"], errors="coerce") / 100.0
    r["local_ret"] = r["local_ret"].fillna(pd.to_numeric(r["ret_price"], errors="coerce"))

    fx = implied_fx_move(positions)
    out = r.merge(fx, on=["iso_currency", "month"], how="left")
    out["fx_move"] = out["fx_move"].fillna(1.0)
    out["ret_usd"] = ((1.0 + out["local_ret"]) * out["fx_move"] - 1.0).clip(*RET_CLIP)
    return out[["fsym_id", "month", "iso_currency", "local_ret", "fx_move", "ret_usd"]]


def begin_weights(positions: pd.DataFrame) -> pd.DataFrame:
    """Begin-of-month weight per (fsym_id, month): the security's share of total
    portfolio value at the **prior** month-end (buy-and-hold within the month).
    The first holdings month has no prior weight, so it carries none (attribution
    starts the following month). New positions enter at weight 0 in their first
    month; that's correct for begin-of-period attribution.

    Returns long: ``fsym_id, month, w_begin``.
    """
    val = (positions.groupby(["fsym_id", "month"])["value_usd"].sum()
           .reset_index())
    wide = val.pivot(index="month", columns="fsym_id", values="value_usd").sort_index()
    w = wide.shift(1)                                   # prior month-end value
    w = w.div(w.sum(axis=1), axis=0)                    # normalize across names
    out = (w.reset_index().melt(id_vars="month", var_name="fsym_id",
                                value_name="w_begin").dropna(subset=["w_begin"]))
    return out[out["w_begin"] != 0.0].reset_index(drop=True)


def security_contributions(positions: pd.DataFrame, stock_returns: pd.DataFrame,
                           profile: pd.DataFrame) -> pd.DataFrame:
    """Per (month, security) begin weight, USD return, and contribution
    (``w_begin * ret_usd``), tagged with sector & country. Securities with a begin
    weight but no USD return get ``ret_usd``/``contrib`` = NaN (tracked as uncovered
    weight, absorbed by the reconciliation residual).
    """
    w = begin_weights(positions)
    rets = security_usd_returns(positions, stock_returns)
    prof = profile[["fsym_id", "sector", "country"]].drop_duplicates("fsym_id")
    df = (w.merge(rets[["fsym_id", "month", "ret_usd"]], on=["fsym_id", "month"],
                  how="left")
          .merge(prof, on="fsym_id", how="left"))
    df["sector"] = df["sector"].fillna("Unclassified")
    df["contrib"] = df["w_begin"] * df["ret_usd"]
    tk = positions[["fsym_id", "ticker"]].drop_duplicates("fsym_id")
    df = df.merge(tk, on="fsym_id", how="left")
    return (df[["month", "ticker", "fsym_id", "sector", "country",
                "w_begin", "ret_usd", "contrib"]]
            .sort_values(["month", "contrib"]).reset_index(drop=True))


def sector_returns(security_contrib: pd.DataFrame) -> pd.DataFrame:
    """Roll security contributions up to (month, sector).

    ``begin_wt`` = full sector weight (all names, incl. those missing a return);
    ``covered_wt`` = weight of names that had a USD return; ``sector_ret`` =
    covered-weighted return of the sector (``Σcontrib / covered_wt``);
    ``contrib`` = Σ contribution (covered names) — these sum across sectors to the
    reconstructed gross return.
    """
    sc = security_contrib
    g = sc.groupby(["month", "sector"])
    covered = sc[sc["ret_usd"].notna()].groupby(["month", "sector"])
    out = pd.DataFrame({
        "begin_wt": g["w_begin"].sum(),
        "covered_wt": covered["w_begin"].sum(),
        "contrib": covered["contrib"].sum(),
    }).reset_index()
    out["covered_wt"] = out["covered_wt"].fillna(0.0)
    out["contrib"] = out["contrib"].fillna(0.0)
    out["sector_ret"] = np.where(out["covered_wt"] > 0,
                                 out["contrib"] / out["covered_wt"], np.nan)
    return out.sort_values(["month", "sector"]).reset_index(drop=True)


def reconcile(sector_ret: pd.DataFrame, fund_ret: pd.Series) -> pd.DataFrame:
    """Per month: reconstructed gross return (Σ sector contributions), the official
    ``fund_ret``, and the ``residual`` (official − reconstructed) that ties them.

    ``fund_ret`` is a Series indexed by 'YYYY-MM' month string.
    """
    recon = (sector_ret.groupby("month")["contrib"].sum()
             .rename("recon_gross").reset_index())
    recon["fund_ret"] = recon["month"].map(fund_ret)
    recon["residual"] = recon["fund_ret"] - recon["recon_gross"]
    return recon


# --- Brinson-Fachler attribution -------------------------------------------

def blend_benchmark_sectors(bench_sector_ret: pd.DataFrame,
                            benches=("efa", "scz", "vss")) -> pd.DataFrame:
    """Equal-weight blend of the per-ETF benchmark sector returns into a single
    ``bench`` series, consistent with ``bench_avg_ret`` (= mean of the ETF total
    returns). Blend sector weight = mean ETF weight; blend sector return =
    value-weighted across ETFs (``Σ w·r / Σ w``). Input/-output columns:
    ``benchmark, month, sector, begin_wt, sector_ret``.
    """
    b = bench_sector_ret[bench_sector_ret["benchmark"].isin(benches)].copy()
    n = len(benches)
    b["wr"] = b["begin_wt"] * b["sector_ret"]
    wt = (b.groupby(["month", "sector"])["begin_wt"].sum() / n).rename("begin_wt")
    cov = (b.dropna(subset=["sector_ret"]).groupby(["month", "sector"])
           .agg(_wcov=("begin_wt", "sum"), _wr=("wr", "sum")))
    blend = wt.to_frame().join(cov).reset_index()
    blend["sector_ret"] = np.where(blend["_wcov"].fillna(0) > 0,
                                   blend["_wr"] / blend["_wcov"], np.nan)
    blend["benchmark"] = "bench"
    return blend[["benchmark", "month", "sector", "begin_wt", "sector_ret"]]


def brinson(port_sector: pd.DataFrame, bench_sector: pd.DataFrame) -> pd.DataFrame:
    """Brinson-Fachler allocation / selection / interaction per (month, sector).

    Inputs are aligned on (month, sector); the sector universe is the union, with
    missing weights filled to 0 (a sector one side never holds) and missing
    returns left NaN (only matters where that side has weight). Per sector:

        allocation  = (w_p - w_b) * (r_b,sec - r_b)
        selection   =  w_b        * (r_p,sec - r_b,sec)
        interaction = (w_p - w_b) * (r_p,sec - r_b,sec)

    where ``r_b`` is the benchmark total (reconstructed Σ w_b·r_b,sec). The three
    effects sum to the reconstructed active return ``r_p - r_b``. Returns long with
    ``w_p, w_b, r_p, r_b_sec, allocation, selection, interaction, active``.
    """
    p = (port_sector.rename(columns={"begin_wt": "w_p", "sector_ret": "r_p"})
         [["month", "sector", "w_p", "r_p"]])
    b = (bench_sector.rename(columns={"begin_wt": "w_b", "sector_ret": "r_b_sec"})
         [["month", "sector", "w_b", "r_b_sec"]])
    m = p.merge(b, on=["month", "sector"], how="outer")
    m["w_p"] = m["w_p"].fillna(0.0)
    m["w_b"] = m["w_b"].fillna(0.0)

    # Benchmark total return per month (reconstructed from its own sectors,
    # value-weighted over sectors that have a return).
    cov = (m.assign(_wr=m["w_b"] * m["r_b_sec"]).dropna(subset=["r_b_sec"])
           .groupby("month").agg(_wcov=("w_b", "sum"), _wr=("_wr", "sum")))
    rb = np.where(cov["_wcov"] > 0, cov["_wr"] / cov["_wcov"], np.nan)
    rb = pd.Series(rb, index=cov.index, name="r_b")
    m = m.merge(rb, on="month", how="left")

    dw = m["w_p"] - m["w_b"]
    m["allocation"] = dw * (m["r_b_sec"] - m["r_b"])
    m["selection"] = m["w_b"] * (m["r_p"] - m["r_b_sec"])
    m["interaction"] = dw * (m["r_p"] - m["r_b_sec"])
    m["active"] = m[["allocation", "selection", "interaction"]].sum(axis=1)
    return (m[["month", "sector", "w_p", "w_b", "r_p", "r_b_sec", "r_b",
               "allocation", "selection", "interaction", "active"]]
            .sort_values(["month", "sector"]).reset_index(drop=True))


# --- convenience: slugged sector labels for joining to combined_monthly -----

def sector_slug(name: str) -> str:
    """RBICS L1 name -> the ``sect_wt_<slug>`` slug used elsewhere in the repo."""
    return _profile.slug(name)
