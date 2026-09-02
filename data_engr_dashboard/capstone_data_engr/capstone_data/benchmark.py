"""Point-in-time benchmark (EFA / SCZ / VSS) sector weights from FactSet Ownership.

The fund's three benchmark ETFs are decomposed into RBICS L1 sector weights the
same way the fund's own holdings are (see ``profile.build_breakdowns``), so the
two are directly comparable. Join path, per ETF:

    factset_own.own_fund_detail_eq   -> each ETF's month-end holdings (fsym_id, adj_mv)
    factset_own.own_sec_entity_eq    -> holding security -> FactSet entity
    factset.sym_entity_sector_rbics  -> entity -> RBICS L1 sector (focus_flag = primary)
        + factset.rbics_structure_l2_curr

Because holdings are reported as of each month-end, the weights are point-in-time
— no look-ahead, no backfilling of a single current snapshot. Coverage is
reported per month via ``wt_classified_sector`` (share of market value that maps
to a sector); in practice this is ~100% for these large, plain-vanilla ETFs.
"""

import numpy as np
import pandas as pd

from . import config, fx, profile, wrds_io

# Sanity bounds for a single constituent's monthly USD total return (after FX).
RET_CLIP = (-0.80, 2.00)


def _chunked_in(db, sql, ids, key="i", chunk=1500, **params):
    """Run an ``... in %(i)s`` query over ``ids`` in chunks, concatenating results.
    Postgres handles large IN lists, but chunking keeps each statement bounded for
    the thousands of benchmark constituents."""
    ids = sorted({i for i in ids if i})
    frames = []
    for j in range(0, len(ids), chunk):
        p = dict(params)
        p[key] = tuple(ids[j:j + chunk])
        frames.append(wrds_io.query(db, sql, params=p))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolve_fund_ids(db, tickers):
    """Map FactSet Ownership fund tickers -> factset_fund_id (active preferred)."""
    rows = wrds_io.query(
        db,
        "select factset_fund_id, fund_ticker, active "
        "from factset_own.own_ent_fund_identifiers where fund_ticker in %(t)s",
        params={"t": tuple(tickers)},
    )
    rows = (rows.sort_values("active", ascending=False)
            .drop_duplicates("fund_ticker"))
    return rows.set_index("fund_ticker")["factset_fund_id"].to_dict()


def fetch_holdings(db, fund_id, start, end):
    """Month-end holdings (report_date, fsym_id, adj_mv) for one fund."""
    return wrds_io.query(
        db,
        "select report_date, fsym_id, adj_mv "
        "from factset_own.own_fund_detail_eq "
        "where factset_fund_id = %(f)s "
        "and report_date between %(s)s and %(e)s and adj_mv is not null",
        params={"f": fund_id, "s": start, "e": end},
    )


def map_sectors(db, fsym_ids):
    """Security ``fsym_id`` -> RBICS L1 sector, via entity + focus_flag primary row.

    Same RBICS structure the fund's own ``profile`` uses, so the sector labels
    line up exactly.
    """
    ids = tuple(sorted({i for i in fsym_ids if i}))
    if not ids:
        return pd.DataFrame(columns=["fsym_id", "sector"])
    ent = wrds_io.query(
        db,
        "select fsym_id, factset_entity_id from factset_own.own_sec_entity_eq "
        "where fsym_id in %(i)s", params={"i": ids})
    eids = tuple(ent["factset_entity_id"].dropna().unique())
    if not eids:
        return ent.assign(sector=pd.NA)[["fsym_id", "sector"]]
    sec = wrds_io.query(
        db,
        "select s.factset_entity_id, s.focus_flag, r.l1_name as sector "
        "from factset.sym_entity_sector_rbics s "
        "join factset.rbics_structure_l2_curr r on r.l2_id = s.l2_id "
        "where s.factset_entity_id in %(e)s", params={"e": eids})
    sec = (sec.sort_values("focus_flag", ascending=False)
           .drop_duplicates("factset_entity_id")[["factset_entity_id", "sector"]])
    return (ent.merge(sec, on="factset_entity_id", how="left")
            [["fsym_id", "sector"]])


def map_entity_meta(db, sec_fsym_ids):
    """Ownership security (``-S``) ``fsym_id`` -> entity name, domicile country
    and RBICS L1 sector.

    Same security -> entity -> RBICS path as :func:`map_sectors`, plus the
    entity's proper name and ``iso_country`` (domicile) from
    ``factset_common.sym_entity`` — so the saved constituents are human-readable
    and support country tilts, not just the sector roll-up.
    """
    cols = ["fsym_id", "name", "country", "sector"]
    ent = _chunked_in(
        db,
        "select fsym_id, factset_entity_id from factset_own.own_sec_entity_eq "
        "where fsym_id in %(i)s", sec_fsym_ids)
    if ent.empty:
        return pd.DataFrame(columns=cols)
    eids = ent["factset_entity_id"].dropna().unique()
    if len(eids) == 0:
        return ent.assign(name=pd.NA, country=pd.NA, sector=pd.NA)[cols]
    meta = _chunked_in(
        db,
        "select factset_entity_id, entity_proper_name as name, iso_country as country "
        "from factset_common.sym_entity where factset_entity_id in %(i)s", eids)
    sec = _chunked_in(
        db,
        "select s.factset_entity_id, s.focus_flag, r.l1_name as sector "
        "from factset.sym_entity_sector_rbics s "
        "join factset.rbics_structure_l2_curr r on r.l2_id = s.l2_id "
        "where s.factset_entity_id in %(i)s", eids)
    if not sec.empty:
        sec = (sec.sort_values("focus_flag", ascending=False)
               .drop_duplicates("factset_entity_id")[["factset_entity_id", "sector"]])
    else:
        sec = pd.DataFrame(columns=["factset_entity_id", "sector"])
    return (ent.merge(meta, on="factset_entity_id", how="left")
            .merge(sec, on="factset_entity_id", how="left")[cols])


def weight_by_sector(holdings, sector_map):
    """Value-weighted RBICS L1 sector weights per month for one ETF.

    One row per month: ``sect_wt_<slug>`` (weights of total market value),
    ``wt_classified_sector`` (share of MV that mapped to a sector), ``n_holdings``.
    A sector absent in a month is 0 weight, not missing — mirrors
    ``profile.build_breakdowns``.
    """
    df = holdings.copy()
    df["month"] = pd.to_datetime(df["report_date"]).dt.to_period("M").astype(str)
    df = df.merge(sector_map, on="fsym_id", how="left")
    rows = []
    for month, g in df.groupby("month"):
        total = g["adj_mv"].sum()
        rec = {"month": month, "n_holdings": int(len(g))}
        if total and total > 0:
            sec = g.dropna(subset=["sector"])
            for s, v in sec.groupby("sector")["adj_mv"].sum().items():
                rec[f"sect_wt_{profile.slug(s)}"] = v / total
            rec["wt_classified_sector"] = float(sec["adj_mv"].sum() / total)
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    wt = [c for c in out.columns if c.startswith("sect_wt_")]
    out[wt] = out[wt].fillna(0.0)
    return out


# --- benchmark sector RETURNS (for Brinson attribution) --------------------

def map_security_to_regional(db, sec_fsym_ids):
    """Ownership security (``-S``) fsym_id -> regional (``-R``) fsym_id via
    ``factset.sym_coverage`` — the cs3 price tables are keyed by the regional id,
    the ownership holdings by the security id."""
    m = _chunked_in(
        db,
        "select fsym_security_id as fsym_id, fsym_regional_id "
        "from factset.sym_coverage where fsym_security_id in %(i)s",
        sec_fsym_ids)
    if m.empty:
        return pd.DataFrame(columns=["fsym_id", "fsym_regional_id"])
    return (m.dropna(subset=["fsym_regional_id"]).drop_duplicates("fsym_id")
            [["fsym_id", "fsym_regional_id"]])


def map_security_ticker(db, sec_fsym_ids):
    """Ownership security (``-S``) ``fsym_id`` -> FactSet ``ticker_region`` (e.g.
    ``7203-JP``), via the ``-S`` -> ``-R`` regional map and
    ``factset.sym_ticker_region`` (keyed by the regional id). Best-effort: a
    security with no current ticker_region maps to NaN."""
    srmap = map_security_to_regional(db, sec_fsym_ids)
    if srmap.empty:
        return pd.DataFrame(columns=["fsym_id", "ticker_region"])
    tr = _chunked_in(
        db,
        "select fsym_id as fsym_regional_id, ticker_region "
        "from factset.sym_ticker_region where fsym_id in %(i)s",
        srmap["fsym_regional_id"].unique())
    if tr.empty:
        return pd.DataFrame(columns=["fsym_id", "ticker_region"])
    tr = tr.drop_duplicates("fsym_regional_id")
    return (srmap.merge(tr, on="fsym_regional_id", how="left")
            .drop_duplicates("fsym_id")[["fsym_id", "ticker_region"]])


def fetch_returns(db, regional_ids, start, end):
    """Per regional-security month-end return: ``fsym_id`` (``-R``), ``month``,
    ``iso_currency``, ``one_month_return`` (percent, local), unioned across the
    int + US/Canada cs3 price tables."""
    frames = []
    for lib, tbl in config.FACTSET_PRICE_TABLES.items():
        frames.append(_chunked_in(
            db,
            f"select fsym_id, price_date, iso_currency, one_month_return "
            f"from {lib}.{tbl} where fsym_id in %(i)s "
            f"and price_date >= %(s)s and price_date <= %(e)s",
            regional_ids, s=start, e=end))
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=["fsym_id", "month", "iso_currency", "one_month_return"])
    out["month"] = pd.to_datetime(out["price_date"]).dt.to_period("M").astype(str)
    return out[["fsym_id", "month", "iso_currency", "one_month_return"]]


def _fx_move_table(fxtab):
    """Monthly FX move (USD per ccy ratio) from :func:`fx.fetch_usd_per_ccy`'s
    date-indexed table -> long ``iso_currency, month, fx_move`` (USD = 1.0)."""
    if fxtab.empty:
        return pd.DataFrame(columns=["iso_currency", "month", "fx_move"])
    lvl = fxtab.copy()
    lvl.index = lvl.index.to_period("M").astype(str)
    move = lvl / lvl.shift(1)
    long = (move.reset_index().melt(id_vars="date", var_name="iso_currency",
                                    value_name="fx_move")
            .rename(columns={"date": "month"}))
    return long.dropna(subset=["fx_move"])


def sector_returns_one_etf(holdings, sector_map, sr_map, returns, fx_move):
    """Value-weighted RBICS L1 **sector returns** for one ETF, per month.

    Begin-of-month sector weight = share of ``adj_mv`` (USD) at the prior month-end;
    constituent USD return = ``(1 + one_month_return/100) * fx_move - 1`` (winsorized).
    Sector return is the begin-weight-weighted mean over covered constituents.
    Returns long: ``month, sector, begin_wt, sector_ret, mv_coverage``.
    """
    h = holdings.copy()
    h["month"] = pd.to_datetime(h["report_date"]).dt.to_period("M").astype(str)
    h = h.merge(sector_map, on="fsym_id", how="left")
    h = h.merge(sr_map, on="fsym_id", how="left")          # -> fsym_regional_id

    # Begin-of-month weight from prior-month adj_mv (USD).
    mv = h.pivot_table(index="month", columns="fsym_id", values="adj_mv",
                       aggfunc="sum").sort_index()
    wb = mv.shift(1)
    wb = wb.div(wb.sum(axis=1), axis=0)
    wlong = (wb.reset_index().melt(id_vars="month", var_name="fsym_id",
                                   value_name="w_begin").dropna(subset=["w_begin"]))
    wlong = wlong[wlong["w_begin"] != 0.0]

    # Constituent USD returns, keyed back to the security (-S) id and its sector.
    r = returns.merge(fx_move, on=["iso_currency", "month"], how="left")
    r["fx_move"] = r["fx_move"].fillna(1.0)
    r["ret_usd"] = ((1.0 + pd.to_numeric(r["one_month_return"], errors="coerce") / 100.0)
                    * r["fx_move"] - 1.0).clip(*RET_CLIP)
    r = r.rename(columns={"fsym_id": "fsym_regional_id"})[["fsym_regional_id", "month", "ret_usd"]]

    sec_meta = h[["fsym_id", "sector", "fsym_regional_id"]].drop_duplicates("fsym_id")
    df = (wlong.merge(sec_meta, on="fsym_id", how="left")
          .merge(r, on=["fsym_regional_id", "month"], how="left"))
    df["sector"] = df["sector"].fillna("Unclassified")
    df["contrib"] = df["w_begin"] * df["ret_usd"]

    g = df.groupby(["month", "sector"])
    cov = df[df["ret_usd"].notna()].groupby(["month", "sector"])
    out = pd.DataFrame({
        "begin_wt": g["w_begin"].sum(),
        "_cov_wt": cov["w_begin"].sum(),
        "_contrib": cov["contrib"].sum(),
    }).reset_index()
    out["_cov_wt"] = out["_cov_wt"].fillna(0.0)
    out["sector_ret"] = np.where(out["_cov_wt"] > 0, out["_contrib"] / out["_cov_wt"], np.nan)
    out["mv_coverage"] = np.where(out["begin_wt"] > 0, out["_cov_wt"] / out["begin_wt"], np.nan)
    return out[["month", "sector", "begin_wt", "sector_ret", "mv_coverage"]]


def build_sector_returns(db, tickers=None, start=None, end=None):
    """Tidy long table of point-in-time benchmark sector **returns** (USD), one row
    per (``benchmark``, ``month``, ``sector``) with ``begin_wt`` (beginning-of-month
    weight), ``sector_ret`` (USD) and ``mv_coverage`` (share of sector weight with a
    return). Scoped to the fund's holdings window by default (Brinson needs only
    that). Reuses the ownership holdings + RBICS sector map from the weights build.
    """
    tickers = tickers or list(config.BENCHMARK_ETFS)
    start = start or "2018-01-01"
    end = end or "2022-12-31"
    fund_ids = resolve_fund_ids(db, [config.BENCHMARK_ETFS[t] for t in tickers])

    frames = []
    for key in tickers:
        tk = config.BENCHMARK_ETFS[key]
        holds = fetch_holdings(db, fund_ids[tk], start, end)
        sids = holds["fsym_id"].dropna().unique()
        smap = map_sectors(db, sids)
        srmap = map_security_to_regional(db, sids)
        rets = fetch_returns(db, srmap["fsym_regional_id"].unique(), start, end)
        fxtab = fx.fetch_usd_per_ccy(rets["iso_currency"].dropna().unique(), start)
        fxmove = _fx_move_table(fxtab)
        sr = sector_returns_one_etf(holds, smap, srmap, rets, fxmove)
        sr.insert(0, "benchmark", key)
        frames.append(sr)
        print(f"  {key.upper()}: {sr['month'].nunique()} months, "
              f"{len(sids)} constituents, MV-coverage {sr['mv_coverage'].mean():.1%}")

    return (pd.concat(frames, ignore_index=True)
            .sort_values(["benchmark", "month", "sector"]).reset_index(drop=True))


def build_sector_weights(db, tickers=None, start=None, end=None):
    """Tidy long table of point-in-time benchmark sector weights.

    One row per (``benchmark``, ``month``) with ``n_holdings``,
    ``wt_classified_sector`` and the RBICS L1 ``sect_wt_*`` columns (0-filled to
    the union of sectors across all benchmarks). Benchmarks default to the three
    in ``config.BENCHMARK_ETFS`` (EFA, SCZ, VSS).
    """
    tickers = tickers or list(config.BENCHMARK_ETFS)
    start = start or config.BENCH_START
    end = end or config.BENCH_END
    fund_ids = resolve_fund_ids(db, [config.BENCHMARK_ETFS[t] for t in tickers])

    frames = []
    for key in tickers:
        tk = config.BENCHMARK_ETFS[key]
        holds = fetch_holdings(db, fund_ids[tk], start, end)
        smap = map_sectors(db, holds["fsym_id"].unique())
        w = weight_by_sector(holds, smap)
        w.insert(0, "benchmark", key)
        frames.append(w)

    out = pd.concat(frames, ignore_index=True)
    wt = sorted(c for c in out.columns if c.startswith("sect_wt_"))
    out[wt] = out[wt].fillna(0.0)  # union of sectors -> 0 where a benchmark lacks one
    front = ["benchmark", "month", "n_holdings", "wt_classified_sector"]
    return (out[front + wt]
            .sort_values(["benchmark", "month"]).reset_index(drop=True))


def build_constituents(db, tickers=None, start=None, end=None):
    """Tidy long table of point-in-time benchmark **constituents** — the
    security-grain holdings that :func:`build_sector_weights` rolls up.

    One row per (``benchmark``, ``month``, holding): ``fsym_id``,
    ``ticker_region``, ``name``, ``country`` (domicile), RBICS L1 ``sector``,
    ``adj_mv`` (USD) and ``weight`` (share of the ETF's month-end market value;
    sums to 1.0 per benchmark-month). Reported at each month-end, so it is
    point-in-time (no look-ahead, no snapshot backfill) — the same holdings the
    sector weights derive from, saved so the constituent detail (country tilts,
    single-name overlap with the fund, concentration) is reusable instead of
    discarded.
    """
    tickers = tickers or list(config.BENCHMARK_ETFS)
    start = start or config.BENCH_START
    end = end or config.BENCH_END
    fund_ids = resolve_fund_ids(db, [config.BENCHMARK_ETFS[t] for t in tickers])

    cols = ["benchmark", "month", "fsym_id", "ticker_region", "name",
            "country", "sector", "adj_mv", "weight"]
    frames = []
    for key in tickers:
        tk = config.BENCHMARK_ETFS[key]
        holds = fetch_holdings(db, fund_ids[tk], start, end)
        if holds.empty:
            continue
        sids = holds["fsym_id"].dropna().unique()
        meta = map_entity_meta(db, sids)
        tkr = map_security_ticker(db, sids)
        df = holds.copy()
        df["month"] = pd.to_datetime(df["report_date"]).dt.to_period("M").astype(str)
        df = (df.merge(meta, on="fsym_id", how="left")
              .merge(tkr, on="fsym_id", how="left"))
        df["weight"] = df["adj_mv"] / df.groupby("month")["adj_mv"].transform("sum")
        df.insert(0, "benchmark", key)
        frames.append(df.reindex(columns=cols))
        print(f"  {key.upper()}: {df['month'].nunique()} months, "
              f"{len(sids)} unique constituents, {len(df):,} rows")
    if not frames:
        return pd.DataFrame(columns=cols)
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["benchmark", "month", "weight"],
                         ascending=[True, True, False]).reset_index(drop=True))
