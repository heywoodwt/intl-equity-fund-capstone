"""FactSet Fundamentals International/US: identifier crosswalk + extraction.

Join path: holding's Yahoo ticker -> FactSet `ticker_region` -> `fsym_id`
(regional, `-R`) -> the `ff_advanced_der_af_*` ratio tables and the monthly
price tables.
"""

import pandas as pd

from . import config, holdings, wrds_io


def map_ticker_regions(db, ticker_regions):
    """Map FactSet `ticker_region` strings to `fsym_id` (current symbology first,
    then the historical table for anything not currently listed).

    Returns a DataFrame: ticker_region, fsym_id, source ('current'|'hist').
    """
    targets = sorted({t for t in ticker_regions if t})
    if not targets:
        return pd.DataFrame(columns=["ticker_region", "fsym_id", "source"])

    cur = wrds_io.query(
        db,
        "select fsym_id, ticker_region from factset.sym_ticker_region "
        "where ticker_region in %(t)s",
        params={"t": tuple(targets)},
    )
    cur["source"] = "current"

    missing = [t for t in targets if t not in set(cur["ticker_region"])]
    frames = [cur]
    if missing:
        try:
            hist = wrds_io.query(
                db,
                "select distinct fsym_id, ticker_region "
                "from factset.sym_ticker_region_hist where ticker_region in %(t)s",
                params={"t": tuple(missing)},
            )
            hist["source"] = "hist"
            frames.append(hist)
        except Exception:  # noqa: BLE001 — hist table is best-effort
            pass
    return pd.concat(frames, ignore_index=True)


def crosswalk(db, yahoo_tickers):
    """Map the fund's Yahoo tickers to FactSet `fsym_id`.

    Returns: yahoo_ticker, ticker_region, fsym_id, source.
    """
    base = pd.DataFrame({"yahoo_ticker": sorted(set(yahoo_tickers))})
    base["ticker_region"] = base["yahoo_ticker"].map(holdings.to_ticker_region)
    mapped = (map_ticker_regions(db, base["ticker_region"].dropna())
              .sort_values("source").drop_duplicates("ticker_region"))
    out = base.merge(mapped, on="ticker_region", how="left")
    out["source"] = out["source"].where(
        out["fsym_id"].notna(),
        out["ticker_region"].isna().map({True: "no_region", False: "unmatched"}),
    )
    return out


def _union_query(db, tables, columns, id_col, date_col, fsym_ids, start, end):
    """Run the same SELECT across region tables and concatenate."""
    cols = ", ".join(columns)
    frames = []
    for lib, tbls in tables.items():
        tbls = tbls if isinstance(tbls, list) else [tbls]
        for tbl in tbls:
            sql = (f"select {id_col}, {date_col}, {cols} from {lib}.{tbl} "
                   f"where {id_col} in %(ids)s "
                   f"and {date_col} >= %(s)s and {date_col} <= %(e)s")
            frames.append(wrds_io.query(
                db, sql, params={"ids": tuple(fsym_ids), "s": start, "e": end}))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_fundamentals(db, fsym_ids, start=None, end=None):
    """Derived annual ratios for the given securities (long: fsym_id, date,
    currency, <features>)."""
    return _union_query(
        db, config.FACTSET_FUND_TABLES,
        ["currency"] + config.FACTSET_FEATURES,
        "fsym_id", "date", fsym_ids,
        start or config.FUND_FETCH_START, end or config.PANEL_END,
    )


def fetch_profile(db, fsym_ids):
    """Per-security profile: country (entity domicile) + RBICS L1 sector.

    Returns: fsym_id, factset_entity_id, name, country, sector.
    """
    parts = []
    for lib, tbl in config.FACTSET_PRICE_TABLES.items():
        parts.append(wrds_io.query(
            db, f"select distinct fsym_id, factset_entity_id from {lib}.{tbl} "
                f"where fsym_id in %(ids)s", params={"ids": tuple(fsym_ids)}))
    prof = (pd.concat([p for p in parts if not p.empty], ignore_index=True)
            .dropna(subset=["factset_entity_id"]).drop_duplicates("fsym_id"))
    eids = tuple(prof["factset_entity_id"].unique())

    country = wrds_io.query(
        db, "select factset_entity_id, iso_country as country, "
            "entity_proper_name as name from factset_common.sym_entity "
            "where factset_entity_id in %(e)s", params={"e": eids})

    sector = wrds_io.query(
        db, "select s.factset_entity_id, s.focus_flag, r.l1_name as sector "
            "from factset.sym_entity_sector_rbics s "
            "join factset.rbics_structure_l2_curr r on r.l2_id = s.l2_id "
            "where s.factset_entity_id in %(e)s", params={"e": eids})
    sector = (sector.sort_values("focus_flag", ascending=False)
              .drop_duplicates("factset_entity_id")[["factset_entity_id", "sector"]])

    return (prof.merge(country, on="factset_entity_id", how="left")
            .merge(sector, on="factset_entity_id", how="left"))


def fetch_prices(db, fsym_ids, start=None, end=None):
    """Month-end price / shares / return per security."""
    return _union_query(
        db, config.FACTSET_PRICE_TABLES,
        ["iso_currency", "price_m", "ff_shs_out", "one_month_return"],
        "fsym_id", "price_date", fsym_ids,
        start or config.PANEL_START, end or config.PANEL_END,
    )


def fetch_unadj_prices(db, fsym_ids, start=None, end=None):
    """Actual (UNADJUSTED) month-end price per holding, keyed by the regional
    ``-R`` fsym_id the holdings use.

    The cs3 ``price_m`` from :func:`fetch_prices` is a split/dividend-**adjusted**
    level — correct for *returns* (ratios) but wrong for *valuation*: for names
    with large cumulative corporate-action factors (e.g. Orpea's 2023
    dilution/reverse-split) the historical adjusted price is ~80x the price that
    actually traded, so ``Shares × price_m`` explodes the position's weight. The
    real price lives in ``factset_own.own_sec_prices_eq.unadj_price``, keyed by the
    security ``-S`` id; we map ``-R -> -S`` via ``factset.sym_coverage`` (keeping
    the ``-S`` with the most observations when a regional id has several listings).
    """
    start = start or config.PANEL_START
    end = end or config.PRICE_FETCH_END
    regional = tuple(sorted({i for i in fsym_ids if i}))
    empty = pd.DataFrame(columns=["fsym_id", "month", "unadj_price"])
    if not regional:
        return empty
    cov = wrds_io.query(
        db,
        "select fsym_security_id, fsym_regional_id from factset.sym_coverage "
        "where fsym_regional_id in %(i)s", params={"i": regional})
    cov = cov.dropna(subset=["fsym_security_id", "fsym_regional_id"])
    sec_ids = tuple(sorted(cov["fsym_security_id"].unique()))
    if not sec_ids:
        return empty
    px = wrds_io.query(
        db,
        "select fsym_id, price_date, unadj_price from factset_own.own_sec_prices_eq "
        "where fsym_id in %(i)s and price_date >= %(s)s and price_date <= %(e)s "
        "and unadj_price is not null",
        params={"i": sec_ids, "s": start, "e": end})
    if px.empty:
        return empty
    px = px.merge(cov, left_on="fsym_id", right_on="fsym_security_id", how="inner")
    # If a regional id maps to several securities, keep the best-covered one.
    best = (px.groupby(["fsym_regional_id", "fsym_security_id"]).size()
            .reset_index(name="n").sort_values("n", ascending=False)
            .drop_duplicates("fsym_regional_id"))
    px = px.merge(best[["fsym_regional_id", "fsym_security_id"]],
                  on=["fsym_regional_id", "fsym_security_id"], how="inner")
    px["month"] = pd.to_datetime(px["price_date"]).dt.to_period("M").astype(str)
    # Drop the -S id (named fsym_id) before renaming the -R id to fsym_id.
    out = px[["fsym_regional_id", "month", "unadj_price"]].rename(
        columns={"fsym_regional_id": "fsym_id"})
    return out.drop_duplicates(["fsym_id", "month"]).reset_index(drop=True)
