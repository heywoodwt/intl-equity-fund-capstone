"""Sector & country profile of the holdings, and value-weighted portfolio
breakdowns (% of portfolio value in each sector / country, per month)."""

import re

import numpy as np
import pandas as pd


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def build_holding_profile(profile_raw, fsym_to_ticker):
    """Static per-holding reference: ticker, fsym_id, entity, name, country, sector."""
    p = profile_raw.copy()
    p["ticker"] = p["fsym_id"].map(fsym_to_ticker)
    return (p.dropna(subset=["ticker"])
            [["ticker", "fsym_id", "factset_entity_id", "name", "country", "sector"]]
            .reset_index(drop=True))


def build_breakdowns(pos, holding_profile):
    """Monthly value-weighted sector & country weights + concentration metrics."""
    df = pos.merge(holding_profile[["fsym_id", "country", "sector"]],
                   on="fsym_id", how="left")
    rows = []
    for month, g in df.groupby("month"):
        total = g["value_usd"].sum()
        rec = {"month": month}
        sec = g.dropna(subset=["sector"])
        ctry = g.dropna(subset=["country"])
        if total and total > 0:
            for s, v in sec.groupby("sector")["value_usd"].sum().items():
                rec[f"sect_wt_{slug(s)}"] = v / total
            for c, v in ctry.groupby("country")["value_usd"].sum().items():
                rec[f"ctry_wt_{c}"] = v / total
            sw = sec.groupby("sector")["value_usd"].sum() / total
            rec["top_sector"] = sw.idxmax() if len(sw) else None
            rec["top_sector_wt"] = float(sw.max()) if len(sw) else np.nan
            rec["sector_hhi"] = float((sw ** 2).sum()) if len(sw) else np.nan
            rec["wt_classified_sector"] = float(sec["value_usd"].sum() / total)
            rec["wt_classified_country"] = float(ctry["value_usd"].sum() / total)
        rec["n_sectors"] = int(sec["sector"].nunique())
        rec["n_countries"] = int(ctry["country"].nunique())
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    # A sector/country absent in a month means 0 weight, not missing.
    wtcols = [c for c in out.columns if c.startswith(("sect_wt_", "ctry_wt_"))]
    out[wtcols] = out[wtcols].fillna(0.0)
    return out
