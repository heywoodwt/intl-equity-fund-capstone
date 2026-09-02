"""Build the Developed-ex-US Fama-French factor table, and verify which Ken
French region the modeling repo's existing ff_factors.csv actually is.

Output columns (decimal monthly returns): Mkt_RF, SMB, HML, RMW, CMA, RF, Mom.
"""

import pandas as pd

from . import config, french

FACTOR_COLS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
# Factors that distinguish one region from another (RF is common; Mom is a
# separate file) — used for region identification.
DISCRIMINATING = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]


def _to_decimal(df):
    """Mask French sentinels (-99.99 / -999) and convert percent -> decimal."""
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= -99.0)
    return df / 100.0


def fetch_region_5f(region):
    """Return a region's 5 factors (decimal, month-end index): Mkt_RF, SMB,
    HML, RMW, CMA, RF."""
    fname = config.FRENCH_5F_REGIONS[region]
    raw = french.parse_monthly(french.fetch_zip_csv(config.FRENCH_BASE_URL + fname))
    out = _to_decimal(raw).rename(columns={"Mkt-RF": "Mkt_RF"})
    return out


def fetch_factors():
    """Developed-ex-US 5 factors + momentum, merged (decimal, month-end)."""
    f5 = fetch_region_5f("Developed_ex_US")
    mom_raw = french.parse_monthly(
        french.fetch_zip_csv(config.FRENCH_BASE_URL + config.FRENCH_MOM_FILE)
    )
    mom = _to_decimal(mom_raw).rename(columns={"WML": "Mom"})
    df = f5.join(mom[["Mom"]], how="left")
    return df[FACTOR_COLS]


def identify_region(existing_path=None, tol=5e-4):
    """Compare the modeling repo's factor file to each candidate Ken French
    region and report which one it matches.

    Returns a dict: per-region {overlap, mean_corr, max_abs_diff} plus the
    best-matching region and whether that is Developed_ex_US.
    """
    existing_path = existing_path or config.MODELING_FF_FACTORS
    theirs = pd.read_csv(existing_path, parse_dates=["date"]).set_index("date")
    cols = [c for c in DISCRIMINATING if c in theirs.columns]

    results = {}
    for region in config.FRENCH_5F_REGIONS:
        ours = fetch_region_5f(region)
        joined = theirs[cols].join(ours[cols], how="inner",
                                   lsuffix="_t", rsuffix="_o").dropna()
        if joined.empty:
            results[region] = {"overlap": 0, "mean_corr": float("nan"),
                               "max_abs_diff": float("nan")}
            continue
        corrs, diffs = [], []
        for c in cols:
            corrs.append(joined[f"{c}_t"].corr(joined[f"{c}_o"]))
            diffs.append((joined[f"{c}_t"] - joined[f"{c}_o"]).abs().max())
        results[region] = {
            "overlap": len(joined),
            "mean_corr": float(pd.Series(corrs).mean()),
            "max_abs_diff": float(max(diffs)),
        }

    ranked = sorted(
        (r for r in results if results[r]["overlap"] > 0),
        key=lambda r: results[r]["max_abs_diff"],
    )
    best = ranked[0] if ranked else None
    matched = bool(
        best and results[best]["max_abs_diff"] < tol
        and results[best]["mean_corr"] > 0.999
    )
    return {
        "per_region": results,
        "best_match": best,
        "is_developed_ex_us": bool(matched and best == "Developed_ex_US"),
        "matched_within_tol": matched,
    }


def save(df, path=None):
    path = path or (config.PROCESSED_DIR / "ff_factors_dev_ex_us.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(path, index=False)
    return path
