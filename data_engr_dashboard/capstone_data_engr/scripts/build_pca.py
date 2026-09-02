#!/usr/bin/env python3
"""PCA feature engineering on the combined monthly dataset (offline).

Keeps `combined_monthly.csv` (the explainable originals) untouched and emits
principal components + loadings as separate, joinable files. Two feature blocks:
- market  : macro + FF factor returns        (full 2014-2025 window)
- full    : market + portfolio fundamentals   (the 2018-2022 holdings window)

    uv run python scripts/build_pca.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from capstone_data import config, pca  # noqa: E402

FACTOR_COLS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]  # exclude RF (rate level)


def _macro_cols():
    cols = pd.read_csv(config.PROCESSED_DIR / "macro_monthly.csv", nrows=0).columns
    return [c for c in cols if c not in ("date", "month")]


def _fundamental_cols():
    # One representative aggregation per base feature: harmonic for price
    # multiples, weighted mean for the rest.
    return [f"{b}_whmean" if b in config.FACTSET_PRICE_MULTIPLES else f"{b}_wmean"
            for b in config.FACTSET_FEATURES]


def main():
    monthly = pd.read_csv(config.PROCESSED_DIR / "combined_monthly.csv")

    market_feats = _macro_cols() + FACTOR_COLS
    full_feats = market_feats + _fundamental_cols()

    fund_cols = [c for c in _fundamental_cols() if c in monthly.columns]
    blocks = {"market": market_feats, "full": full_feats}
    for name, feats in blocks.items():
        feats = [c for c in feats if c in monthly.columns]
        # Restrict the full block to the fundamentals window (else macro/factors
        # are always present and the empty fundamental months pollute the fit).
        data = monthly[monthly[fund_cols].notna().any(axis=1)] if name == "full" else monthly
        comps, load = pca.run(data, feats)
        cpath = config.PROCESSED_DIR / f"pca_{name}_components.csv"
        lpath = config.PROCESSED_DIR / f"pca_{name}_loadings.csv"
        comps.to_csv(cpath, index=False)
        load.to_csv(lpath)
        k95 = int((load.loc["_cumulative"] < 0.95).sum() + 1)
        print(f"[{name}] {len(feats)} features, {len(comps)} months, "
              f"{comps.shape[1] - 1} PCs | {k95} PCs reach 95% var")
        print(f"  top-5 explained var: "
              f"{[round(x, 3) for x in load.loc['_explained_var_ratio'].head(5)]}")
        print(f"  -> {cpath.name}, {lpath.name}")


if __name__ == "__main__":
    main()
