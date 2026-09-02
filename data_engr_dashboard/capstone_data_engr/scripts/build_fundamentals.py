#!/usr/bin/env python3
"""Build the FactSet fundamentals panel + value-weighted monthly portfolio
fundamentals for the fund.

    uv run python scripts/build_fundamentals.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd  # noqa: E402

from capstone_data import config, factset, fundamentals, holdings, profile, wrds_io  # noqa: E402


def main():
    tickers = holdings.read_holding_tickers()
    holdings_df = pd.read_csv(config.HOLDINGS_FILE)

    db = wrds_io.connect()
    try:
        xwalk = factset.crosswalk(db, tickers)
        matched = xwalk.dropna(subset=["fsym_id"])
        print(f"Crosswalk: {len(matched)}/{len(xwalk)} holdings -> fsym_id")
        miss = xwalk.loc[xwalk["fsym_id"].isna(), "yahoo_ticker"].tolist()
        if miss:
            print("  unmatched:", ", ".join(miss))

        fsym_ids = sorted(matched["fsym_id"].unique())
        fsym_to_ticker = dict(zip(matched["fsym_id"], matched["yahoo_ticker"]))
        ticker_to_fsym = dict(zip(matched["yahoo_ticker"], matched["fsym_id"]))

        fund_raw = factset.fetch_fundamentals(db, fsym_ids)
        prices = factset.fetch_prices(db, fsym_ids, end=config.PRICE_FETCH_END)
        unadj = factset.fetch_unadj_prices(db, fsym_ids, end=config.PRICE_FETCH_END)
        profile_raw = factset.fetch_profile(db, fsym_ids)
    finally:
        db.close()

    print(f"Fetched: {len(fund_raw)} fundamental rows, {len(prices)} price rows, "
          f"{len(unadj)} unadjusted-price rows ({unadj['fsym_id'].nunique()} names)")

    panel = fundamentals.build_panel(fund_raw, fsym_to_ticker)
    stock_rets = fundamentals.build_stock_returns(prices, fsym_to_ticker)
    pos = fundamentals.position_values(prices, holdings_df, ticker_to_fsym,
                                       unadj_prices=unadj)
    pos = fundamentals.correct_split_basis(pos, stock_rets)
    portfolio = fundamentals.build_portfolio(panel, pos)
    hold_profile = profile.build_holding_profile(profile_raw, fsym_to_ticker)
    breakdowns = profile.build_breakdowns(pos, hold_profile)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(config.PROCESSED_DIR / "fundamentals_panel.csv", index=False)
    portfolio.to_csv(config.PROCESSED_DIR / "portfolio_fundamentals_monthly.csv", index=False)
    stock_rets.to_csv(config.PROCESSED_DIR / "stock_returns_monthly.csv", index=False)
    hold_profile.to_csv(config.PROCESSED_DIR / "holding_profile.csv", index=False)
    breakdowns.to_csv(config.PROCESSED_DIR / "portfolio_sector_country_monthly.csv", index=False)
    pos[["ticker", "fsym_id", "month", "iso_currency", "price_m", "Shares",
         "value_usd", "usd_per_ccy"]] \
        .to_csv(config.PROCESSED_DIR / "position_values_monthly.csv", index=False)
    print(f"Stock returns: {stock_rets['ticker'].nunique()} tickers "
          f"(ret_price mean {stock_rets['ret_price'].mean():.4f}, "
          f"ret_factset mean {stock_rets['ret_factset'].mean():.4f})")
    print(f"Profile: {hold_profile['country'].nunique()} countries, "
          f"{hold_profile['sector'].nunique()} sectors | "
          f"country known {hold_profile['country'].notna().mean():.0%}, "
          f"sector known {hold_profile['sector'].notna().mean():.0%}")

    any_cov = panel[config.FACTSET_FEATURES].notna().any(axis=1).mean()
    print(f"\nPanel: {panel['ticker'].nunique()} tickers x {panel['month'].nunique()} months "
          f"= {len(panel)} rows")
    print(f"  any-feature populated in {any_cov:.0%} of stock-months")
    print(f"Portfolio: {len(portfolio)} months x {portfolio.shape[1]} cols")
    cols = ["month", "n_valued", "ff_pe_dil_wmean", "ff_pe_dil_whmean",
            "ff_pe_dil_median", "ff_roce_wmean", "ff_div_yld_wmean"]
    print(portfolio[cols].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
