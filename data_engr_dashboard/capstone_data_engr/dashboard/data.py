"""Data loading and column metadata for the dashboard.

Self-contained: imports nothing from the pipeline package and never touches
`.env`/WRDS. It only reads the finished CSVs in `data/processed/`. This is what
makes the `dashboard/` folder liftable into a deploy repo on its own.

Data directory resolution (first hit wins):
  1. ``$CAPSTONE_DATA_DIR``                         (explicit override)
  2. ``dashboard/data/``                            (bundled copy, for deploy)
  3. ``<repo>/data/processed/``                     (this repo, for local dev)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent


def data_dir() -> Path:
    """Locate the directory holding the processed CSVs."""
    env = os.environ.get("CAPSTONE_DATA_DIR")
    candidates = [
        Path(env) if env else None,
        _HERE / "data",                       # bundled for deployment
        _HERE.parent / "data" / "processed",  # repo layout
    ]
    for c in candidates:
        if c and (c / "combined_monthly.csv").exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not find combined_monthly.csv. Looked in:\n  " + tried +
        "\nSet CAPSTONE_DATA_DIR or copy the CSVs into dashboard/data/."
    )


# --- data freeze -----------------------------------------------------------
# The dashboard is frozen to end at December 2024: any row dated later is dropped
# on read, so every view stops at the same month no matter what the underlying
# source holds. This covers the bundled CSVs (via :func:`_clip_to_freeze` in the
# loaders below); its price-side twin, for the live Yahoo pull, is
# ``etf_prices.LAST_DATE``.
LAST_MONTH = "2024-12"
LAST_DATE = pd.Period(LAST_MONTH, freq="M").to_timestamp(how="end")  # 2024-12-31


def _clip_to_freeze(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any row past the data freeze (:data:`LAST_MONTH`).

    Filters on ``date`` when present, else the ``YYYY-MM`` ``month`` string
    (its lexical order is chronological). Frames without either are returned
    unchanged.
    """
    if "date" in df.columns:
        df = df[df["date"] <= LAST_DATE]
    elif "month" in df.columns:
        df = df[df["month"] <= LAST_MONTH]
    return df.reset_index(drop=True)


# --- loaders ---------------------------------------------------------------

@lru_cache(maxsize=1)
def load_combined() -> pd.DataFrame:
    """Monthly rolling-regression dataset, with a parsed ``date`` and derived
    ``excess_ret`` (= fund_ret - RF)."""
    df = pd.read_csv(data_dir() / "combined_monthly.csv")
    df["date"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    if {"fund_ret", "RF"}.issubset(df.columns):
        df["excess_ret"] = df["fund_ret"] - df["RF"]
    return _clip_to_freeze(df.sort_values("date"))


@lru_cache(maxsize=1)
def load_pca_components() -> pd.DataFrame:
    df = pd.read_csv(data_dir() / "pca_market_components.csv")
    df["date"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return _clip_to_freeze(df.sort_values("date"))


@lru_cache(maxsize=1)
def load_pca_loadings() -> pd.DataFrame:
    df = pd.read_csv(data_dir() / "pca_market_loadings.csv")
    return df.rename(columns={df.columns[0]: "feature"})


def load_combined_with_pcs() -> pd.DataFrame:
    """Combined dataset joined with the PCA components on ``month``."""
    base = load_combined()
    pcs = load_pca_components().drop(columns=["date"])
    return base.merge(pcs, on="month", how="left")


def _load_optional(name: str) -> pd.DataFrame | None:
    """Load a processed CSV with a parsed ``date``; ``None`` if it isn't present
    (the attribution layer is optional, so the app degrades gracefully)."""
    path = data_dir() / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "month" in df.columns:
        df["date"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return _clip_to_freeze(df)


@lru_cache(maxsize=1)
def load_sector_returns() -> pd.DataFrame | None:
    """Portfolio sector contribution-to-return (month × sector)."""
    return _load_optional("portfolio_sector_returns_monthly.csv")


@lru_cache(maxsize=1)
def load_security_contrib() -> pd.DataFrame | None:
    """Security-level contribution-to-return (month × ticker)."""
    return _load_optional("security_contrib_monthly.csv")


@lru_cache(maxsize=1)
def load_attribution() -> pd.DataFrame | None:
    """Brinson-Fachler effects per (month, sector) vs the blended benchmark."""
    return _load_optional("attribution_monthly.csv")


@lru_cache(maxsize=1)
def load_attribution_summary() -> pd.DataFrame | None:
    """Monthly Brinson totals + reconciliation to the official active return."""
    return _load_optional("attribution_summary_monthly.csv")


def brinson_dir() -> Path:
    """Locate the directory holding pre-rendered Brinson-Fachler plots."""
    env = os.environ.get("BRINSON_DIR")
    candidates = [
        Path(env) if env else None,
        _HERE / "brinson_fachler",          # bundled copy
        _HERE.parent / "brinson_fachler",   # repo root layout
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return _HERE.parent / "brinson_fachler"


def get_brinson_monthly_plot(plot_type: str, month: str) -> Path | None:
    """Get path to a monthly attribution plot ('sector' or 'country') for a given month (YYYY-MM)."""
    folder = "monthly_sector_plots" if plot_type == "sector" else "monthly_country_plots"
    prefix = "attribution_sector" if plot_type == "sector" else "attribution_country"
    path = brinson_dir() / folder / f"{prefix}_{month}.png"
    return path if path.exists() else None


def get_brinson_holdings_plot(plot_type: str) -> Path | None:
    """Get path to full holdings attribution plot ('sectors' or 'countries')."""
    filename = f"attribution_holdings_{plot_type}.png"
    path = brinson_dir() / filename
    return path if path.exists() else None



# Pretty sector names for the ``sect_wt_<slug>`` family.
def sector_pretty(slug: str) -> str:
    """``sect_wt_consumer_non_cyclicals`` / ``consumer_non_cyclicals`` -> ``Consumer
    Non-Cyclicals``."""
    s = slug
    for p in ("active_sect_wt_", "bench_sect_wt_", "sect_wt_"):
        if s.startswith(p):
            s = s[len(p):]
            break
    return s.replace("_", " ").title().replace("Non ", "Non-")


# A few common ISO country names for the ``ctry_wt_<CC>`` family (fallback = code).
COUNTRY_NAMES = {
    "AT": "Austria", "AU": "Australia", "BE": "Belgium", "CA": "Canada",
    "CH": "Switzerland", "CN": "China", "DE": "Germany", "DK": "Denmark",
    "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom",
    "IE": "Ireland", "IL": "Israel", "IT": "Italy", "JP": "Japan", "MX": "Mexico",
    "NL": "Netherlands", "NO": "Norway", "SE": "Sweden", "TW": "Taiwan",
    "US": "United States", "AR": "Argentina", "UY": "Uruguay",
}


def country_pretty(slug: str) -> str:
    cc = slug[len("ctry_wt_"):] if slug.startswith("ctry_wt_") else slug
    return COUNTRY_NAMES.get(cc, cc)


# --- column groups ---------------------------------------------------------

FF_FACTORS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]

MACRO = [
    "vix", "vix_chg", "cpi_yoy", "core_cpi_yoy", "pce_yoy", "core_pce_yoy",
    "m2_yoy", "y10", "y2", "y3m", "slope_10y_2y", "slope_10y_3m",
    "baa_spread", "baa_aaa_spread", "sp500_ret", "sp500_mom_12m",
]

TARGETS = ["fund_ret", "excess_ret", "alpha_vs_avg", "alpha_vs_efa"]

BENCH_RETS = ["fund_ret", "efa_ret", "scz_ret", "vss_ret", "bench_avg_ret"]

# Selectable benchmarks for the tear sheet, in presentation order. EFA (MSCI
# EAFE, via the iShares ETF) is the default benchmark proxy; SCZ and VSS are
# small-cap alternatives, and the blend is their equal weight.
BENCHMARKS = {
    "efa_ret": "EFA — iShares MSCI EAFE (benchmark proxy)",
    "scz_ret": "SCZ — MSCI EAFE Small-Cap",
    "vss_ret": "VSS — FTSE ex-US Small-Cap",
    "bench_avg_ret": "Blended — EFA/SCZ/VSS (equal-weight)",
}
DEFAULT_BENCHMARK = "efa_ret"


def fundamentals_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ff_")]


def sector_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sect_wt_")]


def country_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ctry_wt_")]


def pc_cols(df: pd.DataFrame) -> list[str]:
    # Exclude degenerate trailing PCs (≈ zero variance — the PCA emits more
    # components than the data's rank), which would make the design singular.
    cols = [c for c in df.columns if c.startswith("PC") and c[2:].isdigit()]
    return [c for c in cols if df[c].std() > 1e-8]


# Human-readable labels for the common regressors/targets. Anything not listed
# falls back to the raw column name via :func:`label`.
LABELS = {
    "fund_ret": "Fund return",
    "excess_ret": "Fund excess return (fund - RF)",
    "alpha_vs_avg": "Active return vs benchmark avg",
    "alpha_vs_efa": "Active return vs EFA",
    "efa_ret": "EFA return", "scz_ret": "SCZ return", "vss_ret": "VSS return",
    "bench_avg_ret": "Benchmark avg return",
    "Mkt_RF": "Market - RF", "SMB": "Size (SMB)", "HML": "Value (HML)",
    "RMW": "Profitability (RMW)", "CMA": "Investment (CMA)", "Mom": "Momentum",
    "vix": "VIX (level)", "vix_chg": "VIX change",
    "cpi_yoy": "CPI YoY", "core_cpi_yoy": "Core CPI YoY",
    "pce_yoy": "PCE YoY", "core_pce_yoy": "Core PCE YoY", "m2_yoy": "M2 YoY",
    "y10": "10y yield", "y2": "2y yield", "y3m": "3m yield",
    "slope_10y_2y": "Curve 10y-2y", "slope_10y_3m": "Curve 10y-3m",
    "baa_spread": "BAA spread", "baa_aaa_spread": "BAA-AAA spread",
    "sp500_ret": "S&P 500 return", "sp500_mom_12m": "S&P 500 12m momentum",
}


# Interpretation of each principal component, from its loadings (see the Glossary
# tab). PC sign is arbitrary; descriptions follow the orientation in the data.
# Explained-variance % is shown live from the loadings file, not hard-coded.
PC_INFO: dict[str, dict[str, str]] = {
    "PC1": {"name": "Rates & inflation regime", "desc":
        "The dominant macro axis. High PC1 = low short- and long-term Treasury "
        "yields, low CPI/PCE inflation, and a steep yield curve (an accommodative / "
        "easing regime); low PC1 = high rates, high inflation, flat-to-inverted "
        "curve (tightening)."},
    "PC2": {"name": "Equity risk appetite (risk-on/off)", "desc":
        "High = strong S&P 500 / market returns with a falling and low VIX and tight "
        "credit spreads (risk-on); low = equity sell-offs, volatility spikes, and "
        "widening credit spreads (risk-off)."},
    "PC3": {"name": "Credit stress & flight-to-quality", "desc":
        "High = wide credit spreads (BAA-AAA, BAA) alongside a positive profitability "
        "premium (RMW) and negative value (HML) and momentum — i.e. stress periods "
        "when high-quality stocks outperform cheap ones."},
    "PC4": {"name": "Inflation & money-supply growth", "desc":
        "High = elevated CPI/PCE inflation and fast M2 growth with calmer volatility "
        "and lower long yields — a reflationary backdrop, distinct from the "
        "rate-level axis (PC1)."},
    "PC5": {"name": "Momentum vs. value style", "desc":
        "High = strong price momentum (stock Mom and S&P 12-month momentum) and "
        "profitability, with the value (HML) and conservative-investment (CMA) "
        "premia negative — a momentum/growth-over-value tilt."},
    "PC6": {"name": "Volatility & liquidity", "desc":
        "High = elevated VIX together with faster M2 growth, a positive size tilt "
        "(SMB), and a flatter curve — a higher-volatility, liquidity-supported regime."},
    "PC7": {"name": "Size premium (SMB)", "desc":
        "Almost entirely the small-minus-big size factor (loading ≈ 0.85). High = "
        "small caps outperforming large caps."},
    "PC8": {"name": "Momentum factor (Mom)", "desc":
        "Dominated by the momentum factor (loading ≈ 0.70) with calmer volatility. "
        "High = winners-minus-losers outperforming."},
    "PC9": {"name": "Profitability & rates", "desc":
        "Minor axis: profitability (RMW) co-moving with long yields and the S&P."},
    "PC10": {"name": "Quality factors vs. curve", "desc":
        "Minor axis: profitability/investment factors (RMW, CMA) opposite the "
        "yield-curve slopes."},
    "PC11": {"name": "Trend & credit residual", "desc":
        "Minor axis: mainly S&P 12-month momentum and credit spreads."},
    "PC12": {"name": "Market-return residual", "desc":
        "Minor axis: residual market-return / VIX-change combination."},
    "PC13": {"name": "Value & credit residual", "desc":
        "Minor axis: value (HML) and credit spreads versus the investment factor (CMA)."},
    "PC14": {"name": "Market vs. value residual", "desc":
        "Minor axis: S&P return / CMA against the value premium (HML)."},
    "PC15": {"name": "Liquidity vs. volatility residual", "desc":
        "Minor axis: M2 growth against the VIX."},
    "PC16": {"name": "Market-factor residual", "desc":
        "Minor axis: offsetting S&P-return vs. Mkt-RF components."},
    "PC17": {"name": "Credit-spread residual", "desc":
        "Minor axis: BAA spread against the S&P level."},
    "PC18": {"name": "Yield-curve shape residual", "desc":
        "Minor axis: 10y-2y versus 10y-3m slope difference."},
    "PC19": {"name": "Inflation-mix residual", "desc":
        "Minor axis: headline-vs-core inflation differences (negligible variance)."},
    "PC20": {"name": "Inflation-component residual", "desc":
        "Minor axis: core CPI vs. core PCE (negligible variance)."},
    "PC21": {"name": "Inflation-component residual", "desc":
        "Minor axis: PCE vs. CPI (negligible variance)."},
}

# Fuller descriptions for the non-PC rolling-regression variables (glossary table).
GLOSSARY: dict[str, str] = {
    # targets
    "fund_ret": "Monthly total return of the fund.",
    "excess_ret": "Fund return minus the risk-free rate (RF). In a factor-return "
                  "model the intercept on this is the fund's alpha.",
    "alpha_vs_avg": "Fund return minus the average of the three benchmark ETFs "
                    "(EFA, SCZ, VSS) — active return vs the blended benchmark.",
    "alpha_vs_efa": "Fund return minus EFA (developed large-cap) — active return vs EFA.",
    # Fama-French (Developed ex-US, decimal monthly)
    "Mkt_RF": "Market excess return (market minus risk-free) — the broad equity factor.",
    "SMB": "Small-minus-big — the size premium (small caps minus large caps).",
    "HML": "High-minus-low — the value premium (cheap minus expensive on book/price).",
    "RMW": "Robust-minus-weak — the profitability premium.",
    "CMA": "Conservative-minus-aggressive — the investment premium (low vs high asset growth).",
    "Mom": "Winners-minus-losers — the momentum premium.",
    # macro
    "vix": "CBOE VIX — implied equity volatility, level.",
    "vix_chg": "Month-over-month change in the VIX.",
    "cpi_yoy": "Headline CPI inflation, year-over-year (%).",
    "core_cpi_yoy": "Core CPI (ex food & energy) inflation, YoY (%).",
    "pce_yoy": "Headline PCE inflation, YoY (%).",
    "core_pce_yoy": "Core PCE inflation, YoY (%).",
    "m2_yoy": "M2 money-supply growth, YoY (%).",
    "y10": "10-year Treasury yield (%).",
    "y2": "2-year Treasury yield (%).",
    "y3m": "3-month Treasury yield (%).",
    "slope_10y_2y": "Yield-curve slope, 10y minus 2y (= y10 − y2).",
    "slope_10y_3m": "Yield-curve slope, 10y minus 3m (= y10 − y3m).",
    "baa_spread": "Moody's BAA corporate yield minus the 10-year Treasury — credit risk premium.",
    "baa_aaa_spread": "BAA minus AAA corporate yield — credit quality spread.",
    "sp500_ret": "S&P 500 monthly return.",
    "sp500_mom_12m": "S&P 500 trailing 12-month momentum.",
}


def label(col: str) -> str:
    if col in PC_INFO:
        return f"{PC_INFO[col]['name']} ({col})"
    return LABELS.get(col, col)


def pc_name(pc: str) -> str:
    return PC_INFO.get(pc, {}).get("name", pc)


def pc_desc(pc: str) -> str:
    return PC_INFO.get(pc, {}).get("desc", "")


def pc_loadings(pc: str) -> pd.Series:
    """Feature loadings (PC coefficients) for one component, metadata rows excluded."""
    L = load_pca_loadings().set_index("feature")
    feats = [f for f in L.index if not str(f).startswith("_")]
    return L.loc[feats, pc]


def pc_explained_var(pc: str) -> float:
    L = load_pca_loadings().set_index("feature")
    return float(L.loc["_explained_var_ratio", pc])


def glossary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Every variable available in the rolling regression, with a description.
    Targets/factors/macro/PCs are listed individually; the large fundamentals and
    sector/country families are summarized at the group level (column-level detail
    lives in engineered_data/README.md)."""
    rows: list[tuple[str, str, str, str]] = []
    for c in TARGETS:
        rows.append(("Target (dependent)", c, label(c), GLOSSARY.get(c, "")))
    for c in FF_FACTORS:
        rows.append(("Fama-French factor", c, label(c), GLOSSARY.get(c, "")))
    for c in MACRO:
        rows.append(("Macro", c, label(c), GLOSSARY.get(c, "")))
    for c in pc_cols(df):
        rows.append(("Principal component", c, label(c),
                     f"{pc_desc(c)} (explains {pc_explained_var(c) * 100:.1f}% of "
                     f"macro/market variance)."))
    rows.append(("Portfolio fundamentals", "ff_*", "Portfolio fundamentals",
                 "Portfolio-weighted aggregates of holding-level FactSet fundamentals "
                 "(valuation, quality, growth). Suffixes: _wmean = value-weighted mean, "
                 "_median = median, _whmean = weighted harmonic mean. 2018-2022 only."))
    rows.append(("Sector weights", "sect_wt_*", "Portfolio sector weights",
                 "Share of the portfolio in each RBICS L1 sector. 2018-2022 only."))
    rows.append(("Country weights", "ctry_wt_*", "Portfolio country weights",
                 "Share of the portfolio by issuer domicile. 2018-2022 only."))
    return pd.DataFrame(rows, columns=["Group", "Code", "Name", "Description"])


# A curated starter model for the rolling regression, kept short enough to fit a
# rolling OLS and plot cleanly. It deliberately mixes the three families so the
# contribution decomposition can answer one question: is the fund's active return
# driven by systematic macro/factor exposure or by the manager's stock selection?
#   Mkt_RF, SMB          — Fama-French: broad market & size (systematic)
#   PC1, PC2             — macro regimes: rates/inflation & risk-on-off (systematic)
#   ff_earn_yld_median   — portfolio earnings yield: the value tilt of the picks
#   ff_roce_median       — portfolio profitability: the quality tilt of the picks
# Six regressors keeps the rolling OLS estimable and the coefficient plot legible.
# The fundamentals restrict the fit to 2018-04 – 2022-12.
ALPHA_DRIVERS_PRESET_NAME = "⭐ Alpha drivers: macro/factor vs. stock-picking"
ALPHA_DRIVERS_PRESET = [
    "Mkt_RF", "SMB", "PC1", "PC2", "ff_earn_yld_median", "ff_roce_median",
]


def regressor_catalog(df: pd.DataFrame) -> dict[str, list[str]]:
    """Named groups of candidate regressors, in presentation order.

    The first entry is the curated, plottable :data:`ALPHA_DRIVERS_PRESET` that
    mixes factors, macro PCs, and portfolio fundamentals; the rest are the raw
    families to fine-tune from.
    """
    return {
        ALPHA_DRIVERS_PRESET_NAME: [c for c in ALPHA_DRIVERS_PRESET if c in df.columns],
        "Fama-French factors": [c for c in FF_FACTORS if c in df.columns],
        "Macro": [c for c in MACRO if c in df.columns],
        "Principal components": pc_cols(df),
        "Portfolio fundamentals": fundamentals_cols(df),
        "Sector weights": sector_cols(df),
        "Country weights": country_cols(df),
    }
