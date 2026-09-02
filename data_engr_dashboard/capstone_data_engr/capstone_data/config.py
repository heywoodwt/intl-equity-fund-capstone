"""Source definitions and paths for the data pipeline."""

from pathlib import Path

# --- Paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# History window. The fund's universe data starts ~2006 and its performance
# series ~2014; we pull from 2000 so trailing-window features (12m YoY,
# momentum) are fully populated before the dates we actually use.
START_DATE = "2000-01-01"

# --- FRED series: FRED id -> output column name -------------------------------
# These are raw levels/indexes. Derived features (YoY, curve slope) live in
# macro.py. Docs: https://fred.stlouisfed.org/
FRED_SERIES = {
    "VIXCLS":       "vix",         # CBOE Volatility Index (daily)
    "CPIAUCSL":     "cpi",         # CPI-U, all items, SA (monthly index)
    "CPILFESL":     "core_cpi",    # Core CPI, ex food & energy
    "PCEPI":        "pce",         # PCE price index
    "PCEPILFE":     "core_pce",    # Core PCE price index
    "M2SL":         "m2",          # M2 money stock, SA ($B, monthly)
    "DGS10":        "y10",         # 10y Treasury constant maturity (%)
    "DGS2":         "y2",          # 2y Treasury (%)
    "DGS3MO":       "y3m",         # 3m Treasury (%)
    "BAA10Y":       "baa_spread",  # Moody's Baa - 10y Treasury (%)
    "AAA":          "aaa_yield",   # Moody's Seasoned Aaa corporate bond yield (%)
    "BAA":          "baa_yield",   # Moody's Seasoned Baa corporate bond yield (%)
}
# NOTE: dropped ICE BofA OAS (BAMLC0A0CM/BAMLH0A0HYM2) — FRED truncated those to
# 2023-06+ (ICE licensing). Moody's Baa-10y and Baa-Aaa cover credit, full history.

# Economic *releases* are published with a lag: the figure for reference month M
# only becomes public in ~M+1. To keep features point-in-time (no look-ahead),
# the derived feature for these is shifted forward by this many months.
# Market series (vix, yields, spreads, sp500) are observable in real time -> 0.
RELEASE_LAG_MONTHS = {
    "cpi": 1, "core_cpi": 1, "pce": 1, "core_pce": 1, "m2": 1,
}

# --- Yahoo Finance: ticker -> output column -----------------------------------
YAHOO_SERIES = {
    "^GSPC": "sp500",   # S&P 500 index level (long history, no API key)
}

# --- Fama-French factors (Ken French Data Library) ----------------------------
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# The region we WANT for this developed-ex-US fund.
FRENCH_5F_FILE = "Developed_ex_US_5_Factors_CSV.zip"
FRENCH_MOM_FILE = "Developed_ex_US_Mom_Factor_CSV.zip"

# Candidate 5-factor regions used to *identify* which set the modeling repo's
# ff_factors.csv actually is. (All Developed-family files start 1990-07; the US
# research file starts 1963-07.)
FRENCH_5F_REGIONS = {
    "Developed_ex_US":        "Developed_ex_US_5_Factors_CSV.zip",
    "Developed":              "Developed_5_Factors_CSV.zip",
    "North_America":          "North_America_5_Factors_CSV.zip",
    "Europe":                 "Europe_5_Factors_CSV.zip",
    "Japan":                  "Japan_5_Factors_CSV.zip",
    "Asia_Pacific_ex_Japan":  "Asia_Pacific_ex_Japan_5_Factors_CSV.zip",
    "US":                     "F-F_Research_Data_5_Factors_2x3_CSV.zip",
}

# Modeling repo (sibling of this one) and its existing factor file, for the
# region verification.
MODELING_REPO = REPO_ROOT.parent / "capstone"
MODELING_FF_FACTORS = MODELING_REPO / "ds6050 project" / "data" / "ff_factors.csv"

# Fund holdings (from the modeling repo) used to build the FactSet crosswalk.
HOLDINGS_FILE = MODELING_REPO / "data" / "monthly_holdings.csv"

# Modeling-repo monthly performance series (% strings) + wide stock returns,
# used by the combine step.
_MODELING_DATA = MODELING_REPO / "ds6050 project" / "data"
PERF_FILES = {
    "fund": _MODELING_DATA / "2014_2025_dataset_Monthly.csv",
    "efa": _MODELING_DATA / "2014_2025_EFA_Monthly.csv",
    "scz": _MODELING_DATA / "2014_2025_SCZ_Monthly.csv",
    "vss": _MODELING_DATA / "2014_2025_VSS_Monthly.csv",
}
MONTHLY_RETURNS_FILE = _MODELING_DATA / "monthly_returns.csv"

# --- Benchmark ETFs: point-in-time sector weights -----------------------------
# The fund is benchmarked vs these three ETFs (same set as PERF_FILES). We pull
# each one's month-end holdings from FactSet Ownership and roll them up to RBICS
# L1 sector weights — the SAME taxonomy as the fund's own holdings — so fund-vs-
# benchmark sector tilts are point-in-time (no look-ahead) and apples-to-apples.
# key -> FactSet Ownership fund ticker. Verified FactSet fund ids (resolved at
# build time from factset_own.own_ent_fund_identifiers):
#   EFA=04BZJ5-E (iShares MSCI EAFE), SCZ=04D200-E (iShares MSCI EAFE Small-Cap),
#   VSS=04DVTB-E (Vanguard FTSE All-World ex-US Small-Cap).
BENCHMARK_ETFS = {"efa": "EFA", "scz": "SCZ", "vss": "VSS"}
# Window for the benchmark holdings pull — spans the performance series in
# combined_monthly. (Fund-vs-benchmark tilts only resolve where the fund's own
# sector weights exist, i.e. the 2018-2022 holdings window.)
BENCH_START = "2014-01-01"
BENCH_END = "2025-12-31"

# --- FactSet Fundamentals -----------------------------------------------------
# Derived annual ratio tables by region (int = EU + Asia-Pacific; usc = US/Canada
# for the fund's US-listed ADRs). Same column set across all three.
FACTSET_FUND_TABLES = {
    "factset_ff_int": ["ff_advanced_der_af_eu", "ff_advanced_der_af_ap"],
    "factset_ff_usc": ["ff_advanced_der_af_am"],
}
FACTSET_PRICE_TABLES = {
    "factset_ff_int": "cs3_monthly_prices_final_int",
    "factset_ff_usc": "cs3_monthly_prices_final_usc",
}
# Valuation / quality / growth ratios to extract (all verified present).
FACTSET_FEATURES = [
    # valuation
    "ff_pe_dil", "ff_pbk_tang", "ff_psales_dil", "ff_pfcf", "ff_div_yld",
    "ff_earn_yld", "ff_fcf_yld", "ff_entrpr_val_ebitda_oper", "ff_entrpr_val_sales",
    # quality  (ff_roea/ff_pfcf_dil are empty in these tables; ff_roce is the ROE proxy)
    "ff_roce", "ff_roic", "ff_roa_ptx", "ff_ebit_oper_mgn", "ff_ebitda_oper_mgn",
    "ff_debt_eq", "ff_tcap_assets", "ff_ebit_oper_int_covg", "ff_zscore", "ff_fscore",
    # growth
    "ff_sales_gr", "ff_eps_dil_gr", "ff_eps_basic_gr", "ff_bps_gr", "ff_com_eq_gr",
    "ff_assets_gr",
]
# Cross-sectional winsorization (per month) before portfolio weighting — ratio
# fields have heavy outliers (e.g. P/E of near-zero-earnings names).
WINSOR_QUANTILES = (0.05, 0.95)

# Price-per-fundamental multiples: these aggregate correctly as a value-weighted
# HARMONIC mean (= aggregate price / aggregate fundamental). The rest (yields,
# margins, returns, growth, scores) use the arithmetic weighted mean.
FACTSET_PRICE_MULTIPLES = [
    "ff_pe_dil", "ff_pbk_tang", "ff_psales_dil", "ff_pfcf",
    "ff_entrpr_val_ebitda_oper", "ff_entrpr_val_sales",
]
# Annual fundamentals are public only months after fiscal year-end; lag by this
# many months so the panel is point-in-time (no look-ahead).
FUND_REPORTING_LAG_MONTHS = 4

# Fund analysis window (holdings/trades cover 2018-2022). Pull fundamentals from
# earlier so point-in-time values exist at the window start.
PANEL_START = "2018-01-01"
PANEL_END = "2022-12-31"
FUND_FETCH_START = "2015-01-01"
# Pull prices a few months past the window so next_ret is defined at the last
# panel month (next_ret at 2022-12 needs the 2023-01 return).
PRICE_FETCH_END = "2023-03-31"

# --- iShares ETF holdings: ticker -> product holdings-CSV URL ------------------
# iShares publishes each product's full holdings as CSV at a stable ajax endpoint.
# Extend the universe by adding a line. URLs are the "Detailed Holdings and
# Analytics" CSV download from each product page (fileType=csv&dataType=fund).
ISHARES_ETFS = {
    "EFA":  "https://www.ishares.com/us/products/239623/ishares-msci-eafe-etf/1467271812596.ajax?fileType=csv&fileName=EFA_holdings&dataType=fund",
    "IEFA": "https://www.ishares.com/us/products/244049/ishares-core-msci-eafe-etf/1467271812596.ajax?fileType=csv&fileName=IEFA_holdings&dataType=fund",
    "ACWX": "https://www.ishares.com/us/products/239641/ishares-msci-acwi-ex-us-etf/1467271812596.ajax?fileType=csv&fileName=ACWX_holdings&dataType=fund",
    "SCZ":  "https://www.ishares.com/us/products/239627/ishares-msci-eafe-smallcap-etf/1467271812596.ajax?fileType=csv&fileName=SCZ_holdings&dataType=fund",
    "IEMG": "https://www.ishares.com/us/products/244050/ishares-core-msci-emerging-markets-etf/1467271812596.ajax?fileType=csv&fileName=IEMG_holdings&dataType=fund",
    "URTH": "https://www.ishares.com/us/products/239696/ishares-msci-world-etf/1467271812596.ajax?fileType=csv&fileName=URTH_holdings&dataType=fund",
}
