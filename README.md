# International Equity Fund Capstone for UVA MSDS '26

**Muhammad Amjad, Thomas Blalock, Finn Sjue, John Twomey, Heywood Williams-Tracy**

**DS-6015 capstone.** Two questions run through every part of the repo:

1. **Explain** the fund's performance — how much comes from market / size / value /
   momentum exposure, from macro regime, and from country/sector allocation vs.
   security selection, and how much is genuine manager alpha.
2. **Predict** it — monthly portfolio return (time-series) and cross-sectional
   stock returns (panel / deep-learning).

The subject is an actively managed, developed-markets **ex-US** small/mid-cap
equity fund (~130 holdings, ~1,184 trades, 2018–2022), judged against **EFA**
(MSCI EAFE), **SCZ** (EAFE small-cap), and **VSS** (FTSE ex-US small-cap).

## Public release & data privacy

This repository has been edited for public release to remove information that
identifies the underlying fund. Specifically:

- **Fund identity removed.** The fund's legal name, ticker, account code, its
  investment adviser, and the names of individuals (trustees, officers, payees)
  have been stripped from all code, documentation, notebooks, and data files and
  replaced with generic placeholders (referred to throughout as "the Fund", and
  as `FUND` where code requires a ticker symbol).
- **Raw custody statements removed.** The source PDF trading/custody statements
  were **deleted in full** to comply with the non-disclosure obligations
  attached to those sensitive records. Only the derived, de-identified
  transaction tables (dates, securities, quantities, prices) remain.
- **History squashed.** Commit history was collapsed into a single commit so no
  earlier revision retains the removed material.

The analytical content — holdings, trades, returns, and every engineered
feature — is unchanged; only identifying metadata was removed.

## What's in this repo

| Path | What it is |
|------|------------|
| **`data/`** | Trade-extraction scripts, cleaned trade / holdings / price / return tables, and `data/engineered_data/` — the modeling-ready datasets (see below). |
| **`data_engr_dashboard/capstone_data_engr/`** | Self-contained `uv` project: the **data-engineering pipeline** (one module per external source → tidy point-in-time tables), a **Streamlit quant tear-sheet dashboard**, ~140 tests, and a numbered process report (`report/1`–`13`). |
| **`kalman/`** | **Time-series factor attribution** — rolling-window OLS, a Time-Varying-Parameter (TVP) Kalman filter, and Kalman macro-regime extraction with regime-conditional models. |
| **`var_v2.ipynb`** | **Vector Autoregression** — tests whether lagged Fama-French factor returns Granger-cause the fund's monthly return. |
| **`ml-models/`** | **Deep-learning prediction** — a baseline 1D-CNN and a trained **hybrid CNN+Transformer** that predict the fund's monthly return from holdings-level characteristics. |
| **`docs/`** | Cross-cutting guides: `models.md` (modeling map), `data-pipeline.md`, `datasets.md` (data catalog), `glossary.md`, plus design specs under `superpowers/`. |

---

## 1. Trade data extraction (`data/`)

Annual custody statements → transaction tables:

```
annual_custody_statement_YYYY.pdf     (removed from this repo — see privacy note)
        │  data/extract transactions.py   (pdfplumber + regex, per year)
        ▼
purchase_and_sale_transactions_full_desc_YYYY.csv
        │  data/split_columns.py           (regex-parse the free-text Description)
        ▼
split_transactions_YYYY.csv
        │  cleaning + ticker resolution + FX  (downstream transform code not committed)
        ▼
clean_transactions.csv → clean_transactions_base_usd.csv → monthly_holdings.csv
```

`docs/datasets.md` is the full catalog: transaction
tables, fund/benchmark monthly performance (`2014_2025_*_Monthly.csv`),
security-level panels (`monthly_prices.csv`, `monthly_returns.csv`,
`characteristics_panel.csv` with the `next_ret` label), and the reference
universe.

> **Units are mixed** — returns/alpha/FF factors are decimals, macro rates are
> percent, weights are fractions, and `2014_2025_*_Monthly.csv` stores percent
> strings. **Do not use `data/ff_factors.csv`** — it is Developed *incl.* US;
> use `data/engineered_data/ff_factors_dev_ex_us.csv`. Details in `docs/datasets.md`.

### Engineered / modeling-ready layer (`data/engineered_data/`)

Built by the pipeline in `data_engr_dashboard/` and copied here with its own
`README.md` (column dictionary) and `REPORT.md` (how/why).

| File | Grain | What |
|------|-------|------|
| `combined_monthly.csv` | month (129 × 132) | Returns/alpha targets + ex-US FF factors + macro + portfolio fundamentals + sector/country + benchmark weights & active tilts. **Time-series / regression dataset.** |
| `combined_panel.csv` | ticker × month (7,740 × 91) | Stock fundamentals + return + `next_ret` + broadcast factors/macro + country/sector one-hots. **Cross-sectional / DL dataset.** |
| `ff_factors_dev_ex_us.csv` | month | Corrected Developed-ex-US 5-factor + momentum (decimal). |
| `pca_{market,full}_{components,loadings}.csv` | month | Orthogonal PCs (additive, full-rank), joinable on `month`. |
| `holding_profile.csv` | security | ticker ↔ FactSet ids, name, country, sector. |

---

## 2. Data-engineering pipeline & dashboard (`data_engr_dashboard/capstone_data_engr/`)

A separate `uv` project (kept isolated from pipeline credentials). Two principles
are enforced in code:

1. **Point-in-time / no look-ahead.** Economic releases (CPI, PCE, M2) are
   published a month in arrears, so their features are shifted forward one month
   (`config.RELEASE_LAG_MONTHS`). A row never holds data that wasn't public by
   that month-end.
2. **Free first.** Macro, FX, factors, and live-name prices are free; paid/sponsor
   access (WRDS/FactSet) is spent only on point-in-time fundamentals, delisted
   names, and a security master.

### Pipeline

```
capstone_data/            # one module per source + assembly
  fred.py  yahoo.py  fx.py               # free: FRED macro, Yahoo prices, Yahoo FX
  french.py  factors.py                  # Ken French factors + ex-US region verification
  wrds_io.py  holdings.py  factset.py    # WRDS query helper, ticker map, FactSet fundamentals
  benchmark.py                           # point-in-time EFA/SCZ/VSS sector weights (FactSet Ownership)
  macro.py  fundamentals.py  profile.py  # builders
  combine.py  pca.py                     # join layers into modeling datasets; PCA features
scripts/                  # one entrypoint per build step
tests/                    # deterministic feature-logic checks (no network)
report/                   # 1..13 — step-by-step process narrative + checkpoint/ figures
brinson_fachler/          # Brinson-Fachler attribution outputs (country/sector, monthly plots)
```

```bash
cd data_engr_dashboard/capstone_data_engr
uv sync
cp .env.example .env          # FRED_API_KEY (free) + WRDS_USERNAME
uv run pytest

# Build order (FRED key for macro/factors; WRDS creds for fundamentals):
uv run python scripts/build_macro.py             # macro_monthly.csv
uv run python scripts/build_factors.py           # ff_factors_dev_ex_us.csv (+ region check)
uv run python scripts/build_fundamentals.py      # fundamentals / returns / profile (WRDS)
uv run python scripts/build_benchmark_sectors.py # benchmark_sector_weights.csv (WRDS)
uv run python scripts/build_etf_holdings.py      # etf_holdings.csv (iShares; see note)
uv run python scripts/build_combined.py          # combined_monthly.csv + combined_panel.csv
uv run python scripts/build_pca.py               # pca_{market,full}_*.csv
```

**WRDS** has no API key — it uses your WRDS username + password over PostgreSQL
(Duo MFA on first connect). UVA's WRDS lacks Compustat Global / Datastream, so
international fundamentals come from **FactSet Fundamentals International**
(`factset_ff_int`); `scripts/wrds_explore.py` lists what you can actually query.

**iShares holdings note:** the iShares holdings-CSV endpoint is gated behind a
terms interstitial. When the fetch returns nothing, download each product's CSV
by hand into `data/raw/etf_holdings/` (filename starting with the ticker) and run
`build_etf_holdings.py --from-files data/raw/etf_holdings`.

> `pandas` is pinned `<3.0` — the `wrds` client's `raw_sql` isn't compatible with
> pandas 3.0. Don't bump it until `wrds` is.

### Dashboard (`dashboard/`)

An interactive Streamlit "quant tear sheet". Self-contained — reads only the
finished CSVs in `data/processed/`, imports nothing from the pipeline package.

```bash
cd data_engr_dashboard/capstone_data_engr
uv sync --group dashboard
uv run streamlit run dashboard/app.py     # http://localhost:8501
```

A sidebar **benchmark** selector (default EFA) and **period** slider drive the tabs:

- **📊 Overview** — headline KPIs (CAGR, vol, Sharpe, Sortino, max DD, Calmar,
  alpha/beta, tracking error, info ratio, capture), growth-of-$1 with an
  underwater panel, trailing/calendar-year return tables, monthly-returns heatmap.
- **🛡️ Risk & drawdown** — full risk table (downside dev, skew/kurtosis, VaR/CVaR,
  hit rate), worst-drawdown episodes, return histogram with VaR markers, rolling
  volatility & Sharpe.
- **🎯 Benchmark & active** — cumulative active return, rolling beta / tracking
  error / information ratio / correlation, capture, fund-vs-benchmark regression.
- **📈 Factor exposures** — the rolling factor model (target, regressors, window),
  defaulting to a clean Fama-French OLS with model/penalty/CV knobs behind an
  **Advanced** expander, plus a contribution decomposition.
- **🧭 Positioning (2018–2022)** — sector/geographic allocation vs. the blended
  benchmark with active tilts, value-weighted style/quality/growth characteristics,
  concentration, and a full **Brinson-Fachler performance-attribution** section
  (cumulative sector & stock contribution, interactive Brinson decomposition,
  full-period and dynamic monthly sector/country attribution plots).
- **🔎 Data explorer** — distributions, returns over time, correlations.
- **🌐 Macro & regime** — fund performance conditioned on macro regimes (VIX,
  rates, curve, credit, inflation, momentum terciles), an interactive
  principal-component explorer, and a dictionary of every regression variable.

Performance/risk math lives in `analytics.py` (pure functions parameterised by
column name — `ret_col`, `bench_col`, `rf_col` — never a hard-coded `fund_ret`).
The dashboard also carries lifted copies of the rolling-regression and Kalman-TVP
engines (`dashboard/rolling_regression.py`, `dashboard/tvp_kalman_filter.py`,
`dashboard/kalman_tvp.py`) and a `fund_rating.py` scorer.

---

## 3. Time-series factor modeling (`kalman/`)

Three views of how the fund's factor exposures and alpha evolve, all on
`data/engineered_data/combined_monthly.csv` (excess return on the Developed-ex-US
`Mkt_RF, SMB, HML, Mom` set):

| File | Model | Idea |
|------|-------|------|
| `rolling_regression.py` (+ `test_rolling_regression.py`) | Rolling-window OLS | Betas re-estimated on a trailing 24-/36-month window — time-varying exposure without a state-space model. The middle ground. |
| `tvp_kalman.ipynb`, `tvp_kalman_filter.py` | **Model 1 — TVP Kalman filter** | State `[alpha, β_mkt, β_smb, β_hml, β_mom]` evolves as a random walk; observation is `r_t − rf_t`. Shows *when* alpha was earned and *how* tilts shifted. |
| `h_kalman.ipynb` | **Models 2 & 3 — regime extraction + conditional TVP** | Model 2 extracts a latent risk-regime index from macro signals (VIX, curve slope, credit spread, S&P momentum) via a Kalman local-linear-trend filter. Model 3a feeds that index back into the TVP model as an extra factor to test whether alpha is regime-dependent. |
| `kalman.ipynb` | Reference | A minimal `pykalman` TVP-factor function. |

Result CSVs (`*_results.csv`, `regime_index.csv`, `regime_conditional_results.csv`)
are committed. Design rationale: `docs/superpowers/specs/2026-07-01-rolling-regression-design.md`.

---

## 4. Vector autoregression (`var_v2.ipynb`)

A VAR on `fund_ret` and a set of Fama-French factor returns, asking whether *lagged*
factor moves (as opposed to contemporaneous exposure, which the Kalman/OLS models
above capture) help predict next month's fund return.

- Stationarity is confirmed with Augmented Dickey-Fuller tests, then lag-order
  selection is run on the 129-month panel (kept short — 6 variables limits how
  many lags are identifiable without overfitting).
- Information criteria select **zero lags**; a parsimonious **VAR(1)** is still
  estimated as a diagnostic, checked for stability (companion-matrix eigenvalues
  inside the unit circle), then probed with **Granger causality tests**, **impulse
  response functions**, and **forecast error variance decomposition (FEVD)**.
- **Result:** none of the factors Granger-cause `fund_ret` at the 5% level, and FEVD
  shows the fund's own forecast-error variance is almost entirely explained by
  shocks to itself, not to the factors. Lagged factor dynamics don't meaningfully
  predict the fund's monthly return in this specification — consistent with the
  fund's performance being better explained by contemporaneous exposure, portfolio
  composition, or trade-level behavior (see the Kalman and ML models).

---

## 5. ML prediction models (`ml-models/`)

Two neural approaches to the same "predict the fund's monthly return" question,
complementary to the linear/state-space models above:

| File | Model | Input → target |
|------|-------|----------------|
| `cnnv1.py` | Tiny **Keras/TensorFlow** 1D-CNN | 6-month window of EFA returns → next monthly fund return. CNN's prediction is treated as the beta/systematic component; the residual is "CNN alpha". |
| `cnnV1.ipynb` | **PyTorch** trade-level CNN (`FundMovementDataset`, `FundCNN`, `RegressionTrainer`, `TradeInfluenceAnalyzer`) with Grad-CAM | Sliding 60-trade window (`Action, Qty, Price_USD, Transaction_Value_USD, FX_Rate_to_USD`) → that month's return proxy. Grad-CAM saliency scores each individual trade as a gain- vs. loss-driver. |
| `hybrid_model.pt` | Trained **hybrid CNN + Transformer** (artifact only — training script not committed; `cnnV1.ipynb` is the closest related code) | Per-holding characteristics → monthly portfolio return, ensembled from a CNN branch and a Transformer branch. |

The hybrid model's out-of-sample outputs are in `portfolio_decomposition.csv` — 57
months (2018-04 → 2022-12) of `actual_ret`, ensemble `pred_ret`, the `cnn_pred` /
`tf_pred` branch predictions, `residual`, and holdings coverage — plotted in
`oos_evaluation.png` and `portfolio_predicted_vs_actual.png`. `cnn_fund_results.png`
is the `cnnv1.py` decomposition plot.

---

## 6. Documentation (`docs/`)

- **`models.md`** — map of all modeling/EDA assets and which dataset to use per
  task (factor attribution, time-series prediction, cross-sectional DL, trade-level
  influence), including the CNN and hybrid CNN+Transformer models in `ml-models/`.
- **`data-pipeline.md`** — how raw statements become the cleaned tables; what is
  and isn't committed.
- **`datasets.md`** — the full data catalog with column lists, coverage, and gotchas.
- **`glossary.md`** — finance/ML terms used across the repo.
- **`superpowers/`** — design specs and implementation plans for the dashboard
  re-platform and the rolling-regression work.

## Academic basis

Fama–French (2015) five-factor, Jegadeesh–Titman (1993) momentum, Ang et al.
(2006) volatility, Gu–Kelly–Xiu (2020) ML asset pricing. The engineered
characteristics and panel models are built around these.
