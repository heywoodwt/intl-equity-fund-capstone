# Models & analysis

The modeling and EDA assets, all under `ML implementation/`. **The notebooks/scripts are the source
of truth**; this is a map. Two goals run through everything: *explain* the fund's performance and
*predict* it.

> Notebooks load data by **bare filename** from `ML implementation/data/` (a working copy of `data/`).
> Run them with that as the working directory, or copy the file in.

## 1. Factor attribution — `data/python_files.ipynb`

Classic, explainable baseline. Pulls FUND monthly returns (`yfinance`) and Ken French
**Developed-ex-US** factors (`pandas_datareader`), then OLS regresses excess return on factors:
- a **3-factor** (Mkt-RF, SMB, HML) and **4-factor** (+ WML momentum) static regression →
  intercept = **alpha** (reported annualized), slopes = factor betas;
- a **rolling 12-month** `RollingOLS` → time-varying annualized alpha.

Use this to answer "how much of performance is market/size/value/momentum vs. manager skill (alpha)."

## 2. EDA & visuals

- **`exploration.ipynb`** — trade-flow EDA on `cleanDataV1.csv`: purchase vs. sale value over time,
  net flow bar chart.
- **`visuals.ipynb`** — fund vs. EFA/SCZ/VSS from the `2014_2025_*_Monthly.csv` files: monthly
  performance lines, return distributions (boxplot), growth-of-$10k, annual bars, drawdowns,
  correlation matrix, and an animated growth race.
- Rendered PNGs in **`ML implementation/visuals/`**: `eda_portfolio_size.png`,
  `eda_sector_allocation.png`, `cnn_fund_results.png`, `oos_evaluation.png`,
  `portfolio_predicted_vs_actual.png`.

## 3. CNN models — `ML implementation/models/`

Two distinct CNNs (don't confuse them):

| File | Idea | Input → target |
|------|------|----------------|
| `cnnv1.py` | Tiny **Keras/TensorFlow** 1D-CNN. Treats EFA as the systematic driver; CNN prediction = beta component, residual = "CNN alpha". | 6-month window of EFA returns → fund's next monthly return. |
| `cnnV1.ipynb` | Substantial **PyTorch** trade-level CNN (`FundMovementDataset`, `FundCNN`, `RegressionTrainer`, `TradeInfluenceAnalyzer`). 3 conv blocks + Grad-CAM. | Sliding window of 60 trades (features: `Action, Qty, Price_USD, Transaction_Value_USD, FX_Rate_to_USD` from `clean_transactions_base_usd.csv`) → that month's return proxy. |

`cnnV1.ipynb` is the richer artifact: Huber loss, OneCycleLR, directional-accuracy tracking, and
**Grad-CAM saliency** that scores each individual trade as a gain- vs. loss-driver — i.e. *which
trades moved the fund*. Run `main()` (or the notebook top to bottom); it saves `cnn_fund_results.png`.
Note the target there is a self-constructed flow proxy (net signed trade value / total value), **not**
the official fund return.

## 4. Hybrid model (artifact only) — `ML implementation/models/hybrid_model.pt`

A trained PyTorch hybrid (**CNN + Transformer**) that predicts the **monthly portfolio return** from
per-holding characteristics, evaluated out-of-sample.

- **Outputs:** `ML implementation/data/portfolio_decomposition.csv` — 57 months **2018-04 → 2022-12**:
  `actual_ret, pred_ret` (ensemble), `cnn_pred`, `tf_pred` (transformer), `residual`, and holdings
  coverage. Plots: `oos_evaluation.png`, `portfolio_predicted_vs_actual.png`.
- ⚠️ **The training script for the hybrid model is not committed to this repo** — only the `.pt`
  weights and the result CSV/plots. `cnnV1.ipynb` is the closest available related code. If you need
  to retrain or change architecture, that code must be located/recreated.

## 5. References — `References.ipynb`

The academic basis: Fama-French (2015) five-factor, Jegadeesh-Titman (1993) momentum, Ang et al.
(2006) volatility, Gu-Kelly-Xiu (2020) ML asset pricing. The engineered characteristics and panel
models are built around these.

## Suggested data per task

- **Explain performance** → `python_files.ipynb` (factors/alpha), `combined_monthly.csv`, `visuals.ipynb`.
- **Predict monthly fund return (time series)** → `combined_monthly.csv` (+ `pca_market_components.csv`).
- **Predict cross-sectional stock returns / DL** → `combined_panel.csv` (label `next_ret`), or
  `characteristics_panel.csv`.
- **Trade-level influence** → `cnnV1.ipynb` on `clean_transactions_base_usd.csv`.
