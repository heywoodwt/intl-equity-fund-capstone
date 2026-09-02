"""Rolling and full-sample factor regressions for the International Equity Fund.

Pure functions over a tidy monthly DataFrame — no Streamlit, no I/O. The
dashboard imports these; a notebook can too::

    import sys; sys.path.insert(0, "dashboard")
    from rolling_regression import rolling_fit, full_sample_fit

Conventions
-----------
* ``df`` has a ``date`` (datetime, month-end) column plus numeric y/X columns.
* Rows with any NaN in the selected y/X are dropped *before* rolling, so the
  window counts observations, not the calendar. For the FF-factor set there are
  no gaps; for the fundamentals (2018-2022 only) the usable span is that block.
* Regularized coefficients are standardized *within each window* (no look-ahead),
  so they're in per-within-window-SD units and comparable across regressors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

MODELS = ("OLS", "Ridge", "ElasticNet")


def annualize(monthly: pd.Series | float):
    """Compound a monthly (alpha) rate to annual: (1+a)^12 - 1."""
    return (1.0 + monthly) ** 12 - 1.0


def _clean(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    cols = ["date", y_col, *x_cols]
    out = df[cols].dropna().sort_values("date").reset_index(drop=True)
    return out


# --- regressor screening ---------------------------------------------------

@dataclass
class Screen:
    kept: list[str]                  # usable, linearly independent regressors
    dropped_zero: list[str]          # ~zero variance over the sample
    dropped_collinear: list[str]     # exact/near linear combinations of kept ones
    condition_number: float          # of the standardized kept design (collinearity)


def screen_regressors(df: pd.DataFrame, y_col: str, x_cols: list[str],
                      var_tol: float = 1e-10) -> Screen:
    """Drop columns that make the design matrix singular: zero-variance columns
    (e.g. degenerate PCs) and exact/near linear combinations (e.g. a yield-curve
    slope that equals ``y10 - y2``). Earlier-listed columns are preferred when a
    dependency must be broken. Reports the condition number of what remains.
    """
    data = _clean(df, y_col, x_cols)
    X = data[x_cols]
    sd = X.std(ddof=0)
    dropped_zero = [c for c in x_cols if sd[c] <= var_tol]
    live = [c for c in x_cols if c not in dropped_zero]

    kept, idx = [], []
    if live:
        Z = ((X[live] - X[live].mean()) / X[live].std(ddof=0)).to_numpy()
        for j, c in enumerate(live):
            if np.linalg.matrix_rank(Z[:, idx + [j]]) == len(idx) + 1:
                idx.append(j)
                kept.append(c)
    dropped_collinear = [c for c in live if c not in kept]

    if len(kept) >= 2:
        Zk = ((X[kept] - X[kept].mean()) / X[kept].std(ddof=0)).to_numpy()
        cond = float(np.linalg.cond(Zk))
    else:
        cond = 1.0
    return Screen(kept, dropped_zero, dropped_collinear, cond)


def alpha_interpretable(df: pd.DataFrame, x_cols: list[str],
                        mean_thresh: float = 0.5, ac_thresh: float = 0.4) -> bool:
    """True only when the intercept is a stable, in-support **alpha** — i.e. 0 is
    in the regressors' support both globally *and* within rolling windows. Every
    regressor must be:

    * **centered** — ``|mean| < mean_thresh · sd`` (rules out level regressors like
      VIX/yields/spreads, whose zero is far outside the data), and
    * **non-persistent** — ``|lag-1 autocorr| < ac_thresh`` (rules out the
      macro-derived PCs: they're centered *globally* but highly autocorrelated, so
      within a 24-month window they sit far from 0 and the intercept extrapolates).

    Factor *returns* pass (mean ≈ 0, autocorr ≈ 0); macro levels and PCs do not —
    for those, read the coefficients and the contribution decomposition, not alpha.
    """
    for c in x_cols:
        s = df[c].dropna()
        sd = s.std(ddof=0)
        if sd == 0:
            continue
        if abs(s.mean()) > mean_thresh * sd:
            return False
        ac = s.autocorr(lag=1)
        if pd.notna(ac) and abs(ac) >= ac_thresh:
            return False
    return True


# --- full sample -----------------------------------------------------------

@dataclass
class FitResult:
    coefs: pd.DataFrame      # index = regressor (+ 'const'); cols depend on model
    r2: float
    n: int
    model: str
    alpha: float | None = None
    l1_ratio: float | None = None
    standardized: bool = False


def full_sample_fit(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    model: str = "OLS",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    standardize: bool | None = None,
) -> FitResult:
    """Fit once over the whole (cleaned) sample.

    Every fit reports, per regressor:
      * ``coef``        — raw coefficient (OLS β = exposure; reg = per-1σ).
      * ``coef_per_sd`` — comparable across scales: Δy per +1 SD of the regressor
                          (= ``coef`` × SD for OLS; native for standardized reg).
      * ``beta_weight`` — fully standardized (× SD(x)/SD(y)); dimensionless.
    OLS adds ``std_err``/``t``/``p``. ``const`` is always the **alpha** (intercept
    at raw regressors = 0); for standardized reg it is de-centered back to alpha.
    Degenerate/collinear regressors are screened out first (see ``screen_regressors``).
    """
    x_cols = screen_regressors(df, y_col, x_cols).kept
    if not x_cols:
        raise ValueError("No usable regressors after dropping degenerate/collinear columns.")
    data = _clean(df, y_col, x_cols)
    if len(data) <= len(x_cols) + 1:
        raise ValueError(f"Not enough rows ({len(data)}) for {len(x_cols)} regressors.")
    y = data[y_col].to_numpy()
    X = data[x_cols].to_numpy()
    sd_x = data[x_cols].std()           # ddof=1
    sd_y = float(data[y_col].std())

    if model == "OLS":
        Xc = sm.add_constant(data[x_cols], has_constant="add")
        res = sm.OLS(data[y_col], Xc).fit()
        coefs = pd.DataFrame({
            "coef": res.params, "std_err": res.bse,
            "t": res.tvalues, "p": res.pvalues,
        })
        coefs.insert(1, "coef_per_sd", coefs["coef"] * sd_x.reindex(coefs.index))
        coefs.insert(2, "beta_weight", coefs["coef_per_sd"] / sd_y)
        return FitResult(coefs=coefs, r2=float(res.rsquared), n=int(res.nobs),
                         model=model, standardized=False)

    std = True if standardize is None else standardize
    scaler = StandardScaler().fit(X) if std else None
    Xz = scaler.transform(X) if std else X
    est = (Ridge(alpha=alpha) if model == "Ridge"
           else ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
    est.fit(Xz, y)
    coefs = pd.DataFrame({"coef": est.coef_}, index=x_cols)
    if std:
        coefs["coef_per_sd"] = coefs["coef"]   # already Δy per 1σ (X z-scored)
        const = float(est.intercept_ - np.sum(est.coef_ * scaler.mean_ / scaler.scale_))
    else:
        coefs["coef_per_sd"] = coefs["coef"] * sd_x.reindex(coefs.index)
        const = float(est.intercept_)
    coefs["beta_weight"] = coefs["coef_per_sd"] / sd_y
    coefs = coefs[["coef", "coef_per_sd", "beta_weight"]]
    coefs.loc["const"] = [const, np.nan, np.nan]
    r2 = float(est.score(Xz, y))
    return FitResult(coefs=coefs, r2=r2, n=len(data), model=model,
                     alpha=alpha, l1_ratio=l1_ratio, standardized=std)


# --- rolling ---------------------------------------------------------------

def rolling_fit(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    window: int = 24,
    model: str = "OLS",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    standardize: bool | None = None,
    coef_scale: str = "raw",
) -> pd.DataFrame:
    """Rolling regression. Returns one row per window end (indexed by ``date``):

    columns: ``const`` (per-period alpha), ``alpha_ann`` (annualized), one column
    per regressor, ``r2``, and ``nobs``.

    ``coef_scale`` (OLS only): ``"raw"`` = native β; ``"std"`` = β × SD over the
    same window (Δy per +1σ — comparable across regressors). Regularized models
    are always reported per-1σ (X is z-scored within each window); their ``const``
    is de-centered back to alpha. Degenerate/collinear regressors are screened
    out first; rolling OLS additionally requires ≥ 2 observations per parameter
    per window (else it raises — increase the window or use Ridge).
    """
    x_cols = screen_regressors(df, y_col, x_cols).kept
    if not x_cols:
        raise ValueError("No usable regressors after dropping degenerate/collinear columns.")
    data = _clean(df, y_col, x_cols)
    if len(data) < window:
        raise ValueError(f"Only {len(data)} usable rows < window {window}.")

    p = len(x_cols)
    if model == "OLS" and window < 2 * (p + 1):
        raise ValueError(
            f"Rolling OLS with {p} regressors needs ≥ {2 * (p + 1)} months per "
            f"window to be stable (got {window}). Increase the window, drop "
            f"regressors, or switch to Ridge."
        )

    if model == "OLS":
        out = _rolling_ols(data, y_col, x_cols, window)
        if coef_scale == "std":
            sd = data[x_cols].rolling(window).std()       # ddof=1
            sd.index = pd.Index(data["date"], name="date")
            sd = sd.reindex(out.index)
            out[x_cols] = out[x_cols].mul(sd[x_cols])
    else:
        std = True if standardize is None else standardize
        out = _rolling_regularized(data, y_col, x_cols, window, model,
                                   alpha, l1_ratio, std)
    out["alpha_ann"] = annualize(out["const"])
    return out


def _rolling_ols(data, y_col, x_cols, window) -> pd.DataFrame:
    from statsmodels.regression.rolling import RollingOLS

    Xc = sm.add_constant(data[x_cols], has_constant="add")
    res = RollingOLS(data[y_col], Xc, window=window, min_nobs=window,
                     expanding=False).fit()
    params = res.params.copy()
    params["r2"] = res.rsquared
    params["nobs"] = window
    params.insert(0, "date", data["date"].to_numpy())
    return params.dropna(subset=["const"]).set_index("date")


def _rolling_regularized(data, y_col, x_cols, window, model, alpha, l1_ratio, std):
    rows = []
    y = data[y_col].to_numpy()
    X = data[x_cols].to_numpy()
    dates = data["date"].to_numpy()
    for end in range(window, len(data) + 1):
        sl = slice(end - window, end)
        Xw, yw = X[sl], y[sl]
        scaler = StandardScaler().fit(Xw) if std else None
        Xwz = scaler.transform(Xw) if std else Xw
        est = (Ridge(alpha=alpha) if model == "Ridge"
               else ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
        est.fit(Xwz, yw)
        # de-center the intercept back to alpha (return when raw regressors = 0)
        const = (float(est.intercept_ - np.sum(est.coef_ * scaler.mean_ / scaler.scale_))
                 if std else float(est.intercept_))
        row = {"date": dates[end - 1], "const": const,
               "r2": est.score(Xwz, yw), "nobs": window}
        row.update(dict(zip(x_cols, est.coef_)))   # coefs are per-1σ when std
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


# --- penalty selection -----------------------------------------------------

def cv_alpha(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    model: str = "Ridge",
    l1_ratio: float = 0.5,
    alphas: np.ndarray | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Time-series CV curve over a grid of penalties (standardized features).

    Returns columns ``alpha`` and ``cv_r2`` (mean across folds). Pick the alpha
    that maximizes ``cv_r2``.
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 25)
    x_cols = screen_regressors(df, y_col, x_cols).kept
    if not x_cols:
        raise ValueError("No usable regressors after dropping degenerate/collinear columns.")
    data = _clean(df, y_col, x_cols)
    y = data[y_col].to_numpy()
    X = data[x_cols].to_numpy()
    n_splits = min(n_splits, max(2, len(data) // (len(x_cols) + 2)))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out = []
    for a in alphas:
        scores = []
        for tr, te in tscv.split(X):
            sc = StandardScaler().fit(X[tr])
            est = (Ridge(alpha=a) if model == "Ridge"
                   else ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000))
            est.fit(sc.transform(X[tr]), y[tr])
            scores.append(est.score(sc.transform(X[te]), y[te]))
        out.append({"alpha": a, "cv_r2": float(np.mean(scores))})
    return pd.DataFrame(out)


# --- attribution -----------------------------------------------------------

def contribution_decomposition(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    """OLS-based attribution of returns to each regressor over the cleaned sample.

    Two complementary, additive decompositions (returns ``(table, summary)``):

    * ``var_share`` = β · cov(x, y) / var(y). Regressor rows **sum to R²**
      (``Residual`` = 1 − R²). *Scale-invariant* — no standardization needed; this
      is the most defensible "what drove the variation in returns" answer.
    * ``mean_contrib`` = β · mean(x), in monthly-return units. Regressor rows plus
      ``Alpha`` **sum to mean(y)**. Intuitive ("X added +0.3%/mo on average") but
      *level-sensitive* for non-return regressors — the split with alpha shifts if
      you re-center them, so read it for factor-return regressors, with care.

    ``summary`` = dict(r2, alpha, mean_y, n).
    """
    x_cols = screen_regressors(df, y_col, x_cols).kept
    if not x_cols:
        raise ValueError("No usable regressors after dropping degenerate/collinear columns.")
    data = _clean(df, y_col, x_cols)
    if len(data) <= len(x_cols) + 1:
        raise ValueError(f"Not enough rows ({len(data)}) for {len(x_cols)} regressors.")
    y = data[y_col]
    X = data[x_cols]
    res = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    beta = res.params.drop("const")
    alpha = float(res.params["const"])
    var_y = float(y.var(ddof=1))
    cov = X.apply(lambda c: c.cov(y))           # cov(x_j, y)
    table = pd.DataFrame({
        "var_share": beta * cov / var_y,        # regressor rows sum to R²
        "mean_contrib": beta * X.mean(),        # rows + alpha sum to mean(y)
    })
    table.loc["Alpha"] = {"var_share": np.nan, "mean_contrib": alpha}
    table.loc["Residual"] = {"var_share": 1.0 - res.rsquared, "mean_contrib": np.nan}
    summary = {"r2": float(res.rsquared), "alpha": alpha,
               "mean_y": float(y.mean()), "n": int(res.nobs)}
    return table, summary
