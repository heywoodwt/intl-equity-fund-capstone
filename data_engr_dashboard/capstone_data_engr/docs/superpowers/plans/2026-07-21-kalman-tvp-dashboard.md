# Kalman TVP Factor Attribution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Kalman (TVP)" option to the dashboard's 📈 Factor exposures tab that fits alpha and every factor beta as random walks and shows their drifting paths with confidence bands, a filtered-vs-smoothed toggle, event annotations, and residual diagnostics.

**Architecture:** Three small units mirroring the existing dashboard split. `dashboard/tvp_kalman_filter.py` is the dependency-free Kalman engine (copied verbatim from branch `heywood`). `dashboard/kalman_tvp.py` is a Streamlit-free, unit-testable wrapper (`fit_tvp` → `TVPResult`). `dashboard/app.py` adds the model option and a dedicated `render_kalman` results renderer, reusing existing regressor-screening and alpha-interpretability guards from `rolling_regression.py`.

**Tech Stack:** Python, numpy, pandas, Streamlit, Plotly (express + graph_objects), statsmodels (Ljung-Box), scipy (EM, transitive). No new dependencies.

---

## File Structure

- **Create** `dashboard/tvp_kalman_filter.py` — `KalmanFilter` engine (filter/smooth/EM/loglikelihood). Copied unchanged from `kalman/tvp_kalman_filter.py` on branch `heywood`.
- **Create** `dashboard/kalman_tvp.py` — `fit_tvp(df, target, x_cols)` returning a `TVPResult` dataclass. The testable adapter; imports the engine, no Streamlit.
- **Create** `tests/test_kalman_tvp.py` — deterministic unit tests for `fit_tvp`.
- **Modify** `dashboard/app.py` — add `import plotly.graph_objects as go` and `import kalman_tvp as K`; restructure the Factor-tab model controls; add `render_kalman`; dispatch to it.

---

## Task 1: Copy the Kalman engine into the dashboard

**Files:**
- Create: `dashboard/tvp_kalman_filter.py`

- [ ] **Step 1: Fetch the file verbatim from branch `heywood`**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
gh api "repos/msdsjht7m/ds6015/contents/kalman/tvp_kalman_filter.py?ref=heywood" \
  --jq '.content' | base64 -d > dashboard/tvp_kalman_filter.py
```
Expected: creates `dashboard/tvp_kalman_filter.py` (~150 lines, a `KalmanFilter` class).

- [ ] **Step 2: Sanity-check it imports and constructs**

Run:
```bash
cd dashboard && python -c "
import numpy as np
from tvp_kalman_filter import KalmanFilter
kf = KalmanFilter(1, 2, np.eye(2), np.zeros((3,1,2)),
                  np.zeros(2), np.eye(2),
                  em_vars=['transition_covariance','observation_covariance'])
print('ok', kf.n_dim_state)
"
```
Expected: prints `ok 2`.

- [ ] **Step 3: Commit**

```bash
git add dashboard/tvp_kalman_filter.py
git commit -m "feat(dashboard): add dependency-free Kalman filter engine"
```

---

## Task 2: `kalman_tvp.fit_tvp` wrapper (TDD)

**Files:**
- Test: `tests/test_kalman_tvp.py`
- Create: `dashboard/kalman_tvp.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_kalman_tvp.py`:
```python
"""Deterministic checks for the Kalman TVP wrapper (no Streamlit, no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import numpy as np
import pandas as pd
import pytest

import kalman_tvp as K


def _synth(n=120, betas=(0.02, 1.1, 0.4), sigma=0.01, seed=0):
    """Stationary factor model: constant alpha/betas + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2015-01", periods=n, freq="M").to_timestamp()
    f1 = rng.normal(0, 0.04, n)
    f2 = rng.normal(0, 0.03, n)
    alpha, b1, b2 = betas
    y = alpha + b1 * f1 + b2 * f2 + rng.normal(0, sigma, n)
    return pd.DataFrame({"date": dates, "excess_ret": y, "Mkt_RF": f1, "SMB": f2})


def test_fit_tvp_shapes():
    res = K.fit_tvp(_synth(), "excess_ret", ["Mkt_RF", "SMB"])
    assert res.n == 120
    assert list(res.smoothed.columns) == ["alpha", "Mkt_RF", "SMB"]
    assert res.smoothed.shape == (120, 3)
    assert res.smoothed_se.shape == (120, 3)
    assert res.filtered.shape == (120, 3)
    assert res.filtered_se.shape == (120, 3)
    assert len(res.residuals) == 120
    assert len(res.resid_std) == 120


def test_fit_tvp_recovers_ols_on_stationary_data():
    res = K.fit_tvp(_synth(betas=(0.02, 1.1, 0.4)), "excess_ret", ["Mkt_RF", "SMB"])
    assert abs(res.smoothed["Mkt_RF"].mean() - res.ols_params["Mkt_RF"]) < 0.15
    assert abs(res.smoothed["SMB"].mean() - res.ols_params["SMB"]) < 0.15
    assert abs(res.smoothed["alpha"].mean() - res.ols_params["alpha"]) < 0.01


def test_standardized_residuals_reasonable():
    res = K.fit_tvp(_synth(), "excess_ret", ["Mkt_RF", "SMB"])
    rstd = res.resid_std.to_numpy()
    rstd = rstd[~np.isnan(rstd)]
    assert abs(rstd.mean()) < 0.5
    assert 0.5 < rstd.std() < 2.0


def test_too_few_months_raises():
    with pytest.raises(ValueError):
        K.fit_tvp(_synth(n=3), "excess_ret", ["Mkt_RF", "SMB"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_kalman_tvp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kalman_tvp'`.

- [ ] **Step 3: Write `dashboard/kalman_tvp.py`**

Create `dashboard/kalman_tvp.py`:
```python
"""Time-varying-parameter (TVP) factor attribution via a Kalman filter.

Dashboard-facing wrapper around the dependency-free ``KalmanFilter`` in
``tvp_kalman_filter.py``. Fits a random-walk TVP regression of ``target`` on
``x_cols`` and returns smoothed/filtered state paths, per-state standard errors,
one-step-ahead residuals, and the log-likelihood. Streamlit-free and
framework-free so it is unit-testable in isolation.

State vector: ``[alpha, beta_1, ..., beta_k]``. Transition is the identity
(random walk); the observation row at ``t`` is ``[1, x_1t, ..., x_kt]``.
Process/observation noise (Q, R) are estimated by EM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tvp_kalman_filter import KalmanFilter


@dataclass
class TVPResult:
    smoothed: pd.DataFrame       # state means, cols ["alpha", *x_cols], index=date
    filtered: pd.DataFrame
    smoothed_se: pd.DataFrame    # sqrt of per-t covariance diagonal
    filtered_se: pd.DataFrame
    residuals: pd.Series         # one-step-ahead prediction errors
    resid_std: pd.Series         # standardized: residual / sqrt(pred_var)
    loglik: float
    ols_params: pd.Series        # static full-sample OLS, index ["alpha", *x_cols]
    x_cols: list[str]
    n: int


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS with intercept via lstsq; returns [alpha, b1, ..., bk]."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def fit_tvp(df: pd.DataFrame, target: str, x_cols: list[str],
            n_iter: int = 20, init_window: int = 12) -> TVPResult:
    """Fit the random-walk TVP model of ``target`` on ``x_cols`` over ``df``.

    ``df`` must have a ``date`` column plus ``target`` and every ``x_cols``.
    Rows with any NA in those columns are dropped.
    """
    x_cols = list(x_cols)
    data = (df[["date", target, *x_cols]]
            .dropna().sort_values("date").reset_index(drop=True))
    dates = pd.DatetimeIndex(data["date"])
    y = data[target].to_numpy(dtype=float)
    X = data[x_cols].to_numpy(dtype=float)

    n_obs = len(y)
    n_states = len(x_cols) + 1
    if n_obs <= n_states:
        raise ValueError(
            f"Need more months ({n_obs}) than states ({n_states}) to fit the "
            f"Kalman TVP model. Widen the period or drop regressors."
        )

    # Time-varying observation matrices: row t = [1, x_1t, ..., x_kt].
    obs = np.zeros((n_obs, 1, n_states))
    obs[:, 0, 0] = 1.0
    obs[:, 0, 1:] = X

    # Initial state from OLS on the first init_window months; inflate uncertainty.
    w = min(init_window, n_obs)
    init_state = _ols(y[:w], X[:w])
    resid0 = y[:w] - (np.column_stack([np.ones(w), X[:w]]) @ init_state)
    scale = max(float(np.var(resid0)), 1e-6)
    init_cov = np.eye(n_states) * scale * 10.0

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=n_states,
        transition_matrices=np.eye(n_states),
        observation_matrices=obs,
        initial_state_mean=init_state,
        initial_state_covariance=init_cov,
        em_vars=["transition_covariance", "observation_covariance"],
    )
    observations = y.reshape(-1, 1)
    kf = kf.em(observations, n_iter=n_iter)

    filt_means, filt_covs = kf.filter(observations)
    smooth_means, smooth_covs = kf.smooth(observations)

    names = ["alpha", *x_cols]
    smoothed = pd.DataFrame(smooth_means, index=dates, columns=names)
    filtered = pd.DataFrame(filt_means, index=dates, columns=names)
    smoothed_se = pd.DataFrame(
        np.sqrt(np.array([np.diag(c) for c in smooth_covs])),
        index=dates, columns=names)
    filtered_se = pd.DataFrame(
        np.sqrt(np.array([np.diag(c) for c in filt_covs])),
        index=dates, columns=names)

    # One-step-ahead residuals from the filter (notebook diagnostics cell).
    y_pred = np.array([float(obs[t] @ filt_means[t]) for t in range(n_obs)])
    residuals = y - y_pred
    r = float(np.asarray(kf.observation_covariance)[0, 0])
    pred_var = np.array(
        [float(obs[t] @ filt_covs[t] @ obs[t].T) + r for t in range(n_obs)])
    resid_std = residuals / np.sqrt(np.maximum(pred_var, 1e-12))

    ols_params = pd.Series(_ols(y, X), index=names)

    return TVPResult(
        smoothed=smoothed, filtered=filtered,
        smoothed_se=smoothed_se, filtered_se=filtered_se,
        residuals=pd.Series(residuals, index=dates),
        resid_std=pd.Series(resid_std, index=dates),
        loglik=float(kf.loglikelihood(observations)),
        ols_params=ols_params, x_cols=x_cols, n=n_obs,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_kalman_tvp.py -v`
Expected: PASS — all 4 tests green. (If `test_fit_tvp_recovers_ols_on_stationary_data` is marginal, the tolerances are intentionally generous; do not loosen them further without noting why in the commit.)

- [ ] **Step 5: Commit**

```bash
git add dashboard/kalman_tvp.py tests/test_kalman_tvp.py
git commit -m "feat(dashboard): add kalman_tvp.fit_tvp TVP wrapper with tests"
```

---

## Task 3: Wire into the Factor exposures tab

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Add the imports**

In `dashboard/app.py`, find (around line 18):
```python
import plotly.express as px
import streamlit as st

import analytics as A
import data as D
import rolling_regression as R
```
Replace with:
```python
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import data as D
import kalman_tvp as K
import rolling_regression as R
```

(No other imports are needed — the `contextlib` approach considered earlier is not used.)

- [ ] **Step 2: Add the module-level events constant and cached fit helper**

In `dashboard/app.py`, add the events constant, the cached-fit helper, and the `render_kalman` renderer as a single block placed directly **above** the existing `def render_regression(` line (around line 113). `render_kalman` uses the module-level `_tight` helper, `A`/`D`/`R` modules, `px`, `go`, `np`, `pd`, and `st`, all already imported:
```python
K_EVENTS = {
    "2018-10": "Q4 2018 selloff",
    "2020-03": "COVID crash",
    "2020-11": "Vaccine rally",
    "2022-01": "Rate-hike cycle",
}


@st.cache_data(show_spinner=False)
def _fit_tvp_cached(df_reg, target, x_cols):
    """Cache the EM fit so toggling filtered/smoothed or beta lines is instant."""
    return K.fit_tvp(df_reg, target, list(x_cols))


def render_kalman(df, target, x_cols):
    """Kalman TVP results body for the Factor exposures tab. ``df`` is already
    period-filtered by the caller. Bails out with ``return`` (never
    ``st.stop()``) so sibling tabs keep rendering."""
    if not x_cols:
        st.info("Select at least one regressor.")
        return

    sc = R.screen_regressors(df, target, x_cols)
    drops = []
    if sc.dropped_zero:
        drops.append("zero-variance: " + ", ".join(D.label(c) for c in sc.dropped_zero))
    if sc.dropped_collinear:
        drops.append("redundant (linear combo of others): "
                     + ", ".join(D.label(c) for c in sc.dropped_collinear))
    if drops:
        st.warning("Dropped to keep the model identified — " + "; ".join(drops) + ".")
    x_fit = sc.kept
    if not x_fit:
        st.warning("No usable regressors left after screening.")
        return

    try:
        with st.spinner("Fitting Kalman filter (EM)…"):
            res = _fit_tvp_cached(df, target, tuple(x_fit))
    except ValueError as e:
        st.warning(str(e))
        return

    st.caption(
        "Random-walk time-varying-parameter model — alpha and every beta drift "
        "each month, estimated by a Kalman filter (EM for the noise variances). "
        "Built for the **Fama-French factor** preset; on persistent/level "
        "regressors the drifting alpha is not meaningful (see below)."
    )

    view = st.radio(
        "Estimate", ["Smoothed (retrospective)", "Filtered (real-time)"],
        index=0, horizontal=True,
        help="Smoothed uses the whole sample — the best hindsight estimate. "
             "Filtered uses only data up to each month: what you'd have known "
             "in real time.",
    )
    smoothed = view.startswith("Smoothed")
    means = res.smoothed if smoothed else res.filtered
    ses = res.smoothed_se if smoothed else res.filtered_se
    alpha_ok = R.alpha_interpretable(df, x_fit)

    c1, c2, c3 = st.columns(3)
    c1.metric("Static OLS alpha (ann.)",
              f"{R.annualize(res.ols_params['alpha']):.2%}" if alpha_ok else "n/a")
    c2.metric("Kalman alpha — mean (ann.)",
              f"{R.annualize(means['alpha']).mean():.2%}" if alpha_ok else "n/a")
    c3.metric("Usable months", res.n)

    # --- Time-varying alpha ------------------------------------------------
    if alpha_ok:
        st.markdown("##### Time-varying alpha (monthly)")
        a, se = means["alpha"], ses["alpha"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=means.index, y=(a + 2 * se).values,
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=means.index, y=(a - 2 * se).values, fill="tonexty",
                                 fillcolor="rgba(31,111,235,0.15)", line=dict(width=0),
                                 name="95% band"))
        fig.add_trace(go.Scatter(x=means.index, y=a.values,
                                 line=dict(color="#1f6feb"), name="Kalman alpha"))
        fig.add_hline(y=res.ols_params["alpha"], line_dash="dash", line_color="gray")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(_tight(fig, 300, yaxis_title="Monthly alpha"), width="stretch")

        st.markdown("##### Annualized alpha with key events")
        ann = R.annualize(a)
        fig = px.line(x=means.index, y=ann.values)
        fig.update_traces(line_color="#1f6feb")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        lo, hi = means.index.min(), means.index.max()
        for ds, lbl in K_EVENTS.items():
            dt = pd.Timestamp(ds)
            if lo <= dt <= hi:
                fig.add_vline(x=dt, line_width=1, line_color="rgba(120,120,120,0.5)")
                fig.add_annotation(x=dt, y=1, yref="paper", text=lbl, showarrow=False,
                                   textangle=-90, xanchor="left", yanchor="top",
                                   font=dict(size=9))
        st.plotly_chart(_tight(fig, 280, yaxis_tickformat=".0%",
                               yaxis_title="Ann. alpha"), width="stretch")
    else:
        st.info(
            "Alpha is hidden because the intercept isn't a meaningful alpha for "
            "these regressors — level regressors (VIX, yields, spreads) and the "
            "persistent macro PCs sit far from 0 within the sample, so the "
            "drifting intercept extrapolates. Use the Fama-French factor preset "
            "for a real alpha; here, read the betas below."
        )

    # --- Time-varying betas ------------------------------------------------
    st.markdown("##### Time-varying betas")
    show = st.multiselect("Lines to plot", x_fit,
                          default=x_fit[: min(4, len(x_fit))],
                          format_func=D.label, key="kalman_betalines")
    for i in range(0, len(show), 2):
        cols = st.columns(2)
        for name, col in zip(show[i:i + 2], cols):
            with col:
                st.markdown(f"**{D.label(name)}**")
                b, se = means[name], ses[name]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=means.index, y=(b + 2 * se).values,
                                         line=dict(width=0), showlegend=False,
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=means.index, y=(b - 2 * se).values,
                                         fill="tonexty", fillcolor="rgba(214,39,40,0.12)",
                                         line=dict(width=0), showlegend=False,
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=means.index, y=b.values,
                                         line=dict(color="#d62728"), showlegend=False))
                fig.add_hline(y=res.ols_params[name], line_dash="dash", line_color="gray")
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(_tight(fig, 240, yaxis_title="Beta"), width="stretch")

    # --- Residual diagnostics ---------------------------------------------
    with st.expander("Residual diagnostics — standardized one-step errors"):
        st.caption("If the model is well-specified these are ~ i.i.d. N(0,1).")
        rstd = res.resid_std
        d1, d2 = st.columns(2)
        with d1:
            fig = px.line(x=rstd.index, y=rstd.values)
            fig.update_traces(line_color="#1f6feb")
            fig.add_hline(y=0, line_color="black", line_width=0.5)
            fig.add_hline(y=2, line_dash="dash", line_color="red", line_width=0.5)
            fig.add_hline(y=-2, line_dash="dash", line_color="red", line_width=0.5)
            st.plotly_chart(_tight(fig, 240, yaxis_title="Std. residual"), width="stretch")
        with d2:
            fig = px.histogram(rstd.dropna().values, nbins=20,
                               histnorm="probability density")
            xg = np.linspace(-4, 4, 100)
            fig.add_trace(go.Scatter(x=xg, y=np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi),
                                     line=dict(color="red"), name="N(0,1)"))
            st.plotly_chart(_tight(fig, 240, showlegend=False), width="stretch")
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(rstd.dropna(), lags=[6, 12], return_df=True)
            st.caption(
                "Ljung-Box (H0: no serial correlation) — "
                + " · ".join(f"lag {int(lag)}: p={row.lb_pvalue:.3f}"
                             for lag, row in lb.iterrows())
                + "."
            )
        except Exception:
            pass
```

- [ ] **Step 3: Add "Kalman (TVP)" to the model radio and hide the penalty UI for it**

In `dashboard/app.py`, find the controls block. It currently reads exactly (lines ~756–796):
```python
        st.divider()
        window = st.slider("Rolling window (months)", 12, 60, 24, 6)

        # Default is the top "Alpha drivers" preset (macro/factor + fundamentals);
        # the modelling knobs (penalty, CV, coefficient scale) live behind "Advanced".
        with st.expander("⚙️ Advanced — model & penalty"):
            model = st.radio("Model", R.MODELS, index=0, horizontal=True)
            alpha = l1_ratio = None
            standardize = False
            if model != "OLS":
                standardize = st.checkbox("Standardize within window", value=True,
                                          help="Z-score per window — required to "
                                               "compare penalized coefficients.")
                log_alpha = st.slider("Penalty (log10 α)", -4.0, 2.0, 0.0, 0.25)
                alpha = float(10.0 ** log_alpha)
                st.caption(f"α = {alpha:.4g}")
                if model == "ElasticNet":
                    l1_ratio = st.slider("L1 ratio (0=Ridge … 1=Lasso)", 0.0, 1.0, 0.5, 0.05)
                if st.button("Suggest α via time-series CV", width="stretch"):
                    with st.spinner("Cross-validating…"):
                        cv = R.cv_alpha(df_reg, target, x_cols, model=model,
                                        l1_ratio=l1_ratio or 0.5)
                    st.session_state["cv"] = cv
                if "cv" in st.session_state:
                    cv = st.session_state["cv"]
                    best = cv.loc[cv.cv_r2.idxmax()]
                    st.caption(f"CV-best α ≈ {best.alpha:.4g} (cv R²={best.cv_r2:.3f})")

            if model == "OLS":
                coef_scale = "std" if st.radio(
                    "Coefficient scale", ["Standardized (Δret per 1σ)", "Raw β"],
                    index=0,
                    help="Standardized = β × SD(regressor): comparable across "
                         "variables on different scales — use this to read off what "
                         "mattered. Raw β = native exposure/sensitivity. Scaling never "
                         "changes R², t-stats, or alpha.",
                ).startswith("Standardized") else "raw"
            else:
                coef_scale = "std"
                st.caption("Penalized coefficients are standardized (Δret per 1σ) by "
                           "construction.")
```
Replace that **entire block** with the following. The model radio moves out in front; everything else is nested under `if not is_kalman:` (the penalty block is the same code, re-indented one level deeper). The inner `if model != "OLS":` is unchanged in meaning because `model` can only be OLS/Ridge/ElasticNet inside this branch:
```python
        st.divider()
        model = st.radio(
            "Model", list(R.MODELS) + ["Kalman (TVP)"], index=0, horizontal=True,
            help="OLS / Ridge / ElasticNet fit a trailing rolling window. "
                 "Kalman (TVP) models alpha and every beta as random walks "
                 "(state-space) — smoothed paths with confidence bands.",
        )
        is_kalman = model == "Kalman (TVP)"

        window = 24
        alpha = l1_ratio = None
        standardize = False
        coef_scale = "raw"
        if not is_kalman:
            window = st.slider("Rolling window (months)", 12, 60, 24, 6)
            # The modelling knobs (penalty, CV, coefficient scale) live behind
            # "Advanced" and only apply to the rolling models.
            with st.expander("⚙️ Advanced — penalty", expanded=False):
                if model != "OLS":
                    standardize = st.checkbox("Standardize within window", value=True,
                                              help="Z-score per window — required to "
                                                   "compare penalized coefficients.")
                    log_alpha = st.slider("Penalty (log10 α)", -4.0, 2.0, 0.0, 0.25)
                    alpha = float(10.0 ** log_alpha)
                    st.caption(f"α = {alpha:.4g}")
                    if model == "ElasticNet":
                        l1_ratio = st.slider("L1 ratio (0=Ridge … 1=Lasso)", 0.0, 1.0, 0.5, 0.05)
                    if st.button("Suggest α via time-series CV", width="stretch"):
                        with st.spinner("Cross-validating…"):
                            cv = R.cv_alpha(df_reg, target, x_cols, model=model,
                                            l1_ratio=l1_ratio or 0.5)
                        st.session_state["cv"] = cv
                    if "cv" in st.session_state:
                        cv = st.session_state["cv"]
                        best = cv.loc[cv.cv_r2.idxmax()]
                        st.caption(f"CV-best α ≈ {best.alpha:.4g} (cv R²={best.cv_r2:.3f})")

                if model == "OLS":
                    coef_scale = "std" if st.radio(
                        "Coefficient scale", ["Standardized (Δret per 1σ)", "Raw β"],
                        index=0,
                        help="Standardized = β × SD(regressor): comparable across "
                             "variables on different scales — use this to read off what "
                             "mattered. Raw β = native exposure/sensitivity. Scaling never "
                             "changes R², t-stats, or alpha.",
                    ).startswith("Standardized") else "raw"
                else:
                    coef_scale = "std"
                    st.caption("Penalized coefficients are standardized (Δret per 1σ) by "
                               "construction.")
```

- [ ] **Step 4: Dispatch to `render_kalman`**

In `dashboard/app.py`, find the results block at the bottom of the Factor tab (currently around lines 802–804):
```python
    with results:
        render_regression(df_reg, target, model, alpha, l1_ratio, standardize,
                          coef_scale, window, x_cols)
```
Replace with:
```python
    with results:
        if is_kalman:
            render_kalman(df_reg, target, x_cols)
        else:
            render_regression(df_reg, target, model, alpha, l1_ratio, standardize,
                              coef_scale, window, x_cols)
```

- [ ] **Step 6: Byte-compile to catch syntax/indentation errors**

Run:
```bash
cd "$(git rev-parse --show-toplevel)" && uv run python -m py_compile dashboard/app.py && echo OK
```
Expected: prints `OK` with no traceback.

- [ ] **Step 7: Smoke-test the app renders headlessly**

Run:
```bash
cd "$(git rev-parse --show-toplevel)" && uv run python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('dashboard/app.py', default_timeout=120).run()
assert not at.exception, at.exception
print('app ran; radios:', [r.label for r in at.radio])
"
```
Expected: no exception; the printed radio labels include `Model` (and the default OLS path renders). This confirms the module imports, the new controls parse, and the default (non-Kalman) path is unbroken.

- [ ] **Step 8: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): add Kalman (TVP) model option and render_kalman to Factor tab"
```

---

## Task 4: Manual verification in the live app

**Files:** none (verification only)

- [ ] **Step 1: Launch the app**

Run: `uv run streamlit run dashboard/app.py`
Open the printed local URL, go to the **📈 Factor exposures** tab.

- [ ] **Step 2: Verify the Kalman happy path (factors)**

- Keep the default target; under **Regressor preset** pick **Fama-French factors**.
- Set **Model** to **Kalman (TVP)**. Confirm: the window slider and penalty expander disappear; a spinner runs once; then the estimate toggle, three metric tiles, the time-varying **alpha** chart (with 95% band + dashed OLS line), the **annualized alpha** chart with event markers inside the date range, per-factor **beta** small-multiples with bands, and the **Residual diagnostics** expander all render.
- Flip **Estimate** between *Smoothed* and *Filtered*: the lines and bands change; smoothed is tighter.

- [ ] **Step 3: Verify the guard path (macro PCs)**

- Pick a preset containing macro PCs or level regressors (e.g. anything with `PC1`/`vix`).
- Confirm the alpha panels are **hidden** and replaced by the explanatory note, while the beta paths and diagnostics still render.

- [ ] **Step 4: Verify no regression to the rolling models**

- Switch **Model** back to **OLS**: the rolling window slider, Advanced penalty expander, rolling-coefficient chart, rolling alpha/R², and contribution decomposition all return unchanged.

- [ ] **Step 5: Confirm the full test suite still passes**

Run: `uv run pytest tests/ -q`
Expected: all tests pass (existing suite + the 4 new Kalman tests).

---

## Self-Review Notes

- **Spec coverage:** engine copy (Task 1) ✓; `fit_tvp`/`TVPResult` wrapper + tests (Task 2) ✓; model-radio option + `render_kalman` with filtered/smoothed toggle, alpha (monthly+annualized), event annotations, beta small-multiples, residual diagnostics, screening + `alpha_interpretable` guard, caching (Task 3) ✓; manual + automated verification (Task 4) ✓. Out-of-scope items (model-comparison panel, forecasting, regime files, rolling middle-ground) are excluded, matching the spec.
- **Type consistency:** `TVPResult` fields used in `render_kalman` (`smoothed`, `filtered`, `smoothed_se`, `filtered_se`, `resid_std`, `ols_params`, `n`, `x_cols`) match the dataclass in Task 2. `res.ols_params["alpha"]` / per-factor keys match the `["alpha", *x_cols]` index. `R.annualize`, `R.screen_regressors`, `R.alpha_interpretable`, and the `_tight` helper are used with their real signatures from `rolling_regression.py` / `app.py`.
- **Dependencies:** no new packages; `plotly.graph_objects` and `statsmodels.stats.diagnostic.acorr_ljungbox` are already available.
- **Penalty-UI gating:** the Kalman branch fully skips the window slider and Advanced expander via `if not is_kalman:`; the penalty widgets are re-indented under that guard (not suppressed by a no-op context), so no penalty widgets render for Kalman.
