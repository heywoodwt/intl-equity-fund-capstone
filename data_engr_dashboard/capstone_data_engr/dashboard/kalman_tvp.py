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
    y_pred = np.array([(obs[t] @ filt_means[t]).item() for t in range(n_obs)])
    residuals = y - y_pred
    r = float(np.asarray(kf.observation_covariance)[0, 0])
    pred_var = np.array(
        [(obs[t] @ filt_covs[t] @ obs[t].T).item() + r for t in range(n_obs)])
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
