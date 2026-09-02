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
