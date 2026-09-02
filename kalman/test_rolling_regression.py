"""Plain-assert tests for rolling_regression (no pytest in this repo).

Run: python3 kalman/test_rolling_regression.py
"""
import numpy as np
import pandas as pd

from rolling_regression import rolling_ols, rolling_one_step_pred, rolling_summary


def _synthetic(n=120, seed=0):
    """y = alpha + b1*x1 + b2*x2 + noise, constant coefficients."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    alpha, b1, b2 = 0.01, 1.20, -0.50
    y = alpha + b1 * x1 + b2 * x2 + rng.normal(scale=0.02, size=n)
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {"excess_ret": y, "x1": x1, "x2": x2}, index=idx
    ), (alpha, b1, b2)


def test_rolling_ols_recovers_constant_coefficients():
    df, (alpha, b1, b2) = _synthetic()
    window = 36
    out = rolling_ols(df, ["x1", "x2"], window)

    # Columns present
    for col in ["alpha", "x1", "x2", "alpha_se", "x1_se", "x2_se", "r2", "nobs"]:
        assert col in out.columns, f"missing column {col}"

    # Warm-up rows are NaN, later rows populated
    assert out["alpha"].iloc[: window - 1].isna().all(), "warm-up not NaN"
    assert out["alpha"].iloc[window - 1 :].notna().all(), "post-window has NaN"

    # Recovers the true (constant) coefficients on populated rows
    populated = out.dropna(subset=["alpha"])
    assert abs(populated["alpha"].mean() - alpha) < 0.01
    assert abs(populated["x1"].mean() - b1) < 0.03
    assert abs(populated["x2"].mean() - b2) < 0.03
    # nobs equals the window everywhere it is defined
    assert (populated["nobs"] == window).all()
    print("PASS test_rolling_ols_recovers_constant_coefficients")


def test_rolling_one_step_pred_out_of_sample():
    df, _ = _synthetic()
    window = 36
    pred = rolling_one_step_pred(df, ["x1", "x2"], window)

    for col in ["fitted", "residual", "excess_ret"]:
        assert col in pred.columns, f"missing column {col}"

    # First `window` rows cannot be predicted (need a full trailing window)
    assert pred["fitted"].iloc[:window].isna().all(), "warm-up not NaN"
    assert pred["fitted"].iloc[window:].notna().all(), "post-warmup has NaN"

    # Residual == excess_ret - fitted where defined
    valid = pred.dropna(subset=["fitted"])
    assert np.allclose(valid["residual"], valid["excess_ret"] - valid["fitted"])

    # Out-of-sample residual variance is >= in-sample fit variance
    insample = rolling_ols(df, ["x1", "x2"], window)
    oos_rmse = np.sqrt((valid["residual"] ** 2).mean())
    insample_rmse = np.sqrt((1 - insample["r2"].dropna()).mean()) * df["excess_ret"].std()
    assert oos_rmse >= insample_rmse * 0.8, "OOS error implausibly small"
    print("PASS test_rolling_one_step_pred_out_of_sample")


def test_rolling_summary_shape_and_values():
    df, _ = _synthetic()
    window = 36
    out = rolling_ols(df, ["x1", "x2"], window)
    ols_params = np.array([0.011, 1.19, -0.49])  # const, x1, x2 (statsmodels order)

    summary = rolling_summary(out, ols_params, ["x1", "x2"])

    assert list(summary.index) == ["alpha", "x1", "x2"]
    for col in ["OLS_estimate", "Rolling_mean", "Rolling_std", "Rolling_min", "Rolling_max"]:
        assert col in summary.columns, f"missing column {col}"
    # OLS_estimate copies the passed params in the right order
    assert np.allclose(summary["OLS_estimate"].values, ols_params)
    # Rolling_mean is between min and max
    assert (summary["Rolling_mean"] >= summary["Rolling_min"] - 1e-9).all()
    assert (summary["Rolling_mean"] <= summary["Rolling_max"] + 1e-9).all()
    print("PASS test_rolling_summary_shape_and_values")


if __name__ == "__main__":
    test_rolling_ols_recovers_constant_coefficients()
    test_rolling_one_step_pred_out_of_sample()
    test_rolling_summary_shape_and_values()
    print("\nAll tests passed.")