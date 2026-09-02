"""Rolling-window OLS factor attribution.

A middle ground between the static full-sample OLS baseline and the Kalman TVP
model used in tvp_kalman.ipynb / h_kalman.ipynb: betas are re-estimated on a
trailing fixed-length window, giving time-varying exposures without a
state-space model.

Companion to tvp_kalman_filter.py. See
docs/superpowers/specs/2026-07-01-rolling-regression-design.md.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def rolling_ols(df_model, factor_cols, window, y_col="excess_ret"):
    """Rolling-window OLS of `y_col` on [const] + `factor_cols`.

    The coefficient row at date t is estimated on the `window` months **ending
    at** t (contemporaneous / in-sample) — the fair comparison to the Kalman
    *smoothed* betas. Rows before a full window are NaN (not back-filled).

    Returns a DataFrame indexed like `df_model` with columns:
      ``alpha``, one per factor beta, ``<name>_se`` for each coefficient,
      ``r2``, ``nobs``.
    """
    data = df_model[[y_col] + list(factor_cols)].dropna()
    y = data[y_col]
    X = sm.add_constant(data[factor_cols])

    model = RollingOLS(y, X, window=window, min_nobs=window).fit()

    names = ["alpha"] + list(factor_cols)
    params = model.params.copy()
    params.columns = names
    bse = model.bse.copy()
    bse.columns = [f"{n}_se" for n in names]

    out = pd.concat([params, bse], axis=1)
    out["r2"] = model.rsquared
    out["nobs"] = out["alpha"].notna() * window
    out["nobs"] = out["nobs"].where(out["alpha"].notna())
    return out.reindex(df_model.index)


def rolling_one_step_pred(df_model, factor_cols, window, y_col="excess_ret"):
    """True out-of-sample one-step-ahead prediction.

    For each date t, fit OLS on months [t-window, t-1] and predict month t.
    This is the fair analogue of the Kalman *filtered* one-step-ahead
    predictions (no look-ahead).

    Returns a DataFrame indexed like `df_model` with columns
    ``fitted``, ``residual``, ``excess_ret``.
    """
    data = df_model[[y_col] + list(factor_cols)].dropna()
    y = data[y_col].values
    X = sm.add_constant(data[factor_cols]).values
    n = len(y)

    fitted = np.full(n, np.nan)
    for t in range(window, n):
        beta = np.linalg.lstsq(X[t - window : t], y[t - window : t], rcond=None)[0]
        fitted[t] = X[t] @ beta

    out = pd.DataFrame(
        {"fitted": fitted, "excess_ret": y}, index=data.index
    )
    out["residual"] = out["excess_ret"] - out["fitted"]
    return out[["fitted", "residual", "excess_ret"]].reindex(df_model.index)


def rolling_summary(rolling_df, ols_params, factor_cols):
    """Static-OLS vs rolling parameter comparison table.

    Mirrors the Cell-20 summary in tvp_kalman.ipynb.

    `ols_params` is the fitted static-OLS parameter vector in statsmodels order
    (const first). Returns a DataFrame indexed by ["alpha"] + factor_cols with
    columns OLS_estimate, Rolling_mean, Rolling_std, Rolling_min, Rolling_max.
    """
    names = ["alpha"] + list(factor_cols)
    coef = rolling_df[names]
    return pd.DataFrame(
        {
            "OLS_estimate": np.asarray(ols_params, dtype=float),
            "Rolling_mean": coef.mean().values,
            "Rolling_std": coef.std().values,
            "Rolling_min": coef.min().values,
            "Rolling_max": coef.max().values,
        },
        index=names,
    )


if __name__ == "__main__":
    # Self-check on the real fund data.
    df = pd.read_csv(
        "../data/engineered_data/combined_monthly.csv", parse_dates=["month"]
    ).set_index("month").sort_index()
    df["excess_ret"] = df["fund_ret"] - df["RF"]
    factors = ["Mkt_RF", "SMB", "HML", "Mom"]
    for w in (24, 36):
        roll = rolling_ols(df, factors, w)
        print(f"window={w}: {roll['alpha'].notna().sum()} rolling estimates")