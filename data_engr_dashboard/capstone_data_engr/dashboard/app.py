"""International Equity Fund — interactive factor & data dashboard.

Run from the repo root:   uv run streamlit run dashboard/app.py
The folder is self-contained (reads only data/processed CSVs), so it can be
lifted into a deploy repo as-is.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling modules importable however the app is launched (streamlit run,
# AppTest, or a notebook adding this dir to the path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import data as D
import kalman_tvp as K
import rolling_regression as R
import etf_analytics as EA
import etf_prices

st.set_page_config(page_title="International Equity Fund Dashboard", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data
def get_data() -> pd.DataFrame:
    return D.load_combined_with_pcs()


@st.cache_data(show_spinner="Loading ticker…")
def get_analytics_frame(ticker: str) -> pd.DataFrame:
    return EA.load_analytics_frame(ticker)


def ticker_selector(key: str) -> str | None:
    """Ticker picker for the analytics tabs. Returns None if input is invalid.

    A blank input resolves to the fund, which bypasses the ETF gate -- it is a
    non-ETF that `etf_prices.validate_etf` rejects by design. Everything else
    is gated, so a mistyped or non-ETF symbol is caught here rather than
    producing an empty fit.
    """
    col_a, col_b = st.columns([1, 3])
    with col_a:
        ticker = st.text_input(
            "Ticker", value="", key=f"ticker_{key}",
            placeholder="Compare an ETF (optional)",
            help="Leave blank to view the fund. Enter any ETF symbol — "
                 "EFA, SCZ, VSS, IEFA, ACWX, VTI … — to run the same analysis on it.",
        ).strip().upper()

    if not ticker:
        return EA.CAPSTONE_TICKER
    if EA.is_capstone(ticker):
        return ticker

    try:
        meta = etf_prices.validate_etf(ticker)
    except etf_prices.EtfDataError as exc:
        with col_b:
            st.error(str(exc))
        return None
    with col_b:
        st.caption(f"**{meta.long_name or ticker}**"
                   + (f" · {meta.category}" if meta.category else ""))
    return ticker


def resolve_frame(ticker: str):
    """Load `ticker`'s analytics frame, rendering any failure as an error.

    Returns ``None`` when the frame could not be built, so callers can stop.
    """
    try:
        frame = get_analytics_frame(ticker)
    except etf_prices.EtfDataError as exc:
        st.error(str(exc))
        return None
    if frame.empty:
        st.error(f"No overlapping monthly data for {ticker}.")
        return None
    if not EA.is_capstone(ticker):
        is_stale, as_of = EA.price_status(ticker)
        if is_stale:
            st.warning(
                f"Yahoo is unreachable — showing cached prices through "
                f"{as_of:%Y-%m-%d}. These are not current."
            )
        st.caption(
            f"{frame['date'].min():%Y-%m} – {frame['date'].max():%Y-%m} · "
            f"{len(frame)} months. Holdings-derived regressors "
            "(fundamentals, sector/country weights) are available for the fund only."
        )
    return frame


try:
    df = get_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

DATES = df["date"]
DMIN, DMAX = DATES.min().to_pydatetime(), DATES.max().to_pydatetime()

st.title("International Equity Fund")
st.caption(
    "Developed ex-US small/midcap portfolio. Monthly performance data 10/2014 "
    "through 12/2024. Daily holdings data 1/2018 through 12/2022. Benchmark MSCI "
    "EAFE, iShares MSCI EAFE ETF (EFA) as proxy. Alternative benchmarks Vanguard "
    "FTSE All-World ex-US Small-Cap ETF (VSS) and iShares MSCI EAFE Small-Cap ETF "
    "(SCZ)."
)

with st.sidebar:
    st.header("Settings")
    bench_keys = [k for k in D.BENCHMARKS if k in df.columns]
    bench_col = st.selectbox(
        "Benchmark", bench_keys,
        index=bench_keys.index(D.DEFAULT_BENCHMARK) if D.DEFAULT_BENCHMARK in bench_keys else 0,
        format_func=lambda k: D.BENCHMARKS[k],
        help="Default is EFA (iShares MSCI EAFE), the benchmark proxy. SCZ and VSS "
             "are small-cap alternatives; the blend is their equal weight.",
    )
    period = st.slider("Period", min_value=DMIN, max_value=DMAX,
                       value=(DMIN, DMAX), format="YYYY-MM")
    st.caption("Benchmark & period apply to the Overview and Risk tabs.")

period_mask = (DATES >= period[0]) & (DATES <= period[1])

(tab_overview, tab_risk, tab_bench, tab_reg, tab_pos, tab_explore,
 tab_regime) = st.tabs([
    "📊 Overview", "🛡️ Risk & drawdown", "🎯 Benchmark & active",
    "📈 Factor exposures", "🧭 Positioning", "🔎 Data explorer",
    "🌐 Macro & regime"])



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


def render_regression(df, target, model, alpha, l1_ratio, standardize, coef_scale,
                      window, x_cols):
    """Render the rolling-regression results over ``df`` (already period-filtered
    by the caller). Uses ``return`` (not ``st.stop()``, which would also halt the
    sibling tabs) to bail out on an invalid spec."""
    if not x_cols:
        st.info("Select at least one regressor.")
        return

    # Screen out degenerate/collinear regressors and report what was dropped.
    sc = R.screen_regressors(df, target, x_cols)
    drops = []
    if sc.dropped_zero:
        drops.append("zero-variance: " + ", ".join(D.label(c) for c in sc.dropped_zero))
    if sc.dropped_collinear:
        drops.append("redundant (exact linear combo of others): "
                     + ", ".join(D.label(c) for c in sc.dropped_collinear))
    if drops:
        st.warning("Dropped to keep the regression identified — " + "; ".join(drops) + ".")
    x_fit = sc.kept
    if not x_fit:
        st.warning("No usable regressors left after screening.")
        return
    if model == "OLS" and sc.condition_number > 30:
        st.warning(
            f"These regressors are highly collinear (condition number "
            f"{sc.condition_number:.0f}); rolling-OLS coefficients and the "
            f"intercept are unstable. Prefer **Ridge** or the **principal "
            f"components**."
        )

    try:
        roll = R.rolling_fit(df, target, x_fit, window=window, model=model,
                             alpha=alpha or 1.0, l1_ratio=l1_ratio or 0.5,
                             standardize=standardize, coef_scale=coef_scale)
        fit = R.full_sample_fit(df, target, x_fit, model=model,
                                alpha=alpha or 1.0, l1_ratio=l1_ratio or 0.5,
                                standardize=standardize)
    except ValueError as e:
        st.warning(str(e))
        return

    alpha_ok = R.alpha_interpretable(df, x_fit)
    if model != "OLS" or coef_scale == "std":
        coef_label = "Coefficient (Δret per 1σ)"
    else:
        coef_label = "Raw β (exposure)"
    c1, c2, c3 = st.columns(3)
    c1.metric("Full-sample R²", f"{fit.r2:.3f}")
    c2.metric("Annualized alpha (full)",
              f"{R.annualize(fit.coefs.loc['const', 'coef']):.2%}" if alpha_ok else "n/a")
    c3.metric("Usable months", fit.n)

    st.markdown(f"##### Rolling coefficients — {window}-month window")
    beta_cols = [c for c in x_fit if c in roll.columns]
    show = st.multiselect("Lines to plot", beta_cols,
                          default=beta_cols[: min(6, len(beta_cols))],
                          format_func=D.label, key="betalines")
    if show:
        long = (roll[show].rename(columns=D.label)
                .reset_index().melt("date", var_name="Regressor",
                                    value_name=coef_label))
        fig = px.line(long, x="date", y=coef_label, color="Regressor")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=380, margin=dict(t=10, b=0, l=0, r=0),
                          legend_title=None)
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("##### Rolling annualized alpha (intercept)")
        if alpha_ok:
            fig = px.area(roll.reset_index(), x="date", y="alpha_ann")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(height=260, margin=dict(t=10, b=0, l=0, r=0),
                              yaxis_tickformat=".0%", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(
                "Alpha is hidden because the intercept isn't a meaningful "
                "alpha here. It's “return when every regressor = 0”, but these "
                "regressors don't sit near 0 within a rolling window — **level** "
                "regressors (VIX, yields, spreads) are off-zero, and the "
                "**macro PCs** are highly persistent (e.g. PC1 autocorr ≈ 0.98) "
                "so they drift far from 0. The intercept then *extrapolates* and "
                "explodes (the 800%+ you saw). For a real alpha use factor "
                "returns (Mkt_RF…Mom); here, read the coefficients and the "
                "contribution decomposition instead."
            )
    with b:
        st.markdown("##### Rolling R²")
        fig = px.line(roll.reset_index(), x="date", y="r2")
        fig.update_layout(height=260, margin=dict(t=10, b=0, l=0, r=0),
                          yaxis_range=[0, 1], yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    # --- contribution decomposition --------------------------------------
    st.markdown("##### What drove returns — contribution decomposition")
    st.caption(
        "OLS on the selected variables over the usable sample. **Left:** share "
        "of return *variance* explained — sums to R², scale-invariant (the most "
        "defensible attribution). **Right:** average monthly return contributed "
        "(β × mean; regressors + alpha sum to mean return) — intuitive but "
        "level-sensitive for non-return regressors, so read it for the factors."
    )
    try:
        contrib, csum = R.contribution_decomposition(df, target, x_fit)
    except Exception as e:  # e.g. p >= n under a Ridge selection
        contrib = None
        st.info(f"Decomposition needs an identified OLS fit (more months than "
                f"regressors): {e}")
    if contrib is not None:
        cc1, cc2 = st.columns(2)
        with cc1:
            vs = contrib["var_share"].drop("Alpha").dropna()
            vs = vs.reindex(vs.abs().sort_values().index)
            names = [("Residual" if i == "Residual" else D.label(i)) for i in vs.index]
            fig = px.bar(x=vs.values, y=names, orientation="h",
                         labels={"x": "Share of return variance", "y": ""})
            fig.update_traces(marker_color=["#bbb" if n == "Residual" else "#1f6feb"
                                            for n in names])
            fig.update_layout(height=max(220, 26 * len(vs) + 70),
                              margin=dict(t=10, b=0, l=0, r=0),
                              xaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Regressor shares sum to R² = {csum['r2']:.2f}; "
                       f"residual = {1 - csum['r2']:.2f}.")
        with cc2:
            mc = contrib["mean_contrib"].dropna()
            mc = mc.reindex(mc.abs().sort_values().index)
            names = [("Alpha" if i == "Alpha" else D.label(i)) for i in mc.index]
            fig = px.bar(x=mc.values, y=names, orientation="h",
                         labels={"x": "Avg monthly return contribution", "y": ""})
            fig.update_traces(marker_color=["#888" if n == "Alpha" else "#1f6feb"
                                            for n in names])
            fig.update_layout(height=max(220, 26 * len(mc) + 70),
                              margin=dict(t=10, b=0, l=0, r=0),
                              xaxis_tickformat=".2%")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Contributions + alpha sum to mean monthly return "
                       f"{csum['mean_y']:.3%}.")

    with st.expander("Full-sample fit table"):
        st.dataframe(fit.coefs.style.format("{:.4f}", na_rep="—"),
                     width="stretch")
        st.caption(
            "`coef` = raw "
            + ("β (exposure). " if not fit.standardized else "(per-1σ) coefficient. ")
            + "`coef_per_sd` = Δ monthly return per +1 SD of the regressor "
            "(comparable across variables). `beta_weight` = fully standardized "
            "(dimensionless). `const` = alpha (intercept at raw regressors = 0)."
            + (" Penalized coefficients are standardized within-sample."
               if fit.standardized else "")
        )


def _pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{d}%}"


def _num(x, d=2):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{d}f}"


def render_overview(sub, bench_col):
    if sub["fund_ret"].dropna().empty:
        st.info("No fund returns in the selected period.")
        return
    s = A.performance_summary(sub, "fund_ret", bench_col, "RF")
    st.caption(f"{s['start']:%Y-%m} – {s['end']:%Y-%m} · {s['n_months']} months · "
               f"vs **{D.BENCHMARKS.get(bench_col, D.label(bench_col))}**")

    head, tog = st.columns([4, 1])
    head.markdown("##### Growth of $1")
    logy = tog.toggle("Log scale", value=False)
    ret_cols = ["fund_ret"] + [c for c in ("efa_ret", "scz_ret", "vss_ret",
                                           "bench_avg_ret") if c in sub.columns]
    g = A.growth_of_dollar(sub, ret_cols)
    long = g.melt("date", var_name="Series", value_name="val")
    long["Series"] = long["Series"].map(D.label)
    fig = px.line(long, x="date", y="val", color="Series")
    if logy:
        fig.update_yaxes(type="log")
    fig.update_layout(height=380, margin=dict(t=10, b=0, l=0, r=0),
                      legend_title=None, yaxis_title="Growth of $1")
    st.plotly_chart(fig, width="stretch")

    uw = A.drawdown_series(A.series(sub, "fund_ret")).rename("dd").reset_index()
    fig = px.area(uw, x="date", y="dd")
    fig.update_traces(line_color="#d62728", fillcolor="rgba(214,39,40,0.2)")
    fig.update_layout(height=190, margin=dict(t=10, b=0, l=0, r=0),
                      yaxis_tickformat=".0%", yaxis_title="Drawdown")
    st.plotly_chart(fig, width="stretch")

    with st.expander("📊 Performance statistics — CAGR, Sharpe, alpha/beta, returns tables…"):
        r1 = st.columns(6)
        for col, (lbl, val) in zip(r1, [
                ("CAGR", _pct(s["cagr"])), ("Ann. volatility", _pct(s["ann_vol"])),
                ("Sharpe", _num(s["sharpe"])), ("Sortino", _num(s["sortino"])),
                ("Max drawdown", _pct(s["max_drawdown"])), ("Calmar", _num(s["calmar"]))]):
            col.metric(lbl, val)
        r2 = st.columns(6)
        for col, (lbl, val) in zip(r2, [
                ("Alpha (ann.)", _pct(s.get("alpha_ann"))), ("Beta", _num(s.get("beta"))),
                ("Tracking error", _pct(s.get("tracking_error"))),
                ("Info ratio", _num(s.get("information_ratio"))),
                ("Up capture", _num(s.get("up_capture"))),
                ("Down capture", _num(s.get("down_capture")))]):
            col.metric(lbl, val)

        cols = ["fund_ret", bench_col]
        left, right = st.columns(2)
        with left:
            st.markdown("##### Trailing returns")
            tr = A.trailing_returns(sub, cols)
            tr["Active"] = tr["fund_ret"] - tr[bench_col]
            tr = tr.rename(columns={"fund_ret": "Fund", bench_col: "Benchmark"})
            st.dataframe(tr.set_index("Period").style.format("{:.1%}", na_rep="—"),
                         width="stretch")
            st.caption("≤ 1Y cumulative; 3Y / 5Y / ITD annualized.")
        with right:
            st.markdown("##### Calendar-year returns")
            cal = A.calendar_returns(sub, cols)
            cal["Active"] = cal["fund_ret"] - cal[bench_col]
            cal = (cal.rename(columns={"fund_ret": "Fund", bench_col: "Benchmark"})
                   .sort_values("Year", ascending=False))
            st.dataframe(cal.set_index("Year").style.format("{:.1%}", na_rep="—"),
                         width="stretch", height=320)

        st.markdown("##### Monthly returns")
        mat = A.monthly_return_matrix(sub, "fund_ret")
        order = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)] + ["Year"]
        mat = mat.reindex(columns=[c for c in order if c in mat.columns])
        fig = px.imshow(mat, color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                        aspect="auto", text_auto=".1%")
        fig.update_layout(height=max(220, 24 * len(mat) + 80),
                          margin=dict(t=10, b=0, l=0, r=0), coloraxis_showscale=False)
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, width="stretch")


def render_risk(sub, bench_col):
    r = A.series(sub, "fund_ret")
    if r.empty:
        st.info("No fund returns in the selected period.")
        return
    s = A.performance_summary(sub, "fund_ret", bench_col, "RF")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Risk & risk-adjusted metrics")
        rows = [
            ("Annualized return (CAGR)", _pct(s["cagr"])),
            ("Annualized volatility", _pct(s["ann_vol"])),
            ("Downside deviation", _pct(A.downside_deviation(r))),
            ("Sharpe ratio", _num(s["sharpe"])),
            ("Sortino ratio", _num(s["sortino"])),
            ("Calmar ratio", _num(s["calmar"])),
            ("Max drawdown", _pct(s["max_drawdown"])),
            ("Best / worst month", f"{_pct(s['best_month'])} / {_pct(s['worst_month'])}"),
            ("% positive months", _pct(s["pct_positive"])),
            ("Skew / excess kurtosis", f"{_num(s['skew'])} / {_num(s['kurtosis'])}"),
            ("Monthly VaR (95%)", _pct(s["var95"])),
            ("Monthly CVaR (95%)", _pct(s["cvar95"])),
            ("Beta vs benchmark", _num(s.get("beta"))),
            ("Tracking error", _pct(s.get("tracking_error"))),
            ("Information ratio", _num(s.get("information_ratio"))),
            ("Batting average", _pct(s.get("batting_avg"))),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric"),
                     width="stretch", height=460)
    with right:
        st.markdown("##### Return distribution")
        fig = px.histogram(r, nbins=30)
        fig.add_vline(x=s["var95"], line_dash="dash", line_color="#d62728",
                      annotation_text="VaR 95%")
        fig.add_vline(x=s["cvar95"], line_dash="dot", line_color="#8b0000",
                      annotation_text="CVaR")
        fig.update_layout(height=290, showlegend=False, margin=dict(t=10, b=0, l=0, r=0),
                          xaxis_tickformat=".0%", xaxis_title="Monthly return")
        st.plotly_chart(fig, width="stretch")
        st.markdown("##### Up / down capture")
        cap = pd.DataFrame({"Capture": ["Up", "Down"],
                            "Ratio": [s.get("up_capture"), s.get("down_capture")]})
        fig = px.bar(cap, x="Ratio", y="Capture", orientation="h")
        fig.update_traces(texttemplate="%{x:.2f}", textposition="outside",
                          marker_color=["#1f6feb", "#d62728"])
        fig.add_vline(x=1.0, line_dash="dot", line_color="gray")
        fig.update_layout(height=170, margin=dict(t=10, b=0, l=0, r=0), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Worst drawdowns")
    dd = A.drawdown_table(sub, "fund_ret", top=5)
    if len(dd):
        disp = dd.copy()
        for c in ("peak", "trough"):
            disp[c] = disp[c].dt.strftime("%Y-%m")
        disp["recovery"] = disp["recovery"].dt.strftime("%Y-%m").fillna("—")
        disp["depth"] = disp["depth"].map(lambda x: f"{x:.1%}")
        for c in ("months_to_trough", "recovery_months", "length_months"):
            disp[c] = disp[c].map(lambda v: "—" if pd.isna(v) else str(int(v)))
        disp = disp.rename(columns={
            "peak": "Peak", "trough": "Trough", "recovery": "Recovery", "depth": "Depth",
            "months_to_trough": "→ Trough (mo)", "recovery_months": "Recovery (mo)",
            "length_months": "Length (mo)"})
        st.dataframe(disp, width="stretch", hide_index=True)
        st.caption("A blank recovery means the drawdown had not fully recovered by the period end.")

    st.markdown("##### Rolling risk")
    win = st.slider("Rolling window (months)", 6, 36, 12, 3, key="risk_win")
    rm = A.rolling_metrics(sub, "fund_ret", bench_col, "RF", window=win)
    if len(rm):
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(rm, x="date", y="vol")
            fig.update_layout(height=240, margin=dict(t=10, b=0, l=0, r=0),
                              yaxis_tickformat=".0%", yaxis_title="Ann. volatility")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.line(rm, x="date", y="sharpe")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(height=240, margin=dict(t=10, b=0, l=0, r=0),
                              yaxis_title="Sharpe")
            st.plotly_chart(fig, width="stretch")


def _tight(fig, height, **kw):
    fig.update_layout(height=height, margin=dict(t=10, b=0, l=0, r=0),
                      legend_title=None, **kw)
    return fig


def render_benchmark(sub, bench_col):
    r = A.series(sub, "fund_ret")
    if r.empty or bench_col not in sub.columns:
        st.info("No data for the selected benchmark / period.")
        return
    s = A.performance_summary(sub, "fund_ret", bench_col, "RF")
    bn = D.BENCHMARKS.get(bench_col, D.label(bench_col))
    st.caption(f"Active management vs **{bn}** · {s['start']:%Y-%m} – {s['end']:%Y-%m} "
               f"· {s['n_months']} months")

    k = st.columns(6)
    for col, (lbl, val) in zip(k, [
            ("Active return (ann.)", _pct(s.get("active_return_ann"))),
            ("Alpha (ann.)", _pct(s.get("alpha_ann"))), ("Beta", _num(s.get("beta"))),
            ("Tracking error", _pct(s.get("tracking_error"))),
            ("Info ratio", _num(s.get("information_ratio"))),
            ("Batting avg", _pct(s.get("batting_avg")))]):
        col.metric(lbl, val)

    st.markdown("##### Cumulative active return (fund vs benchmark)")
    ca = A.cumulative_active(sub, "fund_ret", bench_col)
    fig = px.area(ca, x="date", y="cum_active")
    fig.update_traces(line_color="#1f6feb", fillcolor="rgba(31,111,235,0.15)")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(_tight(fig, 240, yaxis_tickformat=".0%",
                           yaxis_title="Cumulative active"), width="stretch")

    st.markdown("##### Rolling benchmark-relative risk")
    win = st.slider("Rolling window (months)", 6, 36, 12, 3, key="bench_win")
    rm = A.rolling_metrics(sub, "fund_ret", bench_col, "RF", window=win)
    if len(rm):
        cc = st.columns(2)
        plots = [("beta", "Beta", 1.0), ("ir", "Information ratio", 0.0),
                 ("te", "Tracking error", None), ("corr", "Correlation", None)]
        for i, (y, title, hl) in enumerate(plots):
            with cc[i % 2]:
                fig = px.line(rm, x="date", y=y)
                if hl is not None:
                    fig.add_hline(y=hl, line_dash="dot", line_color="gray")
                fmt = ".0%" if y == "te" else None
                st.plotly_chart(_tight(fig, 210, yaxis_title=title,
                                       yaxis_tickformat=fmt), width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("##### Up / down capture")
        cap = pd.DataFrame({"Capture": ["Up", "Down"],
                            "Ratio": [s.get("up_capture"), s.get("down_capture")]})
        fig = px.bar(cap, x="Ratio", y="Capture", orientation="h")
        fig.update_traces(texttemplate="%{x:.2f}", textposition="outside",
                          marker_color=["#1f6feb", "#d62728"])
        fig.add_vline(x=1.0, line_dash="dot", line_color="gray")
        st.plotly_chart(_tight(fig, 200, xaxis_title=None), width="stretch")
        st.caption("Capture > 1 = the fund moved more than the benchmark in those months.")
    with b:
        st.markdown("##### Fund vs benchmark (monthly)")
        d = sub[[bench_col, "fund_ret"]].dropna()
        fig = px.scatter(d, x=bench_col, y="fund_ret", trendline="ols",
                         trendline_color_override="#d62728")
        st.plotly_chart(_tight(fig, 260, xaxis_tickformat=".0%",
                               yaxis_tickformat=".0%", xaxis_title=bn,
                               yaxis_title="Fund return"), width="stretch")


def render_attribution(hold, asof: str | None = None):
    st.divider()
    st.markdown("### Performance attribution")
    sr = D.load_sector_returns()
    sc = D.load_security_contrib()
    summ = D.load_attribution_summary()
    if sr is None:
        st.info("Attribution data not built — run `scripts/build_attribution.py`.")
        return
    st.caption(
        "Holdings-based (gross), reconciled to the official active return with a "
        "**residual**. Fund sector returns are noisy (few names / sector), so read "
        "**allocation** as robust and **selection / interaction** directionally. "
        "AML.L (Aston Martin) corporate-action noise inflates a few months.")

    a, b = st.columns(2)
    with a:
        st.markdown("##### Sector contribution-to-return (cumulative)")
        cum = sr.groupby("sector")["contrib"].sum().sort_values()
        fig = px.bar(x=cum.values, y=cum.index, orientation="h",
                     labels={"x": "Cumulative contribution", "y": ""})
        fig.update_traces(marker_color=["#d62728" if v < 0 else "#1f6feb"
                                        for v in cum.values])
        st.plotly_chart(_tight(fig, 360, xaxis_tickformat=".1%"), width="stretch", key="attr_sector_contrib")
    with b:
        st.markdown("##### Top contributors / detractors")
        if sc is not None:
            tot = sc.groupby("ticker")["contrib"].sum().sort_values()
            top = pd.concat([tot.head(8), tot.tail(8)])
            top = top[~top.index.duplicated()].sort_values()
            fig = px.bar(x=top.values, y=top.index, orientation="h",
                         labels={"x": "Cumulative contribution", "y": ""})
            fig.update_traces(marker_color=["#d62728" if v < 0 else "#1f6feb"
                                            for v in top.values])
            st.plotly_chart(_tight(fig, 360, xaxis_tickformat=".1%"), width="stretch", key="attr_top_contrib")

    if summ is not None:
        st.markdown("##### Brinson attribution vs blended benchmark (cumulative)")
        tot = summ[["allocation", "selection", "interaction", "residual",
                    "true_active"]].sum()
        names = ["Allocation", "Selection", "Interaction", "Residual", "True active"]
        vals = [tot["allocation"], tot["selection"], tot["interaction"],
                tot["residual"], tot["true_active"]]
        fig = px.bar(x=names, y=vals)
        fig.update_traces(marker_color=["#1f6feb", "#2ca02c", "#ff7f0e", "#bbb", "#111"],
                          text=[f"{v:+.1%}" for v in vals], textposition="outside")
        fig.add_hline(y=0, line_color="gray")
        st.plotly_chart(_tight(fig, 300, yaxis_tickformat=".0%"), width="stretch", key="attr_brinson_cum")
        st.caption(f"Allocation **{tot['allocation']:+.1%}** (robust) + selection "
                   f"{tot['selection']:+.1%} + interaction {tot['interaction']:+.1%} + "
                   f"residual {tot['residual']:+.1%} = true active "
                   f"**{tot['true_active']:+.1%}** over 2018-2022.")

    with st.expander("📊 Full-Period Brinson–Fachler Summary Plots (2018–2022)", expanded=False):
        c1, c2 = st.columns(2)
        sect_hold_img = D.get_brinson_holdings_plot("sectors")
        ctry_hold_img = D.get_brinson_holdings_plot("countries")
        with c1:
            if sect_hold_img:
                st.markdown("##### Sector Attribution (Full Period)")
                st.image(str(sect_hold_img), use_container_width=True)
            else:
                st.info("Sector holdings plot not found in `brinson_fachler/`.")
        with c2:
            if ctry_hold_img:
                st.markdown("##### Country Attribution (Full Period)")
                st.image(str(ctry_hold_img), use_container_width=True)
            else:
                st.info("Country holdings plot not found in `brinson_fachler/`.")

    if asof:
        st.markdown(f"#### Monthly Brinson–Fachler Attribution — **{asof}**")
        st.caption("Sector and country attribution breakdown for the selected as-of month.")
        m_sect_img = D.get_brinson_monthly_plot("sector", asof)
        m_ctry_img = D.get_brinson_monthly_plot("country", asof)
        col_s, col_c = st.columns(2)
        with col_s:
            if m_sect_img:
                st.markdown(f"##### Sector Attribution ({asof})")
                st.image(str(m_sect_img), use_container_width=True)
            else:
                st.info(f"No monthly sector plot available for {asof}.")
        with col_c:
            if m_ctry_img:
                st.markdown(f"##### Country Attribution ({asof})")
                st.image(str(m_ctry_img), use_container_width=True)
            else:
                st.info(f"No monthly country plot available for {asof}.")



def render_positioning(full):
    sect_cols = [c for c in full.columns if c.startswith("sect_wt_")]
    hold = (full[full[sect_cols].notna().any(axis=1)] if sect_cols
            else full.iloc[0:0])
    if hold.empty:
        st.info("No holdings-based positioning data available.")
        return
    st.info("📅 Holdings-based views cover **2018-04 – 2022-12** (the period with "
            "position data) — they reflect the portfolio then, not current holdings.")
    months = hold["month"].tolist()
    asof = st.select_slider("As-of month", options=months, value=months[-1])
    row = hold[hold["month"] == asof].iloc[0]

    k = st.columns(4)
    k[0].metric("Holdings", int(row["n_holdings"]) if pd.notna(row.get("n_holdings")) else "—")
    k[1].metric("Top sector", f"{D.sector_pretty(str(row['top_sector']))} "
                f"({row['top_sector_wt']:.0%})" if pd.notna(row.get("top_sector")) else "—")
    k[2].metric("Sector HHI", _num(row.get("sector_hhi")))
    k[3].metric("Sectors / countries",
                f"{int(row['n_sectors'])} / {int(row['n_countries'])}"
                if pd.notna(row.get("n_sectors")) else "—")

    sectors = [c[len("sect_wt_"):] for c in sect_cols]
    rows = []
    for sl in sectors:
        fw = row.get(f"sect_wt_{sl}", 0.0) or 0.0
        bw = row.get(f"bench_sect_wt_{sl}", np.nan)
        rows.append({"Sector": D.sector_pretty(sl), "Fund": fw, "Benchmark": bw,
                     "Active": (fw - bw) if pd.notna(bw) else np.nan})
    sdf = pd.DataFrame(rows).sort_values("Fund")

    a, b = st.columns(2)
    with a:
        st.markdown("##### Sector allocation vs blended benchmark")
        long = sdf.melt("Sector", value_vars=["Fund", "Benchmark"],
                        var_name="Series", value_name="Weight")
        fig = px.bar(long, x="Weight", y="Sector", color="Series",
                     barmode="group", orientation="h")
        st.plotly_chart(_tight(fig, 380, xaxis_tickformat=".0%"), width="stretch")
    with b:
        st.markdown("##### Active sector tilts (fund − benchmark)")
        at = sdf.dropna(subset=["Active"]).sort_values("Active")
        fig = px.bar(at, x="Active", y="Sector", orientation="h")
        fig.update_traces(marker_color=["#d62728" if v < 0 else "#1f6feb"
                                        for v in at["Active"]])
        fig.add_vline(x=0, line_color="gray")
        st.plotly_chart(_tight(fig, 380, xaxis_tickformat=".1%"), width="stretch")

    c, d = st.columns(2)
    with c:
        st.markdown("##### Geographic allocation")
        ctry_cols = [col for col in full.columns if col.startswith("ctry_wt_")]
        cdf = pd.DataFrame([{"Country": D.country_pretty(col),
                             "Weight": row.get(col, 0.0) or 0.0} for col in ctry_cols])
        cdf = cdf[cdf["Weight"] > 0.001].sort_values("Weight")
        fig = px.bar(cdf, x="Weight", y="Country", orientation="h")
        st.plotly_chart(_tight(fig, max(260, 16 * len(cdf) + 60),
                               xaxis_tickformat=".0%"), width="stretch")
    with d:
        st.markdown("##### Sector weights over time")
        ts = hold[["date", *sect_cols]].copy()
        long = ts.melt("date", var_name="Sector", value_name="Weight")
        long["Sector"] = long["Sector"].map(D.sector_pretty)
        fig = px.area(long, x="date", y="Weight", color="Sector")
        st.plotly_chart(_tight(fig, max(260, 16 * len(cdf) + 60),
                               yaxis_tickformat=".0%"), width="stretch")

    st.markdown(f"##### Style & quality characteristics — value-weighted, {asof}")
    groups = [
        ("Valuation", [("ff_pe_dil_whmean", "P/E"), ("ff_pbk_tang_whmean", "P/B"),
                       ("ff_psales_dil_whmean", "P/S"), ("ff_div_yld_wmean", "Div yield %"),
                       ("ff_entrpr_val_ebitda_oper_whmean", "EV/EBITDA")]),
        ("Quality", [("ff_roce_wmean", "ROCE"), ("ff_roic_wmean", "ROIC"),
                     ("ff_ebit_oper_mgn_wmean", "EBIT margin %"),
                     ("ff_debt_eq_wmean", "Debt/Equity"), ("ff_zscore_wmean", "Altman Z")]),
        ("Growth", [("ff_sales_gr_wmean", "Sales growth %"),
                    ("ff_eps_dil_gr_wmean", "EPS growth %"),
                    ("ff_bps_gr_wmean", "Book/sh growth %")]),
    ]
    gc = st.columns(3)
    for (title, specs), col in zip(groups, gc):
        with col:
            st.caption(title)
            rr = [{"Metric": lbl, "Value": (f"{row[c]:.2f}" if pd.notna(row.get(c)) else "—")}
                  for c, lbl in specs if c in full.columns]
            st.dataframe(pd.DataFrame(rr).set_index("Metric"), width="stretch")

    render_attribution(hold, asof=asof)


def render_regime(frame):
    st.subheader("Performance by macro regime")
    st.caption("Fund monthly return conditioned on the macro environment "
               "(low / mid / high terciles of the chosen variable).")
    regime_opts = {
        "VIX (volatility)": "vix", "10y Treasury yield": "y10",
        "Yield curve (10y−2y)": "slope_10y_2y", "BAA credit spread": "baa_spread",
        "CPI inflation (YoY)": "cpi_yoy", "S&P 500 12m momentum": "sp500_mom_12m",
    }
    pick = st.selectbox("Regime variable", list(regime_opts))
    col = regime_opts[pick]
    rb = A.regime_buckets(frame, "fund_ret", col, q=3, labels=["Low", "Mid", "High"])
    if len(rb):
        left, right = st.columns([2, 1])
        with left:
            fig = px.bar(rb, x="bucket", y="ann_return")
            fig.update_traces(texttemplate="%{y:.1%}", textposition="outside",
                              marker_color="#1f6feb")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(_tight(fig, 300, yaxis_tickformat=".0%",
                                   xaxis_title=f"{pick} tercile",
                                   yaxis_title="Annualized return"), width="stretch")
        with right:
            disp = rb.copy()
            disp["ann_return"] = disp["ann_return"].map("{:.1%}".format)
            disp["hit_rate"] = disp["hit_rate"].map("{:.0%}".format)
            disp["ann_vol"] = disp["ann_vol"].map("{:.1%}".format)
            disp = disp[["bucket", "n", "ann_return", "hit_rate", "ann_vol"]].rename(
                columns={"bucket": "Regime", "ann_return": "Ann. return",
                         "hit_rate": "Hit rate", "ann_vol": "Ann. vol"})
            st.dataframe(disp.set_index("Regime"), width="stretch")
    st.divider()


with tab_overview:
    render_overview(df[period_mask], bench_col)

with tab_risk:
    render_risk(df[period_mask], bench_col)

with tab_bench:
    render_benchmark(df[period_mask], bench_col)

with tab_pos:
    render_positioning(df)



# =========================================================================
# TAB 1 — Rolling regression
# =========================================================================
with tab_reg:
    reg_ticker = ticker_selector("reg")
    if reg_ticker is None:
        st.stop()
    reg_frame = resolve_frame(reg_ticker)
    if reg_frame is None:
        st.stop()
    REG_DATES = reg_frame["date"]
    RMIN = REG_DATES.min().to_pydatetime()
    RMAX = REG_DATES.max().to_pydatetime()

    controls, results = st.columns([1, 3], gap="large")

    with controls:
        st.subheader("Specification")
        target = st.selectbox(
            "Dependent variable", D.TARGETS, index=1,
            format_func=D.label,
            help="Excess = fund_ret - RF (intercept = alpha). "
                 "alpha_vs_* = active return over benchmark.",
        )

        reg_period = st.slider(
            "Sample period", min_value=RMIN, max_value=RMAX,
            value=(RMIN, RMAX), format="YYYY-MM", key="reg_period",
            help="Restricts the rolling fit, full-sample fit, and contribution "
                 "decomposition to this window — independent of the sidebar "
                 "period, which only drives the Overview/Risk/Benchmark tabs. "
                 "Narrow this to isolate a regime (e.g. post-2020) instead of "
                 "diluting it into the full-sample numbers.",
        )
        df_reg = reg_frame[(REG_DATES >= reg_period[0])
                           & (REG_DATES <= reg_period[1])].reset_index(drop=True)
        st.caption(f"{df_reg['date'].min():%Y-%m} – {df_reg['date'].max():%Y-%m} "
                  f"· {len(df_reg)} months")
        st.caption(
            "The fit drops any month missing a selected regressor, so the "
            "effective sample can be shorter than the range above — the PCs "
            "cover 2014-10 – 2024-12 only."
        )

        catalog = EA.regressor_catalog(df_reg)
        preset = st.radio(
            "Regressor preset", list(catalog) + ["Custom"], index=0,
            help="Start from a group, then fine-tune the exact columns below. "
                 "The top preset is a compact mixed model for asking whether the "
                 "fund's active return comes from macro/factor exposure or the "
                 "manager's stock selection.",
        )
        if preset == "Custom":
            default_x = catalog["Fama-French factors"]
        else:
            default_x = catalog[preset]
        if preset == D.ALPHA_DRIVERS_PRESET_NAME:
            st.caption(
                "**Systematic side** — Mkt-RF, SMB (factors) and PC1, PC2 "
                "(rates/inflation & risk-on-off regimes). **Stock-picking side** — "
                "the portfolio's earnings yield & profitability (the value/quality "
                "tilt of the actual holdings). Read the **contribution decomposition** "
                "below: a high variance-share on the systematic block means the "
                "active return is mostly exposure; the rest is selection. "
                "Fundamentals restrict the sample to 2018–2022."
            )
        all_regressors = sorted({c for cols in catalog.values() for c in cols})
        x_cols = st.multiselect(
            "Regressors", all_regressors, default=default_x,
            format_func=D.label,
        )

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

    with results:
        if is_kalman:
            render_kalman(df_reg, target, x_cols)
        else:
            render_regression(df_reg, target, model, alpha, l1_ratio, standardize,
                              coef_scale, window, x_cols)


# =========================================================================
# TAB 2 — Data explorer
# =========================================================================
with tab_explore:
    st.subheader("Explore variables")

    drange = st.slider("Date range", min_value=DMIN, max_value=DMAX,
                       value=(DMIN, DMAX), format="YYYY-MM")
    mask = (DATES >= drange[0]) & (DATES <= drange[1])
    sub = df[mask]
    st.caption(f"{len(sub)} months selected.")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("##### Distribution")
        numeric = [c for c in df.columns
                   if c not in ("month", "date") and pd.api.types.is_numeric_dtype(df[c])]
        var = st.selectbox("Variable", numeric,
                           index=numeric.index("fund_ret") if "fund_ret" in numeric else 0,
                           format_func=D.label)
        series = sub[var].dropna()
        if series.empty:
            st.info("No data for this variable in the selected range.")
        else:
            fig = px.histogram(series, nbins=30, marginal="box")
            fig.update_layout(height=320, showlegend=False,
                              margin=dict(t=10, b=0, l=0, r=0),
                              xaxis_title=D.label(var))
            st.plotly_chart(fig, width="stretch")
            s = series.describe()
            st.caption(
                f"n={int(s['count'])} · mean={s['mean']:.4g} · sd={s['std']:.4g} "
                f"· min={s['min']:.4g} · median={series.median():.4g} · max={s['max']:.4g}"
            )

    with right:
        st.markdown("##### Returns over time")
        ret_opts = [c for c in D.BENCH_RETS if c in df.columns]
        picks = st.multiselect("Series", ret_opts,
                               default=[c for c in ("fund_ret", "bench_avg_ret") if c in ret_opts],
                               format_func=D.label)
        cumulative = st.toggle("Cumulative growth of $1", value=False)
        if picks:
            plot = sub[["date", *picks]].copy()
            if cumulative:
                for c in picks:
                    plot[c] = (1 + plot[c].fillna(0)).cumprod()
            long = plot.melt("date", var_name="Series", value_name="Return")
            long["Series"] = long["Series"].map(D.label)
            fig = px.line(long, x="date", y="Return", color="Series")
            fig.update_layout(height=320, margin=dict(t=10, b=0, l=0, r=0),
                              legend_title=None,
                              yaxis_tickformat="" if cumulative else ".0%")
            if not cumulative:
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig, width="stretch")

    st.divider()
    with st.expander("Correlation heatmap"):
        catalog = D.regressor_catalog(df)
        default_corr = catalog["Fama-French factors"] + ["fund_ret"]
        corr_cols = st.multiselect(
            "Variables", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                          and c not in ("month", "date")],
            default=[c for c in default_corr if c in df.columns],
            format_func=D.label,
        )
        if len(corr_cols) >= 2:
            corr = sub[corr_cols].corr()
            fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            aspect="auto", text_auto=".2f")
            fig.update_xaxes(tickvals=list(range(len(corr_cols))),
                             ticktext=[D.label(c) for c in corr_cols])
            fig.update_yaxes(tickvals=list(range(len(corr_cols))),
                             ticktext=[D.label(c) for c in corr_cols])
            fig.update_layout(height=480, margin=dict(t=10, b=0, l=0, r=0))
            st.plotly_chart(fig, width="stretch")


# =========================================================================
# Macro & regime — regime buckets, then the PCA explorer + variable dictionary
# =========================================================================
with tab_regime:
    regime_ticker = ticker_selector("regime")
    if regime_ticker is not None:
        regime_frame = resolve_frame(regime_ticker)
        if regime_frame is not None:
            render_regime(regime_frame)

    st.subheader("Principal components")
    st.caption(
        "The PCs are orthogonal combinations of the 23 macro / market features, "
        "ordered by how much variance they capture. They're the stable, "
        "non-collinear way to use the macro block in the rolling regression. "
        "Names below are interpretations of each component's loadings; the sign of "
        "a PC is arbitrary, so read “high/low” relative to the loadings shown."
    )

    pcs = D.pc_cols(df)
    sel = st.selectbox("Select a principal component", pcs, format_func=D.label)
    evr = D.pc_explained_var(sel)
    st.markdown(f"### {D.pc_name(sel)}  ·  `{sel}`")
    st.caption(f"Explains **{evr * 100:.1f}%** of the macro/market variance.")
    st.write(D.pc_desc(sel))

    loads = D.pc_loadings(sel).sort_values()
    fig = px.bar(
        x=loads.values, y=[D.label(f) for f in loads.index], orientation="h",
        labels={"x": "Loading (PC coefficient)", "y": ""},
    )
    fig.update_traces(marker_color=["#d62728" if v < 0 else "#1f6feb"
                                    for v in loads.values])
    fig.update_layout(height=max(320, 22 * len(loads) + 60),
                      margin=dict(t=10, b=0, l=0, r=0))
    fig.add_vline(x=0, line_color="gray")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Blue = positive loading, red = negative. A variable's loading is its weight "
        "in this component; larger |loading| = more influence. Squared loadings sum "
        "to 1 across all components."
    )
    with st.expander("Loadings table (sorted by magnitude)"):
        s = D.pc_loadings(sel)
        tbl = pd.DataFrame({"variable": [D.label(i) for i in s.index],
                            "loading": s.values}) \
            .sort_values("loading", key=lambda c: c.abs(), ascending=False)
        st.dataframe(tbl.style.format({"loading": "{:+.3f}"}),
                     width="stretch", hide_index=True)

    st.divider()
    st.subheader("Glossary — rolling-regression variables")
    st.caption(
        "Every variable selectable in the rolling regression. The portfolio "
        "fundamentals and sector/country weights are large families summarized at "
        "the group level — see `engineered_data/README.md` for column-level detail."
    )
    st.dataframe(D.glossary_dataframe(df), width="stretch", hide_index=True)
