"""Deterministic checks for the 5-star fund rating (no network, no Streamlit)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import numpy as np
import pandas as pd
import pytest

import fund_rating as FR


def test_linear_score_endpoints_and_midpoint():
    assert FR.linear_score(0.0, 0.0, 1.5) == pytest.approx(0.0)
    assert FR.linear_score(1.5, 0.0, 1.5) == pytest.approx(100.0)
    assert FR.linear_score(0.75, 0.0, 1.5) == pytest.approx(50.0)


def test_linear_score_clamps_outside_range():
    assert FR.linear_score(-5.0, 0.0, 1.5) == pytest.approx(0.0)
    assert FR.linear_score(99.0, 0.0, 1.5) == pytest.approx(100.0)


def test_linear_score_inverted_scale_scores_in_the_right_direction():
    """Drawdown: -45% scores 0, -5% scores 100 (hi < lo)."""
    assert FR.linear_score(-0.45, -0.45, -0.05) == pytest.approx(0.0)
    assert FR.linear_score(-0.05, -0.45, -0.05) == pytest.approx(100.0)
    assert FR.linear_score(-0.25, -0.45, -0.05) == pytest.approx(50.0)
    # Worse than the floor and better than the ceiling both clamp.
    assert FR.linear_score(-0.90, -0.45, -0.05) == pytest.approx(0.0)
    assert FR.linear_score(0.0, -0.45, -0.05) == pytest.approx(100.0)


def test_linear_score_concentration_scale():
    """Top-10 concentration: 60% scores 0, 20% scores 100."""
    assert FR.linear_score(0.60, 0.60, 0.20) == pytest.approx(0.0)
    assert FR.linear_score(0.20, 0.60, 0.20) == pytest.approx(100.0)
    assert FR.linear_score(0.40, 0.60, 0.20) == pytest.approx(50.0)


def test_linear_score_nan_propagates():
    assert np.isnan(FR.linear_score(float("nan"), 0.0, 1.5))


def test_thresholds_match_the_locked_roadmap():
    assert FR.SHARPE_RANGE == (0.0, 1.5)
    assert FR.RETURN_RANGE == (0.0, 0.15)
    assert FR.DRAWDOWN_RANGE == (-0.45, -0.05)
    assert FR.TOP10_RANGE == (0.60, 0.20)
    assert FR.ENTROPY_RANGE == (0.50, 0.90)
    assert FR.METRIC_WEIGHTS == {"sharpe": 0.55, "ann_return": 0.20,
                                 "max_drawdown": 0.25}
    assert FR.WINDOWS == {"1y": (12, 0.20), "3y": (36, 0.40), "5y": (60, 0.40)}
    assert (FR.PERFORMANCE_WEIGHT, FR.DIVERSIFICATION_WEIGHT) == (0.75, 0.25)


def _monthly(values, start="2020-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


def _daily_from_path(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def _monthly_max_dd(r):
    """Reference drawdown from monthly returns, for the fallback assertion."""
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def test_window_metrics_uses_monthly_for_sharpe_and_return():
    r = _monthly([0.01] * 12)
    m = FR.window_metrics(r, rf=0.0)
    assert m.n_months == 12
    assert m.ann_return == pytest.approx((1.01 ** 12) - 1.0)
    assert np.isnan(m.sharpe)          # zero variance -> undefined, not 0
    assert m.ann_vol == pytest.approx(0.0)


def test_window_metrics_sharpe_is_positive_for_a_rising_series():
    r = _monthly([0.02, -0.01, 0.03, 0.01, 0.02, -0.005] * 2)
    m = FR.window_metrics(r, rf=0.0)
    assert m.sharpe > 0


def test_window_metrics_daily_drawdown_beats_monthly_for_intramonth_crash():
    """A trough inside a month is invisible monthly but visible daily."""
    # Monthly: flat then -2%. Daily: dives to -30% mid-window and recovers.
    r = _monthly([0.0, -0.02])
    path = ([100.0] * 15 + [70.0] + [100.0] * 20 + [98.0] * 25)
    daily = _daily_from_path(path)

    monthly_only = FR.window_metrics(r, rf=0.0)
    hybrid = FR.window_metrics(r, rf=0.0, daily_close=daily)

    assert hybrid.max_drawdown < monthly_only.max_drawdown
    assert hybrid.max_drawdown == pytest.approx(-0.30)


def test_window_metrics_falls_back_to_monthly_drawdown_without_daily():
    r = _monthly([0.10, -0.20, 0.05])
    m = FR.window_metrics(r, rf=0.0, daily_close=None)
    assert m.max_drawdown == pytest.approx(_monthly_max_dd(r))


def test_score_window_blends_the_three_metrics():
    m = FR.WindowMetrics(sharpe=1.5, ann_return=0.15, max_drawdown=-0.05,
                         ann_vol=0.10, n_months=12)
    assert FR.score_window(m) == pytest.approx(100.0)

    worst = FR.WindowMetrics(sharpe=0.0, ann_return=0.0, max_drawdown=-0.45,
                             ann_vol=0.10, n_months=12)
    assert FR.score_window(worst) == pytest.approx(0.0)


def test_score_window_skips_nan_metrics_and_renormalizes():
    """Undefined Sharpe must not drag the window to zero."""
    m = FR.WindowMetrics(sharpe=float("nan"), ann_return=0.15,
                         max_drawdown=-0.05, ann_vol=0.0, n_months=12)
    # Only return (0.20) and drawdown (0.25) survive; both score 100.
    assert FR.score_window(m) == pytest.approx(100.0)


def test_score_window_all_nan_is_nan():
    m = FR.WindowMetrics(sharpe=float("nan"), ann_return=float("nan"),
                         max_drawdown=float("nan"), ann_vol=float("nan"),
                         n_months=12)
    assert np.isnan(FR.score_window(m))


def test_performance_composite_blends_windows_by_weight():
    w = {"1y": 100.0, "3y": 50.0, "5y": 0.0}
    # 0.20*100 + 0.40*50 + 0.40*0 = 40
    assert FR.performance_composite(w) == pytest.approx(40.0)


def test_performance_composite_renormalizes_when_a_window_is_missing():
    """Only 1y and 3y available -> weights 0.20/0.40 renormalize to 1/3, 2/3."""
    w = {"1y": 90.0, "3y": 30.0}
    assert FR.performance_composite(w) == pytest.approx((0.20 * 90 + 0.40 * 30)
                                                        / 0.60)


def test_performance_composite_single_window():
    assert FR.performance_composite({"1y": 77.0}) == pytest.approx(77.0)


def test_performance_composite_empty_is_nan():
    assert np.isnan(FR.performance_composite({}))


def test_diversification_composite_blends_concentration_and_entropy():
    d = FR.DiversificationInputs(top10_weight=0.20, sector_entropy=0.90,
                                 n_holdings=100, as_of="2022-12")
    assert FR.diversification_composite(d) == pytest.approx(100.0)

    worst = FR.DiversificationInputs(top10_weight=0.60, sector_entropy=0.50,
                                     n_holdings=10, as_of="2022-12")
    assert FR.diversification_composite(worst) == pytest.approx(0.0)


def test_diversification_composite_matches_verified_fund_numbers():
    """Real FUND values at 2022-12: top-10 47.9%, entropy 0.889."""
    d = FR.DiversificationInputs(top10_weight=0.479, sector_entropy=0.889,
                                 n_holdings=64, as_of="2022-12")
    assert FR.diversification_composite(d) == pytest.approx(63.7, abs=0.2)


def test_final_score_blends_75_25():
    assert FR.final_score(80.0, 40.0) == pytest.approx(0.75 * 80 + 0.25 * 40)


def test_final_score_without_diversification_is_performance_alone():
    assert FR.final_score(62.0, None) == pytest.approx(62.0)


def test_stars_half_star_mapping_and_floor():
    assert FR.stars(100.0) == 5.0
    assert FR.stars(0.0) == 0.5           # floor: never below half a star
    assert FR.stars(50.0) == 2.5
    assert FR.stars(61.4) == 3.0
    assert FR.stars(27.6) == 1.5
    assert FR.stars(float("nan")) is None


def test_stars_never_exceeds_five():
    assert FR.stars(1000.0) == 5.0


def test_rate_end_to_end_performance_only():
    monthly = _monthly([0.01] * 60)
    r = FR.rate(monthly, rf=0.0, daily_close=None, div_inputs=None)
    assert r.performance_only is True
    assert r.diversification is None
    assert r.final == pytest.approx(r.performance)
    assert r.stars is not None
    assert set(r.windows) == {"1y", "3y", "5y"}


def test_rate_with_diversification_uses_both_composites():
    monthly = _monthly([0.01] * 60)
    d = FR.DiversificationInputs(top10_weight=0.479, sector_entropy=0.889,
                                 n_holdings=64, as_of="2022-12")
    r = FR.rate(monthly, rf=0.0, daily_close=None, div_inputs=d)
    assert r.performance_only is False
    assert r.diversification == pytest.approx(63.7, abs=0.2)
    assert r.as_of == "2022-12"


def test_rate_not_rated_under_twelve_months():
    r = FR.rate(_monthly([0.01] * 6), rf=0.0, daily_close=None, div_inputs=None)
    assert r.stars is None
    assert "12 months" in r.not_rated_reason


def test_rate_uses_only_windows_with_full_history():
    """40 months -> 1y and 3y qualify, 5y does not."""
    r = FR.rate(_monthly([0.01] * 40), rf=0.0, daily_close=None,
                div_inputs=None)
    assert set(r.windows) == {"1y", "3y"}
