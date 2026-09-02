"""0-5 star fund rating: risk-adjusted performance plus holdings diversification.

Pure by design -- every function takes series or dataclasses and returns
numbers. No file reads, no network, no Streamlit, so each threshold can be
tested at its exact boundary. All I/O lives in :mod:`etf_analytics`.

The scoring is *absolute*, not peer-relative: a fund is measured against fixed
thresholds, so two funds' stars are directly comparable. Every threshold below
is locked by the roadmap; change them only deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import analytics as A

# --- locked thresholds -----------------------------------------------------
# Each range is (value scoring 0, value scoring 100). For drawdown and
# concentration the second element is *smaller* -- less drawdown and less
# concentration are better. `linear_score` handles that inversion.

SHARPE_RANGE = (0.0, 1.5)
RETURN_RANGE = (0.0, 0.15)
DRAWDOWN_RANGE = (-0.45, -0.05)
TOP10_RANGE = (0.60, 0.20)
ENTROPY_RANGE = (0.50, 0.90)

# Volatility is displayed but deliberately unweighted -- Sharpe already prices
# it in, so scoring it again would double-count risk.
METRIC_WEIGHTS = {"sharpe": 0.55, "ann_return": 0.20, "max_drawdown": 0.25}

# window -> (months required, weight)
WINDOWS = {"1y": (12, 0.20), "3y": (36, 0.40), "5y": (60, 0.40)}

PERFORMANCE_WEIGHT = 0.75
DIVERSIFICATION_WEIGHT = 0.25

MIN_MONTHS = 12         # below this a fund is not rated at all


def linear_score(x: float, lo: float, hi: float) -> float:
    """Piecewise-linear 0-100 score, clamped.

    `lo` is the value scoring 0 and `hi` the value scoring 100. When
    ``hi < lo`` the scale is inverted (lower is better), which is how drawdown
    and concentration are expressed. NaN propagates, so an undefined metric
    never masquerades as a zero.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    if hi == lo:
        return float("nan")
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 100.0)


# --- per-window metrics ----------------------------------------------------

@dataclass(frozen=True)
class WindowMetrics:
    """The four numbers a window is judged on. `ann_vol` is shown, not scored."""
    sharpe: float
    ann_return: float
    max_drawdown: float
    ann_vol: float
    n_months: int


# Below this, a series' "volatility" is floating-point residue rather than real
# dispersion. `analytics.sharpe` only guards against an exact `sd > 0`, so a
# constant series slips through with sd ~1e-18 and yields a Sharpe of ~1e16 --
# which would clamp to a perfect 100 on the heaviest-weighted metric.
_MIN_STD = 1e-12


def _daily_max_drawdown(close: pd.Series) -> float:
    """Worst peak-to-trough on a daily close path."""
    if close is None or len(close) < 2:
        return float("nan")
    return float((close / close.cummax() - 1.0).min())


def _sharpe_or_nan(monthly_ret: pd.Series, rf: pd.Series | float) -> float:
    """Sharpe, or NaN when excess returns have no meaningful dispersion."""
    excess = monthly_ret - rf
    if len(excess) < 2 or not np.isfinite(excess.std(ddof=1)):
        return float("nan")
    if excess.std(ddof=1) < _MIN_STD:
        return float("nan")
    value = A.sharpe(monthly_ret, rf)
    return value if np.isfinite(value) else float("nan")


def window_metrics(monthly_ret: pd.Series, rf: pd.Series | float = 0.0,
                   daily_close: pd.Series | None = None) -> WindowMetrics:
    """Metrics for one trailing window.

    Sharpe, annualized return and volatility come from `monthly_ret` (via
    ``analytics``, which annualizes with 12 and sqrt(12)) so they align with the
    monthly RF series. Max drawdown prefers `daily_close`: monthly returns hide
    intra-month troughs and would systematically flatter a fund that crashed and
    recovered inside a month.
    """
    dd = (_daily_max_drawdown(daily_close) if daily_close is not None
          else A.max_drawdown(monthly_ret))
    return WindowMetrics(
        sharpe=_sharpe_or_nan(monthly_ret, rf),
        ann_return=A.cagr(monthly_ret),
        max_drawdown=dd,
        ann_vol=A.ann_vol(monthly_ret),
        n_months=len(monthly_ret),
    )


_METRIC_RANGES = {
    "sharpe": SHARPE_RANGE,
    "ann_return": RETURN_RANGE,
    "max_drawdown": DRAWDOWN_RANGE,
}


def score_window(m: WindowMetrics) -> float:
    """0-100 for one window: 55% Sharpe, 20% return, 25% drawdown.

    Metrics that are NaN (an undefined Sharpe on a zero-variance series, say)
    drop out and the surviving weights renormalize, so an undefined metric
    never scores as a bad one.
    """
    total, acc = 0.0, 0.0
    for name, weight in METRIC_WEIGHTS.items():
        score = linear_score(getattr(m, name), *_METRIC_RANGES[name])
        if not np.isnan(score):
            acc += weight * score
            total += weight
    return acc / total if total else float("nan")


# --- composites ------------------------------------------------------------

def performance_composite(window_scores: dict[str, float]) -> float:
    """Weighted blend across the windows that had enough history.

    Missing windows drop out and the surviving weights renormalize, so a fund
    with only three years of history is judged on what it has rather than
    penalized for the years it could not have.
    """
    total, acc = 0.0, 0.0
    for name, score in window_scores.items():
        if np.isnan(score):
            continue
        _, weight = WINDOWS[name]
        acc += weight * score
        total += weight
    return acc / total if total else float("nan")


@dataclass(frozen=True)
class DiversificationInputs:
    """Holdings-derived inputs. `as_of` is the snapshot month (YYYY-MM)."""
    top10_weight: float
    sector_entropy: float
    n_holdings: int
    as_of: str


def diversification_composite(d: DiversificationInputs) -> float:
    """50% top-10 concentration, 50% sector spread."""
    conc = linear_score(d.top10_weight, *TOP10_RANGE)
    spread = linear_score(d.sector_entropy, *ENTROPY_RANGE)
    parts = [p for p in (conc, spread) if not np.isnan(p)]
    return float(np.mean(parts)) if parts else float("nan")


def final_score(performance: float, diversification: float | None) -> float:
    """0.75/0.25 blend; performance alone when diversification is unavailable.

    Redistributing rather than scoring a missing component as zero is what
    stops "we have no holdings data" from reading as "this fund is
    concentrated".
    """
    if diversification is None or np.isnan(diversification):
        return performance
    return (PERFORMANCE_WEIGHT * performance
            + DIVERSIFICATION_WEIGHT * diversification)


def stars(final: float) -> float | None:
    """Linear half-star mapping with a 0.5 floor. None when unscoreable."""
    if final is None or np.isnan(final):
        return None
    return float(np.clip(round(final / 100.0 * 5.0 * 2.0) / 2.0, 0.5, 5.0))


# --- orchestration ---------------------------------------------------------

@dataclass(frozen=True)
class Rating:
    """A complete rating. `stars is None` means not rated, which is not 0 stars."""
    stars: float | None
    final: float
    performance: float
    diversification: float | None
    windows: dict[str, WindowMetrics] = field(default_factory=dict)
    window_scores: dict[str, float] = field(default_factory=dict)
    performance_only: bool = True
    as_of: str | None = None
    not_rated_reason: str = ""


def rate(monthly_ret: pd.Series, rf: pd.Series | float = 0.0,
         daily_close: pd.Series | None = None,
         div_inputs: DiversificationInputs | None = None) -> Rating:
    """Score a fund. Never raises on thin data -- returns an unrated Rating.

    `monthly_ret` and `rf` must share an index. Each window takes the trailing
    N months; a window qualifies only with its full observation count.
    """
    monthly_ret = monthly_ret.dropna()
    if len(monthly_ret) < MIN_MONTHS:
        return Rating(stars=None, final=float("nan"), performance=float("nan"),
                      diversification=None,
                      not_rated_reason=f"needs at least {MIN_MONTHS} months of "
                                       f"history, has {len(monthly_ret)}")

    windows: dict[str, WindowMetrics] = {}
    scores: dict[str, float] = {}
    for name, (months, _weight) in WINDOWS.items():
        if len(monthly_ret) < months:
            continue
        sub = monthly_ret.tail(months)
        sub_rf = rf.reindex(sub.index) if isinstance(rf, pd.Series) else rf
        daily = (daily_close[daily_close.index >= sub.index[0]]
                 if daily_close is not None else None)
        m = window_metrics(sub, sub_rf, daily)
        windows[name] = m
        scores[name] = score_window(m)

    performance = performance_composite(scores)
    div = diversification_composite(div_inputs) if div_inputs else None
    final = final_score(performance, div)
    return Rating(
        stars=stars(final),
        final=final,
        performance=performance,
        diversification=div,
        windows=windows,
        window_scores=scores,
        performance_only=div is None,
        as_of=div_inputs.as_of if div_inputs else None,
    )
