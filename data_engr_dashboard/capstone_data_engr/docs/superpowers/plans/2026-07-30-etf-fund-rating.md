# 5-Star Fund Rating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score any fund 0–5 stars on risk-adjusted absolute performance plus holdings diversification, and surface it in a new dashboard tab.

**Architecture:** A pure scoring module, `dashboard/fund_rating.py`, holds every threshold and every calculation — no I/O, no Streamlit, so it is trivially testable. Two loader functions added to `dashboard/etf_analytics.py` supply its inputs: monthly returns + `RF` + daily closes from `etf_prices`, and diversification inputs from the engineered layer. `app.py` wires them into a ⭐ Rating tab. Returns come from yfinance for **every** ticker so all stars use identical methodology.

**Tech Stack:** Python 3.13, pandas 2.3, numpy, Streamlit 1.58, pytest 8, `uv` for running everything.

**Spec:** `docs/superpowers/specs/2026-07-30-etf-fund-rating-design.md`

---

## Critical context for the implementer

Read this before Task 1.

1. **`dashboard/` must not import `capstone_data`.** Read processed CSVs through `data.py`'s `data_dir()`.
2. **Intra-dashboard imports are bare.** Write `import fund_rating as FR`, not `from dashboard import fund_rating`. Precedent: `dashboard/app.py:25-27`.
3. **`fund_rating.py` must stay pure.** No file reads, no network, no Streamlit import. Every function takes series/dataclasses and returns numbers. This is what makes the thresholds testable at their exact boundaries. All I/O belongs in `etf_analytics.py`.
4. **Import direction is one-way.** `etf_analytics` imports `fund_rating`; never the reverse. Reversing it creates a cycle.
5. **Use the ungated `etf_prices.load()`, not `load_etf`/`validate_etf`.** FUND is a `MUTUALFUND` and `validate` rejects it by design. The rating must work for it.
6. **Do not source rating returns from `combined_monthly.csv`.** Its FUND `fund_ret` is exactly `0.0375` for all six months 2025-01 → 2025-06 — a placeholder. The rating deliberately uses yfinance for every ticker. (`combined_monthly` remains correct-and-authoritative for the factor/regime tabs; that bug is separate follow-up work.)
7. **Reuse `analytics.py` for the monthly metrics.** `A.sharpe(r, rf)`, `A.cagr(r)`, `A.ann_vol(r)`, `A.max_drawdown(r)` already annualize with `PPY = 12`. Do not reimplement them. Note `A.sharpe` returns `NaN` when the excess-return std is 0.
8. **`uv.lock` gotcha.** Local `uv` is 0.7.3, older than whatever wrote `uv.lock` (`revision = 3`). Running `uv` rewrites the lock down to `revision = 2`. **Never commit that churn** — run `git checkout -- uv.lock` if it appears in `git status`.

**Commit rules:** stage only files you changed — never `git add -A`. The `.idea/` files stay unstaged. Never put AI/Claude/Anthropic attribution in a commit message.

**Run everything with `uv`:** `uv run pytest`, `uv run python`.

## Source data (verified)

| File | Columns used | Notes |
|---|---|---|
| `position_values_monthly.csv` | `ticker`, `month`, `value_usd` | FUND positions, 2018-04 → 2022-12, 3595 rows. Top-10 weight = share of `value_usd`. |
| `portfolio_sector_country_monthly.csv` | `month`, `sect_wt_*` | FUND sector weights, 2018-04 → 2022-12. |
| `etf_holdings.csv` | — | **Does not exist.** iShares is gated. Every ETF therefore takes the performance-only fallback. |

Verified FUND values at 2022-12: 64 positions, top-10 = 47.9%, 10 sectors, normalized entropy = 0.889.

## File structure

| File | Status | Responsibility |
|---|---|---|
| `dashboard/fund_rating.py` | Create | Pure scoring: constants, `linear_score`, window metrics, composites, star mapping, `rate()` |
| `tests/test_fund_rating.py` | Create | Deterministic no-network tests |
| `dashboard/etf_analytics.py` | Modify | `rating_returns`, `diversification_inputs` |
| `dashboard/app.py` | Modify | ⭐ Rating tab |
| `dashboard/README.md` | Modify | Document the module |
| `docs/superpowers/etf-replatform-STATUS.md` | Modify | Mark sub-project #4 done |

Expected final size of `fund_rating.py`: ~220 lines.

---

### Task 1: Thresholds and the piecewise-linear scorer

Everything else composes from `linear_score`, so it is built and boundary-tested first. The inverted scales (drawdown, concentration) are where this usually goes wrong: `lo` is the value scoring 0 and `hi` the value scoring 100, and for those metrics `hi < lo`.

**Files:**
- Create: `dashboard/fund_rating.py`
- Create: `tests/test_fund_rating.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fund_rating.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fund_rating.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fund_rating'`

- [ ] **Step 3: Create the module**

Create `dashboard/fund_rating.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fund_rating.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/fund_rating.py tests/test_fund_rating.py
git commit -m "feat(dashboard): add rating thresholds and piecewise-linear scorer"
```

---

### Task 2: Window metrics (the hybrid frequency)

The one place two return frequencies meet. Sharpe/return/vol come from monthly data (aligned with the monthly `RF`); drawdown comes from daily closes so intra-month troughs are visible.

**Files:**
- Modify: `dashboard/fund_rating.py`
- Test: `tests/test_fund_rating.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fund_rating.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fund_rating.py -v`
Expected: FAIL with `AttributeError: module 'fund_rating' has no attribute 'window_metrics'`

- [ ] **Step 3: Implement window metrics and window scoring**

Append to `dashboard/fund_rating.py`:

```python
# --- per-window metrics ----------------------------------------------------

@dataclass(frozen=True)
class WindowMetrics:
    """The four numbers a window is judged on. `ann_vol` is shown, not scored."""
    sharpe: float
    ann_return: float
    max_drawdown: float
    ann_vol: float
    n_months: int


def _daily_max_drawdown(close: pd.Series) -> float:
    """Worst peak-to-trough on a daily close path."""
    if close is None or len(close) < 2:
        return float("nan")
    return float((close / close.cummax() - 1.0).min())


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
        sharpe=A.sharpe(monthly_ret, rf),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fund_rating.py -v`
Expected: PASS, 13 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/fund_rating.py tests/test_fund_rating.py
git commit -m "feat(dashboard): add window metrics with hybrid-frequency drawdown"
```

---

### Task 3: Composites, star mapping, and `rate()`

**Files:**
- Modify: `dashboard/fund_rating.py`
- Test: `tests/test_fund_rating.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fund_rating.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fund_rating.py -v`
Expected: FAIL with `AttributeError: module 'fund_rating' has no attribute 'performance_composite'`

- [ ] **Step 3: Implement the composites, stars, and `rate`**

Append to `dashboard/fund_rating.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fund_rating.py -v`
Expected: PASS, 27 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/fund_rating.py tests/test_fund_rating.py
git commit -m "feat(dashboard): add rating composites, star mapping, and rate()"
```

---

### Task 4: Loaders in `etf_analytics.py`

The I/O half. Kept out of `fund_rating` so the scoring stays pure.

**Files:**
- Modify: `dashboard/etf_analytics.py`
- Test: `tests/test_etf_analytics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_analytics.py`:

```python
def test_rating_returns_gives_monthly_with_rf_and_daily(monkeypatch):
    monthly = _monthly([0.01] * 24)
    daily = pd.Series(range(1, 101), dtype=float,
                      index=pd.date_range("2020-01-01", periods=100, freq="D"))

    class _P:
        ret_monthly = monthly
        close_daily = daily
        is_stale = False
        as_of = daily.index[-1]

    monkeypatch.setattr(EA.etf_prices, "load", lambda t, start=None: _P())
    m, d = EA.rating_returns("EFA")
    assert list(m.columns) == ["month", "fund_ret", "RF"]
    assert len(m) == 24
    assert m["RF"].notna().all()
    assert d is daily


def test_rating_returns_uses_the_ungated_load(monkeypatch):
    """FUND is a MUTUALFUND; validate_etf would reject it."""
    called = []

    class _P:
        ret_monthly = _monthly([0.01] * 12)
        close_daily = pd.Series([1.0, 2.0],
                                index=pd.date_range("2020-01-01", periods=2))
        is_stale = False
        as_of = None

    monkeypatch.setattr(EA.etf_prices, "load",
                        lambda t, start=None: called.append(t) or _P())
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: pytest.fail("must not use load_etf"))
    EA.rating_returns("FUND")
    assert called == ["FUND"]


def test_diversification_inputs_fund_matches_verified_values():
    d = EA.diversification_inputs("FUND")
    assert d is not None
    assert d.as_of == "2022-12"
    assert d.n_holdings == 64
    assert d.top10_weight == pytest.approx(0.479, abs=0.005)
    assert d.sector_entropy == pytest.approx(0.889, abs=0.005)


def test_diversification_inputs_is_none_without_holdings():
    """No etf_holdings.csv exists, so every ETF takes the fallback."""
    assert EA.diversification_inputs("EFA") is None


def test_sector_entropy_is_zero_for_a_single_sector():
    assert EA.sector_entropy(pd.Series([1.0])) == pytest.approx(0.0)


def test_sector_entropy_is_one_for_perfectly_even_weights():
    assert EA.sector_entropy(pd.Series([0.25] * 4)) == pytest.approx(1.0)


def test_sector_entropy_ignores_zero_and_negative_weights():
    even = EA.sector_entropy(pd.Series([0.5, 0.5]))
    padded = EA.sector_entropy(pd.Series([0.5, 0.5, 0.0, -0.1]))
    assert padded == pytest.approx(even)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: FAIL with `AttributeError: module 'etf_analytics' has no attribute 'rating_returns'`

- [ ] **Step 3: Implement the loaders**

Add the import at the top of `dashboard/etf_analytics.py`, after `import etf_prices`:

```python
import fund_rating
```

Add `import numpy as np` after `import pandas as pd` in the same file.

Then append to `dashboard/etf_analytics.py`:

```python
# --- rating inputs ---------------------------------------------------------

def rating_returns(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """``(monthly, daily_close)`` for the rating (network).

    Returns come from yfinance for **every** ticker, FUND included, so all
    stars use identical methodology. Deliberately not sourced from
    ``combined_monthly.csv``: its FUND ``fund_ret`` is a constant placeholder
    for 2025-01 - 2025-06, which would inflate the trailing-year score.

    Uses the ungated :func:`etf_prices.load`, not ``load_etf``/``validate``,
    because FUND is a MUTUALFUND that the ETF gate rejects by design.
    """
    ticker = ticker.upper()
    prices = etf_prices.load(ticker)
    ret = prices.ret_monthly
    monthly = pd.DataFrame({
        "month": pd.PeriodIndex(ret.index, freq="M").astype(str),
        "fund_ret": ret.to_numpy(dtype=float),
    })
    rf = load_market_frame()[["month", "RF"]]
    return monthly.merge(rf, on="month", how="inner"), prices.close_daily


def sector_entropy(weights: pd.Series) -> float:
    """Shannon entropy of sector weights, normalized to [0, 1].

    Zero/negative weights are dropped and the rest renormalized. A single
    sector returns 0.0 rather than dividing by ``ln(1) == 0`` -- one sector is
    maximal concentration, which is exactly what a 0 means here.
    """
    w = weights.dropna()
    w = w[w > 0]
    if len(w) <= 1:
        return 0.0
    w = w / w.sum()
    return float(-(w * np.log(w)).sum() / np.log(len(w)))


def diversification_inputs(ticker: str):
    """Holdings-derived rating inputs, or ``None`` when unavailable.

    FUND resolves from the engineered layer (position values + sector
    weights). Other tickers need ``etf_holdings.csv``, which does not exist
    while the iShares endpoint is gated -- so they return None and the rating
    redistributes that weight to performance.
    """
    if not is_capstone(ticker):
        path = D.data_dir() / "etf_holdings.csv"
        if not path.exists():
            return None
        holdings = pd.read_csv(path)
        rows = holdings[holdings["etf_ticker"].str.upper() == ticker.upper()]
        if rows.empty:
            return None
        w = rows["weight"].dropna()
        w = w[w > 0] / w[w > 0].sum()
        sectors = rows.groupby("sector")["weight"].sum()
        return fund_rating.DiversificationInputs(
            top10_weight=float(w.nlargest(10).sum()),
            sector_entropy=sector_entropy(sectors),
            n_holdings=int(len(rows)),
            as_of=str(rows["as_of_date"].iloc[0]),
        )

    d = D.data_dir()
    pos = pd.read_csv(d / "position_values_monthly.csv")
    sect = pd.read_csv(d / "portfolio_sector_country_monthly.csv")
    months = set(pos["month"]) & set(sect["month"])
    if not months:
        return None
    as_of = max(months)

    p = pos[pos["month"] == as_of]
    weights = p["value_usd"] / p["value_usd"].sum()
    sect_cols = [c for c in sect.columns if c.startswith("sect_wt_")]
    row = sect[sect["month"] == as_of][sect_cols].iloc[0]

    return fund_rating.DiversificationInputs(
        top10_weight=float(weights.nlargest(10).sum()),
        sector_entropy=sector_entropy(row),
        n_holdings=int(len(p)),
        as_of=as_of,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: PASS, 32 passed.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, 136 passed (102 pre-existing + 27 fund_rating + 7 etf_analytics).

If `uv.lock` shows as modified, run `git checkout -- uv.lock`.

- [ ] **Step 6: Commit**

```bash
git add dashboard/etf_analytics.py tests/test_etf_analytics.py
git commit -m "feat(dashboard): add rating return and diversification loaders"
```

---

### Task 5: The ⭐ Rating tab

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Import the module**

In `dashboard/app.py`, after `import etf_prices` (added in sub-project #3), add:

```python
import fund_rating as FR
```

- [ ] **Step 2: Add the tab to the tab list**

Replace:

```python
(tab_overview, tab_risk, tab_bench, tab_reg, tab_pos, tab_explore,
 tab_regime) = st.tabs([
    "📊 Overview", "🛡️ Risk & drawdown", "🎯 Benchmark & active",
    "📈 Factor exposures", "🧭 Positioning", "🔎 Data explorer",
    "🌐 Macro & regime"])
```

with:

```python
(tab_overview, tab_risk, tab_bench, tab_reg, tab_pos, tab_explore,
 tab_regime, tab_rating) = st.tabs([
    "📊 Overview", "🛡️ Risk & drawdown", "🎯 Benchmark & active",
    "📈 Factor exposures", "🧭 Positioning", "🔎 Data explorer",
    "🌐 Macro & regime", "⭐ Rating"])
```

- [ ] **Step 3: Add the renderer**

Add this function immediately before the `with tab_overview:` line:

```python
def render_rating(ticker: str):
    """0-5 star rating for `ticker`, with its components shown."""
    try:
        monthly, daily = EA.rating_returns(ticker)
    except etf_prices.EtfDataError as exc:
        st.error(str(exc))
        return

    is_stale, as_of = EA.price_status(ticker)
    if is_stale:
        st.warning(
            f"Yahoo is unreachable — rating computed from cached prices "
            f"through {as_of:%Y-%m-%d}. These are not current."
        )

    div = EA.diversification_inputs(ticker)
    rating = FR.rate(monthly.set_index("month")["fund_ret"],
                     rf=monthly.set_index("month")["RF"],
                     daily_close=daily, div_inputs=div)

    if rating.stars is None:
        st.info(f"Not rated — {rating.not_rated_reason}.")
        return

    left, right = st.columns([1, 2])
    with left:
        st.metric("Rating", "★" * int(rating.stars)
                  + ("½" if rating.stars % 1 else ""),
                  f"{rating.stars} / 5.0")
        st.caption(f"Composite score {rating.final:.1f} / 100")
    with right:
        st.metric("Performance", f"{rating.performance:.1f}")
        if rating.performance_only:
            st.caption(
                "Rated on performance only — no holdings data for this fund, "
                "so the diversification weight is redistributed."
            )
        else:
            st.metric("Diversification", f"{rating.diversification:.1f}")
            st.caption(
                f"Diversification from holdings as of **{rating.as_of}** "
                f"({div.n_holdings} positions, top-10 {div.top10_weight:.1%}, "
                f"sector entropy {div.sector_entropy:.2f}). Holdings are a "
                "point-in-time snapshot and may lag the return window."
            )

    st.divider()
    st.subheader("By window")
    rows = []
    for name, m in rating.windows.items():
        rows.append({
            "Window": name,
            "Score": round(rating.window_scores[name], 1),
            "Sharpe": round(m.sharpe, 2),
            "Ann. return": f"{m.ann_return:.2%}",
            "Max drawdown": f"{m.max_drawdown:.2%}",
            "Ann. vol": f"{m.ann_vol:.2%}",
            "Months": m.n_months,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Windows are blended 0.20 · 1y + 0.40 · 3y + 0.40 · 5y; missing windows "
        "redistribute their weight. Within a window: Sharpe 55%, annualized "
        "return 20%, max drawdown 25% (volatility is shown, not scored). "
        "Drawdown uses daily prices; Sharpe and return use monthly."
    )
```

- [ ] **Step 4: Wire the tab**

Add at the end of `dashboard/app.py`:

```python
with tab_rating:
    rating_ticker = ticker_selector("rating")
    if rating_ticker is not None:
        render_rating(rating_ticker)
```

- [ ] **Step 5: Verify the app parses and the suite passes**

Run: `uv run python -c "import ast; ast.parse(open('dashboard/app.py').read()); print('app.py parses')"`
Expected: `app.py parses`

Run: `uv run pytest -q`
Expected: PASS, 136 passed.

- [ ] **Step 6: Smoke-test the real rating path**

Run:

```bash
uv run python -c "
import sys; sys.path.insert(0,'dashboard')
import etf_analytics as EA, fund_rating as FR
for t in ['FUND','EFA','SCZ','VSS']:
    m,d = EA.rating_returns(t)
    r = FR.rate(m.set_index('month')['fund_ret'], m.set_index('month')['RF'], d,
                EA.diversification_inputs(t))
    print(f'{t:6s} perf={r.performance:5.1f} div={r.diversification} '
          f'final={r.final:5.1f} -> {r.stars} stars')
"
```

Expected (approximately — live prices move, so stars may shift by a half):

```
FUND  perf= 15.5 div=63.7 final= 27.6 -> 1.5 stars
EFA    perf= 61.4 div=None final= 61.4 -> 3.0 stars
SCZ    perf= 52.8 div=None final= 52.8 -> 2.5 stars
VSS    perf= 56.5 div=None final= 56.5 -> 3.0 stars
```

If `uv.lock` shows as modified, run `git checkout -- uv.lock`.

- [ ] **Step 7: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): add 5-star rating tab"
```

---

### Task 6: Documentation

**Files:**
- Modify: `dashboard/README.md`
- Modify: `docs/superpowers/etf-replatform-STATUS.md`

- [ ] **Step 1: Document the module in the dashboard README**

Append a section to `dashboard/README.md`:

```markdown
## Fund rating (`fund_rating.py`)

0–5 stars from risk-adjusted absolute performance plus holdings diversification.

- `rate(monthly_ret, rf, daily_close, div_inputs)` -> `Rating(stars, final,
  performance, diversification, windows, …)`. `stars is None` means **not rated**
  (under 12 months of history) — which is not the same as zero stars.
- Pure: no I/O, no network, no Streamlit. Inputs come from
  `etf_analytics.rating_returns()` and `etf_analytics.diversification_inputs()`.

**Scoring** (absolute thresholds, not peer-relative). Windows blend
0.20 · 1y + 0.40 · 3y + 0.40 · 5y, with missing windows redistributing their
weight. Within a window: Sharpe 55% (0 at ≤0.0, 100 at ≥1.5), annualized return
20% (0 at ≤0%, 100 at ≥15%), max drawdown 25% (0 at ≤−45%, 100 at ≥−5%).
Volatility is shown but not scored. Diversification is top-10 concentration 50%
(0 at ≥60%, 100 at ≤20%) plus normalized sector entropy 50% (0 at ≤0.50, 100 at
≥0.90). `final = 0.75·performance + 0.25·diversification`, or performance alone
when holdings are unavailable.

**Two deliberate choices worth knowing:**

- **Returns come from yfinance for every ticker, FUND included**, so all stars
  use identical methodology. `combined_monthly.csv` is *not* used here: its FUND
  `fund_ret` is a constant `0.0375` placeholder for 2025-01 – 2025-06, which
  would inflate the trailing-year score. That bug still affects the factor/regime
  tabs and is separate follow-up work.
- **Drawdown uses daily prices, Sharpe and return use monthly.** Monthly returns
  hide intra-month troughs and would systematically flatter a fund that crashed
  and recovered inside a month.

No ETF has holdings data while the iShares endpoint is gated, so every ETF is
currently rated on performance only. FUND's diversification comes from the
engineered layer and is a 2022-12 snapshot — the tab labels the as-of date.
```

- [ ] **Step 2: Mark sub-project #4 done in the STATUS doc**

In `docs/superpowers/etf-replatform-STATUS.md`:

- Change the sub-project #4 row of the status table from `⬜ not started (design locked in roadmap)` to `✅ **DONE**`.
- Update **Last updated** to `2026-07-30`.
- In "How to resume", change the recommendation to **#5 (dashboard re-platform)** — the last remaining sub-project.
- Add a short note under the table recording the two open data issues: the
  `combined_monthly.csv` FUND placeholder months (2025-01 – 2025-06), and that
  no ETF holdings exist until the iShares gate is solved, so every ETF is rated
  on performance only.

- [ ] **Step 3: Run the full suite one final time**

Run: `uv run pytest -q`
Expected: PASS, 136 passed.

- [ ] **Step 4: Confirm the working tree is clean**

Run: `git status --short`
Expected: only the pre-existing `.idea/` entries. If `uv.lock` appears, run `git checkout -- uv.lock`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/README.md docs/superpowers/etf-replatform-STATUS.md
git commit -m "docs: document fund_rating module and mark sub-project #4 done"
```

---

## Done when

- `uv run pytest -q` reports 136 passed.
- `dashboard/fund_rating.py` contains no `import streamlit`, no file reads, and no network calls (verify: `grep -nE "streamlit|read_csv|requests|yfinance" dashboard/fund_rating.py` returns nothing).
- The smoke test in Task 5 Step 6 rates FUND at roughly 1.5 stars and EFA/SCZ/VSS between 2.5 and 3.0.
- The ⭐ Rating tab renders stars, both composites, and the per-window table; a fund with no holdings shows the "rated on performance only" note, and FUND shows its diversification as-of date.
- No file in `dashboard/` imports `capstone_data` (verify: `grep -rn "^import capstone_data\|^from capstone_data" dashboard/` returns nothing).
- `git status --short` shows only the `.idea/` entries that were already there.
