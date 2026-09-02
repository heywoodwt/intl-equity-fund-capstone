# ETF Factor Analytics Recompute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Factor-exposures and Macro-&-regime tabs run their existing factor / rolling-regression / Kalman-TVP / regime machinery against any user-selected ETF, not just FUND.

**Architecture:** One new module, `dashboard/etf_analytics.py`, builds a fund-independent "market frame" (Fama-French factors ⋈ macro ⋈ PCA components ⋈ benchmark ETF returns, joined on `month`) once, then joins it per-ticker to monthly returns from `etf_prices` (sub-project #2). The selected ticker's returns land in a column still named `fund_ret`, so the analytics engines and every `app.py` call site work unchanged. FUND short-circuits to the existing combined frame to preserve the capstone's exact numbers and its holdings-derived regressors.

**Tech Stack:** Python 3.13, pandas 2.3, Streamlit 1.58, pytest 8, `uv` for running everything.

**Spec:** `docs/superpowers/specs/2026-07-30-etf-factor-analytics-design.md`

---

## Critical context for the implementer

Read this before Task 1.

1. **`dashboard/` must not import `capstone_data`.** The folder is liftable into a deploy repo on its own. Read processed CSVs through `data.py`'s `data_dir()`; never import the pipeline package.
2. **Intra-dashboard imports are bare.** Write `import data as D`, `import etf_prices`, not `from dashboard import ...`. Precedent: `dashboard/app.py:25` does `import rolling_regression as R`.
3. **Tests put `dashboard/` on `sys.path`.** Precedent: `tests/test_etf_prices.py:6`. Tests are deterministic and never touch the network. Reading a local processed CSV *is* allowed — it is deterministic and network-free.
4. **The engines already take column names.** `analytics.py`, `rolling_regression.py`, and `kalman_tvp.py` never hard-code `fund_ret`. Do not modify them. This plan does not touch them at all.
5. **Units.** Every return column is a **decimal**. `etf_prices.monthly_returns()` also emits decimals, so no conversion is needed. ⚠️ `sp500_ret` is in **percent** in both `macro_monthly.csv` and `combined_monthly.csv` — a pre-existing quirk. Pass it through unchanged; do not "fix" it.
6. **`uv.lock` gotcha.** Local `uv` is 0.7.3, older than whatever wrote `uv.lock` (`revision = 3`). Running `uv` rewrites the lock down to `revision = 2`. **Never commit that churn** — if `uv.lock` appears in `git status`, run `git checkout -- uv.lock`.

**Commit rules:** stage only files you changed — never `git add -A`. The `.idea/` files in the working tree are unrelated and must stay unstaged. Never put AI/Claude/Anthropic attribution in a commit message.

**Run everything with `uv`:** `uv run pytest`, `uv run python`.

## Source table schemas (verified)

| File | Key column(s) | Notes |
|---|---|---|
| `ff_factors_dev_ex_us.csv` | `date` (month-end) | `Mkt_RF, SMB, HML, RMW, CMA, RF, Mom`. **No `month` column** — derive it. 1990-07 → 2026-04. |
| `macro_monthly.csv` | `date` **and** `month` | 17 macro columns. Drop `date`, keep `month`. 2000-01 → 2026-05. |
| `pca_market_components.csv` | `month` | `PC1`…`PC23`. No `date` column. 2014-10 → 2025-06. |
| `benchmark_etf_returns_monthly.csv` | `month` | `efa_ret, scz_ret, vss_ret`. 2014-09 → 2026-06. |

Joined market frame: **432 rows, 1990-07 → 2026-06.**

## File structure

| File | Status | Responsibility |
|---|---|---|
| `dashboard/etf_analytics.py` | Create | Market frame, ETF returns adapter, derived columns, `load_analytics_frame`, catalog wrapper |
| `tests/test_etf_analytics.py` | Create | Deterministic no-network tests |
| `dashboard/app.py` | Modify | Ticker selector; wire the two analytics tabs; error/stale handling |
| `dashboard/README.md` | Modify | Document the module |
| `docs/superpowers/etf-replatform-STATUS.md` | Modify | Mark sub-project #3 done |

Expected final size of `etf_analytics.py`: ~200 lines.

---

### Task 1: Scaffolding — module skeleton and the market frame

**Files:**
- Create: `dashboard/etf_analytics.py`
- Create: `tests/test_etf_analytics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_etf_analytics.py`:

```python
"""Deterministic checks for the ETF analytics frame (no network, no Streamlit)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import pandas as pd
import pytest

import etf_analytics as EA


def test_market_frame_month_range_and_uniqueness():
    mf = EA.load_market_frame()
    assert mf["month"].is_unique
    assert mf["month"].min() == "1990-07"
    assert mf["month"].max() == "2026-06"
    assert len(mf) == 432


def test_market_frame_has_all_four_source_families():
    mf = EA.load_market_frame()
    for col in ("Mkt_RF", "SMB", "RF", "Mom"):        # Fama-French
        assert col in mf.columns
    for col in ("vix", "y10", "cpi_yoy", "sp500_ret"):  # macro
        assert col in mf.columns
    for col in ("PC1", "PC2"):                         # PCA
        assert col in mf.columns
    for col in ("efa_ret", "scz_ret", "vss_ret"):      # benchmark ETFs
        assert col in mf.columns


def test_market_frame_has_no_date_column():
    """`date` is derived per-frame by load_analytics_frame, not carried here."""
    assert "date" not in EA.load_market_frame().columns


def test_market_frame_coverage_matches_sources():
    mf = EA.load_market_frame()
    assert mf["Mkt_RF"].notna().sum() == 430
    assert mf["PC1"].notna().sum() == 129
    assert mf["efa_ret"].notna().sum() == 142
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'etf_analytics'`

- [ ] **Step 3: Create the module with the market frame**

Create `dashboard/etf_analytics.py`:

```python
"""Per-ETF analytics frames for the factor / regime tabs.

Sits between :mod:`etf_prices` (which fetches prices) and the analytics engines
(``analytics``, ``rolling_regression``, ``kalman_tvp``), which are already
column-generic. This module's job is to hand them a frame whose ``fund_ret``
column holds the *selected* ticker's monthly returns.

Self-contained by design: imports nothing from the pipeline package, so
``dashboard/`` stays liftable into a deploy repo on its own.

Two paths. FUND -- the capstone fund -- short-circuits to the existing
``combined_monthly`` frame so its numbers and its holdings-derived regressors
are preserved exactly. Any other ticker is joined onto a fund-independent
"market frame" that spans far more history than ``combined_monthly`` does.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

import data as D

# The capstone fund. It is a MUTUALFUND, so `etf_prices.validate` rejects it by
# design; it is served from the existing combined frame instead of yfinance.
CAPSTONE_TICKER = "FUND"


@lru_cache(maxsize=1)
def load_market_frame() -> pd.DataFrame:
    """Fund-independent monthly frame: factors, macro, PCs, benchmark returns.

    Joined on ``month`` (``YYYY-MM``) with an outer join, so each source keeps
    its own natural coverage: factors reach back to 1990-07, macro to 2000-01,
    benchmark ETFs to 2014-09, and the PCA components only span
    2014-10 - 2025-06. Callers must tolerate NaN outside a column's range --
    the engines already drop incomplete rows.

    Verified to reproduce every shared column of ``combined_monthly.csv``
    exactly over their overlap; see ``test_market_frame_matches_combined``.
    """
    d = D.data_dir()

    ff = pd.read_csv(d / "ff_factors_dev_ex_us.csv")
    # This is the only source keyed by a month-end date rather than `month`.
    ff["month"] = pd.PeriodIndex(pd.to_datetime(ff["date"]), freq="M").astype(str)
    ff = ff.drop(columns=["date"])

    macro = pd.read_csv(d / "macro_monthly.csv").drop(columns=["date"])
    pcs = pd.read_csv(d / "pca_market_components.csv")
    bench = pd.read_csv(d / "benchmark_etf_returns_monthly.csv")

    out = (ff.merge(macro, on="month", how="outer")
             .merge(pcs, on="month", how="outer")
             .merge(bench, on="month", how="outer"))
    return out.sort_values("month").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_analytics.py tests/test_etf_analytics.py
git commit -m "feat(dashboard): add fund-independent market frame for ETF analytics"
```

---

### Task 2: Regression guard — the market frame must match `combined_monthly`

This is the single most important test in the plan. It is what stops the ETF path and the FUND path from silently disagreeing about a factor or macro value. Write it as its own task so it is never skipped.

**Files:**
- Test: `tests/test_etf_analytics.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_etf_analytics.py`:

```python
def test_market_frame_matches_combined_monthly_exactly():
    """Every column the market frame shares with combined_monthly.csv must be
    identical over their overlap. If this fails, the ETF path and the FUND
    path have diverged and every cross-ticker comparison is suspect."""
    mf = EA.load_market_frame()
    combined = pd.read_csv(D_DIR / "combined_monthly.csv")

    shared = [c for c in mf.columns if c in combined.columns and c != "month"]
    assert len(shared) == 27, f"expected 27 shared columns, got {len(shared)}"

    joined = combined[["month"] + shared].merge(
        mf[["month"] + shared], on="month", suffixes=("_c", "_m"))
    assert len(joined) == 129

    mismatched = []
    for col in shared:
        diff = (joined[f"{col}_c"] - joined[f"{col}_m"]).abs().max()
        if pd.notna(diff) and diff >= 1e-9:
            mismatched.append((col, diff))
    assert mismatched == [], f"columns diverged: {mismatched}"
```

Add this import helper near the top of the test file, immediately after `import etf_analytics as EA`:

```python
import data as D

D_DIR = D.data_dir()
```

- [ ] **Step 2: Run the test to verify it passes**

This test guards existing behavior, so it should pass immediately. If it fails, **stop** — the market frame join is wrong and every later task inherits the error.

Run: `uv run pytest tests/test_etf_analytics.py::test_market_frame_matches_combined_monthly_exactly -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_etf_analytics.py
git commit -m "test(dashboard): guard market frame against combined_monthly drift"
```

---

### Task 3: ETF monthly returns adapter

Converts `etf_prices`' month-end `DatetimeIndex` series into a two-column frame keyed the way every processed table is keyed.

**Files:**
- Modify: `dashboard/etf_analytics.py`
- Test: `tests/test_etf_analytics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_analytics.py`:

```python
def _monthly(values, start="2020-01-31"):
    """Month-end-stamped monthly returns, the shape etf_prices returns."""
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


class _FakePrices:
    """Stand-in for etf_prices.EtfPrices carrying only what we consume."""

    def __init__(self, ret_monthly, is_stale=False, as_of=None):
        self.ret_monthly = ret_monthly
        self.is_stale = is_stale
        # Real EtfPrices stamps as_of from the daily close; for these tests the
        # last monthly point is close enough and keeps the fixtures short.
        self.as_of = as_of if as_of is not None else (
            ret_monthly.index[-1] if len(ret_monthly) else None)


def test_etf_monthly_returns_maps_month_end_to_month_key(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01, -0.02, 0.03])))
    out = EA.etf_monthly_returns("efa")
    assert list(out.columns) == ["month", "fund_ret"]
    assert list(out["month"]) == ["2020-01", "2020-02", "2020-03"]
    assert out["fund_ret"].tolist() == [0.01, -0.02, 0.03]


def test_etf_monthly_returns_uppercases_ticker(monkeypatch):
    seen = []
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: seen.append(t) or _FakePrices(_monthly([0.01])))
    EA.etf_monthly_returns("efa")
    assert seen == ["EFA"]


def test_etf_monthly_returns_on_empty_series(monkeypatch):
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(empty))
    out = EA.etf_monthly_returns("EFA")
    assert out.empty
    assert list(out.columns) == ["month", "fund_ret"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: FAIL with `AttributeError: module 'etf_analytics' has no attribute 'etf_prices'`

- [ ] **Step 3: Implement the adapter**

Add the import to `dashboard/etf_analytics.py`, immediately after `import data as D`:

```python
import etf_prices
```

Then append:

```python
# --- per-ticker returns ----------------------------------------------------

def etf_monthly_returns(ticker: str) -> pd.DataFrame:
    """Monthly total returns for `ticker` as ``(month, fund_ret)`` (network).

    ``etf_prices`` stamps monthly returns on a month-end ``DatetimeIndex``;
    every processed table here is keyed by a ``YYYY-MM`` string, so convert.
    The column is named ``fund_ret`` because the analytics engines and the
    ``app.py`` call sites already use that name -- see the module docstring.
    """
    ticker = ticker.upper()
    ret = etf_prices.load_etf(ticker).ret_monthly
    if ret.empty:
        return pd.DataFrame({"month": pd.Series(dtype=str),
                             "fund_ret": pd.Series(dtype=float)})
    return pd.DataFrame({
        "month": pd.PeriodIndex(ret.index, freq="M").astype(str),
        "fund_ret": ret.to_numpy(dtype=float),
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_analytics.py tests/test_etf_analytics.py
git commit -m "feat(dashboard): add ETF monthly returns adapter"
```

---

### Task 4: Derived return columns

Pure function, no I/O. The `skipna=False` requirement is the subtle part: pandas' default would silently average whichever benchmark legs exist and produce a two-ETF "blended benchmark".

**Files:**
- Modify: `dashboard/etf_analytics.py`
- Test: `tests/test_etf_analytics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_analytics.py`:

```python
def _frame(**cols):
    """Minimal frame with a `month` key plus whatever columns a test needs."""
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"month": [f"2020-{i + 1:02d}" for i in range(n)], **cols})


def test_derive_returns_computes_all_four_identities():
    df = _frame(fund_ret=[0.05], RF=[0.01],
                efa_ret=[0.02], scz_ret=[0.03], vss_ret=[0.04])
    out = EA.derive_returns(df)
    assert out["excess_ret"].iloc[0] == pytest.approx(0.04)
    assert out["bench_avg_ret"].iloc[0] == pytest.approx(0.03)
    assert out["alpha_vs_avg"].iloc[0] == pytest.approx(0.02)
    assert out["alpha_vs_efa"].iloc[0] == pytest.approx(0.03)


def test_derive_returns_bench_avg_is_nan_when_any_leg_missing():
    """skipna=False: a partial month must not yield a 2-ETF blended benchmark."""
    df = _frame(fund_ret=[0.05], RF=[0.01],
                efa_ret=[0.02], scz_ret=[float("nan")], vss_ret=[0.04])
    out = EA.derive_returns(df)
    assert pd.isna(out["bench_avg_ret"].iloc[0])
    assert pd.isna(out["alpha_vs_avg"].iloc[0])
    assert out["alpha_vs_efa"].iloc[0] == pytest.approx(0.03)   # still computable


def test_derive_returns_adds_month_start_date():
    out = EA.derive_returns(_frame(fund_ret=[0.05, 0.01], RF=[0.0, 0.0],
                                   efa_ret=[0.0, 0.0], scz_ret=[0.0, 0.0],
                                   vss_ret=[0.0, 0.0]))
    assert out["date"].tolist() == [pd.Timestamp("2020-01-01"),
                                    pd.Timestamp("2020-02-01")]


def test_derive_returns_does_not_mutate_input():
    df = _frame(fund_ret=[0.05], RF=[0.01],
                efa_ret=[0.02], scz_ret=[0.03], vss_ret=[0.04])
    EA.derive_returns(df)
    assert "excess_ret" not in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: FAIL with `AttributeError: module 'etf_analytics' has no attribute 'derive_returns'`

- [ ] **Step 3: Implement the derived columns**

Append to `dashboard/etf_analytics.py`:

```python
# --- derived columns -------------------------------------------------------

BENCH_LEGS = ["efa_ret", "scz_ret", "vss_ret"]


def derive_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``excess_ret``, ``bench_avg_ret``, ``alpha_vs_*`` and ``date``.

    Mirrors the identities baked into ``combined_monthly.csv``, all verified
    exact: the blended benchmark is the simple mean of the three benchmark
    ETFs, and each alpha is the fund return minus its benchmark.

    ``skipna=False`` on the blend is deliberate. Pandas would otherwise average
    whichever legs happen to be present, silently reporting a two-ETF blend as
    the three-ETF benchmark for any month with a gap.
    """
    out = frame.copy()
    out["date"] = pd.PeriodIndex(out["month"], freq="M").to_timestamp()

    if "RF" in out.columns:
        out["excess_ret"] = out["fund_ret"] - out["RF"]
    if all(c in out.columns for c in BENCH_LEGS):
        out["bench_avg_ret"] = out[BENCH_LEGS].mean(axis=1, skipna=False)
        out["alpha_vs_avg"] = out["fund_ret"] - out["bench_avg_ret"]
    if "efa_ret" in out.columns:
        out["alpha_vs_efa"] = out["fund_ret"] - out["efa_ret"]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: PASS, 12 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_analytics.py tests/test_etf_analytics.py
git commit -m "feat(dashboard): add derived return columns for ETF frames"
```

---

### Task 5: The `load_analytics_frame` orchestrator

**Files:**
- Modify: `dashboard/etf_analytics.py`
- Test: `tests/test_etf_analytics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_analytics.py`:

```python
def test_load_analytics_frame_fund_returns_the_combined_frame():
    """The capstone fund must keep its exact numbers and holdings regressors."""
    out = EA.load_analytics_frame("FUND")
    assert len(out) == 129
    assert any(c.startswith("ff_") for c in out.columns)
    assert any(c.startswith("sect_wt_") for c in out.columns)
    assert "PC1" in out.columns


def test_load_analytics_frame_fund_is_case_insensitive():
    assert len(EA.load_analytics_frame("fund")) == 129


def test_load_analytics_frame_etf_has_no_holdings_columns(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    out = EA.load_analytics_frame("EFA")
    assert not [c for c in out.columns if c.startswith("ff_")]
    assert not [c for c in out.columns if c.startswith("sect_wt_")]
    assert not [c for c in out.columns if c.startswith("ctry_wt_")]


def test_load_analytics_frame_etf_joins_onto_market_data(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    out = EA.load_analytics_frame("EFA")
    assert len(out) == 24
    assert out["month"].iloc[0] == "2020-01"
    for col in ("Mkt_RF", "vix", "PC1", "efa_ret", "date",
                "excess_ret", "bench_avg_ret", "alpha_vs_avg", "alpha_vs_efa"):
        assert col in out.columns
    assert out["fund_ret"].notna().all()


def test_load_analytics_frame_etf_keeps_only_months_with_returns(monkeypatch):
    """Inner join: no rows for months the ticker did not trade."""
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01, 0.02])))
    out = EA.load_analytics_frame("EFA")
    assert len(out) == 2


def test_load_analytics_frame_etf_pre_pca_months_have_nan_pcs(monkeypatch):
    """Ragged coverage: PCs start 2014-10, factors far earlier."""
    monkeypatch.setattr(
        EA.etf_prices, "load_etf",
        lambda t, start=None: _FakePrices(_monthly([0.01] * 6, start="2005-01-31")))
    out = EA.load_analytics_frame("EFA")
    assert out["PC1"].isna().all()
    assert out["Mkt_RF"].notna().all()   # factors reach back to 1990


def test_load_analytics_frame_propagates_price_errors(monkeypatch):
    def boom(t, start=None):
        raise etf_prices.NoPriceHistory("no price history for ZZZZ")
    monkeypatch.setattr(EA.etf_prices, "load_etf", boom)
    with pytest.raises(etf_prices.NoPriceHistory):
        EA.load_analytics_frame("ZZZZ")


def test_price_status_reports_staleness(monkeypatch):
    stale = _FakePrices(_monthly([0.01, 0.02]), is_stale=True)
    monkeypatch.setattr(EA.etf_prices, "load_etf", lambda t, start=None: stale)
    is_stale, as_of = EA.price_status("EFA")
    assert is_stale is True
    assert as_of == pd.Timestamp("2020-02-29")


def test_price_status_is_never_stale_for_the_capstone_fund():
    """FUND comes from a local CSV, so the yfinance cache never applies."""
    assert EA.price_status("FUND") == (False, None)
```

Add this import near the top of the test file, after `import data as D`:

```python
import etf_prices
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: FAIL with `AttributeError: module 'etf_analytics' has no attribute 'load_analytics_frame'`

- [ ] **Step 3: Implement the orchestrator**

Append to `dashboard/etf_analytics.py`:

```python
# --- orchestration ---------------------------------------------------------

def is_capstone(ticker: str) -> bool:
    """True for the capstone fund, which takes the combined-frame path."""
    return ticker.upper() == CAPSTONE_TICKER


def load_analytics_frame(ticker: str) -> pd.DataFrame:
    """Monthly analytics frame for `ticker`, with returns in ``fund_ret``.

    FUND returns the existing combined frame untouched -- same 129 months,
    same numbers, and the holdings-derived regressors (``ff_*``, ``sect_wt_*``,
    ``ctry_wt_*``) the capstone tabs depend on.

    Any other ticker is inner-joined onto the market frame, so rows exist only
    for months where both the ticker traded and market data exists. Holdings
    columns are absent by construction -- they are never joined in, because
    they describe FUND's portfolio and mean nothing for another fund.

    Raises whatever :func:`etf_prices.load_etf` raises (``TickerNotFound``,
    ``NoPriceHistory``, ``EtfDataError``); the caller renders the message.
    """
    if is_capstone(ticker):
        return D.load_combined_with_pcs()

    returns = etf_monthly_returns(ticker)
    merged = returns.merge(load_market_frame(), on="month", how="inner")
    return derive_returns(merged).sort_values("month").reset_index(drop=True)


def price_status(ticker: str) -> tuple[bool, pd.Timestamp | None]:
    """``(is_stale, as_of)`` for `ticker`'s price data.

    ``load_analytics_frame`` returns a bare frame, which drops the staleness
    flag that :class:`etf_prices.EtfPrices` carries -- but the UI must never
    present cached data as current, so expose it separately. The underlying
    ``load_etf`` is memoized, so this does not refetch.

    The capstone fund is read from a local CSV, so the yfinance cache never
    applies to it.
    """
    if is_capstone(ticker):
        return False, None
    prices = etf_prices.load_etf(ticker.upper())
    return prices.is_stale, prices.as_of
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: PASS, 21 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/etf_analytics.py tests/test_etf_analytics.py
git commit -m "feat(dashboard): add load_analytics_frame orchestrator"
```

---

### Task 6: Regressor catalog wrapper and the ETF preset

`data.regressor_catalog` already builds every group as `[c for c in FAMILY if c in df.columns]`, so on an ETF frame the holdings groups come back empty on their own. The wrapper drops the empties and swaps the preset.

**Files:**
- Modify: `dashboard/etf_analytics.py`
- Test: `tests/test_etf_analytics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_analytics.py`:

```python
def test_regressor_catalog_fund_keeps_holdings_families_and_preset():
    frame = EA.load_analytics_frame("FUND")
    cat = EA.regressor_catalog(frame)
    assert D.ALPHA_DRIVERS_PRESET_NAME in cat
    assert "Portfolio fundamentals" in cat
    assert "Sector weights" in cat


def test_regressor_catalog_etf_drops_holdings_and_swaps_preset(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    frame = EA.load_analytics_frame("EFA")
    cat = EA.regressor_catalog(frame)

    assert "Portfolio fundamentals" not in cat
    assert "Sector weights" not in cat
    assert "Country weights" not in cat
    assert D.ALPHA_DRIVERS_PRESET_NAME not in cat
    assert EA.ETF_ALPHA_PRESET_NAME in cat
    assert cat[EA.ETF_ALPHA_PRESET_NAME] == ["Mkt_RF", "SMB", "HML", "PC1", "PC2"]


def test_regressor_catalog_never_returns_empty_groups(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    cat = EA.regressor_catalog(EA.load_analytics_frame("EFA"))
    assert all(cols for cols in cat.values())


def test_regressor_catalog_etf_preset_columns_all_exist(monkeypatch):
    monkeypatch.setattr(EA.etf_prices, "load_etf",
                        lambda t, start=None: _FakePrices(_monthly([0.01] * 24)))
    frame = EA.load_analytics_frame("EFA")
    cat = EA.regressor_catalog(frame)
    for col in cat[EA.ETF_ALPHA_PRESET_NAME]:
        assert col in frame.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: FAIL with `AttributeError: module 'etf_analytics' has no attribute 'regressor_catalog'`

- [ ] **Step 3: Implement the catalog wrapper**

Append to `dashboard/etf_analytics.py`:

```python
# --- regressor catalog -----------------------------------------------------

# The FUND preset mixes in portfolio fundamentals (`ff_earn_yld_median`,
# `ff_roce_median`) that exist only for that fund. For an ETF the equivalent
# question -- systematic exposure vs. everything else -- is answered with
# factor and macro axes alone. Five regressors keeps the rolling OLS estimable
# and the coefficient plot legible, the same reasoning behind the FUND preset.
ETF_ALPHA_PRESET_NAME = "⭐ Factor & macro exposure"
ETF_ALPHA_PRESET = ["Mkt_RF", "SMB", "HML", "PC1", "PC2"]


def regressor_catalog(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Named regressor groups for `frame`, with empty families removed.

    ``data.regressor_catalog`` filters every group against the frame's columns,
    so an ETF frame already yields empty holdings groups; they are dropped here
    rather than shown as unusable menu entries. The starred preset is swapped
    for :data:`ETF_ALPHA_PRESET` when the frame has no portfolio fundamentals.
    """
    catalog = D.regressor_catalog(frame)
    has_holdings = bool(D.fundamentals_cols(frame))

    if not has_holdings:
        catalog.pop(D.ALPHA_DRIVERS_PRESET_NAME, None)
        preset = [c for c in ETF_ALPHA_PRESET if c in frame.columns]
        catalog = {ETF_ALPHA_PRESET_NAME: preset, **catalog}

    return {name: cols for name, cols in catalog.items() if cols}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_analytics.py -v`
Expected: PASS, 25 passed.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, 102 passed (77 pre-existing + 25 new).

If `uv.lock` now shows as modified, run `git checkout -- uv.lock`.

- [ ] **Step 6: Commit**

```bash
git add dashboard/etf_analytics.py tests/test_etf_analytics.py
git commit -m "feat(dashboard): add ETF-aware regressor catalog and preset"
```

---

### Task 7: Wire the selector into the two analytics tabs

**Files:**
- Modify: `dashboard/app.py`

The two tabs currently read the module-level `df` (FUND). They will read a per-ticker frame instead. `render_regime` already takes a `frame` argument; the factor tab builds `df_reg` inline.

- [ ] **Step 1: Import the module**

In `dashboard/app.py`, after `import rolling_regression as R` (line 25), add:

```python
import etf_analytics as EA
import etf_prices
```

- [ ] **Step 2: Add the cached loader and the selector widget**

After the existing `get_data()` function (around line 33), add:

```python
@st.cache_data(show_spinner="Loading ticker…")
def get_analytics_frame(ticker: str) -> pd.DataFrame:
    return EA.load_analytics_frame(ticker)


def ticker_selector(key: str) -> str | None:
    """Ticker picker for the analytics tabs. Returns None if input is invalid.

    FUND is the default and bypasses the ETF gate -- it is a MUTUALFUND, which
    `etf_prices.validate_etf` rejects by design. Everything else is gated, so a
    mistyped or non-ETF symbol is caught here rather than producing an empty fit.
    """
    col_a, col_b = st.columns([1, 3])
    with col_a:
        ticker = st.text_input(
            "Ticker", value=EA.CAPSTONE_TICKER, key=f"ticker_{key}",
            help="FUND is the capstone fund. Any ETF symbol works — "
                 "EFA, SCZ, VSS, IEFA, ACWX, VTI …",
        ).strip().upper()

    if not ticker:
        return None
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
```

- [ ] **Step 3: Add the frame-resolution helper**

Immediately after `ticker_selector`, add:

```python
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
            "(fundamentals, sector/country weights) are FUND-only."
        )
    return frame
```

- [ ] **Step 4: Point the factor-exposures tab at the selected ticker**

In the `with tab_reg:` block, replace this line (currently the first statement in the block):

```python
    controls, results = st.columns([1, 3], gap="large")
```

with:

```python
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
```

Then, still inside `with tab_reg:`, change the sample-period slider and the frame it slices so they follow the selected ticker rather than FUND's date range. Replace:

```python
        reg_period = st.slider(
            "Sample period", min_value=DMIN, max_value=DMAX,
            value=(DMIN, DMAX), format="YYYY-MM", key="reg_period",
```

with:

```python
        reg_period = st.slider(
            "Sample period", min_value=RMIN, max_value=RMAX,
            value=(RMIN, RMAX), format="YYYY-MM", key="reg_period",
```

and replace:

```python
        df_reg = df[(DATES >= reg_period[0]) & (DATES <= reg_period[1])].reset_index(drop=True)
```

with:

```python
        df_reg = reg_frame[(REG_DATES >= reg_period[0])
                           & (REG_DATES <= reg_period[1])].reset_index(drop=True)
```

- [ ] **Step 5: Use the ETF-aware catalog**

Still inside `with tab_reg:`, replace:

```python
        catalog = D.regressor_catalog(df_reg)
```

with:

```python
        catalog = EA.regressor_catalog(df_reg)
```

- [ ] **Step 6: Surface the effective fitted window**

The spec requires this: selecting a PC regressor on a long-history ETF silently cuts the sample from ~300 months to 129, and nothing else tells the user.

Still inside `with tab_reg:`, immediately after the existing caption:

```python
        st.caption(f"{df_reg['date'].min():%Y-%m} – {df_reg['date'].max():%Y-%m} "
                  f"· {len(df_reg)} months")
```

add:

```python
        st.caption(
            "The fit drops any month missing a selected regressor, so the "
            "effective sample can be shorter than the range above — the PCs "
            "cover 2014-10 – 2025-06 only."
        )
```

- [ ] **Step 7: Point the regime tab at the selected ticker**

Replace the `with tab_regime:` block (currently `render_regime(df)`):

```python
with tab_regime:
    render_regime(df)
```

with:

```python
with tab_regime:
    regime_ticker = ticker_selector("regime")
    if regime_ticker is not None:
        regime_frame = resolve_frame(regime_ticker)
        if regime_frame is not None:
            render_regime(regime_frame)
```

- [ ] **Step 8: Verify the app imports and the suite still passes**

Run: `uv run python -c "import sys; sys.path.insert(0, 'dashboard'); import ast; ast.parse(open('dashboard/app.py').read()); print('app.py parses')"`
Expected: `app.py parses`

Run: `uv run pytest -q`
Expected: PASS, 102 passed.

If `uv.lock` shows as modified, run `git checkout -- uv.lock`.

- [ ] **Step 9: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): add ticker selector to factor and regime tabs"
```

---

### Task 8: Documentation

**Files:**
- Modify: `dashboard/README.md`
- Modify: `docs/superpowers/etf-replatform-STATUS.md`

- [ ] **Step 1: Document the module in the dashboard README**

Append a section to `dashboard/README.md`:

```markdown
## ETF factor analytics (`etf_analytics.py`)

Builds the monthly frame the Factor-exposures and Macro-&-regime tabs analyze,
for any ticker.

- `load_analytics_frame(ticker)` -> monthly frame with the ticker's returns in
  `fund_ret`. **FUND** short-circuits to the existing `combined_monthly` frame
  (129 months, holdings regressors intact); any other ticker is inner-joined onto
  the market frame.
- `load_market_frame()` -> the fund-independent join of Fama-French factors, macro,
  PCA components, and benchmark ETF returns (432 months, 1990-07 – 2026-06). A test
  guards that it reproduces `combined_monthly.csv` exactly on the overlap.
- `regressor_catalog(frame)` -> `data.regressor_catalog` with empty families dropped
  and, for ETF frames, the holdings-based preset swapped for `ETF_ALPHA_PRESET`.

**Coverage is deliberately ragged.** Factors reach back to 1990-07, macro to 2000-01,
benchmark ETFs to 2014-09, and the PCs only span 2014-10 – 2025-06. The engines drop
incomplete rows, so selecting a PC regressor on a long-history ETF can quietly shrink
the fitted sample — which is why both tabs caption the effective window.
```

- [ ] **Step 2: Mark sub-project #3 done in the STATUS doc**

In `docs/superpowers/etf-replatform-STATUS.md`:

- Change the sub-project #3 row of the status table from `⬜ not started` to `✅ **DONE**`.
- Update **Last updated** to `2026-07-30`.
- In "How to resume", change the recommendation from "#3 or #4" to **#4 (the rating)**, noting that #5 remains last.

- [ ] **Step 3: Run the full suite one final time**

Run: `uv run pytest -q`
Expected: PASS, 102 passed.

- [ ] **Step 4: Confirm the working tree is clean**

Run: `git status --short`
Expected: only the pre-existing `.idea/` entries. If `uv.lock` appears, run `git checkout -- uv.lock`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/README.md docs/superpowers/etf-replatform-STATUS.md
git commit -m "docs: document etf_analytics module and mark sub-project #3 done"
```

---

## Done when

- `uv run pytest -q` reports 102 passed.
- `test_market_frame_matches_combined_monthly_exactly` passes — the ETF and FUND paths agree on all 27 shared columns.
- Selecting `FUND` on the Factor-exposures tab shows the holdings families and the original starred preset; selecting `EFA` shows neither, offers `⭐ Factor & macro exposure`, and spans more months than 129.
- Entering a non-ETF symbol (e.g. `AAPL`) on either tab renders an error rather than an empty fit.
- No file in `dashboard/` imports `capstone_data` (verify: `grep -rn "^import capstone_data\|^from capstone_data" dashboard/` returns nothing).
- `git status --short` shows only the `.idea/` entries that were already there.
