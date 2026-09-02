# ETF Holdings Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest full published holdings for a curated list of iShares ETFs and write a tidy latest-snapshot table to `data/processed/etf_holdings.csv`.

**Architecture:** One source module `capstone_data/etf_holdings.py` (network `fetch` isolated from a pure `parse`; `build` orchestrates with skip-on-failure; `write` emits CSV), a curated `ISHARES_ETFS` dict in `config.py`, and a build-step entrypoint `scripts/build_etf_holdings.py`. Mirrors the existing `french.py` / `build_attribution.py` conventions.

**Tech Stack:** Python 3.10+, pandas, requests. Tests via `uv run pytest` (deterministic, no network).

**Spec:** `docs/superpowers/specs/2026-07-22-etf-holdings-ingestion-design.md`

---

### Task 1: Add the curated iShares universe to config

**Files:**
- Modify: `capstone_data/config.py` (append a new section)

- [ ] **Step 1: Add the `ISHARES_ETFS` dict**

Append to `capstone_data/config.py`:

```python
# --- iShares ETF holdings: ticker -> product holdings-CSV URL ------------------
# iShares publishes each product's full holdings as CSV at a stable ajax endpoint.
# Extend the universe by adding a line. URLs are the "Detailed Holdings and
# Analytics" CSV download from each product page (fileType=csv&dataType=fund).
ISHARES_ETFS = {
    "EFA":  "https://www.ishares.com/us/products/239623/ishares-msci-eafe-etf/1467271812596.ajax?fileType=csv&fileName=EFA_holdings&dataType=fund",
    "IEFA": "https://www.ishares.com/us/products/244049/ishares-core-msci-eafe-etf/1467271812596.ajax?fileType=csv&fileName=IEFA_holdings&dataType=fund",
    "ACWX": "https://www.ishares.com/us/products/239641/ishares-msci-acwi-ex-us-etf/1467271812596.ajax?fileType=csv&fileName=ACWX_holdings&dataType=fund",
    "SCZ":  "https://www.ishares.com/us/products/239627/ishares-msci-eafe-smallcap-etf/1467271812596.ajax?fileType=csv&fileName=SCZ_holdings&dataType=fund",
    "IEMG": "https://www.ishares.com/us/products/244050/ishares-core-msci-emerging-markets-etf/1467271812596.ajax?fileType=csv&fileName=IEMG_holdings&dataType=fund",
    "URTH": "https://www.ishares.com/us/products/239696/ishares-msci-world-etf/1467271812596.ajax?fileType=csv&fileName=URTH_holdings&dataType=fund",
}
```

> Note for implementer: these are the canonical iShares product paths. If any URL 404s at run time, open the product page on ishares.com and copy the "Download Holdings" CSV link — only the numeric product id / slug differs. This does not block the parser or tests (which never hit the network).

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "from capstone_data import config; print(len(config.ISHARES_ETFS))"`
Expected: `6`

- [ ] **Step 3: Commit**

```bash
git add capstone_data/config.py
git commit -m "feat(pipeline): add curated ISHARES_ETFS universe to config"
```

---

### Task 2: Parse iShares holdings CSV → tidy schema

**Files:**
- Create: `capstone_data/etf_holdings.py`
- Test: `tests/test_etf_holdings.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_etf_holdings.py`:

```python
"""Deterministic checks for iShares holdings ingestion (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from capstone_data import etf_holdings as H

# Real-shaped iShares export: text preamble, blank line, header, 2 equity rows,
# 1 cash/derivative row. Market values are quoted with thousands separators;
# the cash row has "-" for ticker.
SAMPLE = b'''iShares MSCI EAFE ETF
Fund Holdings as of,"Jul 18, 2026"
Inception Date,"Aug 14, 2001"
Shares Outstanding,"1,000,000"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,CUSIP,ISIN,SEDOL,Price,Location,Exchange,Currency
NESN,NESTLE SA,Consumer Staples,Equity,"1,234,567.89","2.15","1,234,567.89","12,345",-,CH0038863350,-,85.20,Switzerland,SIX Swiss Exchange,CHF
ASML,ASML HOLDING NV,Information Technology,Equity,"2,000,000.00","3.50","2,000,000.00","5,000",-,NL0010273215,-,700.00,Netherlands,Euronext,EUR
-,BLK CSH FND TREASURY,Cash and/or Derivatives,Money Market,"500,000.00","0.30","500,000.00","500,000",-,-,-,1.00,United States,-,USD
'''


def test_parse_schema_and_row_count():
    df = H.parse(SAMPLE, "EFA")
    assert list(df.columns) == H.COLUMNS
    assert len(df) == 3
    assert (df["etf_ticker"] == "EFA").all()


def test_parse_as_of_date():
    df = H.parse(SAMPLE, "EFA")
    assert (df["as_of_date"] == "2026-07-18").all()


def test_parse_numeric_coercion():
    df = H.parse(SAMPLE, "EFA").set_index("name")
    row = df.loc["NESTLE SA"]
    assert row["market_value"] == pytest.approx(1234567.89)
    assert row["weight"] == pytest.approx(2.15)
    assert row["shares"] == pytest.approx(12345.0)


def test_parse_cash_row_kept_and_ticker_blanked():
    df = H.parse(SAMPLE, "EFA").set_index("name")
    cash = df.loc["BLK CSH FND TREASURY"]
    assert cash["asset_class"] == "Money Market"
    assert cash["constituent_ticker"] == ""


def test_parse_missing_header_raises():
    with pytest.raises(ValueError, match="header row not found"):
        H.parse(b"just some text\nno,constituent,table\n", "EFA")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_holdings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'capstone_data.etf_holdings'`

- [ ] **Step 3: Write the module (config, fetch, parse)**

Create `capstone_data/etf_holdings.py`:

```python
"""Ingest full published holdings for a curated list of iShares ETFs.

iShares publishes each product's holdings as CSV at a stable ajax URL. The file
has a short text preamble (fund name, "Fund Holdings as of", share buckets) and
a blank line before the real header row, which starts with "Ticker,Name,Sector,".
We locate that header, read the constituent table, and normalize to one tidy row
per (ETF, constituent). Cash/derivative rows are kept (identified by asset_class)
so weights still sum to ~100%.
"""

import io
import re
import time

import pandas as pd
import requests

from capstone_data import config

_HEADERS = {"User-Agent": "capstone-data-engr/0.1"}
# Capture the whole date (it contains a comma, e.g. "Jul 18, 2026"), quotes optional.
_ASOF_RE = re.compile(r'Fund Holdings as of,\s*"?(.+?)"?\s*$', re.IGNORECASE)

# Output columns, in order.
COLUMNS = [
    "etf_ticker", "as_of_date", "constituent_ticker", "name",
    "sector", "asset_class", "weight", "market_value", "shares",
]

# iShares source column -> our column.
_COLMAP = {
    "Ticker": "constituent_ticker",
    "Name": "name",
    "Sector": "sector",
    "Asset Class": "asset_class",
    "Weight (%)": "weight",
    "Market Value": "market_value",
    "Shares": "shares",
}


def _get(url, timeout=30, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise last


def fetch(ticker, url):
    """Download the raw iShares holdings CSV bytes for one ETF (network)."""
    return _get(url).content


def _to_float(s):
    """Coerce an iShares numeric column (quoted strings, thousands commas) to float."""
    cleaned = (s.astype(str).str.replace(",", "", regex=False).str.strip()
               .replace({"-": None, "": None, "nan": None}))
    return pd.to_numeric(cleaned, errors="coerce")


def parse(raw, etf_ticker):
    """Parse raw iShares holdings CSV bytes into the tidy schema (pure)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = text.splitlines()

    m = next((_ASOF_RE.search(ln) for ln in lines if _ASOF_RE.search(ln)), None)
    as_of = pd.to_datetime(m.group(1).strip()).date().isoformat() if m else None

    hdr = next((i for i, ln in enumerate(lines)
                if ln.lstrip().startswith("Ticker,")), None)
    if hdr is None:
        raise ValueError(f"{etf_ticker}: iShares holdings header row not found")

    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df = df[[c for c in _COLMAP if c in df.columns]].rename(columns=_COLMAP)
    df = df.dropna(how="all")
    df = df.loc[df["name"].notna()].copy()

    for col in ("weight", "market_value", "shares"):
        df[col] = _to_float(df[col])
    df["constituent_ticker"] = (df["constituent_ticker"].astype(str).str.strip()
                                .replace({"-": "", "nan": ""}))
    df.insert(0, "as_of_date", as_of)
    df.insert(0, "etf_ticker", etf_ticker)
    return df[COLUMNS].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_holdings.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add capstone_data/etf_holdings.py tests/test_etf_holdings.py
git commit -m "feat(pipeline): parse iShares holdings CSV into tidy schema"
```

---

### Task 3: Orchestrate build with skip-on-failure

**Files:**
- Modify: `capstone_data/etf_holdings.py` (add `build`)
- Test: `tests/test_etf_holdings.py` (add cases)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etf_holdings.py`:

```python
def test_build_concats_and_skips_failures(monkeypatch):
    def fake_fetch(ticker, url):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return SAMPLE
    monkeypatch.setattr(H, "fetch", fake_fetch)

    df, failures = H.build({"EFA": "u1", "BAD": "u2"})
    assert list(df["etf_ticker"].unique()) == ["EFA"]
    assert len(df) == 3
    assert "BAD" in failures and "boom" in failures["BAD"]


def test_build_all_fail_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(H, "fetch",
                        lambda t, u: (_ for _ in ()).throw(RuntimeError("x")))
    df, failures = H.build({"EFA": "u1"})
    assert df.empty
    assert list(df.columns) == H.COLUMNS
    assert set(failures) == {"EFA"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etf_holdings.py -k build -v`
Expected: FAIL — `AttributeError: module 'capstone_data.etf_holdings' has no attribute 'build'`

- [ ] **Step 3: Add `build` to the module**

Append to `capstone_data/etf_holdings.py`:

```python
def build(etfs=None):
    """Fetch+parse each ETF, concatenating results. Returns (frame, failures).

    A network or parse failure for one ETF is recorded in ``failures`` and skipped;
    the rest still build.
    """
    etfs = etfs if etfs is not None else config.ISHARES_ETFS
    frames, failures = [], {}
    for ticker, url in etfs.items():
        try:
            frames.append(parse(fetch(ticker, url), ticker))
        except Exception as exc:  # network (requests) or parse (ValueError)
            failures[ticker] = str(exc)
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=COLUMNS))
    return out, failures
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_holdings.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add capstone_data/etf_holdings.py tests/test_etf_holdings.py
git commit -m "feat(pipeline): add build() with per-ETF skip-on-failure"
```

---

### Task 4: Write the processed CSV

**Files:**
- Modify: `capstone_data/etf_holdings.py` (add `write`)
- Test: `tests/test_etf_holdings.py` (add case)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_etf_holdings.py`:

```python
def test_write_roundtrip(tmp_path):
    df = H.parse(SAMPLE, "EFA")
    out = H.write(df, tmp_path / "etf_holdings.csv")
    assert out.exists()
    back = pd.read_csv(out)
    assert list(back.columns) == H.COLUMNS
    assert len(back) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_etf_holdings.py -k write -v`
Expected: FAIL — `AttributeError: module 'capstone_data.etf_holdings' has no attribute 'write'`

- [ ] **Step 3: Add `write` to the module**

Append to `capstone_data/etf_holdings.py`:

```python
def write(df, path=None):
    """Write the holdings frame to CSV (default data/processed/etf_holdings.csv)."""
    path = path if path is not None else (config.PROCESSED_DIR / "etf_holdings.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etf_holdings.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add capstone_data/etf_holdings.py tests/test_etf_holdings.py
git commit -m "feat(pipeline): write etf_holdings.csv to data/processed"
```

---

### Task 5: Build-step entrypoint

**Files:**
- Create: `scripts/build_etf_holdings.py`

- [ ] **Step 1: Write the entrypoint**

Create `scripts/build_etf_holdings.py`:

```python
#!/usr/bin/env python3
"""Fetch full iShares ETF holdings and write the processed snapshot.

    uv run python scripts/build_etf_holdings.py

Downloads each ETF in config.ISHARES_ETFS from its iShares holdings-CSV URL,
normalizes to a tidy per-(ETF, constituent) table, and writes
data/processed/etf_holdings.csv (latest snapshot). ETFs that fail to fetch or
parse are skipped and reported; the rest still write.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capstone_data import config, etf_holdings  # noqa: E402


def main():
    df, failures = etf_holdings.build(config.ISHARES_ETFS)
    out = etf_holdings.write(df)

    n_etfs = df["etf_ticker"].nunique() if not df.empty else 0
    print(f"etf_holdings: {len(df)} rows across {n_etfs} ETFs -> {out}")
    if not df.empty:
        by_etf = df.groupby("etf_ticker").size()
        for ticker, n in by_etf.items():
            asof = df.loc[df["etf_ticker"] == ticker, "as_of_date"].iloc[0]
            print(f"  {ticker:<6} {n:>4} holdings  as of {asof}")
    if failures:
        print("\nSkipped (fetch/parse failed):")
        for ticker, msg in failures.items():
            print(f"  {ticker:<6} {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports and wires up (offline smoke test)**

Run:
```bash
uv run python -c "import sys; from pathlib import Path; \
sys.path.insert(0, '.'); \
from capstone_data import etf_holdings as H; \
df, f = H.build({}); print('empty build ok:', df.empty, 'cols', list(df.columns))"
```
Expected: `empty build ok: True cols ['etf_ticker', 'as_of_date', 'constituent_ticker', 'name', 'sector', 'asset_class', 'weight', 'market_value', 'shares']`

- [ ] **Step 3: Live run (network — verification, may be skipped if offline)**

Run: `uv run python scripts/build_etf_holdings.py`
Expected: a per-ETF holdings/as-of summary and a written `data/processed/etf_holdings.csv`. If iShares blocks the request or a URL 404s, that ETF is listed under "Skipped"; refresh the URL from the product page (see Task 1 note) and re-run. This step confirms the real iShares format still matches the parser.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_etf_holdings.py
git commit -m "feat(pipeline): add build_etf_holdings entrypoint"
```

---

### Task 6: Full regression + README build-order note

**Files:**
- Modify: `README.md` (add the new step to the build-order list, if present)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (existing suite + 8 new `test_etf_holdings` tests).

- [ ] **Step 2: Add the build step to the README**

In `README.md`, locate the "Build order" / build-step listing and add a line for the new entrypoint alongside the others (match the existing formatting), e.g.:

```
uv run python scripts/build_etf_holdings.py   # iShares ETF holdings -> data/processed/etf_holdings.csv
```

If no build-order list exists, skip this step.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: note etf_holdings build step"
```