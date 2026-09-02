# Data Access: What to Look Into and Ask For

Goal: enrich the fund's holdings/trades with **fundamentals, macro regime indicators, and
sector/country data** (see `roadmap.md` → Data Engineering & Enrichment). Most gaps need data we
don't have yet. This lists what to ask the **school (UVA / library)** and the **fund (our capstone
sponsor)** for, what each source gives us, and why.

## Two principles to repeat in every request

1. **Point-in-time / as-reported.** We need each metric *as it was known on that date* (e.g., the
   P/E of a stock in March 2019, using the financials reported by then). Free tools (yfinance,
   OpenBB) return *today's* fundamentals — using them for a 2019 trade is **look-ahead bias** and
   silently corrupts the analysis. Always ask: "Is this point-in-time / does it avoid restated
   data?"
2. **Includes delisted & acquired names.** ~25 of the fund's 130 holdings have already dropped out
   of our universe, and they skew toward blowups/buyouts (Wirecard, GW Pharma, Farfetch, Kahoot…).
   A source that only has *currently listed* companies gives us **survivorship bias**. Always ask:
   "Does this include delisted/merged securities and delisting returns?"

> Context that makes this harder: the fund is **developed-markets ex-US** (Europe/Japan/Asia-Pac).
> Many US-centric databases (e.g. CRSP) won't cover the holdings. We specifically need **global /
> international** coverage.

---

## Ask the School (UVA MSDS program + Library / Finance lab)

These are institutional subscriptions universities often have. Find out **which we already have**
and **which databases are included** before asking the sponsor.

| Resource | What it gives us | Why we need it | Priority |
|---|---|---|---|
| **WRDS** (Wharton Research Data Services) | Umbrella access to Compustat, CRSP, IBES, etc. via one login | The single most valuable thing to confirm. Many MSDS programs include it. | 🔴 Ask first |
| → **Compustat Global** (within WRDS) | Point-in-time fundamentals for **non-US** companies (revenue, earnings, book value, margins, debt, cash flow) | Source for valuation **and** quality ratios on the actual (international) holdings | 🔴 Critical |
| → **Datastream / Worldscope** (Refinitiv, often via WRDS or library) | Deep international fundamentals + macro + FX, long history | Best international coverage; backup/complement to Compustat Global | 🔴 Critical |
| → **IBES** (within WRDS) | Analyst estimates / forward earnings | Forward valuation (forward P/E) and growth/revision signals | 🟡 Nice-to-have |
| **Bloomberg Terminal** (library / finance lab) | Historical fundamentals, FX, macro, classifications, holdings | Strong fallback if no WRDS; extraction is manual (Excel `BDH/BDP`), so scope it | 🟡 Good fallback |
| **Capital IQ (S&P)** | Fundamentals, screening, classifications | Alternative fundamentals source | 🟢 Optional |
| **Morningstar Direct** | Fund-level holdings history, style/sector classifications, attribution | Could give us clean point-in-time *holdings* and sector tags for the fund | 🟡 Useful |
| **MSCI / GICS classification** access | Official sector + sub-industry codes | Clean sector breakdowns (roadmap: "% in each sector") tied to the fund's benchmark family | 🟡 Useful |

**Key questions to email the program/library:**
- "Do students have **WRDS** access, and does our subscription include **Compustat Global** and/or
  **Refinitiv Datastream/Worldscope**?"
- "Is there a **Bloomberg Terminal** available to students, and can we export historical
  fundamentals/FX for a list of ~130 international tickers?"

---

## Ask the Fund (our capstone sponsor)

The sponsor can often hand us exactly what we need far faster than we can rebuild it — they likely
already license a data vendor. Frame these as "to do the attribution well, it would help to have…".

| Request | What it gives us | Why we need it | Priority |
|---|---|---|---|
| **Full point-in-time holdings history** (weights + shares, monthly or finer), *including positions later sold/delisted* | Authoritative portfolio composition over time | Fixes our reconstructed-holdings gaps **and** the survivorship problem (#2 above). Foundation of everything. | 🔴 Critical |
| **Official return / NAV series** (gross **and** net of fees, monthly + daily if possible) | Ground-truth fund performance | Our performance series is scraped/derived; theirs is authoritative and lets us define alpha precisely | 🔴 Critical |
| **Their data vendor + a fundamentals export for the holdings** (Bloomberg / FactSet / Refinitiv) | Point-in-time fundamentals for exactly the names we care about | Could make the entire "ask the school" path unnecessary | 🔴 High-leverage |
| **Cash, FX-hedge, and derivative positions** | The non-equity part of the book | We only have equity trades; performance also comes from cash drag and currency. Needed for honest attribution | 🟡 Important |
| **Benchmark(s) they officially report against** | The right baseline for alpha | We're guessing EFA/SCZ/VSS; confirm so alpha is defined correctly | 🟡 Important |
| **Existing attribution reports** (e.g. Brinson sector/country attribution) | Their own breakdown of what drove returns | Ground truth to validate our models against | 🟡 Important |
| **Mandate / prospectus / strategy doc** | Universe, constraints, style, currency policy | Defines the investable universe and how to frame factors | 🟢 Context |
| **Fee schedule** | Expense ratio / fee timing | Convert between gross and net returns cleanly | 🟢 Context |
| **Their sector/country/style classification scheme** (GICS? in-house?) | Consistent category labels | So our sector/country aggregates match how they think about the book | 🟢 Context |

**Key questions to ask the sponsor:**
- "Can you share **point-in-time holdings** (with weights), including names you've since exited?"
- "What **data provider** do you use, and could you export **historical fundamentals** for the
  holdings — or let us pull them?"
- "What **benchmark** do you officially measure against, and do you have **attribution reports** we
  could compare our findings to?"

---

## Already free — don't spend an ask on these

We can get these ourselves; listed so we don't request them by mistake.

| Need (roadmap) | Free source | Notes |
|---|---|---|
| Macro: VIX, CPI, M2, yield curve, credit spreads, S&P 500, USD index | **FRED** (`fredapi`) | Series e.g. `VIXCLS`, `CPIAUCSL`, `M2SL`, `DGS10`, `T10Y2Y`, `BAMLC0A0CM`, `SP500`, `DTWEXBGS`. Historical + point-in-time. |
| FF Developed ex-US factors | **Ken French Data Library** / `pandas-datareader` | Already have `ff_factors.csv` — just **confirm it's the ex-US/World set**, not US. |
| Equity prices/returns | **yfinance / Stooq** | Already have for ~80% of holdings. ⚠️ yfinance **fundamentals are current-only** — not usable for history. |
| Sector/country/currency tags (rough) | **yfinance `.info`**, OpenBB | Usable as a fallback classification, but current snapshot only — not point-in-time. |
| International country macro (M2, rates) | **OECD, ECB, World Bank** APIs | For per-country money supply / rates if we add country macro. |

---

## Suggested first moves (in order)

1. **Confirm WRDS** access + which databases (Compustat Global / Datastream). One email to the
   program — this unblocks the most.
2. **Ask the sponsor** for point-in-time holdings + their data vendor. This may make most other
   asks moot and fixes the survivorship gap.
3. While waiting, **build the free layer** (FRED macro + verify FF region + yfinance returns) so
   the pipeline scaffold is ready to drop fundamentals into.
4. Decide the **fundamentals scope** once we know the source: full point-in-time (WRDS/sponsor) vs.
   a reduced, clearly-caveated set (free only).
