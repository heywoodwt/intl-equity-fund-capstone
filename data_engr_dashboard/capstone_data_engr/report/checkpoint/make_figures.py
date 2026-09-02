"""Generate checkpoint-report figures for the data-processing / aggregation section.

Run from the repo root with the project's uv environment:

    uv run python report/checkpoint/make_figures.py

Reads the finished tables in data/processed/ and writes PNGs to
report/checkpoint/figures/. Self-contained; no network access.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- paths --------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
GROUP_FF = REPO.parent / "ds6015" / "data" / "ff_factors.csv"  # Developed incl. US

# --- house style --------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "figure.autolayout": False,
})
FUND_C = "#1b4965"     # fund
BENCH_C = ["#9aa7b0", "#c0876a", "#6a9c78"]  # EFA, SCZ, VSS
ACCENT = "#c1666b"
GOOD = "#3f7d5e"


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path.relative_to(REPO))


# ------------------------------------------------------------------------------
# Fig 1 — data-processing pipeline / architecture
# ------------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc, tc="white", fs=9.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
            linewidth=0, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
            color="#5a5a5a", lw=1.4, zorder=1,
            connectionstyle="arc3,rad=0.0"))

    ax.text(50, 96, "Data pipeline: thin gift  →  enriched, point-in-time modeling tables",
            ha="center", fontsize=14, weight="bold")

    # Column 1: the gift + external sources
    ax.text(13, 88, "RAW INPUTS", ha="center", fontsize=10, color="#444", weight="bold")
    box(2, 76, 22, 7, "Fund gift:\nholdings + trades\n(~130 secs, ~1,184 trades)", "#1b4965", fs=8.8)
    box(2, 66, 22, 6, "FRED  (macro)", "#577590", fs=9)
    box(2, 58, 22, 6, "Ken French  (factors)", "#577590", fs=9)
    box(2, 50, 22, 6, "Yahoo  (prices / FX)", "#577590", fs=9)
    box(2, 42, 22, 6, "WRDS — FactSet Intl\n(fundamentals, delisted)", "#577590", fs=8.6)

    # Column 2: per-source tidy layer
    ax.text(50, 88, "TIDY, POINT-IN-TIME LAYERS", ha="center", fontsize=10, color="#444", weight="bold")
    box(34, 74, 32, 6, "macro_monthly  (17 regime features)", "#43806c", fs=8.8)
    box(34, 65, 32, 6, "ff_factors_dev_ex_us  (6 factors)", "#43806c", fs=8.8)
    box(34, 56, 32, 6, "stock_returns_monthly  (survivorship-free)", "#43806c", fs=8.4)
    box(34, 47, 32, 6, "fundamentals_panel  (25 ratios, lagged 4mo)", "#43806c", fs=8.2)
    box(34, 38, 32, 6, "portfolio_* / sector-country  (value-weighted)", "#43806c", fs=8.0)

    # Column 3: combined modeling datasets
    ax.text(86, 88, "MODELING DATASETS", ha="center", fontsize=10, color="#444", weight="bold")
    box(74, 70, 24, 8, "combined_monthly\n129 mo × 132 cols\n(time-series / regression)", ACCENT, fs=8.6)
    box(74, 57, 24, 8, "combined_panel\n7,740 rows × 91 cols\n(cross-section / DL)", ACCENT, fs=8.6)
    box(74, 44, 24, 8, "pca_{market,full}\northogonal components\n(additive features)", "#7a5195", fs=8.6)

    # arrows raw -> tidy (left column box centers -> matching tidy box centers)
    raw_y = [79.5, 69, 61, 53, 45]          # raw box centers
    tidy_y = [77, 68, 59, 50, 41]           # tidy box centers
    for ry, ty in zip(raw_y, tidy_y):
        arrow(24, ry, 34, ty)

    # tidy -> combined (fan in to the two combined tables)
    for ty in tidy_y:
        arrow(66, ty, 74, 74)
        arrow(66, ty, 74, 61)
    # combined tables feed the additive PCA features
    arrow(86, 70, 86, 52)

    # join key note
    ax.text(50, 30, "Join key:  month  (YYYY-MM)   •   panel also keys on  ticker",
            ha="center", fontsize=9.5, color="#333", style="italic")
    ax.text(50, 24,
            "Enforced throughout:  (1) point-in-time / no look-ahead  "
            "•  (2) survivorship-aware (delisted names kept)",
            ha="center", fontsize=9.5, color="#333")

    _save(fig, "fig1_pipeline.png")


# ------------------------------------------------------------------------------
# Fig 2 — cumulative growth of $1: fund vs benchmarks (the modeling target)
# ------------------------------------------------------------------------------
def fig_cumulative():
    m = pd.read_csv(PROC / "combined_monthly.csv")
    m["dt"] = pd.PeriodIndex(m["month"], freq="M").to_timestamp("M")
    cols = {"fund_ret": "Fund", "efa_ret": "EFA (dev large)",
            "scz_ret": "SCZ (dev small)", "vss_ret": "VSS (ex-US small)"}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    colors = [FUND_C] + BENCH_C
    lws = [2.6, 1.5, 1.5, 1.5]
    for (c, lbl), col, lw in zip(cols.items(), colors, lws):
        growth = (1 + m[c].fillna(0)).cumprod()
        ax.plot(m["dt"], growth, label=lbl, color=col, lw=lw)
    ax.axhline(1, color="#999", lw=0.8, ls="--")
    ax.set_title("Growth of $1 invested: fund vs. three benchmark ETFs (2014–10 → 2025–06)")
    ax.set_ylabel("Cumulative value of $1")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.margins(x=0.01)
    # annotate end values
    for (c, lbl), col in zip(cols.items(), colors):
        end = (1 + m[c].fillna(0)).cumprod().iloc[-1]
        ax.annotate(f"{end:.2f}×", (m["dt"].iloc[-1], end),
                    xytext=(6, 0), textcoords="offset points",
                    color=col, fontsize=9, va="center", weight="bold")
    _save(fig, "fig2_cumulative_returns.png")


# ------------------------------------------------------------------------------
# Fig 3 — portfolio P/E by aggregation method (justification for aggregation)
# ------------------------------------------------------------------------------
def fig_pe_aggregation():
    m = pd.read_csv(PROC / "combined_monthly.csv")
    sub = m[m["ff_pe_dil_whmean"].notna()].copy()
    sub["dt"] = pd.PeriodIndex(sub["month"], freq="M").to_timestamp("M")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    ax.plot(sub["dt"], sub["ff_pe_dil_wmean"], color=ACCENT, lw=2,
            label="Arithmetic value-weighted mean")
    ax.plot(sub["dt"], sub["ff_pe_dil_median"], color="#577590", lw=2,
            label="Value-weighted median")
    ax.plot(sub["dt"], sub["ff_pe_dil_whmean"], color=GOOD, lw=2.4,
            label="Harmonic value-weighted mean (used)")
    ax.set_title("Portfolio P/E by aggregation method")
    ax.set_ylabel("Portfolio price / earnings")
    ax.legend(frameon=False, fontsize=9)
    ax.margins(x=0.02)

    ax2 = axes[1]
    means = [sub["ff_pe_dil_wmean"].mean(), sub["ff_pe_dil_median"].mean(),
             sub["ff_pe_dil_whmean"].mean()]
    labels = ["Arithmetic\nmean", "Median", "Harmonic\nmean"]
    bars = ax2.bar(labels, means, color=[ACCENT, "#577590", GOOD], width=0.62)
    ax2.set_title("Window average")
    ax2.set_ylabel("Portfolio P/E")
    ax2.grid(axis="x", alpha=0)
    for b, v in zip(bars, means):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}",
                 ha="center", fontsize=10, weight="bold")
    fig.suptitle("Aggregating a price multiple: arithmetic mean inflates it; harmonic ≈ median",
                 fontsize=13, weight="bold", y=1.02)
    _save(fig, "fig3_pe_aggregation.png")


# ------------------------------------------------------------------------------
# Fig 4 — value-weighted sector & country composition (holdings -> portfolio)
# ------------------------------------------------------------------------------
def fig_sector_country():
    m = pd.read_csv(PROC / "combined_monthly.csv")
    sub = m[m["ff_pe_dil_whmean"].notna()]
    sect = sub[[c for c in m.columns if c.startswith("sect_wt_")]].mean() * 100
    ctry = sub[[c for c in m.columns if c.startswith("ctry_wt_")]].mean() * 100
    sect = sect[sect > 0].sort_values()
    ctry = ctry.sort_values(ascending=False).head(10).sort_values()

    def clean_sect(idx):
        return [i.replace("sect_wt_", "").replace("_", " ").title() for i in idx]

    def clean_ctry(idx):
        return [i.replace("ctry_wt_", "").upper() for i in idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    a = axes[0]
    a.barh(clean_sect(sect.index), sect.values, color="#43806c")
    a.set_title("Avg. value-weighted sector weight")
    a.set_xlabel("% of portfolio")
    for y, v in enumerate(sect.values):
        a.text(v + 0.4, y, f"{v:.0f}%", va="center", fontsize=8.5)
    a.grid(axis="y", alpha=0)

    b = axes[1]
    b.barh(clean_ctry(ctry.index), ctry.values, color="#577590")
    b.set_title("Avg. value-weighted country weight (top 10)")
    b.set_xlabel("% of portfolio")
    for y, v in enumerate(ctry.values):
        b.text(v + 0.4, y, f"{v:.0f}%", va="center", fontsize=8.5)
    b.grid(axis="y", alpha=0)

    fig.suptitle("Holdings aggregated to portfolio exposures (value-weighted, 2018–04 → 2022–12)",
                 fontsize=13, weight="bold", y=1.01)
    _save(fig, "fig4_sector_country.png")


# ------------------------------------------------------------------------------
# Fig 5 — candidate-predictor correlation with fund return & alpha
# ------------------------------------------------------------------------------
def fig_predictor_corr():
    m = pd.read_csv(PROC / "combined_monthly.csv")
    preds = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom",
             "vix", "vix_chg", "cpi_yoy", "m2_yoy", "y10",
             "slope_10y_2y", "baa_spread", "sp500_ret", "sp500_mom_12m"]
    targets = {"fund_ret": "Fund return", "alpha_vs_avg": "Alpha vs bench"}
    C = m[preds + list(targets)].corr().loc[preds, list(targets)]

    fig, ax = plt.subplots(figsize=(5.6, 7.2))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(targets)), list(targets.values()), fontsize=10)
    ax.set_yticks(range(len(preds)), preds, fontsize=9.5)
    for i in range(len(preds)):
        for j in range(len(targets)):
            v = C.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.45 else "#222", fontsize=9)
    ax.set_title("Candidate predictors vs. monthly\nfund return and alpha", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label("Pearson correlation", fontsize=9)
    ax.grid(False)
    _save(fig, "fig5_predictor_correlation.png")


# ------------------------------------------------------------------------------
# Fig 6 — the factor correction: Developed (incl US) vs Developed-ex-US market
# ------------------------------------------------------------------------------
def fig_factor_correction():
    ex = pd.read_csv(PROC / "ff_factors_dev_ex_us.csv", parse_dates=["date"])
    if not GROUP_FF.exists():
        print("skip fig6: group ff_factors.csv not found")
        return
    incl = pd.read_csv(GROUP_FF, parse_dates=["date"])
    lo = "2014-10-01"
    ex = ex[ex["date"] >= lo]
    incl = incl[incl["date"] >= lo]
    merged = ex[["date", "Mkt_RF"]].merge(
        incl[["date", "Mkt_RF"]], on="date", suffixes=("_exus", "_incl"))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(merged["date"], (1 + merged["Mkt_RF_incl"]).cumprod(),
            color=ACCENT, lw=2, label="Developed incl. US  (group repo's ff_factors.csv)")
    ax.plot(merged["date"], (1 + merged["Mkt_RF_exus"]).cumprod(),
            color=GOOD, lw=2.4, label="Developed ex-US  (corrected — matches the fund)")
    corr = merged["Mkt_RF_incl"].corr(merged["Mkt_RF_exus"])
    ax.set_title("Why the factor set matters: the two 'market' factors diverge")
    ax.set_ylabel("Cumulative market factor (growth of $1)")
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.01)
    ax.text(0.99, 0.04, f"monthly corr = {corr:.2f}", transform=ax.transAxes,
            ha="right", fontsize=10, color="#444",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    _save(fig, "fig6_factor_correction.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_cumulative()
    fig_pe_aggregation()
    fig_sector_country()
    fig_predictor_corr()
    fig_factor_correction()
    print("done ->", OUT)
