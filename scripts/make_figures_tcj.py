"""Main figures for the TCJ manuscript (v13). Every plotted value is read from a committed result file (or computed from
one with the pooling functions of the analysis); the values drawn are written to results/figdata_tcj/<figure>.csv.

Fig. 1  Study overview: data in time, forward split, the two scenarios, scoring and verdict.
Fig. 2  Sparse testing: value of the target-year phenotypes and the G x E increment, by dataset; new vs old lines.
Fig. 3  CLAC: target years, ablations, the arc-cosine kernel in cell-level GBLUP by dataset.
Fig. 4  New lines: method x dataset differences, pooled estimates (DerSimonian-Laird and Hartung-Knapp), across-environment metric.
Fig. 5  Other information and published networks.
Fig. 6  Evaluation practices that supply information from the test data.
Fig. S1 Tested lines and seasonal forecasts in G2F (exploratory; Supplementary Information).

Style: Okabe-Ito colours, Arial 7-8 pt, panel letters 9 pt bold, width 174 mm (double column).
Run from the repository root:  python3 scripts/make_figures_tcj.py [fig1 ...]  ->  figures/tcj/"""
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, "src")
from dartgxe.forward.pool import floor_se, hartung_knapp  # noqa: E402

if os.environ.get("FIGURE_QA_DIR"):
    sys.path.insert(0, os.environ["FIGURE_QA_DIR"])
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
except ImportError:
    require_matplotlib_panel_alignment = None

R = Path("results")
OUT = Path("figures/tcj"); OUT.mkdir(parents=True, exist_ok=True)
SRC = R / "figdata_tcj"; SRC.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
W2 = 174 * MM

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none",
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False, "lines.linewidth": 0.8})
# Okabe-Ito
BLUE, SKY, GREEN, ORANGE, VERM, PURPLE, YELLOW = "#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#F0E442"
GREY, GREY_L, DARK = "#7F7F7F", "#DDDDDD", "#222222"
DS = ["G2F", "NUST", "URSN", "ESWYT", "GEM_IA", "MU_SOY"]
DS_LAB = {"G2F": "G2F\nmaize", "NUST": "NUST\nsoybean", "URSN": "URSN\nwheat", "ESWYT": "ESWYT\nwheat", "GEM_IA": "GEM_IA\nmaize",
          "MU_SOY": "MU_SOY\nsoybean"}
NAMES = {"reml": "Two-stage GBLUP", "reml_x0.1": "Ridge, 0.1× shrinkage", "reml_x10": "Ridge, 10× shrinkage", "reml_x100": "Ridge, 100× shrinkage",
         "ridge_pc20": "Ridge on 20 PCs", "rf": "Random forest", "gbm": "Gradient boosting", "knn10": "kNN (k = 10)", "knn30": "kNN (k = 30)",
         "mlp": "Multilayer perceptron", "gxe_gbm": "Covariate LightGBM", "rn_ridge": "Covariate RN ridge", "dl_g": "Within-env.-loss network",
         "R_STK_FW": "Stacking (forward history)"}
BENCH_LAB = {f"Benchmark: {v} − cell-level GBLUP": k for k, v in {
    "reml": "Two-stage GBLUP", "reml_x0.1": "Two-stage ridge, shrinkage x0.1", "reml_x10": "Two-stage ridge, shrinkage x10",
    "reml_x100": "Two-stage ridge, shrinkage x100", "ridge_pc20": "Ridge on 20 PCs", "rf": "Random forest", "gbm": "Gradient boosting",
    "knn10": "kNN, k = 10", "knn30": "kNN, k = 30", "mlp": "Multilayer perceptron", "gxe_gbm": "Covariate LightGBM",
    "rn_ridge": "Covariate reaction-norm ridge", "dl_g": "Within-environment-loss network", "R_STK_FW": "Stacking on the forward history"}.items()}


def row(path, **cond):
    d = pd.read_csv(R / path)
    for k, v in cond.items():
        d = d[d[k].astype(str) == str(v)]
    assert len(d) == 1, (path, cond, len(d))
    return d.iloc[0]


def jget(path, *keys):
    x = json.load(open(R / path))
    for k in keys:
        x = x[k]
    return x


def label(fig, ax, s, dx=-0.06, dy=0.02):
    p = ax.get_position()
    fig.text(p.x0 + dx, p.y1 + dy, s, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name, records):
    pd.DataFrame(records).to_csv(SRC / f"{name}.csv", index=False)
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None and name not in ("fig1",):
        require_matplotlib_panel_alignment(fig, json_out=str(OUT / f"{name}.alignment.json"), overlay_svg=str(OUT / f"{name}.alignment.svg"),
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    with Image.open(OUT / f"{name}.tiff") as im:
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    bg.save(OUT / f"{name}.tiff", compression="tiff_lzw", dpi=(600, 600))
    fig.savefig(OUT / f"{name}.preview.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def interval(ax, x, est, lo, hi, color, marker="o", ms=4, mfc=None, horizontal=False, lw=1.0, z=3):
    if horizontal:
        ax.errorbar(est, x, xerr=[[est - lo], [hi - est]], fmt=marker, color=color, mfc=mfc or color, ms=ms, capsize=1.6, elinewidth=lw, mew=0.8, zorder=z)
    else:
        ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt=marker, color=color, mfc=mfc or color, ms=ms, capsize=1.6, elinewidth=lw, mew=0.8, zorder=z)


def resbar(ax, x, res, half=0.3, horizontal=False):
    if horizontal:
        ax.add_patch(Rectangle((-res, x - half), 2 * res, 2 * half, color=GREY_L, lw=0, zorder=0))
    else:
        ax.plot([x - half, x + half], [res, res], color="#BBBBBB", lw=2.2, solid_capstyle="butt", zorder=0)


# ================================================================== Fig. 1 overview
def env_counts(d):
    """Environments per year of a dataset: from the benchmark cells (built from the public data), or, where these are not
    available (public release), from the stored source data of panel 1a."""
    f = R / f"benchmark/cells_{d}.parquet"
    if f.exists():
        return pd.read_parquet(f).groupby("year")["env"].nunique()
    s = pd.read_csv(SRC / "fig1.csv")
    s = s[(s["panel"] == "a") & (s["dataset"] == d)]
    return s.set_index("year")["environments"].astype(int)


def fig1():
    rec = []
    fig = plt.figure(figsize=(W2, 125 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.15, 1.0, 1.0], hspace=0.55, wspace=0.35)
    a = fig.add_subplot(gs[0, :])
    spl = json.load(open(R / "benchmark/splits.json"))
    for i, d in enumerate(DS):
        g = env_counts(d)
        tg = set(spl[d]["target_years"])
        for y, n in g.items():
            t = y in tg
            a.scatter(y, len(DS) - 1 - i, s=6 + 2.2 * n, color=BLUE if t else "white", edgecolor=BLUE if t else GREY, lw=0.6, zorder=3)
            rec.append(dict(panel="a", dataset=d, year=int(y), environments=int(n), target_year=t))
    a.set_yticks(range(len(DS))); a.set_yticklabels([DS_LAB[d].replace("\n", " ") for d in DS][::-1])
    a.set_xlim(1992, 2025.5); a.set_ylim(-0.7, len(DS) - 0.3); a.set_xlabel("Year")
    a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    for n in (5, 20, 50):
        a.scatter([], [], s=6 + 2.2 * n, color="white", edgecolor=GREY, lw=0.6, label=f"{n} environments")
    a.scatter([], [], s=30, color=BLUE, label="target year (forward benchmark)")
    a.legend(loc="lower left", ncol=4, bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, columnspacing=1.0)
    # b forward split
    b = fig.add_subplot(gs[1, 0]); b.axis("off"); b.set_xlim(0, 10); b.set_ylim(0, 10)
    for k in range(5):
        b.add_patch(Rectangle((0.3 + 1.55 * k, 6.2), 1.35, 1.6, color=GREY_L, lw=0))
        b.text(1.0 + 1.55 * k, 7.0, f"Y−{5 - k}" if k < 4 else "Y−1", ha="center", va="center", fontsize=6.5)
    b.add_patch(Rectangle((8.05, 6.2), 1.6, 1.6, color=BLUE, lw=0)); b.text(8.85, 7.0, "Y", color="white", ha="center", va="center", fontweight="bold")
    b.text(3.8, 8.5, "training: all years before Y", ha="center", fontsize=6.2)
    b.text(8.85, 9.4, "target\nyear", ha="center", va="center", fontsize=6.2)
    b.add_patch(FancyArrowPatch((4.0, 5.8), (8.6, 5.8), arrowstyle="-|>", mutation_scale=8, color=DARK, lw=0.8))
    b.text(0.3, 4.2, "Old lines: records before Y", fontsize=6.5, color=GREY)
    b.text(0.3, 3.0, "New lines: first tested in Y", fontsize=6.5, color=BLUE)
    b.text(0.3, 1.4, "No phenotype of Y or later enters\ntraining or tuning", fontsize=6.5)
    # c scenarios
    c = fig.add_subplot(gs[1, 1]); c.axis("off"); c.set_xlim(0, 10); c.set_ylim(0, 10)
    rng = np.random.default_rng(3)
    for ox, title, frac in ((0.2, "Sparse testing:\n25 % or 50 %\nphenotyped", 0.25), (5.3, "New lines:\nno target-year\nphenotypes", 0.0)):
        c.text(ox + 2.2, 9.3, title, ha="center", va="center", fontsize=6.0)
        for i in range(6):
            for j in range(5):
                obs = rng.random() < frac if frac else False
                c.add_patch(Rectangle((ox + 0.2 + 0.8 * j, 1.6 + 0.95 * i), 0.7, 0.85, facecolor=BLUE if obs else "white",
                                      edgecolor=GREY, lw=0.5, hatch=None if obs else "////"))
        c.text(ox + 2.2, 0.9, "environments →", ha="center", fontsize=6)
    c.text(0.0, 4.6, "lines", rotation=90, ha="center", va="center", fontsize=6)
    c.add_patch(Rectangle((0.4, -0.3), 0.5, 0.45, facecolor=BLUE, edgecolor=GREY, lw=0.5)); c.text(1.1, -0.08, "phenotyped, used for training", fontsize=6, va="center")
    c.add_patch(Rectangle((0.4, -1.3), 0.5, 0.45, facecolor="white", edgecolor=GREY, hatch="////", lw=0.5)); c.text(1.1, -1.08, "predicted and scored", fontsize=6, va="center")
    c.set_ylim(-1.6, 10.4)
    # d scoring flow
    d = fig.add_subplot(gs[1, 2]); d.axis("off"); d.set_xlim(0, 10); d.set_ylim(0, 10)
    steps = ["Spearman correlation\nwithin each environment", "Method − reference;\nyear effects (bootstrap)",
             "Pool years: DerSimonian–\nLaird (Hartung–Knapp)", "Resolved: Δ ≥ 3/√N and\n95 % interval excludes 0"]
    for k, s in enumerate(steps):
        y = 8.6 - 2.45 * k
        d.add_patch(FancyBboxPatch((0.0, y - 0.85), 10.0, 1.7, boxstyle="round,pad=0.05,rounding_size=0.25",
                                   facecolor="#EAF3FA" if k < 3 else "#E6F4EF", edgecolor=BLUE if k < 3 else GREEN, lw=0.7))
        d.text(5.0, y, s, ha="center", va="center", fontsize=5.9)
        if k < 3:
            d.add_patch(FancyArrowPatch((5.0, y - 0.9), (5.0, y - 1.55), arrowstyle="-|>", mutation_scale=7, color=DARK, lw=0.7))
    fig.canvas.draw()
    for ax_, s in ((a, "a"), (b, "b"), (c, "c"), (d, "d")):
        label(fig, ax_, s, dx=-0.04 if s != "a" else -0.11, dy=0.035 if s == "a" else 0.0)
    save(fig, "fig1", rec)


# ================================================================== Fig. 2 sparse testing decomposition
def fig2():
    P = pd.read_csv(R / "sparse_value/pooled.csv")
    get = lambda f, sc, c, rg: P[(P.fraction == f) & (P.scope == sc) & (P.contrast == c) & (P["range"] == rg)].iloc[0]
    rec = []
    fig = plt.figure(figsize=(W2, 120 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1.6, 1], hspace=0.6, wspace=0.3)
    a = fig.add_subplot(gs[0, :])
    cols = [("all", "Pooled")] + [(d, d) for d in DS]
    for i, (rg, lab) in enumerate(cols):
        for j, (f, c, colr, mk) in enumerate(((0.25, "m0-ref", BLUE, "o"), (0.25, "m1-m0", SKY, "o"), (0.5, "m0-ref", BLUE, "s"), (0.5, "m1-m0", SKY, "s"))):
            sub = P[(P.fraction == f) & (P.scope == "all") & (P.contrast == c) & (P["range"] == rg)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            x = i + (-0.27, -0.09, 0.09, 0.27)[j]
            resbar(a, x, r.resolution, half=0.07)
            interval(a, x, r.est, r.lo, r.hi, colr, marker=mk, ms=3.6)
            rec.append(dict(panel="a", range=rg, fraction=f, contrast=c, est=r.est, lo=r.lo, hi=r.hi, resolution=r.resolution, verdict=r.verdict))
    a.axhline(0, color=DARK, lw=0.6); a.set_xticks(range(len(cols)))
    a.set_xticklabels(["Pooled\n(46/36 years)"] + [DS_LAB[d] for d in DS]); a.set_xlim(-0.6, len(cols) - 0.4)
    a.set_ylabel("Δ within-environment Spearman")
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="Phenotypes themselves (with − without)"),
         plt.Line2D([], [], marker="o", ls="", color=SKY, ms=4, label="G×E increment (M×E − main effect)"),
         plt.Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=4, label="25 % phenotyped"),
         plt.Line2D([], [], marker="s", ls="", color=GREY, mfc="white", ms=4, label="50 % phenotyped"),
         plt.Line2D([], [], color="#BBBBBB", lw=2.2, label="resolution of each estimate")]
    a.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, handletextpad=0.3, columnspacing=0.9)
    # b new vs old lines
    b = fig.add_subplot(gs[1, 0])
    items = [(0.25, "m0-ref", "Phenotypes, 25 %"), (0.5, "m0-ref", "Phenotypes, 50 %"), (0.25, "m1-m0", "G×E increment, 25 %"), (0.5, "m1-m0", "G×E increment, 50 %")]
    for i, (f, c, lab) in enumerate(items):
        y = len(items) - 1 - i
        for sc, colr, dy in (("old", GREY, -0.13), ("new", BLUE if c == "m0-ref" else SKY, 0.13)):
            r = get(f, sc, c, "all" if not (sc == "old" and f == 0.5 and c == "m0-ref") else "all")
            interval(b, y + dy, r.est, r.lo, r.hi, colr, ms=3.6, horizontal=True)
            rec.append(dict(panel="b", fraction=f, contrast=c, lines=sc, est=r.est, lo=r.lo, hi=r.hi, resolution=r.resolution, verdict=r.verdict))
    b.axvline(0, color=DARK, lw=0.6); b.set_yticks(range(len(items))); b.set_yticklabels([t for *_, t in items][::-1])
    b.set_xlabel("Δ within-environment Spearman"); b.tick_params(axis="y", length=0)
    b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=3.6, label="new lines"),
                      plt.Line2D([], [], marker="o", ls="", color=GREY, ms=3.6, label="old lines")], loc="lower right")
    # c G x E increment on common years
    c_ = fig.add_subplot(gs[1, 1])
    cc = [(0.25, "all", "25 %\n46 y"), (0.25, "common_years", "25 %\n36 y*"), (0.5, "all", "50 %\n36 y")]
    for i, (f, rg, lab) in enumerate(cc):
        r = get(f, "all", "m1-m0", rg)
        resbar(c_, i, r.resolution, half=0.25)
        interval(c_, i, r.est, r.lo, r.hi, SKY, ms=4)
        rec.append(dict(panel="c", fraction=f, range=rg, contrast="m1-m0", est=r.est, lo=r.lo, hi=r.hi, resolution=r.resolution, verdict=r.verdict))
    c_.axhline(0, color=DARK, lw=0.6); c_.set_xticks(range(3)); c_.set_xticklabels([t for *_, t in cc]); c_.set_xlim(-0.6, 2.6)
    c_.set_ylabel("G×E increment"); c_.set_xlabel("*target years analysed at both fractions", fontsize=6)
    fig.canvas.draw()
    label(fig, a, "a", dx=-0.07, dy=0.06); label(fig, b, "b", dx=-0.2); label(fig, c_, "c", dx=-0.09)
    save(fig, "fig2", rec)


# ================================================================== Fig. 4 new lines: methods x datasets
def fig4():
    PD = pd.read_csv(R / "revision_tcj/per_dataset.csv")
    PD = PD[PD["analysis"].str.startswith("Benchmark:")].copy()
    PD["item"] = PD["analysis"].map(BENCH_LAB)
    ye = pd.read_csv(R / "summary48/year_effects.csv"); ye = ye[ye["metric"] == "spearman"]
    order = ["reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp", "gxe_gbm", "rn_ridge", "dl_g", "R_STK_FW"]
    rec = []
    pooled = {}
    for k in order:
        g = ye[ye["item"] == k][["dataset", "d", "se"]].reset_index(drop=True)
        h = hartung_knapp(floor_se(g))
        pooled[k] = h
    fig = plt.figure(figsize=(W2, 150 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1.3, 1], hspace=0.62, wspace=0.75)
    a = fig.add_subplot(gs[0, 0])
    M = np.full((len(order), len(DS)), np.nan); V = np.empty((len(order), len(DS)), dtype=object)
    for i, k in enumerate(order):
        for j, d in enumerate(DS):
            s = PD[(PD["item"] == k) & (PD["dataset"] == d)]
            if len(s):
                M[i, j] = s.iloc[0]["est"]; V[i, j] = s.iloc[0]["verdict"]
                rec.append(dict(panel="a", item=k, dataset=d, est=s.iloc[0]["est"], lo=s.iloc[0]["lo"], hi=s.iloc[0]["hi"],
                                resolution=s.iloc[0]["resolution"], verdict=V[i, j]))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("ok", [VERM, "#F7F7F7", BLUE])
    im = a.imshow(np.clip(M, -0.15, 0.15), cmap=cmap, vmin=-0.15, vmax=0.15, aspect="auto")
    for i in range(len(order)):
        for j in range(len(DS)):
            if np.isnan(M[i, j]):
                a.text(j, i, "–", ha="center", va="center", fontsize=6, color=GREY); continue
            res = V[i, j] in ("resolved loss", "resolved gain")
            a.text(j, i, f"{M[i, j]:+.3f}".replace("-", "−"), ha="center", va="center", fontsize=5.6,
                   fontweight="bold" if res else "normal", color=DARK)
            if res:
                a.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, edgecolor=DARK, lw=0.9))
    a.set_xticks(range(len(DS))); a.set_xticklabels([DS_LAB[d] for d in DS], fontsize=6.3)
    a.set_yticks(range(len(order))); a.set_yticklabels([NAMES[k] for k in order], fontsize=6.5)
    a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]
    cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.045, pad=0.13, aspect=35)
    cb.set_label("Δ vs cell-level GBLUP (clipped at ±0.15)", fontsize=6.3); cb.ax.tick_params(labelsize=6)
    a.set_title("boxed: resolved against the dataset's own 3/√N", fontsize=6.3, loc="left")
    # b pooled DL vs HK
    b = fig.add_subplot(gs[0, 1])
    ks = sorted(order, key=lambda k: pooled[k]["est"])
    for i, k in enumerate(ks):
        h = pooled[k]
        resbar(b, i, 0.0087, half=0.35, horizontal=True)
        b.plot([h["hk_lo"], h["hk_hi"]], [i - 0.17, i - 0.17], color=ORANGE, lw=1.0)
        interval(b, i + 0.08, h["est"], h["lo"], h["hi"], GREY if h["est"] < 0 else BLUE, ms=3.2, horizontal=True)
        rec.append(dict(panel="b", item=k, est=h["est"], dl_lo=h["lo"], dl_hi=h["hi"], hk_lo=h["hk_lo"], hk_hi=h["hk_hi"], k=h["k"]))
    b.axvline(0, color=DARK, lw=0.6); b.set_yticks(range(len(ks))); b.set_yticklabels([NAMES[k] for k in ks], fontsize=6.3)
    b.set_xlabel("Pooled Δ vs cell-level GBLUP"); b.tick_params(axis="y", length=0); b.set_ylim(-0.7, len(ks) - 0.3)
    b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GREY, ms=3.2, label="DerSimonian–Laird 95 %"),
                      plt.Line2D([], [], color=ORANGE, lw=1.0, label="Hartung–Knapp 95 %"),
                      plt.Line2D([], [], color=GREY_L, lw=4, label="±3/√N")], loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=6, columnspacing=0.8)
    # c within vs across environments
    c_ = fig.add_subplot(gs[1, :])
    T = pd.read_csv(R / "tpe_metric/pooled.csv"); T = T[T["range"] == "all"].set_index("item")
    W = pd.read_csv(R / "summary48/pooled.csv"); W = W[(W["metric"] == "spearman") & (W["scope"] == "all48")].set_index("item")
    ks2 = [k for k in order if k in T.index]
    for i, k in enumerate(ks2):
        w, t = W.loc[k], T.loc[k]
        interval(c_, i - 0.12, w["est"], w["lo"], w["hi"], GREY, ms=3.4)
        interval(c_, i + 0.12, t["est"], t["lo"], t["hi"], PURPLE, marker="D", ms=3.2)
        rec.append(dict(panel="c", item=k, within_est=w["est"], within_lo=w["lo"], within_hi=w["hi"], across_est=t["est"], across_lo=t["lo"],
                        across_hi=t["hi"], across_verdict=t["verdict"]))
    c_.axhline(0, color=DARK, lw=0.6); c_.set_xticks(range(len(ks2))); c_.set_xticklabels([NAMES[k] for k in ks2], fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
    c_.set_xlim(-0.6, len(ks2) - 0.4); c_.set_ylabel("Pooled Δ vs cell-level GBLUP")
    c_.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GREY, ms=3.4, label="within environments (primary)"),
                       plt.Line2D([], [], marker="D", ls="", color=PURPLE, ms=3.2, label="line means across environments (post hoc)")],
              loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    fig.canvas.draw()
    label(fig, a, "a", dx=-0.17, dy=0.03); label(fig, b, "b", dx=-0.2, dy=0.03); label(fig, c_, "c", dx=-0.07)
    save(fig, "fig4", rec)


# ================================================================== Fig. 3 CLAC
def fig3():
    rec = []
    fig = plt.figure(figsize=(W2, 105 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.95)
    a = fig.add_subplot(gs[0, 0])
    y = pd.read_csv(R / "reanalysis/year_effects.csv"); y = y[(y.method == "clac") & (y.metric == "sp")].sort_values("target")
    v = jget("reanalysis/verdict.json", "verdict", "clac")
    items = [(str(r.target), r.d, r.d - 1.96 * r.se, r.d + 1.96 * r.se) for r in y.itertuples()] + [("Pooled", v["est"], v["ci"][0], v["ci"][1])]
    for i, (nm, e, lo, hi) in enumerate(items):
        yy = len(items) - 1 - i
        interval(a, yy, e, lo, hi, GREEN, marker="D" if nm == "Pooled" else "o", ms=4 if nm == "Pooled" else 3.4, horizontal=True)
        rec.append(dict(panel="a", item=nm, est=e, lo=lo, hi=hi))
    a.add_patch(Rectangle((-v["resolution"], -0.4), 2 * v["resolution"], 0.8, color=GREY_L, lw=0, zorder=0))
    a.axvline(0, color=DARK, lw=0.6); a.set_yticks(range(len(items))); a.set_yticklabels([i[0] for i in items][::-1])
    a.set_xlabel("CLAC − cell-level GBLUP"); a.tick_params(axis="y", length=0)
    b = fig.add_subplot(gs[0, 1])
    H = pd.read_csv(R / "revision_tcj/hk_sensitivity.csv").set_index("contrast")
    abl = [("loss: linear kernel", "Linear kernel"), ("loss: single main effect", "Single main effect"), ("loss: equal weights", "Equal env. weights"),
           ("loss: no phenotype cleaning", "No phenotype cleaning"), ("loss: no spatial adjustment", "No spatial adjustment")]
    for i, (k, nm) in enumerate(abl):
        h = H.loc[f"CLAC ablation, {k}"]; yy = len(abl) - 1 - i
        resbar(b, yy, h["resolution"], half=0.35, horizontal=True)
        b.plot([h["hk_lo"], h["hk_hi"]], [yy - 0.18, yy - 0.18], color=ORANGE, lw=1.0)
        interval(b, yy + 0.08, h["est"], h["dl_lo"], h["dl_hi"], GREEN if h["dl_verdict"] == "resolved gain" else GREY, ms=3.4, horizontal=True)
        rec.append(dict(panel="b", item=nm, level=h["level"], est=h["est"], dl_lo=h["dl_lo"], dl_hi=h["dl_hi"], hk_lo=h["hk_lo"], hk_hi=h["hk_hi"],
                        dl_verdict=h["dl_verdict"], hk_verdict=h["hk_verdict"]))
    b.axvline(0, color=DARK, lw=0.6); b.set_yticks(range(len(abl))); b.set_yticklabels([n for _, n in abl][::-1])
    b.set_xlabel("Loss in ranking when removed"); b.tick_params(axis="y", length=0)
    b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GREY, ms=3.4, label="Holm-adjusted (DL)"),
                      plt.Line2D([], [], color=ORANGE, lw=1.0, label="Hartung–Knapp")], loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2, fontsize=6)
    c_ = fig.add_subplot(gs[0, 2])
    T = pd.read_csv(R / "transfer_arc/pooled.csv"); T = T[(T.metric == "spearman") & (T.scope == "all")].set_index("range")
    rr = ["all48"] + DS
    for i, k in enumerate(rr):
        t = T.loc[k]; yy = len(rr) - 1 - i
        resbar(c_, yy, min(t["resolution"], 0.06), half=0.35, horizontal=True)
        if t["resolution"] > 0.06:
            c_.text(0.058, yy + 0.38, f"3/√N = {t['resolution']:.2f}", ha="right", va="bottom", fontsize=5.5, color=GREY)
        interval(c_, yy, t["est"], t["lo"], t["hi"], GREEN if k == "G2F" else GREY, marker="D" if k == "all48" else "o", ms=3.4, horizontal=True)
        rec.append(dict(panel="c", range=k, est=t["est"], lo=t["lo"], hi=t["hi"], resolution=t["resolution"]))
    c_.axvline(0, color=DARK, lw=0.6); c_.set_yticks(range(len(rr))); c_.set_yticklabels(["Pooled (48 y)"] + DS[:])[::-1] if False else None
    c_.set_yticklabels((["Pooled (48 y)"] + DS)[::-1]); c_.set_xlim(-0.06, 0.06)
    c_.set_xlabel("Arc-cosine − linear"); c_.tick_params(axis="y", length=0)
    fig.canvas.draw()
    label(fig, a, "a", dx=-0.09); label(fig, b, "b", dx=-0.16); label(fig, c_, "c", dx=-0.12)
    save(fig, "fig3", rec)


# ================================================================== Fig. 5 other information, networks
def fig5():
    rec = []
    ol = jget("headroom_oldlines/verdict.json", "decisions")
    ol10 = row("headroom_oldlines/pooled.csv", contrast="ol1-ol0", metric="sp", range="all")
    sx = jget("headroom_secondary/verdict.json", "H1")
    ia = jget("headroom_ia_ib/verdict.json")
    s2 = jget("headroom_sparse2/verdict.json", "decisions"); m2 = jget("headroom_sparse/verdict.json", "main_m2_m1")
    items = [("vs cell-level GBLUP", None),
             ("Own-history residual (old lines)", (ol10.est, ol10.lo, ol10.hi, ol10.resolution)),
             ("Line × location history (old lines)", (ol["H2"]["est"], *ol["H2"]["ci"], ol["H2"]["resolution"])),
             ("Same-season silking dates (G2F)", (sx["est"], sx["lo"], sx["hi"], jget("headroom_secondary/verdict.json", "resolution"))),
             ("Marker × location effects (43 y)", (ia["I-B"]["loc43"]["est"], ia["I-B"]["loc43"]["lo"], ia["I-B"]["loc43"]["hi"], ia["I-B"]["loc43"]["resolution"])),
             ("Heteroscedastic GBLUP (48 y)", (ia["I-A"]["all48"]["est"], ia["I-A"]["all48"]["lo"], ia["I-A"]["all48"]["hi"], ia["I-A"]["all48"]["resolution"])),
             ("vs M×E GBLUP (25 % phenotyped)", None),
             ("Learned env. correlation", (m2["est"], m2["lo"], m2["hi"], row("headroom_sparse/pooled.csv", fraction=0.25, contrast="m1-m0", metric="sp", range="all48").resolution)),
             ("Stacking with LightGBM", (s2["stk"]["est"], *s2["stk"]["ci"], s2["stk"]["resolution"])),
             ("Residual network", (s2["dlres"]["est"], *s2["dlres"]["ci"], s2["dlres"]["resolution"])),
             ("Covariate env. correlation", (s2["ecmxe"]["est"], *s2["ecmxe"]["ci"], s2["ecmxe"]["resolution"]))]
    fig = plt.figure(figsize=(W2, 95 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.85)
    a = fig.add_subplot(gs[0, 0])
    n = len(items)
    for i, (nm, v) in enumerate(items):
        yy = n - 1 - i
        if v is None:
            a.text(-0.02, yy, nm, fontsize=6.5, fontweight="bold", va="center", ha="right", transform=a.get_yaxis_transform()); continue
        e, lo, hi, res = v
        resbar(a, yy, res, half=0.32, horizontal=True)
        verdict = "resolved" if (lo > 0 and e >= res) else ("detectable" if lo > 0 or hi < 0 else "tied")
        interval(a, yy, e, lo, hi, BLUE if verdict == "resolved" else (SKY if verdict == "detectable" else GREY), ms=3.4, horizontal=True)
        rec.append(dict(panel="a", item=nm, est=e, lo=lo, hi=hi, resolution=res, verdict=verdict))
    a.axvline(0, color=DARK, lw=0.6); a.set_yticks([n - 1 - i for i, (nm, v) in enumerate(items) if v is not None])
    a.set_yticklabels([nm for nm, v in items if v is not None]); a.tick_params(axis="y", length=0); a.set_xlim(-0.062, 0.04)
    a.set_xlabel("Δ within-environment Spearman")
    a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=3.4, label="resolved"),
                      plt.Line2D([], [], marker="o", ls="", color=SKY, ms=3.4, label="detectable, below 3/√N"),
                      plt.Line2D([], [], marker="o", ls="", color=GREY, ms=3.4, label="tied")], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=6)
    b = fig.add_subplot(gs[0, 1])
    pub = []
    for sc, f in (("2024", "tables/g2f_gate2_F2024m_vs_B1r_gblup_reml_main.csv"), ("2022", "tables/g2f_gate2_F2022m_vs_B1r_gblup_reml_main.csv")):
        r = row(f, metric="spearman", method="GEFormer_fair")
        cells = row("e1/geformer_rerun_vs_gate2.csv", scenario=f"F{sc}m", protocol="own", seed=147).cells
        pub.append((f"GEFormer, {sc}", r.delta, r.ci2_lo, r.ci2_hi, 3 / cells ** 0.5))
    for sc in ("2024", "2022"):
        r = row("a1/summary.csv", scenario=f"F{sc}m", contrast="C_fair", metric="spearman")
        pub.append((f"GE-BiFormer, {sc}", r.delta, r.ci2_lo, r.ci2_hi, r.resolution))
    lc = jget("reanalysis/verdict.json", "verdict", "lc_real")
    pub.append(("Lopez-Cruz RN, 2022", lc["est"], lc["ci"][0], lc["ci"][1], lc["resolution"]))
    for i, (nm, e, lo, hi, res) in enumerate(pub):
        yy = len(pub) - 1 - i
        resbar(b, yy, res, half=0.32, horizontal=True)
        interval(b, yy, e, lo, hi, BLUE if (lo > 0 and e >= res) else (VERM if (hi < 0 and -e >= res) else (SKY if lo > 0 else GREY)), ms=3.4, horizontal=True)
        rec.append(dict(panel="b", item=nm, est=e, lo=lo, hi=hi, resolution=res))
    b.axvline(0, color=DARK, lw=0.6); b.set_yticks(range(len(pub))); b.set_yticklabels([p[0] for p in pub][::-1]); b.tick_params(axis="y", length=0)
    b.set_xlabel("Δ vs GBLUP (G2F)")
    fig.canvas.draw()
    label(fig, a, "a", dx=-0.3); label(fig, b, "b", dx=-0.2)
    save(fig, "fig5", rec)


# ================================================================== Fig. 6 exploratory reaction norms
def figS1():
    rec = []
    rs = "rn_paper/ge_mean_shift_decomposition.json"
    cells = [("cv0_strict", "Tested lines, new env."), ("cohort_matched_forward", "Same cohort, future years"), ("cv00", "New lines, new env."),
             ("forward_year", "Forward years, mostly new lines"), ("forward_year_v2", "Same, genomic reference")]
    fig = plt.figure(figsize=(W2, 75 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.75)
    a = fig.add_subplot(gs[0, 0])
    for i, (key, nm) in enumerate(cells):
        s = jget(rs, "cells", key, "vs_additive_g_e", "spearman"); n = jget(rs, "cells", key, "n_prediction_rows"); res = 3 / n ** 0.5
        yy = len(cells) - 1 - i; resbar(a, yy, res, half=0.32, horizontal=True)
        interval(a, yy, s["estimate"], s["ci_low"], s["ci_high"], BLUE if s["ci_low"] > 0 else (VERM if s["ci_high"] < 0 else GREY), ms=3.4, horizontal=True)
        rec.append(dict(panel="a", item=nm, est=s["estimate"], lo=s["ci_low"], hi=s["ci_high"], resolution=res,
                        environments=jget(rs, "cells", key, "n_environments")))
    a.axvline(0, color=DARK, lw=0.6); a.set_yticks(range(len(cells)))
    a.set_yticklabels([f"{nm} ({jget(rs, 'cells', k, 'n_environments')})" for k, nm in cells][::-1]); a.tick_params(axis="y", length=0)
    a.set_xlabel("Kernel-gated RN − additive reference")
    b = fig.add_subplot(gs[0, 1])
    rp = "rn_paper/ge_mean_primary_evidence.json"
    sts = [("observed", "Observed weather"), ("inseason", "Forecast, 1 June"), ("preseason", "Forecast, 1 March")]
    for i, (st, nm) in enumerate(sts):
        n = jget(rp, "states", st, "n_genotype_environment_units"); res = 3 / n ** 0.5
        f = jget(rp, "states", st, "fw_vs_additive", "spearman"); k = jget(rp, "states", st, "kernel_gated_vs_fw", "spearman")
        yy = len(sts) - 1 - i; resbar(b, yy, res, half=0.35, horizontal=True)
        interval(b, yy + 0.13, f["estimate"], f["ci_low"], f["ci_high"], BLUE, ms=3.4, horizontal=True)
        interval(b, yy - 0.13, k["estimate"], k["ci_low"], k["ci_high"], GREY, ms=3.4, mfc="white", horizontal=True)
        rec += [dict(panel="b", item=f"FW vs additive, {nm}", est=f["estimate"], lo=f["ci_low"], hi=f["ci_high"], resolution=res),
                dict(panel="b", item=f"Kernel-gated vs FW, {nm}", est=k["estimate"], lo=k["ci_low"], hi=k["ci_high"], resolution=res)]
    b.axvline(0, color=DARK, lw=0.6); b.set_yticks(range(len(sts))); b.set_yticklabels([s[1] for s in sts][::-1]); b.tick_params(axis="y", length=0)
    b.set_xlabel("Δ within-env. Spearman (tested lines)")
    b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=3.4, label="FW − additive"),
                      plt.Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=3.4, label="kernel-gated − FW")], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6)
    fig.canvas.draw()
    label(fig, a, "a", dx=-0.27); label(fig, b, "b", dx=-0.17)
    save(fig, "figS1", rec)


# ================================================================== Fig. 7 evaluation practices
def fig6():
    rec = []
    r7 = json.load(open(R / "c_paper/workspace/handoff/r7_h1h2.json"))[0]["structured_output"]["key_numbers"]
    r9 = jget("c_paper/amax_runs/A/final/R9_A_results.json", "H9a", "per_setting", "S4_G2F_CV00")
    r10 = "c_paper/amax_runs/trackA/final/R10_A_results.json"
    ext = jget(r10, "G1_enhancements_external", "contrasts", "D_noEC_minus_MG_2k", "EXT3_pooled")
    tun = jget(r10, "G4_tuning_audit", "MG_2k", "P2_minus_P3", "EXT3_pooled")
    fig = plt.figure(figsize=(W2, 120 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.65, wspace=0.55)
    a = fig.add_subplot(gs[0, 0])
    sets = [("S1_G2F_LOYO", "G2F,\nyear out"), ("S2_G2F_LOLO", "G2F,\nlocation out"), ("S3_BRIWECS_LOYO", "Wheat,\nyear out")]
    for i, (k, nm) in enumerate(sets):
        e, lo, hi = r7[f"{k}|H1_optimism"], r7[f"{k}|H1_ci_low"], r7[f"{k}|H1_ci_high"]; w = r7[f"{k}|C2_within_env_leaky_minus_nested"]
        interval(a, i - 0.12, e, lo, hi, VERM, ms=3.4); a.plot(i + 0.12, w, "s", color=BLUE, ms=3.4)
        rec += [dict(panel="a", item=nm.replace("\n", " "), quantity="environment-mean r, leaky − nested", est=e, lo=lo, hi=hi),
                dict(panel="a", item=nm.replace("\n", " "), quantity="within-environment r, leaky − nested", est=w)]
    a.axhline(0, color=DARK, lw=0.6); a.set_xticks(range(3)); a.set_xticklabels([s[1] for s in sets]); a.set_ylabel("Leaky − nested window search")
    a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=VERM, ms=3.4, label="environment means"),
                      plt.Line2D([], [], marker="s", ls="", color=BLUE, ms=3.4, label="within environments")], loc="upper left", fontsize=6)
    b = fig.add_subplot(gs[0, 1])
    nS4, nEXT = 82082, 24995
    for i, (nm, e, lo, hi, res) in enumerate([("Internal split", r9["mean"], r9["ci_low"], r9["ci_high"], 3 / nS4 ** 0.5),
                                              ("External years", ext["mean"], ext["ci_low"], ext["ci_high"], 3 / nEXT ** 0.5)]):
        resbar(b, i, res, half=0.25); interval(b, i, e, lo, hi, BLUE if i == 0 else GREY, ms=3.6)
        rec.append(dict(panel="b", item=nm, est=e, lo=lo, hi=hi, resolution=res))
    b.annotate("", xy=(1, ext["mean"]), xytext=(0, r9["mean"]), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7, linestyle="--"))
    b.axhline(0, color=DARK, lw=0.6); b.set_xticks(range(2)); b.set_xticklabels(["Internal\nsplit", "External\nyears"]); b.set_xlim(-0.6, 1.6)
    b.set_ylabel("Non-linear stage − ridge\n(within-env. Pearson)")
    c_ = fig.add_subplot(gs[0, 2])
    orc = row("summary48/pooled.csv", item="Oracle", metric="spearman", scope="all48")
    for i, (q, nm) in enumerate((("naive", "Same\nenvironments"), ("cross_fit", "Different\nhalves"))):
        r = orc if q == "naive" else row("ideas_diag/oracle_pooled.csv", metric="spearman", scope="all48", quantity=q)
        resbar(c_, i, 0.0087, half=0.25); interval(c_, i, r.est, r.lo, r.hi, BLUE if i == 0 else GREY, ms=3.6)
        rec.append(dict(panel="c", item=nm.replace("\n", " "), est=r.est, lo=r.lo, hi=r.hi, resolution=0.0087))
    c_.annotate("", xy=(1, row("ideas_diag/oracle_pooled.csv", metric="spearman", scope="all48", quantity="cross_fit").est),
                xytext=(0, orc.est),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7, linestyle="--"))
    c_.axhline(0, color=DARK, lw=0.6); c_.set_xticks(range(2)); c_.set_xticklabels(["Same\nenvironments", "Different\nhalves"]); c_.set_xlim(-0.6, 1.6)
    c_.set_ylabel("Best method in hindsight\n− cell-level GBLUP")
    d = fig.add_subplot(gs[1, :2])
    items = [("Ridge λ tuned on the test year\n(within-env. Pearson)", tun["mean"], tun["ci_low"], tun["ci_high"], 3 / nEXT ** 0.5)]
    for sc in ("2024", "2022"):
        r = row(f"tables/g2f_gate2_F{sc}m_vs_GEFormer_fair_main.csv", metric="spearman", method="GEFormer_own")
        cells = row("e1/geformer_rerun_vs_gate2.csv", scenario=f"F{sc}m", protocol="own", seed=147).cells
        items.append((f"GEFormer, own − validation-year choices, {sc}", r.delta, r.ci2_lo, r.ci2_hi, 3 / cells ** 0.5))
    for sc in ("2024", "2022"):
        r = row("a1/summary.csv", scenario=f"F{sc}m", contrast="O", metric="spearman")
        items.append((f"GE-BiFormer, own − validation-year choices, {sc}", r.delta, r.ci2_lo, r.ci2_hi, r.resolution))
    for i, (nm, e, lo, hi, res) in enumerate(items):
        yy = len(items) - 1 - i; resbar(d, yy, res, half=0.32, horizontal=True)
        interval(d, yy, e, lo, hi, VERM if e < 0 else BLUE, ms=3.4, horizontal=True)
        rec.append(dict(panel="d", item=nm.replace("\n", " "), est=e, lo=lo, hi=hi, resolution=res))
    d.axvline(0, color=DARK, lw=0.6); d.set_yticks(range(len(items))); d.set_yticklabels([i[0] for i in items][::-1]); d.tick_params(axis="y", length=0)
    d.set_xlabel("Shift in the reported value when choices use the test year")
    e_ = fig.add_subplot(gs[1, 2])
    cls = [("Choose on test data", 6, VERM), ("Train/validation\nonly", 5, BLUE), ("Not auditable", 3, GREY), ("No public code", 3, GREY_L)]
    e_.barh(range(len(cls))[::-1], [c[1] for c in cls], color=[c[2] for c in cls], height=0.6)
    for i, c in enumerate(cls):
        e_.text(c[1] + 0.2, len(cls) - 1 - i, str(c[1]), va="center", fontsize=6.5)
        rec.append(dict(panel="e", item=c[0].replace("\n", " "), count=c[1]))
    e_.set_yticks(range(len(cls))[::-1]); e_.set_yticklabels([c[0] for c in cls]); e_.set_xlabel("Published models (of 17)"); e_.tick_params(axis="y", length=0)
    e_.set_xlim(0, 8)
    fig.canvas.draw()
    label(fig, a, "a", dx=-0.075); label(fig, b, "b", dx=-0.085); label(fig, c_, "c", dx=-0.085)
    label(fig, d, "d", dx=-0.3); label(fig, e_, "e", dx=-0.14)
    save(fig, "fig6", rec)


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "figS1"]
    for f in which:
        globals()[f](); print("done", f)
