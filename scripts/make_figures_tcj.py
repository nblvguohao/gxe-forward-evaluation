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
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03)   # any raster element (heat map) at 600 dpi
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
    """Fig. 1 is the authors' study-design schematic (figures/tcj/fig1_schematic_v5/, drawn in BioRender and PowerPoint). Its panel a
    embeds the timeline drawn here from the benchmark (results/benchmark/splits.json and the benchmark cells): fig1a_85mm, 85 mm wide.
    This function draws that timeline, writes the values drawn to results/figdata_tcj/fig1.csv, and exports the schematic as Fig. 1."""
    rec = []
    fig, a = plt.subplots(figsize=(85 * MM, 52 * MM))
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
        a.scatter([], [], s=6 + 2.2 * n, color="white", edgecolor=GREY, lw=0.6, label=f"{n} env.")
    a.scatter([], [], s=30, color=BLUE, label="target year")
    a.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.0)
    pd.DataFrame(rec).to_csv(SRC / "fig1.csv", index=False)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"fig1a_85mm.{ext}", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig1a_85mm_600dpi.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    # the schematic: vector PDF and SVG as delivered; TIFF (600 dpi, RGB, LZW) and the docx preview from its 600 dpi PNG
    import shutil
    from PIL import Image
    S = OUT / "fig1_schematic_v5"
    shutil.copyfile(S / "schematic_v5_overview.pdf", OUT / "fig1.pdf")
    shutil.copyfile(S / "schematic_v5_overview.svg", OUT / "fig1.svg")
    with Image.open(S / "schematic_v5_overview_600dpi.png") as im:
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    bg.save(OUT / "fig1.tiff", compression="tiff_lzw", dpi=(600, 600))
    bg.resize((bg.width // 2, bg.height // 2), Image.LANCZOS).save(OUT / "fig1.preview.png", dpi=(300, 300))










# ================================================================== Fig. S1 exploratory reaction norms
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




# ================================================================== style shared by Figs. 2-6 (matched to the Fig. 1 schematic)
NAVY, TAB_FILL, TAB_EDGE = "#1F3864", "#DAE3F3", "#8EA9DB"
HKC = "#4D4D4D"   # Hartung-Knapp intervals: dark grey dashed, not a verdict colour
TINT = {"blue": "#EEF2F9", "green": "#EFF6EB", "yellow": "#FDF8E8", "pink": "#FCF0F1", "grey": "#F4F5F7"}
VERM_L = "#EBA98A"
VSTYLE = {"gain": dict(color=NAVY, mfc=NAVY), "det+": dict(color=SKY, mfc=SKY), "tied": dict(color=GREY, mfc="white"),
          "det-": dict(color=VERM_L, mfc=VERM_L), "loss": dict(color=VERM, mfc=VERM)}
VLABEL = {"gain": "resolved gain", "det+": "detectable gain, below 3/√N", "tied": "tied", "det-": "detectable loss, below 3/√N", "loss": "resolved loss"}


def vclass(est, lo, hi, res, verdict=None):
    """Verdict class; uses the stored verdict string where the result file has one, otherwise the rule of Section 2.1."""
    if verdict is not None and isinstance(verdict, str):
        v = verdict.lower()
        if v.startswith("resolved gain"): return "gain"
        if v.startswith("resolved loss"): return "loss"
        if v.startswith("detectable"): return "det+" if est > 0 else "det-"
        if v.startswith("tied"): return "tied"
    if lo > 0 and est >= res: return "gain"
    if hi < 0 and -est >= res: return "loss"
    if lo > 0: return "det+"
    if hi < 0: return "det-"
    return "tied"


def vpoint(ax, y, est, lo, hi, cls, marker="o", ms=3.4, horizontal=True, z=3):
    st = VSTYLE[cls]
    kw = dict(fmt=marker, color=st["color"], mfc=st["mfc"], mec=st["color"], ms=ms, capsize=1.6, elinewidth=0.9, mew=0.8, zorder=z)
    if horizontal:
        ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], **kw)
    else:
        ax.errorbar(y, est, yerr=[[est - lo], [hi - est]], **kw)


def tab(fig, ax, letter, title, dx=0.0, dy=0.012):
    """Panel tab in the style of Fig. 1: letter and short title in a rounded light-blue box above the top-left corner of the axes."""
    p = ax.get_position()
    t = fig.text(p.x0 + dx, p.y1 + dy, f"$\\bf{{{letter}}}$  {title}", fontsize=7.5, color=NAVY, va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.25", fc=TAB_FILL, ec=TAB_EDGE, lw=0.5))
    return t


def verdict_legend(fig, classes=("gain", "det+", "tied", "det-", "loss"), y=0.0, extra=()):
    h = [plt.Line2D([], [], marker="o", ls="", color=VSTYLE[c]["color"], mfc=VSTYLE[c]["mfc"], ms=3.6, label=VLABEL[c]) for c in classes]
    h += [mpl.patches.Patch(color=GREY_L, label="±3/√N")] + list(extra)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(h), fontsize=6.3, handletextpad=0.3, columnspacing=1.0, frameon=False)


def note(ax, x, y, s, **kw):
    kw = {**dict(fontsize=6.5, color=NAVY, fontweight="bold", va="center"), **kw}
    return ax.text(x, y, s, **kw)


def fmt(x, d=3):
    return f"{x:+.{d}f}".replace("-", "−")


# ================================================================== Fig. 2 phenotypes of part of the target year
def fig2():
    P = pd.read_csv(R / "sparse_value/pooled.csv")
    get = lambda f, sc, c, rg: P[(P.fraction == f) & (P.scope == sc) & (P.contrast == c) & (P["range"] == rg)].iloc[0]
    COL = {"m0-ref": NAVY, "m1-m0": SKY}
    rec = []
    fig = plt.figure(figsize=(W2, 128 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], width_ratios=[0.78, 1.2, 1.0], hspace=0.62, wspace=0.42,
                          left=0.08, right=0.99, top=0.93, bottom=0.17)
    # a decomposition (headline)
    a = fig.add_subplot(gs[0, 0])
    for i, f in enumerate((0.25, 0.5)):
        p0, g0 = get(f, "all", "m0-ref", "all"), get(f, "all", "m1-m0", "all")
        for dx, r, colr in ((-0.19, p0, NAVY), (0.19, g0, SKY)):
            a.bar(i + dx, r.est, width=0.34, color=colr, zorder=2)
            a.errorbar(i + dx, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="none", ecolor=DARK, elinewidth=0.8, capsize=1.6, zorder=3)
            a.text(i + dx, r.hi + 0.004, fmt(r.est), ha="center", va="bottom", fontsize=5.8, color=DARK)
        ratio = p0.est / g0.est
        a.text(i, max(p0.hi, g0.hi) + 0.03, f"{ratio:.1f}×", ha="center", va="bottom", fontsize=8, color=NAVY, fontweight="bold")
        rec.append(dict(panel="a", fraction=f, phenotypes_est=p0.est, phenotypes_lo=p0.lo, phenotypes_hi=p0.hi, gxe_est=g0.est, gxe_lo=g0.lo,
                        gxe_hi=g0.hi, ratio=ratio))
    a.set_xticks([0, 1]); a.set_xticklabels(["25 %", "50 %"]); a.set_xlim(-0.5, 1.5); a.set_ylim(0, 0.19)
    a.set_xlabel("Target-year cells phenotyped"); a.set_ylabel("Δ within-environment Spearman")
    # b by dataset
    b = fig.add_subplot(gs[0, 1:])
    cols = [("all", "Pooled")] + [(d, d) for d in DS]
    for i, (rg, lab) in enumerate(cols):
        for j, (f, c, mk) in enumerate(((0.25, "m0-ref", "o"), (0.25, "m1-m0", "o"), (0.5, "m0-ref", "s"), (0.5, "m1-m0", "s"))):
            sub = P[(P.fraction == f) & (P.scope == "all") & (P.contrast == c) & (P["range"] == rg)]
            if sub.empty:
                continue
            r = sub.iloc[0]; x = i + (-0.27, -0.09, 0.09, 0.27)[j]
            resbar(b, x, r.resolution, half=0.07)
            cls = vclass(r.est, r.lo, r.hi, r.resolution, r.verdict)
            mfc = COL[c] if cls == "gain" else "white"
            b.errorbar(x, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt=mk, color=COL[c], mfc=mfc, mec=COL[c], ms=3.2, capsize=1.4,
                       elinewidth=0.8, mew=0.8, alpha=1.0 if cls != "tied" else 0.55, zorder=3)
            rec.append(dict(panel="b", range=rg, fraction=f, contrast=c, est=r.est, lo=r.lo, hi=r.hi, resolution=r.resolution, verdict=r.verdict))
    b.axhline(0, color=DARK, lw=0.6); b.set_xticks(range(len(cols)))
    b.set_xticklabels(["Pooled\n(46/36 y)"] + [DS_LAB[d] for d in DS], fontsize=6.3); b.set_xlim(-0.55, len(cols) - 0.45)
    b.set_ylabel("Δ within-environment Spearman")
    # c new vs old lines
    c_ = fig.add_subplot(gs[1, :2])
    items = [(0.25, "m0-ref", "Phenotypes themselves, 25 %"), (0.5, "m0-ref", "Phenotypes themselves, 50 %"),
             (0.25, "m1-m0", "G×E increment, 25 %"), (0.5, "m1-m0", "G×E increment, 50 %")]
    for i, (f, c, lab) in enumerate(items):
        y = len(items) - 1 - i
        for sc, mk, dy in (("old", "s", -0.15), ("new", "o", 0.15)):
            r = get(f, sc, c, "all")
            cls = vclass(r.est, r.lo, r.hi, r.resolution, r.verdict)
            colr = COL[c] if sc == "new" else GREY
            c_.errorbar(r.est, y + dy, xerr=[[r.est - r.lo], [r.hi - r.est]], fmt=mk, color=colr, mfc=colr if cls == "gain" else "white", mec=colr,
                        ms=3.2, capsize=1.4, elinewidth=0.8, mew=0.8, alpha=1.0 if cls != "tied" else 0.55, zorder=3)
            rec.append(dict(panel="c", fraction=f, contrast=c, lines=sc, est=r.est, lo=r.lo, hi=r.hi, resolution=r.resolution, verdict=r.verdict))
    c_.axvline(0, color=DARK, lw=0.6); c_.set_yticks(range(len(items))); c_.set_yticklabels([t for *_, t in items][::-1])
    c_.set_xlabel("Δ within-environment Spearman"); c_.tick_params(axis="y", length=0); c_.set_ylim(-0.6, len(items) - 0.4)
    c_.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=NAVY, ms=3.2, label="new lines"),
                       plt.Line2D([], [], marker="s", ls="", color=GREY, ms=3.2, label="old lines")], loc="lower right", fontsize=6.3)
    # d G x E increment on common years
    d = fig.add_subplot(gs[1, 2])
    cc = [(0.25, "all", "25 %\n46 y"), (0.25, "common_years", "25 %\n36 y*"), (0.5, "all", "50 %\n36 y")]
    for i, (f, rg, lab) in enumerate(cc):
        r = get(f, "all", "m1-m0", rg)
        resbar(d, i, r.resolution, half=0.25)
        cls = vclass(r.est, r.lo, r.hi, r.resolution, r.verdict)
        d.errorbar(i, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", color=SKY, mfc=SKY if cls == "gain" else "white", mec=SKY, ms=3.6,
                   capsize=1.6, elinewidth=0.9, mew=0.8, zorder=3)
        rec.append(dict(panel="d", fraction=f, range=rg, contrast="m1-m0", est=r.est, lo=r.lo, hi=r.hi, resolution=r.resolution, verdict=r.verdict))
    d.axhline(0, color=DARK, lw=0.6); d.set_xticks(range(3)); d.set_xticklabels([t for *_, t in cc], fontsize=6.3); d.set_xlim(-0.6, 2.6)
    d.set_ylabel("G×E increment"); d.set_xlabel("*years analysed\nat both fractions", fontsize=6)
    fig.canvas.draw()
    tab(fig, a, "a", "Phenotypes vs model"); tab(fig, b, "b", "By dataset"); tab(fig, c_, "c", "New and old lines"); tab(fig, d, "d", "Same target years")
    h = [mpl.patches.Patch(color=NAVY, label="phenotypes themselves (with − without)"), mpl.patches.Patch(color=SKY, label="G×E increment (M×E − main effect)"),
         plt.Line2D([], [], marker="o", ls="", color=GREY, mfc=GREY, ms=3.4, label="filled: resolved"),
         plt.Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=3.4, label="open: below 3/√N or tied (faded)"),
         plt.Line2D([], [], marker="s", ls="", color=GREY, mfc="white", ms=3.4, label="squares in b: 50 %"),
         plt.Line2D([], [], color="#BBBBBB", lw=2.2, label="3/√N")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=6.2, handletextpad=0.3, columnspacing=1.0, frameon=False)
    save(fig, "fig2", rec)


# ================================================================== Fig. 3 CLAC
def fig3():
    rec = []
    fig = plt.figure(figsize=(W2, 92 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.62, left=0.075, right=0.985, top=0.88, bottom=0.2)
    a = fig.add_subplot(gs[0, 0])
    y = pd.read_csv(R / "reanalysis/year_effects.csv"); y = y[(y.method == "clac") & (y.metric == "sp")].sort_values("target")
    v = jget("reanalysis/verdict.json", "verdict", "clac")
    items = [(str(r.target), r.d, r.d - 1.96 * r.se, r.d + 1.96 * r.se) for r in y.itertuples()]
    for i, (nm, e, lo, hi) in enumerate(items):
        yy = len(items) - i
        a.errorbar(e, yy, xerr=[[e - lo], [hi - e]], fmt="o", color=NAVY, mfc="white", mec=NAVY, ms=3.2, capsize=1.4, elinewidth=0.8, mew=0.8, zorder=3)
        rec.append(dict(panel="a", item=nm, est=e, lo=lo, hi=hi))
    a.add_patch(Rectangle((-v["resolution"], -0.35), 2 * v["resolution"], 0.7, color=GREY_L, lw=0, zorder=0))
    cls = vclass(v["est"], v["ci"][0], v["ci"][1], v["resolution"])
    vpoint(a, 0, v["est"], v["ci"][0], v["ci"][1], cls, marker="D", ms=4)
    note(a, v["ci"][1] + 0.006, 0, fmt(v["est"]), ha="left")
    rec.append(dict(panel="a", item="Pooled", est=v["est"], lo=v["ci"][0], hi=v["ci"][1], resolution=v["resolution"], verdict=cls))
    a.axhline(0.5, color=GREY_L, lw=0.5)
    a.axvline(0, color=DARK, lw=0.6); a.set_yticks(range(len(items) + 1)); a.set_yticklabels(["Pooled"] + [i[0] for i in items][::-1])
    a.set_xlabel("CLAC − cell-level GBLUP"); a.tick_params(axis="y", length=0); a.set_xlim(-0.03, 0.15)
    b = fig.add_subplot(gs[0, 1])
    H = pd.read_csv(R / "revision_tcj/hk_sensitivity.csv").set_index("contrast")
    abl = [("loss: linear kernel", "Linear kernel"), ("loss: single main effect", "Single main effect"), ("loss: equal weights", "Equal env. weights"),
           ("loss: no phenotype cleaning", "No phenotype cleaning"), ("loss: no spatial adjustment", "No spatial adjustment")]
    for i, (k, nm) in enumerate(abl):
        h = H.loc[f"CLAC ablation, {k}"]; yy = len(abl) - 1 - i
        resbar(b, yy, h["resolution"], half=0.38, horizontal=True)
        cls = vclass(h["est"], h["dl_lo"], h["dl_hi"], h["resolution"], h["dl_verdict"])
        b.barh(yy, h["est"], height=0.42, color=VSTYLE[cls]["color"] if cls != "tied" else "#C9CCD1", zorder=2)
        b.errorbar(h["est"], yy, xerr=[[h["est"] - h["dl_lo"]], [h["dl_hi"] - h["est"]]], fmt="none", ecolor=DARK, elinewidth=0.8, capsize=1.6, zorder=3)
        b.plot([h["hk_lo"], h["hk_hi"]], [yy - 0.32, yy - 0.32], color=HKC, lw=0.9, ls=(0, (2.2, 1.2)), zorder=3)
        rec.append(dict(panel="b", item=nm, level=h["level"], est=h["est"], dl_lo=h["dl_lo"], dl_hi=h["dl_hi"], hk_lo=h["hk_lo"], hk_hi=h["hk_hi"],
                        dl_verdict=h["dl_verdict"], hk_verdict=h["hk_verdict"]))
    b.axvline(0, color=DARK, lw=0.6); b.set_yticks(range(len(abl))); b.set_yticklabels([n for _, n in abl][::-1])
    b.set_xlabel("Loss in ranking when removed"); b.tick_params(axis="y", length=0); b.set_ylim(-0.6, len(abl) - 0.4)
    c_ = fig.add_subplot(gs[0, 2])
    T = pd.read_csv(R / "transfer_arc/pooled.csv"); T = T[(T.metric == "spearman") & (T.scope == "all")].set_index("range")
    rr = ["all48"] + DS
    for i, k in enumerate(rr):
        t = T.loc[k]; yy = len(rr) - 1 - i
        resbar(c_, yy, min(t["resolution"], 0.06), half=0.38, horizontal=True)
        if t["resolution"] > 0.06:
            c_.text(0.058, yy + 0.4, f"3/√N = {t['resolution']:.2f}", ha="right", va="bottom", fontsize=5.5, color=GREY)
        cls = vclass(t["est"], t["lo"], t["hi"], t["resolution"], t["judgement"])
        vpoint(c_, yy, t["est"], t["lo"], t["hi"], cls, marker="D" if k == "all48" else "o", ms=3.6 if k == "all48" else 3.4)
        if k == "G2F":
            note(c_, t["hi"] + 0.004, yy, fmt(t["est"]), ha="left")
        rec.append(dict(panel="c", range=k, est=t["est"], lo=t["lo"], hi=t["hi"], resolution=t["resolution"], verdict=cls))
    c_.axhline(len(rr) - 1.5, color=GREY_L, lw=0.5)
    c_.axvline(0, color=DARK, lw=0.6); c_.set_yticks(range(len(rr))); c_.set_yticklabels((["Pooled (48 y)"] + DS)[::-1]); c_.set_xlim(-0.06, 0.06)
    c_.set_xlabel("Arc-cosine − linear kernel"); c_.tick_params(axis="y", length=0); c_.set_ylim(-0.6, len(rr) - 0.4); c_.set_xticks([-0.05, 0, 0.05])
    fig.canvas.draw()
    tab(fig, a, "a", "CLAC, year by year"); tab(fig, b, "b", "Ablations"); tab(fig, c_, "c", "Kernel in cell-level GBLUP")
    verdict_legend(fig, classes=("gain", "tied"), y=0.0,
                   extra=[plt.Line2D([], [], marker="o", ls="", color=NAVY, mfc="white", ms=3.2, label="single target year"),
                          plt.Line2D([], [], color=DARK, lw=0.8, label="95 % / Holm (DL)"), plt.Line2D([], [], color=HKC, lw=0.9, ls=(0, (2.2, 1.2)), label="Hartung–Knapp 95 %")])
    save(fig, "fig3", rec)


# ================================================================== Fig. 4 new lines: methods x datasets
def fig4():
    PD = pd.read_csv(R / "revision_tcj/per_dataset.csv")
    PD = PD[PD["analysis"].str.startswith("Benchmark:")].copy()
    PD["item"] = PD["analysis"].map(BENCH_LAB)
    ye = pd.read_csv(R / "summary48/year_effects.csv"); ye = ye[ye["metric"] == "spearman"]
    order0 = ["reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp", "gxe_gbm", "rn_ridge", "dl_g", "R_STK_FW"]
    rec = []
    pooled = {k: hartung_knapp(floor_se(ye[ye["item"] == k][["dataset", "d", "se"]].reset_index(drop=True))) for k in order0}
    order = sorted(order0, key=lambda k: pooled[k]["est"], reverse=True)   # shared row order for a, b and c
    n = len(order)
    fig = plt.figure(figsize=(W2, 98 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.95, 0.8], wspace=0.08, left=0.2, right=0.985, top=0.93, bottom=0.27)
    a = fig.add_subplot(gs[0, 0])
    M = np.full((n, len(DS)), np.nan); V = np.empty((n, len(DS)), dtype=object)
    for i, k in enumerate(order):
        for j, d in enumerate(DS):
            s_ = PD[(PD["item"] == k) & (PD["dataset"] == d)]
            if len(s_):
                M[i, j] = s_.iloc[0]["est"]; V[i, j] = s_.iloc[0]["verdict"]
                rec.append(dict(panel="a", item=k, dataset=d, est=s_.iloc[0]["est"], lo=s_.iloc[0]["lo"], hi=s_.iloc[0]["hi"],
                                resolution=s_.iloc[0]["resolution"], verdict=V[i, j]))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("vn", [VERM, "#FFFFFF", NAVY])
    im = a.imshow(np.clip(M, -0.15, 0.15), cmap=cmap, vmin=-0.15, vmax=0.15, aspect="auto")
    for i in range(n):
        for j in range(len(DS)):
            if np.isnan(M[i, j]):
                a.text(j, i, "–", ha="center", va="center", fontsize=6, color=GREY); continue
            res = V[i, j] in ("resolved loss", "resolved gain")
            a.text(j, i, f"{M[i, j]:+.3f}".replace("-", "−"), ha="center", va="center", fontsize=5.4,
                   fontweight="bold" if res else "normal", color="white" if abs(M[i, j]) > 0.1 else DARK)
            if res:
                a.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, edgecolor=DARK, lw=0.8))
    a.set_xticks(range(len(DS))); a.set_xticklabels([DS_LAB[d] for d in DS], fontsize=6)
    a.set_yticks(range(n)); a.set_yticklabels([NAMES[k] for k in order], fontsize=6.4)
    a.tick_params(length=0); [sp.set_visible(False) for sp in a.spines.values()]
    cax = fig.add_axes([a.get_position().x0 + 0.02, 0.135, a.get_position().width - 0.04, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Δ within dataset (clipped at ±0.15); boxed = resolved", fontsize=6)
    cb.ax.tick_params(labelsize=5.8)
    b = fig.add_subplot(gs[0, 1])
    for i, k in enumerate(order):
        h = pooled[k]; yy = i
        b.add_patch(Rectangle((-0.0087, yy - 0.38), 2 * 0.0087, 0.76, color=GREY_L, lw=0, zorder=0))
        b.plot([h["hk_lo"], h["hk_hi"]], [yy + 0.28, yy + 0.28], color=HKC, lw=0.9, ls=(0, (2.2, 1.2)))
        cls = vclass(h["est"], h["lo"], h["hi"], 0.0087)
        vpoint(b, yy, h["est"], h["lo"], h["hi"], cls, ms=3.2)
        rec.append(dict(panel="b", item=k, est=h["est"], dl_lo=h["lo"], dl_hi=h["hi"], hk_lo=h["hk_lo"], hk_hi=h["hk_hi"], k=h["k"], verdict=cls))
    b.axvline(0, color=DARK, lw=0.6); b.set_yticks([]); b.set_ylim(n - 0.5, -0.5); b.set_xlim(-0.13, 0.03)
    b.set_xlabel("Pooled Δ, 48 target years\n(none resolved above the reference)"); b.spines["left"].set_visible(False)
    c_ = fig.add_subplot(gs[0, 2])
    T = pd.read_csv(R / "tpe_metric/pooled.csv"); T = T[T["range"] == "all"].set_index("item")
    for i, k in enumerate(order):
        if k not in T.index:
            c_.text(-0.08, i, "not available", ha="center", va="center", fontsize=5.6, color=GREY); continue
        t = T.loc[k]
        cls = vclass(t["est"], t["lo"], t["hi"], t["resolution"], t["verdict"])
        c_.add_patch(Rectangle((-t["resolution"], i - 0.38), 2 * t["resolution"], 0.76, color=GREY_L, lw=0, zorder=0))
        vpoint(c_, i, t["est"], t["lo"], t["hi"], cls, marker="D", ms=3.0)
        rec.append(dict(panel="c", item=k, across_est=t["est"], across_lo=t["lo"], across_hi=t["hi"], across_verdict=t["verdict"]))
    c_.axvline(0, color=DARK, lw=0.6); c_.set_yticks([]); c_.set_ylim(n - 0.5, -0.5); c_.set_xlim(-0.16, 0.08)
    c_.set_xlabel("Δ on line means\nacross environments"); c_.spines["left"].set_visible(False)
    fig.canvas.draw()
    tab(fig, a, "a", "Within each dataset", dx=-0.13); tab(fig, b, "b", "Pooled"); tab(fig, c_, "c", "Post hoc metric")
    verdict_legend(fig, classes=("tied", "det-", "loss"), y=0.0, extra=[plt.Line2D([], [], color=HKC, lw=0.9, ls=(0, (2.2, 1.2)), label="Hartung–Knapp 95 %")])
    save(fig, "fig4", rec)


# ================================================================== Fig. 5 other information, networks
def fig5():
    rec = []
    ol = jget("headroom_oldlines/verdict.json", "decisions")
    ol10 = row("headroom_oldlines/pooled.csv", contrast="ol1-ol0", metric="sp", range="all")
    sx = jget("headroom_secondary/verdict.json", "H1")
    ia = jget("headroom_ia_ib/verdict.json")
    s2 = jget("headroom_sparse2/verdict.json", "decisions"); m2 = jget("headroom_sparse/verdict.json", "main_m2_m1")
    groups = [("vs cell-level GBLUP", TINT["green"], [
                  ("Own-history residual (old lines)", (ol10.est, ol10.lo, ol10.hi, ol10.resolution)),
                  ("Line × location history (old lines)", (ol["H2"]["est"], *ol["H2"]["ci"], ol["H2"]["resolution"])),
                  ("Same-season silking dates (G2F)", (sx["est"], sx["lo"], sx["hi"], jget("headroom_secondary/verdict.json", "resolution"))),
                  ("Marker × location effects (43 y)", (ia["I-B"]["loc43"]["est"], ia["I-B"]["loc43"]["lo"], ia["I-B"]["loc43"]["hi"], ia["I-B"]["loc43"]["resolution"])),
                  ("Heteroscedastic GBLUP (48 y)", (ia["I-A"]["all48"]["est"], ia["I-A"]["all48"]["lo"], ia["I-A"]["all48"]["hi"], ia["I-A"]["all48"]["resolution"]))]),
              ("vs M×E GBLUP (25 %)", TINT["yellow"], [
                  ("Learned env. correlation", (m2["est"], m2["lo"], m2["hi"], row("headroom_sparse/pooled.csv", fraction=0.25, contrast="m1-m0", metric="sp", range="all48").resolution)),
                  ("Stacking with LightGBM", (s2["stk"]["est"], *s2["stk"]["ci"], s2["stk"]["resolution"])),
                  ("Residual network", (s2["dlres"]["est"], *s2["dlres"]["ci"], s2["dlres"]["resolution"])),
                  ("Covariate env. correlation", (s2["ecmxe"]["est"], *s2["ecmxe"]["ci"], s2["ecmxe"]["resolution"]))])]
    fig = plt.figure(figsize=(W2, 80 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.85], wspace=0.75, left=0.3, right=0.985, top=0.88, bottom=0.22)
    a = fig.add_subplot(gs[0, 0])
    rows_, yy = [], 0
    for gname, tint, its in groups:
        y0 = yy
        a.text(0.01, yy - 0.05, gname, transform=a.get_yaxis_transform(), ha="left", va="center", fontsize=6, color=NAVY, style="italic")
        yy += 0.75
        for nm, v in its:
            rows_.append((yy, nm, v)); yy += 1
        a.axhspan(y0 - 0.5, yy - 0.5, color=tint, zorder=-1, lw=0)
        yy += 0.3
    for y_, nm, (e, lo, hi, res) in rows_:
        resbar(a, y_, res, half=0.3, horizontal=True)
        cls = vclass(e, lo, hi, res)
        vpoint(a, y_, e, lo, hi, cls)
        rec.append(dict(panel="a", item=nm, est=e, lo=lo, hi=hi, resolution=res, verdict=VLABEL[cls]))
    a.axvline(0, color=DARK, lw=0.6); a.set_yticks([r[0] for r in rows_]); a.set_yticklabels([r[1] for r in rows_]); a.tick_params(axis="y", length=0)
    a.set_ylim(yy - 0.4 - 0.5 + 0.0, -0.6); a.set_xlim(-0.035, 0.035); a.set_xlabel("Δ within-environment Spearman")
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
        resbar(b, i, res, half=0.3, horizontal=True)
        cls = vclass(e, lo, hi, res)
        vpoint(b, i, e, lo, hi, cls)
        if nm == "GEFormer, 2022":
            note(b, hi + 0.012, i, fmt(e), ha="left")
        rec.append(dict(panel="b", item=nm, est=e, lo=lo, hi=hi, resolution=res, verdict=VLABEL[cls]))
    b.axvline(0, color=DARK, lw=0.6); b.set_yticks(range(len(pub))); b.set_yticklabels([p_[0] for p_ in pub]); b.tick_params(axis="y", length=0)
    b.set_ylim(len(pub) - 0.4, -0.6); b.set_xlim(-0.38, 0.14); b.set_xlabel("Δ vs GBLUP (G2F)")
    fig.canvas.draw()
    tab(fig, a, "a", "Other information and additions", dx=-0.25); tab(fig, b, "b", "Published methods, re-run", dx=-0.13)
    verdict_legend(fig, y=0.0)
    save(fig, "fig5", rec)


# ================================================================== Fig. 6 evaluation practices
def fig6():
    rec = []
    r7 = json.load(open(R / "c_paper/workspace/handoff/r7_h1h2.json"))[0]["structured_output"]["key_numbers"]
    r9 = jget("c_paper/amax_runs/A/final/R9_A_results.json", "H9a", "per_setting", "S4_G2F_CV00")
    r10 = "c_paper/amax_runs/trackA/final/R10_A_results.json"
    ext = jget(r10, "G1_enhancements_external", "contrasts", "D_noEC_minus_MG_2k", "EXT3_pooled")
    tun = jget(r10, "G4_tuning_audit", "MG_2k", "P2_minus_P3", "EXT3_pooled")
    fig = plt.figure(figsize=(W2, 108 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.075, right=0.985, top=0.93, bottom=0.6)
    gs2 = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.95], wspace=0.75, left=0.215, right=0.985, top=0.47, bottom=0.15)
    a = fig.add_subplot(gs[0, 0])
    sets = [("S1_G2F_LOYO", "G2F,\nyear out"), ("S2_G2F_LOLO", "G2F,\nlocation out"), ("S3_BRIWECS_LOYO", "Wheat,\nyear out")]
    for i, (k, nm) in enumerate(sets):
        e, lo, hi = r7[f"{k}|H1_optimism"], r7[f"{k}|H1_ci_low"], r7[f"{k}|H1_ci_high"]; w = r7[f"{k}|C2_within_env_leaky_minus_nested"]
        a.errorbar(i - 0.12, e, yerr=[[e - lo], [hi - e]], fmt="o", color=PURPLE, ms=3.4, capsize=1.6, elinewidth=0.9, mew=0.8)
        a.plot(i + 0.12, w, "s", color=GREEN, ms=3.4)
        rec += [dict(panel="a", item=nm.replace("\n", " "), quantity="environment-mean r, leaky − nested", est=e, lo=lo, hi=hi),
                dict(panel="a", item=nm.replace("\n", " "), quantity="within-environment r, leaky − nested", est=w)]
    a.axhline(0, color=DARK, lw=0.6); a.set_xticks(range(3)); a.set_xticklabels([s_[1] for s_ in sets], fontsize=6.3); a.set_ylabel("Leaky − nested window search")
    a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=PURPLE, ms=3.4, label="environment means"),
                      plt.Line2D([], [], marker="s", ls="", color=GREEN, ms=3.4, label="within environments")], loc="upper left", fontsize=6)
    b = fig.add_subplot(gs[0, 1])
    nS4, nEXT = 82082, 24995
    for i, (nm, e, lo, hi, res) in enumerate([("Internal split", r9["mean"], r9["ci_low"], r9["ci_high"], 3 / nS4 ** 0.5),
                                              ("External years", ext["mean"], ext["ci_low"], ext["ci_high"], 3 / nEXT ** 0.5)]):
        resbar(b, i, res, half=0.25)
        b.errorbar(i, e, yerr=[[e - lo], [hi - e]], fmt="o", color=DARK if i == 0 else GREY, mfc=DARK if i == 0 else "white", ms=3.6, capsize=1.6, elinewidth=0.9, mew=0.8)
        rec.append(dict(panel="b", item=nm, est=e, lo=lo, hi=hi, resolution=res))
    b.annotate("", xy=(1, ext["mean"]), xytext=(0, r9["mean"]), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7, linestyle="--"))
    b.axhline(0, color=DARK, lw=0.6); b.set_xticks(range(2)); b.set_xticklabels(["Internal\nsplit", "External\nyears"], fontsize=6.3); b.set_xlim(-0.6, 1.6)
    b.set_ylabel("Non-linear stage − ridge\n(within-env. Pearson)")
    c_ = fig.add_subplot(gs[0, 2])
    orc = row("summary48/pooled.csv", item="Oracle", metric="spearman", scope="all48")
    cf = row("ideas_diag/oracle_pooled.csv", metric="spearman", scope="all48", quantity="cross_fit")
    for i, (r, nm) in enumerate(((orc, "Same\nenvironments"), (cf, "Different\nhalves"))):
        resbar(c_, i, 0.0087, half=0.25)
        c_.errorbar(i, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", color=DARK if i == 0 else GREY, mfc=DARK if i == 0 else "white", ms=3.6,
                    capsize=1.6, elinewidth=0.9, mew=0.8)
        rec.append(dict(panel="c", item=nm.replace("\n", " "), est=r.est, lo=r.lo, hi=r.hi, resolution=0.0087))
    c_.annotate("", xy=(1, cf.est), xytext=(0, orc.est), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7, linestyle="--"))
    c_.axhline(0, color=DARK, lw=0.6); c_.set_xticks(range(2)); c_.set_xticklabels(["Same\nenvironments", "Different\nhalves"], fontsize=6.3); c_.set_xlim(-0.6, 1.6)
    c_.set_ylabel("Best method in hindsight\n− cell-level GBLUP")
    d = fig.add_subplot(gs2[0, :2])
    items = [("Ridge λ tuned on the test year (Pearson)", tun["mean"], tun["ci_low"], tun["ci_high"], 3 / nEXT ** 0.5)]
    for sc in ("2024", "2022"):
        r = row(f"tables/g2f_gate2_F{sc}m_vs_GEFormer_fair_main.csv", metric="spearman", method="GEFormer_own")
        cells = row("e1/geformer_rerun_vs_gate2.csv", scenario=f"F{sc}m", protocol="own", seed=147).cells
        items.append((f"GEFormer, own pipeline, {sc}", r.delta, r.ci2_lo, r.ci2_hi, 3 / cells ** 0.5))
    for sc in ("2024", "2022"):
        r = row("a1/summary.csv", scenario=f"F{sc}m", contrast="O", metric="spearman")
        items.append((f"GE-BiFormer, own pipeline, {sc}", r.delta, r.ci2_lo, r.ci2_hi, r.resolution))
    for i, (nm, e, lo, hi, res) in enumerate(items):
        resbar(d, i, res, half=0.3, horizontal=True)
        colr = PURPLE if e < 0 else GREEN
        d.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="o", color=colr, ms=3.4, capsize=1.6, elinewidth=0.9, mew=0.8)
        rec.append(dict(panel="d", item=nm, est=e, lo=lo, hi=hi, resolution=res))
    d.axvline(0, color=DARK, lw=0.6); d.set_yticks(range(len(items))); d.set_yticklabels([i_[0] for i_ in items]); d.tick_params(axis="y", length=0)
    d.set_ylim(len(items) - 0.4, -0.6)
    d.set_xlabel("Shift in the reported value (choices on the test year − on a validation year)")
    e_ = fig.add_subplot(gs2[0, 2])
    cls_ = [("Choose on test data", 6, PURPLE), ("Train/validation only", 5, GREEN), ("Not auditable", 3, GREY), ("No public code", 3, GREY_L)]
    e_.barh(range(len(cls_)), [c[1] for c in cls_], color=[c[2] for c in cls_], height=0.6)
    for i, c in enumerate(cls_):
        e_.text(c[1] + 0.2, i, str(c[1]), va="center", fontsize=6.5)
        rec.append(dict(panel="e", item=c[0], count=c[1]))
    e_.set_yticks(range(len(cls_))); e_.set_yticklabels([c[0] for c in cls_], fontsize=6.3); e_.set_ylim(len(cls_) - 0.4, -0.6)
    e_.set_xlabel("Published models (of 17)"); e_.tick_params(axis="y", length=0); e_.set_xlim(0, 7.5); e_.set_xticks([0, 2, 4, 6])
    fig.canvas.draw()
    tab(fig, a, "a", "Covariate windows"); tab(fig, b, "b", "Internal vs external"); tab(fig, c_, "c", "Picking the best")
    tab(fig, d, "d", "Tuning on the test year", dx=-0.2); tab(fig, e_, "e", "Public code", dx=-0.13)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=DARK, ms=3.4, label="b, c: as usually reported"),
                        plt.Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=3.4, label="b, c: on independent data"),
                        plt.Line2D([], [], color="#BBBBBB", lw=2.2, label="3/√N"),
                        plt.Line2D([], [], marker="o", ls="", color=PURPLE, ms=3.4, label="d: lower when the test year is used"),
                        plt.Line2D([], [], marker="o", ls="", color=GREEN, ms=3.4, label="d: higher"),
                        mpl.patches.Patch(color=GREY_L, label="d: ±3/√N")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6, handletextpad=0.3, columnspacing=0.9, frameon=False)
    save(fig, "fig6", rec)



# ================================================================== Fig. S2 GPverdict screenshots (public page, soybean example)
def figS2():
    """Screenshots captured by scripts/capture_gpverdict_screens.py from the public page; only cropped, never edited."""
    from PIL import Image
    D = OUT / "figS2_screens"
    a_im = Image.open(D / "s2a_card.png"); a_im = a_im.crop((0, 640, a_im.width, a_im.height))  # from the column list to the buttons
    b_im = Image.open(D / "s2b_report.png"); c_im = Image.open(D / "s2c_footer.png")
    rec = [dict(panel=p, file=f, width_px=im.width, height_px=im.height, crop=cr) for p, f, im, cr in
           (("a", "s2a_card.png", a_im, "rows 640 to end"), ("b", "s2b_report.png", b_im, "none"), ("c", "s2c_footer.png", c_im, "none"))]
    x0, wa, wb = 0.045, 0.955, 0.70
    ha, hb, hc = wa * a_im.height / a_im.width, wb * b_im.height / b_im.width, wa * c_im.height / c_im.width
    gap = 0.04
    H = ha + hb + hc + 2 * gap
    fig = plt.figure(figsize=(W2, W2 * H))
    axes = []
    y = H
    for im, w, h in ((a_im, wa, ha), (b_im, wb, hb), (c_im, wa, hc)):
        y -= h
        ax = fig.add_axes([x0, y / H, w, h / H]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(GREY); sp.set_linewidth(0.4)
        axes.append(ax)
        y -= gap
    for ax, l in zip(axes, "abc"):
        fig.text(0.0, ax.get_position().y1, l, fontsize=9, fontweight="bold", va="top", ha="left")
    pd.DataFrame(rec).to_csv(SRC / "figS2.csv", index=False)
    for ext in ("pdf",):
        fig.savefig(OUT / f"figS2.{ext}", dpi=600, pad_inches=0.03)
    fig.savefig(OUT / "figS2.tiff", dpi=600, pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(OUT / "figS2.tiff") as im:   # no alpha channel in the TIFF
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    bg.save(OUT / "figS2.tiff", compression="tiff_lzw", dpi=(600, 600))
    fig.savefig(OUT / "figS2.preview.png", dpi=300, pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "figS1", "figS2"]
    for f in which:
        globals()[f](); print("done", f)
