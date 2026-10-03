"""Main figures for the TAG manuscript (draft v5). Every plotted value is read from a committed result file; the
values actually drawn are also written to results/figdata_tag/<figure>.csv as source data.

Run from the repository root:  python3 scripts/make_figures_tag.py
Outputs: figures/tag/fig{1..4}.{pdf,svg,tiff} and <name>.alignment.json (multi-panel geometry check).

Fig. 1  Kind of information vs added model complexity (hero overview).
Fig. 2  New lines, new year: published G2F methods (deployable pipelines), CLAC ablations, library methods (48 years).
Fig. 3  One reaction norm across scenarios, and simple vs complex reaction norms for tested lines (G2F).
Fig. 4  Four protocol effects."""
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Optional render-time panel-alignment check; set FIGURE_QA_DIR to the directory holding audit_panel_alignment.py.
if os.environ.get("FIGURE_QA_DIR"):
    sys.path.insert(0, os.environ["FIGURE_QA_DIR"])
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
except ImportError:
    require_matplotlib_panel_alignment = None

R = Path("results")
OUT = Path("figures/tag"); OUT.mkdir(parents=True, exist_ok=True)
SRC = R / "figdata_tag"; SRC.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
FIG_WIDTH_MM = 174  # TAG double column
W2 = FIG_WIDTH_MM * MM

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,  # TAG: lettering 8-12 pt at final size
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "lines.linewidth": 0.8,
})
BLUE, BLUE2, TEAL = "#0F4D92", "#3775BA", "#42949E"
GREY, GREY_L, DARK, RED = "#767676", "#CFCECE", "#272727", "#B64342"


# ------------------------------------------------------------------ readers
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


def save(fig, name, records):
    pd.DataFrame(records).to_csv(SRC / f"{name}.csv", index=False)
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(fig, json_out=str(OUT / f"{name}.alignment.json"),
                                           overlay_svg=str(OUT / f"{name}.alignment.svg"),
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name.capitalize()}.eps", bbox_inches="tight", pad_inches=0.02)  # TAG: EPS preferred, named Fig1.eps
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image  # journals expect RGB without an alpha channel
    with Image.open(OUT / f"{name}.tiff") as im:
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    bg.save(OUT / f"{name}.tiff", compression="tiff_lzw", dpi=(600, 600))
    fig.savefig(OUT / f"{name}.preview.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def label(ax, s, x=-0.02):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def forest(ax, items, xlab, xlim=None):
    """items: dicts with name, est, lo, hi, res, kind ('info'|'complex'|'pipeline'|'ref'|'flag'). Rows top-down."""
    n = len(items)
    for k, it in enumerate(items):
        y = n - 1 - k
        if it.get("resol") is not None:
            ax.add_patch(plt.Rectangle((-it["resol"], y - 0.32), 2 * it["resol"], 0.64, color="#E8E8E8", lw=0, zorder=0))
        style = {"info": dict(fmt="o", color=BLUE, mfc=BLUE), "complex": dict(fmt="o", color=GREY, mfc="white"),
                 "pipeline": dict(fmt="D", color=TEAL, mfc=TEAL), "flag": dict(fmt="o", color=GREY, mfc=GREY_L)}[it["kind"]]
        ax.errorbar(it["est"], y, xerr=[[it["est"] - it["lo"]], [it["hi"] - it["est"]]], ms=4, capsize=1.8,
                    elinewidth=0.8, mew=0.8, zorder=3, **style)
    ax.axvline(0, color=DARK, lw=0.6, zorder=1)
    ax.set_yticks(range(n)); ax.set_yticklabels([it["name"] for it in items][::-1])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(xlab)
    if xlim:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="y", length=0)


# ------------------------------------------------------------------ shared values
res48 = jget("headroom_ia_ib/verdict.json", "I-A", "all48", "resolution")
s48 = lambda item: row("summary48/pooled.csv", item=item, metric="spearman", scope="all48")
rv = lambda m: jget("reanalysis/verdict.json", "verdict", m)
NAMES = {"reml": "Two-stage GBLUP", "reml_x0.1": "Ridge, 0.1× shrinkage", "reml_x10": "Ridge, 10× shrinkage",
         "reml_x100": "Ridge, 100× shrinkage", "ridge_pc20": "Ridge on 20 PCs", "rf": "Random forest",
         "gbm": "Gradient boosting", "knn10": "kNN (k = 10)", "knn30": "kNN (k = 30)", "mlp": "Multilayer perceptron",
         "rn_ridge": "EC reaction-norm ridge", "gxe_gbm": "EC LightGBM", "dl_g": "Within-env.-loss network",
         "R_CV": "Pick by CV", "R_FW": "Pick by forward history", "R_EQ": "Equal-weight ensemble",
         "R_STK_CV": "Stacking (CV)", "R_STK_FW": "Stacking (forward)"}


# ================================================================== Fig. 1
def fig1():
    sp = lambda frac, con: row("headroom_sparse/pooled.csv", fraction=frac, contrast=con, metric="sp", range="all48")
    ol = jget("headroom_oldlines/verdict.json", "decisions")
    ol10 = row("headroom_oldlines/pooled.csv", contrast="ol1-ol0", metric="sp", range="all")
    sx = jget("headroom_secondary/verdict.json", "H1")
    rp = "rn_paper/ge_mean_primary_evidence.json"
    fw = jget(rp, "states", "observed", "fw_vs_additive", "spearman")
    kg = jget(rp, "states", "observed", "kernel_gated_vs_fw", "spearman")
    nobs = jget(rp, "states", "observed", "n_genotype_environment_units")
    s2 = jget("headroom_sparse2/verdict.json", "decisions"); m2 = jget("headroom_sparse/verdict.json", "main_m2_m1")
    clac = rv("clac")
    cats = [
        ("None\n(new lines,\nnew year)", None, res48,
         [(NAMES[k], s48(k).est, s48(k).lo, s48(k).hi) for k in ("rf", "gbm", "mlp", "rn_ridge", "gxe_gbm", "dl_g", "R_STK_FW")]),
        ("Line's own\nhistory", (ol10.est, ol10.lo, ol10.hi), ol10.resolution, []),
        ("Line ×\nlocation\nhistory", (ol["H2"]["est"], *ol["H2"]["ci"]), ol["H2"]["resolution"],
         [("Marker G×location", ol["H1"]["est"], *ol["H1"]["ci"])]),
        ("Same-season\nflowering\n(G2F)", (sx["est"], sx["lo"], sx["hi"]), jget("headroom_secondary/verdict.json", "resolution"), []),
        ("Weather\nreaction norm,\ntested lines\n(G2F)", (fw["estimate"], fw["ci_low"], fw["ci_high"]), 3 / nobs ** 0.5,
         [("Kernel-gated", kg["estimate"], kg["ci_low"], kg["ci_high"])]),
        ("Target year,\n25 %\nphenotyped", tuple(sp("0.25", "m1-m0")[["est", "lo", "hi"]]), sp("0.25", "m1-m0").resolution,
         [("Learned correlation", m2["est"], m2["lo"], m2["hi"])] + [(k, s2[k]["est"], *s2[k]["ci"]) for k in ("stk", "dlres", "ecmxe")]),
        ("Target year,\n50 %\nphenotyped", tuple(sp("0.5", "m1-m0")[["est", "lo", "hi"]]), sp("0.5", "m1-m0").resolution, []),
    ]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W2, 150 * MM), gridspec_kw=dict(height_ratios=[95, 50]))
    rec = []
    for i, (lab, info, res, comp) in enumerate(cats):
        x0 = i - 0.02 if i == 0 else i - 0.42  # in "None" the left half carries CLAC's own resolution bar
        ax.plot([x0, i + 0.42], [res, res], color=GREY_L, lw=2.4, solid_capstyle="butt", zorder=0)
        if info is not None:
            e, lo, hi = info
            ax.errorbar(i - 0.14, e, yerr=[[e - lo], [hi - e]], fmt="o", color=BLUE, ms=5, capsize=2, elinewidth=1.0, zorder=3)
            rec.append(dict(category=lab.replace("\n", " "), kind="information", item="information added", est=e, lo=lo, hi=hi, resolution=res))
        for j, (cl, e, lo, hi) in enumerate(comp):
            x = i + 0.06 + 0.3 * j / max(len(comp) - 1, 1) if len(comp) > 1 else i + 0.12
            ax.errorbar(x, e, yerr=[[e - lo], [hi - e]], fmt="o", mfc="white", color=GREY, ms=3, capsize=1.2, elinewidth=0.7, zorder=3)
            rec.append(dict(category=lab.replace("\n", " "), kind="complexity", item=cl, est=e, lo=lo, hi=hi, resolution=res))
        if i == 0:
            e, (lo, hi) = clac["est"], clac["ci"]
            ax.errorbar(i - 0.24, e, yerr=[[e - lo], [hi - e]], fmt="D", color=TEAL, ms=3.8, capsize=1.8, elinewidth=0.9, zorder=4)
            ax.plot([i - 0.42, i - 0.08], [clac["resolution"]] * 2, color=GREY_L, lw=2.4, solid_capstyle="butt", zorder=0)
            rec.append(dict(category="None", kind="published pipeline", item="CLAC (G2F, 97.5 % Holm)", est=e, lo=lo, hi=hi, resolution=clac["resolution"]))
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[0] for c in cats])
    ax.set_xlim(-0.6, len(cats) - 0.4); ax.set_ylim(-0.115, 0.09)
    ax.set_ylabel("Δ within-environment Spearman")
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="Information added"),
         plt.Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=3, label="Model complexity added"),
         plt.Line2D([], [], marker="D", ls="", color=TEAL, ms=3.8, label="Published pipeline (CLAC, G2F)"),
         plt.Line2D([], [], color=GREY_L, lw=2.4, label="Resolution 3/√N")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.04), ncol=2, columnspacing=1.2, handletextpad=0.4)
    # panel b: the sparse-testing gain (MxE GBLUP minus main-effect GBLUP) in each dataset
    sets = [("G2F", "G2F\nmaize"), ("GEM_IA", "GEM_IA\nmaize"), ("NUST", "NUST\nsoybean"), ("MU_SOY", "MU_SOY\nsoybean"),
            ("ESWYT", "ESWYT\nwheat"), ("URSN", "URSN\nwheat")]
    for i, (ds, _) in enumerate(sets):
        for frac, dx, col in (("0.25", -0.12, BLUE), ("0.5", 0.12, TEAL)):
            r = row("headroom_sparse/pooled.csv", fraction=frac, contrast="m1-m0", metric="sp", range=ds)
            bx.errorbar(i + dx, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", color=col, ms=4.5, capsize=2, elinewidth=1.0, zorder=3)
            rec.append(dict(category=ds, kind="sparse testing by dataset", item=f"{frac} of cells phenotyped ({int(r.k)} target years)",
                            est=r.est, lo=r.lo, hi=r.hi, resolution=None))
    bx.axhline(0, color=DARK, lw=0.6)
    bx.set_xticks(range(len(sets))); bx.set_xticklabels([t for _, t in sets])
    bx.set_xlim(-0.6, len(sets) - 0.4)
    bx.set_ylabel("Δ within-environment Spearman")
    bx.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5, label="25 % of target-year cells phenotyped"),
                       plt.Line2D([], [], marker="o", ls="", color=TEAL, ms=4.5, label="50 % phenotyped")],
              loc="upper right", handletextpad=0.4)
    fig.align_ylabels([ax, bx])
    fig.tight_layout(pad=0.4, h_pad=1.5)
    for a_, s_ in ((ax, "a"), (bx, "b")):
        fig.text(0.005, a_.get_position().y1 + 0.008, s_, fontsize=10, fontweight="bold", va="bottom", ha="left")
    save(fig, "fig1", rec)


# ================================================================== Fig. 2
def fig2():
    lib = [k for k in ("reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp",
                       "rn_ridge", "gxe_gbm", "dl_g", "R_CV", "R_FW", "R_EQ", "R_STK_CV", "R_STK_FW")]
    ia = []
    for k in lib:
        r = s48(k)
        ia.append(dict(name=f"{NAMES[k]} ({int(r.k)} y)" if int(r.k) != 48 else NAMES[k], est=r.est, lo=r.lo, hi=r.hi, resol=res48,
                       kind="complex"))
    ia.sort(key=lambda d: d["est"], reverse=True)
    pub = []
    for sc, f in (("2024", "tables/g2f_gate2_F2024m_vs_B1r_gblup_reml_main.csv"), ("2022", "tables/g2f_gate2_F2022m_vs_B1r_gblup_reml_main.csv")):
        r = row(f, metric="spearman", method="GEFormer_fair")
        cells = row("e1/geformer_rerun_vs_gate2.csv", scenario=f"F{sc}m", protocol="own", seed=147).cells
        pub.append(dict(name=f"GEFormer, {sc}", est=r.delta, lo=r.ci2_lo, hi=r.ci2_hi, resol=3 / cells ** 0.5, kind="complex"))
    for sc in ("2024", "2022"):
        r = row("a1/summary.csv", scenario=f"F{sc}m", contrast="C_fair", metric="spearman")
        pub.append(dict(name=f"GE-BiFormer, {sc}", est=r.delta, lo=r.ci2_lo, hi=r.ci2_hi, resol=r.resolution, kind="complex"))
    c = rv("clac"); pub.insert(0, dict(name="CLAC, 5 years", est=c["est"], lo=c["ci"][0], hi=c["ci"][1], resol=c["resolution"], kind="pipeline"))
    lc = rv("lc_real"); pub.insert(1, dict(name="Lopez-Cruz RN, 2022", est=lc["est"], lo=lc["ci"][0], hi=lc["ci"][1], resol=lc["resolution"], kind="complex"))
    dv = jget("clac_decomp/verdict.json", "verdict")
    abl = [dict(name=nm, est=dv[k]["est"], lo=dv[k]["ci"][0], hi=dv[k]["ci"][1], resol=dv[k]["resolution"],
                kind="pipeline" if dv[k]["judgement"].startswith("carries") else "complex")
           for k, nm in (("loss_D4_linear_kernel", "Linear instead of arc-cosine kernel"), ("loss_D5_single_main_effect", "Single main-effect GBLUP"),
                         ("loss_D1_equal_weights", "Equal weights over training env."), ("loss_D3_no_cleaning", "No phenotype cleaning"),
                         ("loss_D2_no_spatial", "No spatial adjustment"))]
    fig, (a, b, c_) = plt.subplots(3, 1, figsize=(W2, 215 * MM), gridspec_kw=dict(height_ratios=[6, 5, 18]))
    forest(a, pub, "Δ within-environment Spearman vs GBLUP (G2F)", xlim=(-0.37, 0.1))
    forest(b, abl, "Loss in within-environment Spearman when the component is removed (G2F)", xlim=(-0.03, 0.09))
    forest(c_, ia, "Δ within-environment Spearman vs cell-level GBLUP", xlim=(-0.12, 0.03))
    fig.tight_layout(pad=0.4, h_pad=1.5)
    for ax, s_ in ((a, "a"), (b, "b"), (c_, "c")):
        fig.text(0.005, ax.get_position().y1 + 0.006, s_, fontsize=10, fontweight="bold", va="bottom", ha="left")
    save(fig, "fig2", [dict(panel="a", **d) for d in pub] + [dict(panel="b", **d) for d in abl] + [dict(panel="c", **d) for d in ia])


# ================================================================== Fig. 3
def fig3():
    rs = "rn_paper/ge_mean_shift_decomposition.json"
    cells = [("cv0_strict", "Tested lines, new environments"), ("cohort_matched_forward", "Same cohort, future years"),
             ("cv00", "New lines, new environments"), ("forward_year", "Forward years, mostly new lines"),
             ("forward_year_v2", "Same, genomic reference")]
    a_items = []
    for key, nm in cells:
        s = jget(rs, "cells", key, "vs_additive_g_e", "spearman"); n = jget(rs, "cells", key, "n_prediction_rows")
        a_items.append(dict(name=f"{nm} ({jget(rs, 'cells', key, 'n_environments')})", est=s["estimate"], lo=s["ci_low"],
                            hi=s["ci_high"], resol=3 / n ** 0.5, kind="info" if key in ("cv0_strict", "cohort_matched_forward") else "complex"))
    rp = "rn_paper/ge_mean_primary_evidence.json"
    b_items = []
    for st, nm in (("observed", "observed weather"), ("preseason", "1 March forecast"), ("inseason", "1 June forecast")):
        n = jget(rp, "states", st, "n_genotype_environment_units")
        f = jget(rp, "states", st, "fw_vs_additive", "spearman"); k = jget(rp, "states", st, "kernel_gated_vs_fw", "spearman")
        b_items.append(dict(name=f"FW vs additive, {nm}", est=f["estimate"], lo=f["ci_low"], hi=f["ci_high"], resol=3 / n ** 0.5, kind="info"))
        b_items.append(dict(name=f"Kernel-gated vs FW, {nm}", est=k["estimate"], lo=k["ci_low"], hi=k["ci_high"], resol=3 / n ** 0.5, kind="complex"))
    fig, (a, b) = plt.subplots(2, 1, figsize=(W2, 140 * MM), gridspec_kw=dict(height_ratios=[5, 6]))
    forest(a, a_items, "Δ within-env. Spearman, kernel-gated RN vs additive", xlim=(-0.075, 0.075))
    forest(b, b_items, "Δ within-env. Spearman (tested lines, G2F)", xlim=(-0.03, 0.04))
    fig.tight_layout(pad=0.4, h_pad=1.5)
    for ax, s_ in ((a, "a"), (b, "b")):
        fig.text(0.005, ax.get_position().y1 + 0.01, s_, fontsize=10, fontweight="bold", va="bottom", ha="left")
    save(fig, "fig3", [dict(panel="a", **d) for d in a_items] + [dict(panel="b", **d) for d in b_items])


# ================================================================== Fig. 4
def fig4():
    r7 = json.load(open(R / "c_paper/workspace/handoff/r7_h1h2.json"))[0]["structured_output"]["key_numbers"]
    r9 = jget("c_paper/amax_runs/A/final/R9_A_results.json", "H9a", "per_setting", "S4_G2F_CV00")
    r10 = "c_paper/amax_runs/trackA/final/R10_A_results.json"
    ext = jget(r10, "G1_enhancements_external", "contrasts", "D_noEC_minus_MG_2k", "EXT3_pooled")
    tun = jget(r10, "G4_tuning_audit", "MG_2k", "P2_minus_P3", "EXT3_pooled")
    rec = []
    fig, axs = plt.subplots(2, 2, figsize=(W2, 150 * MM))
    # a leakage
    a = axs[0, 0]
    sets = [("S1_G2F_LOYO", "G2F, leave-year-out"), ("S2_G2F_LOLO", "G2F, leave-location-out"), ("S3_BRIWECS_LOYO", "Wheat, leave-year-out")]
    for i, (k, nm) in enumerate(sets):
        e, lo, hi = r7[f"{k}|H1_optimism"], r7[f"{k}|H1_ci_low"], r7[f"{k}|H1_ci_high"]
        w = r7[f"{k}|C2_within_env_leaky_minus_nested"]
        a.errorbar(i - 0.12, e, yerr=[[e - lo], [hi - e]], fmt="o", color=RED, ms=3.5, capsize=1.6, elinewidth=0.8)
        a.plot(i + 0.12, w, "s", color=BLUE, ms=3.5)
        rec += [dict(panel="a", item=nm, quantity="environment-mean r inflation", est=e, lo=lo, hi=hi),
                dict(panel="a", item=nm, quantity="within-environment r change", est=w, lo=None, hi=None)]
    a.axhline(0, color=DARK, lw=0.6); a.set_xticks(range(3)); a.set_xticklabels(["G2F,\nyear out", "G2F,\nlocation out", "Wheat,\nyear out"])
    a.set_ylabel("Leaky − nested window search")
    a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=RED, ms=3.5, label="Environment-mean r"),
                      plt.Line2D([], [], marker="s", ls="", color=BLUE, ms=3.5, label="Within-environment r")], loc="upper left")
    # b internal vs external
    b = axs[0, 1]
    nS4 = 82082; nEXT = 24995  # cells; derivation and check in scripts/check_sequel_sources.py (C6)
    for i, (nm, e, lo, hi, res) in enumerate([("Internal split\n(CV00, 211 env.)", r9["mean"], r9["ci_low"], r9["ci_high"], 3 / nS4 ** 0.5),
                                              ("External years\n2022–2024 (75 env.)", ext["mean"], ext["ci_low"], ext["ci_high"], 3 / nEXT ** 0.5)]):
        b.plot([i - 0.3, i + 0.3], [res, res], color=GREY_L, lw=2.4, solid_capstyle="butt", zorder=0)
        b.errorbar(i, e, yerr=[[e - lo], [hi - e]], fmt="o", color=BLUE if i == 0 else GREY, ms=3.5, capsize=1.6, elinewidth=0.8)
        rec.append(dict(panel="b", item=nm.replace("\n", " "), quantity="non-linear stage − ridge (within-env. Pearson)", est=e, lo=lo, hi=hi, resolution=res))
    b.axhline(0, color=DARK, lw=0.6); b.set_xticks(range(2)); b.set_xticklabels(["Internal split\n(211 env.)", "External years\n(75 env.)"])
    b.set_xlim(-0.6, 1.6); b.set_ylabel("Δ within-env. Pearson")
    # c choosing on test data
    c = axs[1, 0]
    items = [dict(name="Ridge λ tuned on test year*", est=tun["mean"], lo=tun["ci_low"], hi=tun["ci_high"], resol=3 / nEXT ** 0.5, kind="complex")]
    for sc in ("2024", "2022"):
        r = row(f"tables/g2f_gate2_F{sc}m_vs_GEFormer_fair_main.csv", metric="spearman", method="GEFormer_own")
        cells = row("e1/geformer_rerun_vs_gate2.csv", scenario=f"F{sc}m", protocol="own", seed=147).cells
        items.append(dict(name=f"GEFormer own − deployable, {sc}", est=r.delta, lo=r.ci2_lo, hi=r.ci2_hi, resol=3 / cells ** 0.5, kind="complex"))
    for sc in ("2024", "2022"):
        r = row("a1/summary.csv", scenario=f"F{sc}m", contrast="O", metric="spearman")
        items.append(dict(name=f"GE-BiFormer own − deployable, {sc}", est=r.delta, lo=r.ci2_lo, hi=r.ci2_hi, resol=r.resolution, kind="complex"))
    forest(c, items, "Shift in reported within-env. correlation", xlim=(-0.2, 0.3))
    rec += [dict(panel="c", **d) for d in items]
    # d hindsight
    d = axs[1, 1]
    for i, (q, nm) in enumerate((("naive", "Picked and scored\non same environments"), ("cross_fit", "Picked and scored\non different halves"))):
        r = row("ideas_diag/oracle_pooled.csv", metric="spearman", scope="all48", quantity=q)
        d.plot([i - 0.3, i + 0.3], [res48, res48], color=GREY_L, lw=2.4, solid_capstyle="butt", zorder=0)
        d.errorbar(i, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", color=BLUE if i == 0 else GREY, ms=3.5, capsize=1.6, elinewidth=0.8)
        rec.append(dict(panel="d", item=nm.replace("\n", " "), quantity="best method in hindsight − cell-level GBLUP", est=r.est, lo=r.lo, hi=r.hi, resolution=res48))
    d.axhline(0, color=DARK, lw=0.6); d.set_xticks(range(2)); d.set_xticklabels(["Same\nenvironments", "Different\nhalves"])
    d.set_xlim(-0.6, 1.6); d.set_ylabel("Δ within-env. Spearman"); d.set_xlabel("Best method picked and scored on")
    fig.tight_layout(pad=0.4, h_pad=1.6, w_pad=2.0)
    for ax, s in zip(axs.flat, "abcd"):  # column-aligned labels in figure coordinates
        col = 0 if s in "ac" else 1
        x = 0.005 if col == 0 else axs[0, 1].get_position().x0 - 0.075
        fig.text(x, ax.get_position().y1 + 0.015, s, fontsize=10, fontweight="bold", va="bottom", ha="left")
    save(fig, "fig4", rec)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f(); print("done", f.__name__)
