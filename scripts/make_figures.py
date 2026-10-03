"""Manuscript figures (docs: manuscript/figure_plan.md). Inputs are committed CSV/JSON result files only; every figure is
written as PDF (vector) and PNG (300 dpi) into figures/. Run from the repository root: python3 scripts/make_figures.py [fig ...]

Colour rules (dataviz skill, reference palette): one categorical pair only where two things are compared (blue #2a78d6 and
orange #eb6834, adjacent colour-blind separation 9.1 OKLab x100 in the reference palette); blue = better / orange = worse
with mid-grey for 'tied' where a sign is encoded; status colours (green/amber/red) only for PASS/WARN/FAIL, always with the
word; text in ink colours; thin marks; no dual axes; direct labels instead of legends where possible."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

R = Path("results")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
INK, INK2, GRID = "#1d2430", "#5b6574", "#dde2ea"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8f98"
OK, WARN, FAIL = "#2e8b57", "#d99a00", "#c0392b"
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def panel_label(ax, s, dx=-0.09, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------------------------------------- Figure 1
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    rng = np.random.default_rng(3)
    n_lines, n_env = 8, 9  # 3 years x 3 environments
    for ax, kind in zip(axes, ("random", "forward")):
        for i in range(n_lines):
            for j in range(n_env):
                year = j // 3
                if kind == "random":
                    test = rng.random() < 0.2
                    fc, ec, ls = (ORANGE if test else BLUE), "white", "-"
                else:
                    new = i >= 4
                    if year < 2:
                        fc, ec, ls = ("white", GREY, (0, (2, 2))) if new else (BLUE, "white", "-")
                    else:
                        fc, ec, ls = ORANGE, "white", "-"
                ax.add_patch(Rectangle((j + 0.04 + 0.28 * (j // 3), -i - 1 + 0.06), 0.92, 0.88, facecolor=fc, edgecolor=ec,
                                       linewidth=0.7 if fc == "white" and kind == "forward" else 0.5, linestyle=ls))
        ax.set_xlim(-0.3, n_env + 0.9)
        ax.set_ylim(-n_lines - 1.6, 1.6)
        ax.axis("off")
        for y in range(3):
            ax.text(y * 3 + 1.5 + 0.28 * y, -n_lines - 0.35, f"year {'Y-2 Y-1 Y'.split()[y]}", ha="center", va="top", color=INK2)
        ax.text(-0.15, 0.55, "lines", ha="right", va="bottom", color=INK2, rotation=0)
        ax.text(4.5 + 0.28, 1.45, "environments (location–year)", ha="center", va="bottom", color=INK2)
    axes[0].set_title("Random cross-validation", loc="left", pad=14)
    axes[1].set_title("Deployment: new lines in a new year", loc="left", pad=14)
    axes[0].text(4.7, -n_lines - 1.25, "test cells are scattered over the same lines\nand the same environments as training cells",
                 ha="center", va="top", color=INK, fontsize=7)
    axes[1].text(4.7, -n_lines - 1.25, "all cells of year Y are unseen; at least half of the lines\nhave no earlier record; markers exist for every line",
                 ha="center", va="top", color=INK, fontsize=7)
    axes[1].add_patch(Rectangle((6.02 + 0.56, -n_lines - 0.05), 3.0, n_lines + 0.1, fill=False, edgecolor=ORANGE, linewidth=1.0))
    axes[1].text(6.0 + 0.56 + 1.5 + 0.03, 0.55, "target year", ha="center", va="bottom", color=ORANGE, fontsize=7.5, fontweight="bold")
    axes[1].text(-0.15, -2.0, "known\nlines", ha="right", va="center", color=INK2, fontsize=6.5)
    axes[1].text(-0.15, -6.0, "new\nlines", ha="right", va="center", color=INK2, fontsize=6.5)
    handles = [Rectangle((0, 0), 1, 1, facecolor=BLUE), Rectangle((0, 0), 1, 1, facecolor=ORANGE),
               Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GREY, linestyle=(0, (2, 2)))]
    fig.legend(handles, ["phenotype used for training", "predicted and scored", "no phenotype yet"], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig1_scenario")


# ---------------------------------------------------------------------------------------------------------- Figure 2
def fig2():
    E = pd.read_csv(R / "figdata/fig2_epochs.csv")
    run = E[(E.model == "GEFormer") & (E.scenario == "F2024m") & (E.seed == 147)].set_index("epoch")
    eP, eS = int(run.pooled_r.idxmax()), int(run.within_spearman.idxmax())
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.95], wspace=0.55)
    # (a) pooled r as a function of the within-environment scale for the two chosen epochs
    ax = fig.add_subplot(gs[0])
    u = np.linspace(0, 4.2, 400)
    for e, col, lab in ((eP, ORANGE, f"epoch {eP}\n(chosen by pooled r)"), (eS, BLUE, f"epoch {eS}\n(chosen by Spearman)")):
        r_ = run.loc[e]
        curve = (r_.a + r_.k * u) / np.sqrt((r_.v_b + u ** 2) * r_.S)
        ax.plot(u, curve, color=col, linewidth=1.2)
        ax.plot([r_.u], [r_.pooled_r], "o", color=col, markersize=5, markeredgecolor="white", markeredgewidth=0.6)
        if np.isfinite(r_.u_star) and r_.u_star < 4.2:
            ax.plot([r_.u_star], [np.sqrt((r_.a ** 2 / r_.v_b + r_.k ** 2) / r_.S)], marker="^", color=col, markersize=5.5,
                    markerfacecolor="white", markeredgewidth=0.9, linestyle="none")
        if col == ORANGE:
            xl = 1.3
            yl = float((r_.a + r_.k * xl) / np.sqrt((r_.v_b + xl ** 2) * r_.S)) + 0.03
            ax.text(xl, yl, lab, color=col, ha="left", va="bottom", fontsize=6.8)
        else:
            ax.text(4.15, 0.235, lab, color=col, ha="right", va="bottom", fontsize=6.8)
    ax.set_xlabel("within-environment scale u")
    ax.set_ylabel("pooled Pearson r")
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 0.62)
    ax.text(0.98, 0.98, "● where the epoch is\n△ maximum of the curve", transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK2)
    panel_label(ax, "a", -0.2)
    # (b) trajectory over epochs
    ax = fig.add_subplot(gs[1])
    for seed, m in ((1, "o"), (2, "o")):
        o = E[(E.model == "GEFormer") & (E.scenario == "F2024m") & (E.seed == seed)]
        ax.plot(o.within_spearman, o.pooled_r, marker=m, markersize=1.6, linestyle="none", color=GREY, alpha=0.5)
    ax.plot(run.within_spearman, run.pooled_r, color=INK2, linewidth=0.5, alpha=0.6)
    ax.scatter(run.within_spearman, run.pooled_r, c=run.index, cmap="Greys", s=6, zorder=3, vmin=-30, vmax=100)
    for e, col, off, ha in ((eP, ORANGE, (7, 2), "left"), (eS, BLUE, (-6, 12), "right")):
        ax.plot(run.loc[e, "within_spearman"], run.loc[e, "pooled_r"], "o", color=col, markersize=7, markeredgecolor="white",
                zorder=5)
        ax.annotate(f"epoch {e}", (run.loc[e, "within_spearman"], run.loc[e, "pooled_r"]), xytext=off,
                    textcoords="offset points", color=col, fontsize=7, ha=ha,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    ax.set_xlabel("within-environment Spearman")
    ax.set_ylabel("pooled Pearson r")
    ax.text(0.97, 0.03, "each dot: one epoch\n(dark = late);\n3 seeds, one in full", transform=ax.transAxes, fontsize=6.3, color=INK2, va="bottom", ha="right")
    panel_label(ax, "b", -0.2)
    # (c) environment-mean share at the two chosen epochs, 18 runs
    ax = fig.add_subplot(gs[2])
    X = pd.read_csv(R / "wp4/e1_exact_runs.csv")
    for _, r_ in X.iterrows():
        ax.plot([0, 1], [r_.phi_e_S, r_.phi_e_P], color=GREY, linewidth=0.7, alpha=0.8, zorder=1)
    ax.scatter(np.zeros(len(X)), X.phi_e_S, color=BLUE, s=16, zorder=3, edgecolor="white", linewidth=0.4)
    ax.scatter(np.ones(len(X)), X.phi_e_P, color=ORANGE, s=16, zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["within-env.\nSpearman", "pooled\nPearson"], fontsize=6.8)
    ax.set_xlabel("epoch chosen by", fontsize=7)
    ax.set_ylabel("environment-mean share φ")
    n_up = int((X.phi_e_P > X.phi_e_S).sum())
    ax.set_title(f"higher in {n_up} of {len(X)} runs", fontsize=7, color=INK)
    panel_label(ax, "c", -0.32)
    save(fig, "fig2_tstar")


# ---------------------------------------------------------------------------------------------------------- Figure 3
def fig3():
    L = pd.read_csv(R / "a1/levels.csv")
    m = L.groupby(["scenario", "method"])[["within_spearman", "pooled_pearson"]].mean()
    # GEFormer: gate-2 report (docs/gate2_geformer_2026-09-26.md, results section); means over three seeds
    geo = {("F2024m", "own"): (0.078, 0.389), ("F2024m", "fair"): (0.162, 0.187), ("F2022m", "own"): (0.216, 0.363),
           ("F2022m", "fair"): (0.274, 0.162)}
    geo_reml = {"F2024m": 0.225, "F2022m": 0.223}
    cols = []
    for model in ("GEFormer", "GE-BiFormer"):
        for sc in ("F2024m", "F2022m"):
            if model == "GEFormer":
                own, fair = geo[(sc, "own")], geo[(sc, "fair")]
                reml = geo_reml[sc]
            else:
                own = tuple(m.loc[(sc, "GEBiFormer_own"), ["within_spearman", "pooled_pearson"]])
                fair = tuple(m.loc[(sc, "GEBiFormer_fair"), ["within_spearman", "pooled_pearson"]])
                reml = m.loc[(sc, "B1r_gblup_reml"), "within_spearman"]
            cols.append((model, sc, own, fair, reml))
    # resolved contrasts O = own - fair (95 % CIs, docs): spearman, pooled
    resolved = {("GEFormer", "F2024m"): (True, True), ("GEFormer", "F2022m"): (False, False),
                ("GE-BiFormer", "F2024m"): (True, False), ("GE-BiFormer", "F2022m"): (False, False)}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    xs = np.array([0, 1, 2.6, 3.6])
    for k, (ax, title, idx) in enumerate(zip(axes, ("Within-environment Spearman", "Pooled Pearson r"), (0, 1))):
        for x, (model, sc, own, fair, reml) in zip(xs, cols):
            ax.plot([x, x], [own[idx], fair[idx]], color=GREY, linewidth=1.0, zorder=1)
            ax.plot(x, own[idx], "o", color=ORANGE, markersize=6.5, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
            ax.plot(x, fair[idx], "o", color=BLUE, markersize=6.5, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
            if idx == 0:
                ax.plot([x - 0.28, x + 0.28], [reml, reml], color=INK, linewidth=1.0, linestyle=(0, (2, 1.5)), zorder=2)
            if resolved[(model, sc)][idx]:
                ax.text(x + 0.14, (own[idx] + fair[idx]) / 2, "resolved", fontsize=6.3, color=INK, va="center", ha="left")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c[1]}" for c in cols], fontsize=7)
        for x0, x1, name in ((0, 1, "GEFormer"), (2.6, 3.6, "GE-BiFormer")):
            ax.text((x0 + x1) / 2, -0.15, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.5, color=INK, fontweight="bold")
        ax.set_xlim(-0.6, 4.3)
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.axhline(0, color=INK2, linewidth=0.5)
        ax.set_ylabel("Spearman" if idx == 0 else "Pearson r")
        panel_label(ax, "ab"[idx], -0.12)
    axes[1].set_ylim(0, 0.66)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=6.5),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=6.5),
               Line2D([0], [0], color=INK, linewidth=1, linestyle=(0, (2, 1.5)))]
    fig.legend(handles, ["public code's pipeline (choices on the test year)", "matched pipeline (choices before the test year)",
                         "REML-GBLUP (same environments)"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06),
               fontsize=7)
    fig.subplots_adjust(bottom=0.26, wspace=0.28)
    save(fig, "fig3_paired_reruns")


# ---------------------------------------------------------------------------------------------------------- Figure 4
LABEL = {"dl_g": "Network, within-env. loss", "dl_ge": "Network, within-env. loss + covariates", "gxe_gbm": "G×E boosting (hist. covariates)",
         "gxe_gbm_real": "G×E boosting (observed covariates)", "rn_ridge": "Reaction-norm ridge (hist. cov.)",
         "rn_ridge_real": "Reaction-norm ridge (observed cov.)", "reml": "Two-stage GBLUP (REML)", "reml_x0.1": "Two-stage GBLUP, λ×0.1",
         "reml_x10": "Two-stage GBLUP, λ×10", "reml_x100": "Two-stage GBLUP, λ×100", "ridge_pc20": "Ridge on 20 PCs", "rf": "Random forest",
         "gbm": "Gradient boosting", "knn10": "kNN (10)", "knn30": "kNN (30)", "mlp": "Two-stage perceptron",
         "R_CV": "Pick by cross-validation", "R_FW": "Pick by forward history", "R_EQ": "Equal-weight ensemble",
         "R_STK_CV": "Stacking on cross-validation", "R_STK_FW": "Stacking on forward history", "Oracle": "Oracle (best single, hindsight)"}
GROUPS = [("Rules for picking or combining", ["R_CV", "R_FW", "R_EQ", "R_STK_CV", "R_STK_FW", "Oracle"]),
          ("Single methods", ["reml", "reml_x0.1", "reml_x10", "reml_x100", "ridge_pc20", "rf", "gbm", "knn10", "knn30", "mlp"]),
          ("Covariate learners and networks", ["rn_ridge", "rn_ridge_real", "gxe_gbm", "gxe_gbm_real", "dl_g", "dl_ge"])]


def sign_colour(r):
    return BLUE if r.lo > 0 else (ORANGE if r.hi < 0 else GREY)


def forest(P, scope, metrics, fname, title, figsize=(7.4, 6.6), res=None, two_scopes=None):
    Q = P[(P.scope == scope)] if two_scopes is None else P
    ypos, labels, y = {}, [], 0
    order = []
    for g, items in GROUPS:
        y -= 0.9
        order.append(("hdr", g, y))
        sub = P[(P.scope == "all48") & (P.metric == metrics[0]) & P["item"].isin(items)].sort_values("est", ascending=False)
        for it in sub["item"]:
            y -= 1
            order.append(("row", it, y))
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, met in zip(axes, metrics):
        for kind, name, yy in order:
            if kind == "hdr":
                if ax is axes[0]:
                    ax.text(-0.02, yy, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.2,
                            fontweight="bold", color=INK)
                continue
            if two_scopes is None:
                r = Q[(Q["item"] == name) & (Q.metric == met)].iloc[0]
                ax.errorbar(r.est, yy, xerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o" if not name.startswith(("R_", "Oracle")) else "s",
                            color=sign_colour(r), ms=3.6, capsize=1.6, lw=0.8, capthick=0.8)
            else:
                for sc, mk, dy, fc in zip(two_scopes, ("o", "D"), (0.16, -0.16), (INK, "white")):
                    rr = Q[(Q["item"] == name) & (Q.metric == met) & (Q.scope == sc)]
                    if rr.empty:
                        continue
                    r = rr.iloc[0]
                    ax.errorbar(r.est, yy + dy, xerr=[[r.est - r.lo], [r.hi - r.est]], fmt=mk, color=INK, mfc=fc, ms=3.4,
                                capsize=1.4, lw=0.7, capthick=0.7)
        ax.axvline(0, color=INK, linewidth=0.7)
        if res:
            ax.axvspan(-res, res, color=GRID, alpha=0.7, zorder=0, linewidth=0)
        ax.set_xlabel({"spearman": "difference from GBLUP (Spearman)", "sel_diff_f10": "difference from GBLUP (SD units)"}[met])
        ax.set_title(title[metrics.index(met)], loc="left", fontsize=8)
        ax.grid(axis="x", color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if two_scopes is not None:
            ax.set_xticks(np.arange(-0.15, 0.051, 0.05))
    axes[0].set_yticks([yy for k, n, yy in order if k == "row"])
    axes[0].set_yticklabels([f"{LABEL[n]}  ({int(Q[(Q['item'] == n) & (Q.metric == metrics[0])]['k'].iloc[0]) if two_scopes is None else ''})".replace("  ()", "")
                             for k, n, yy in order if k == "row"], fontsize=7)
    return fig, axes


def fig4():
    P = pd.read_csv(R / "summary48/pooled.csv")
    res = 3 / np.sqrt(118438)
    fig, axes = forest(P, "all48", ["spearman", "sel_diff_f10"], "fig4",
                       ["Within-environment Spearman", "Top-10 % selection differential"], res=res)
    handles = [Line2D([0], [0], marker="o", color=BLUE, linestyle="none", ms=4), Line2D([0], [0], marker="o", color=GREY, linestyle="none", ms=4),
               Line2D([0], [0], marker="o", color=ORANGE, linestyle="none", ms=4),
               Rectangle((0, 0), 1, 1, facecolor=GRID)]
    fig.legend(handles, ["better (interval above 0)", "tied", "worse (interval below 0)", "±3/√N, the resolution"],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 0.015), fontsize=7)
    fig.suptitle("No method or rule available in advance beats cell-level GBLUP (48 forward target years, 6 datasets, 3 crops)", fontsize=8.3,
                 x=0.02, ha="left", y=0.995)
    fig.subplots_adjust(left=0.30, bottom=0.11, top=0.94, wspace=0.08)
    fig.text(0.02, 0.005, "Difference from cell-level GBLUP; numbers in brackets are the target years pooled (DerSimonian–Laird).",
             fontsize=6.3, color=INK2)
    save(fig, "fig4_forest48")
    # supplementary: original set versus independent set
    fig, axes = forest(P, None, ["spearman"], "figS1", ["Within-environment Spearman: original 31 years (●) and independent 17 years (◆)"],
                       figsize=(6.0, 6.2), two_scopes=["orig31", "new17"])
    fig.subplots_adjust(left=0.42, bottom=0.08, top=0.95)
    save(fig, "figS1_orig_vs_independent")


# ---------------------------------------------------------------------------------------------------------- Figure 5
def fig5():
    W = pd.read_csv(R / "b3d/choices_weights.csv")
    cols = [c for c in W.columns if c.startswith("w_fw_")]
    S = W[cols].div(W[cols].sum(axis=1), axis=0)
    unstable = ["w_fw_rf", "w_fw_gbm", "w_fw_knn10", "w_fw_knn30", "w_fw_mlp"]
    W["cell"] = S["w_fw_cell_reml"]
    W["flex"] = S[unstable].sum(axis=1)
    W["other"] = 1 - W["cell"] - W["flex"]
    Y = pd.read_csv(R / "b3d/year_effects.csv")
    Y = Y[(Y.contrast == "H7") & (Y.metric == "spearman")][["dataset", "target", "d", "se"]]
    D = W.merge(Y, on=["dataset", "target"])
    mk = {"ESWYT": "o", "GEM_IA": "s", "MU_SOY": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    for ds, g in D.groupby("dataset"):
        ax.plot(g.n_fw_history_envs, g["flex"], mk[ds], color=ORANGE, ms=5, mec="white", mew=0.5)
        ax.plot(g.n_fw_history_envs, g["cell"], mk[ds], color=BLUE, ms=5, mec="white", mew=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("environments in the forward history")
    ax.set_ylabel("share of the stacking weight")
    ax.set_ylim(-0.03, 1.03)
    ax.text(0.03, 0.625, "random forest, boosting,\nnearest neighbours, perceptron", color=ORANGE, transform=ax.transAxes, fontsize=6.8, va="center")
    ax.text(0.60, 0.13, "cell-level GBLUP", color=BLUE, transform=ax.transAxes, fontsize=6.8, ha="left", va="center")
    ax.legend([Line2D([0], [0], marker=m_, color="w", markerfacecolor=GREY, markersize=5) for m_ in ("o", "s", "^")],
              ["ESWYT", "GEM_IA", "MU_SOY"], frameon=False, fontsize=6.5, loc="upper right", handletextpad=0.2, borderpad=0.2)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    panel_label(ax, "a", -0.16)
    ax = axes[1]
    for ds, g in D.groupby("dataset"):
        ax.errorbar(g.n_fw_history_envs, g.d, yerr=1.96 * g.se, fmt=mk[ds], color=INK, mfc=INK, ms=4.4, capsize=1.5, lw=0.7,
                    label={"ESWYT": "ESWYT (12 years)", "GEM_IA": "GEM_IA (4)", "MU_SOY": "MU_SOY (1)"}[ds])
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("environments in the forward history")
    ax.set_ylabel("stacking minus cell-level GBLUP,\nwithin-env. Spearman")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right", handletextpad=0.3)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    panel_label(ax, "b", -0.2)
    ax.text(0.98, 0.97, "descriptive: 17 independent target years,\nnot a tested hypothesis", transform=ax.transAxes, fontsize=6.3, color=INK2, va="top", ha="right")
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig5_history_length")


# ---------------------------------------------------------------------------------------------------------- Figure 6
def fig6():
    v1 = json.load(open(R / "wp2/validation/verdict.json"))
    v2 = json.load(open(R / "wp2/validation_v2/verdict.json"))
    fig = plt.figure(figsize=(7.4, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.3], wspace=1.0)
    ax = fig.add_subplot(gs[0])
    rows = [("A1 detects a choice on\nthe reported data", "V1_A1_fail_rate_P1P3", True),
            ("A1 false alarm\n(clean pipeline)", "V1_A1_nonpass_rate_P0P2", False),
            ("A2 detects a pooled criterion\nwith a poor pick", "V2_A2_sensitivity", True),
            ("A2 false alarm", "V2_A2_false_alarm", False)]
    for i, (lab, key, good) in enumerate(rows):
        for v, dy, mk, name in ((v1, 0.14, "o", "round 1"), (v2, -0.14, "D", "round 2 (new seed)")):
            p, lo, hi = v[key]
            ax.errorbar(p, -i + dy, xerr=[[p - lo], [hi - p]], fmt=mk, color=BLUE if good else ORANGE, ms=4.2, capsize=1.6, lw=0.8)
    ax.set_yticks([-i for i in range(4)])
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.set_xlim(-0.03, 1.03)
    ax.axvline(0.8, color=GREY, linewidth=0.6, linestyle=":")
    ax.axvline(0.1, color=GREY, linewidth=0.6, linestyle=":")
    ax.text(0.8, 0.62, "≥ 0.8", fontsize=6, color=INK2, ha="center")
    ax.text(0.1, 0.62, "≤ 0.1", fontsize=6, color=INK2, ha="center")
    ax.set_xlabel("proportion of 500 replicates (95 % Wilson interval)")
    ax.grid(axis="x", color=GRID, linewidth=0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Simulation validation", loc="left", fontsize=8)
    ax.text(0.5, -0.24, "● round 1     ◆ round 2 (new random seed)\nblue: should be high; orange: should be low", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.3, color=INK2)
    panel_label(ax, "a", -0.5)
    ax.set_ylim(-3.4, 0.9)
    # (b) real cases
    ax = fig.add_subplot(gs[1])
    cs = json.load(open(R / "wp2/cases/cases_summary.json"))
    g2 = json.load(open(R / "wp2/case_gate2_summary.json"))
    groups = [("GEFormer, public pipeline", [g2["F2024m_own"], g2["F2022m_own"]], "own"),
              ("GEFormer, matched pipeline", [g2["F2024m_fair"], g2["F2022m_fair"]], "fair"),
              ("GE-BiFormer, public pipeline", [v for k, v in cs.items() if k.startswith("gebiformer") and "_own_seed" in k and "clean" not in k], "own"),
              ("GE-BiFormer, own_clean", [v for k, v in cs.items() if "own_clean" in k], "oc"),
              ("GE-BiFormer, matched pipeline", [v for k, v in cs.items() if k.startswith("gebiformer") and "_fair_seed" in k], "fair"),
              ("Benchmark forward panels,\nclean pipeline", [v for k, v in cs.items() if k.startswith("b3_")], "clean")]
    checks = [("A1_overlap", "A1\nchosen on\nreported\ndata"), ("A2_criterion", "A2\ncriterion"), ("A3_source", "A3\nenv.-mean\nshare"),
              ("A4_optimism", "A4\ninflation"), ("A5_baseline", "A5\nvs GBLUP")]
    colour = {"PASS": OK, "WARN": WARN, "FAIL": FAIL, "better": BLUE, "tied": GREY, "worse": ORANGE, "UNKNOWN": GREY}
    for i, (name, runs, _) in enumerate(groups):
        for j, (ck, _) in enumerate(checks):
            vals = [r[ck]['verdict'] if isinstance(r[ck], dict) else r[ck] for r in runs]
            c = pd.Series(vals).value_counts()
            top = c.index[0]
            n = len(vals)
            mixed = len(c) > 1 and c.iloc[0] == c.iloc[1]
            if mixed:
                txt = "\n".join(f"{k} {v}" for k, v in c.items())
                fc = "#6b7280"
            else:
                txt = f"{top}\n{c.iloc[0]}/{n}" if n > 1 else top
                fc = colour[top]
            ax.add_patch(Rectangle((j + 0.04, -i - 1 + 0.06), 0.92, 0.88, facecolor=fc, edgecolor="white", alpha=0.92 if ck != "A5_baseline" else 0.85))
            ax.text(j + 0.5, -i - 0.5, txt, ha="center", va="center", fontsize=6.0, color="white", fontweight="bold")
    ax.set_xlim(0, 5)
    ax.set_ylim(-len(groups), 0)
    ax.set_yticks([-i - 0.5 for i in range(len(groups))])
    ax.set_yticklabels([g[0] for g in groups], fontsize=6.8)
    ax.xaxis.tick_top()
    ax.set_xticks([j + 0.5 for j in range(5)])
    ax.set_xticklabels([c[1] for c in checks], fontsize=6.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Real cases", loc="left", fontsize=8, pad=38)
    ax.text(0.0, -0.03, "each cell: most frequent verdict and its count over runs (seeds and scenarios); grey lists both when equally frequent;\nA5 compares with REML-GBLUP",
            transform=ax.transAxes, fontsize=6.0, color=INK2, va="top")
    panel_label(ax, "b", -0.75, 1.2)
    save(fig, "fig6_tool")


FIGS = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6}
if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for w in which:
        FIGS[w]()
