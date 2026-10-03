"""Figure: within-environment Spearman gain by KIND of information added (filled) and by model complexity added on top
(open), with 95 % intervals and the 3/sqrt(N) resolution of each information contrast (grey bar).
Every value is read from a committed result file; nothing is typed in. Run from the repository root:
    python3 scripts/make_fig_info_kind.py   ->  figures/fig_info_kind.{pdf,png}"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("results")


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


s48 = lambda item: row("summary48/pooled.csv", item=item, metric="spearman", scope="all48")
res48 = jget("headroom_ia_ib/verdict.json", "I-A", "all48", "resolution")
sp = lambda frac, con: row("headroom_sparse/pooled.csv", fraction=frac, contrast=con, metric="sp", range="all48")
ol_h = jget("headroom_oldlines/verdict.json", "decisions")
sx = jget("headroom_secondary/verdict.json", "H1")
rp = "rn_paper/ge_mean_primary_evidence.json"
fw = jget(rp, "states", "observed", "fw_vs_additive", "spearman")
kgfw = jget(rp, "states", "observed", "kernel_gated_vs_fw", "spearman")
n_obs = jget(rp, "states", "observed", "n_genotype_environment_units")
clac = jget("reanalysis/verdict.json", "verdict", "clac")
ol10 = row("headroom_oldlines/pooled.csv", contrast="ol1-ol0", metric="sp", range="all")
s2 = jget("headroom_sparse2/verdict.json", "decisions")
m2 = jget("headroom_sparse/verdict.json", "main_m2_m1")

# (label, info point or None, resolution, [(complexity label, est, lo, hi), ...], note)
cats = [
    ("None\n(new lines,\nnew year)", (0.0, 0.0, 0.0), res48,
     [(k, s48(k).est, s48(k).lo, s48(k).hi) for k in ("rf", "gbm", "mlp", "rn_ridge", "gxe_gbm", "dl_g", "R_STK_FW")],
     "reference: cell-level GBLUP"),
    ("Line's own\nhistory", (ol10.est, ol10.lo, ol10.hi), ol10.resolution, [], ""),
    ("Line ×\nlocation\nhistory", (ol_h["H2"]["est"], *ol_h["H2"]["ci"]), ol_h["H2"]["resolution"],
     [("marker G×loc", ol_h["H1"]["est"], *ol_h["H1"]["ci"])], "97.5 % (Holm)"),
    ("Same-season\nflowering\n(G2F)", (sx["est"], sx["lo"], sx["hi"]), jget("headroom_secondary/verdict.json", "resolution"), [], ""),
    ("Weather\nreaction\nnorm, tested\nlines (G2F)", (fw["estimate"], fw["ci_low"], fw["ci_high"]), 3 / n_obs ** 0.5,
     [("kernel-gated", kgfw["estimate"], kgfw["ci_low"], kgfw["ci_high"])], "vs additive reference"),
    ("Target year,\n25 %\nphenotyped", (sp("0.25", "m1-m0").est, sp("0.25", "m1-m0").lo, sp("0.25", "m1-m0").hi),
     sp("0.25", "m1-m0").resolution,
     [("learned corr.", m2["est"], m2["lo"], m2["hi"])] + [(k, s2[k]["est"], *s2[k]["ci"]) for k in ("stk", "dlres", "ecmxe")], ""),
    ("Target year,\n50 %\nphenotyped", (sp("0.5", "m1-m0").est, sp("0.5", "m1-m0").lo, sp("0.5", "m1-m0").hi),
     sp("0.5", "m1-m0").resolution, [], ""),
]

fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.axhline(0, color="0.4", lw=0.8)
for i, (lab, info, res, comp, note) in enumerate(cats):
    if i > 0:
        ax.plot([i - 0.32, i + 0.32], [res, res], color="0.6", lw=3, alpha=0.5, solid_capstyle="butt")
        e, lo, hi = info
        ax.errorbar(i - 0.12, e, yerr=[[e - lo], [hi - e]], fmt="o", color="#1f5f99", ms=6, capsize=3, zorder=3)
    for j, (cl, e, lo, hi) in enumerate(comp):
        x = i + 0.06 + (0.36 / max(len(comp), 1)) * j if len(comp) > 1 else i + 0.12
        ax.errorbar(x, e, yerr=[[e - lo], [hi - e]], fmt="o", mfc="white", color="#b3541e", ms=4.5, capsize=2, lw=0.9, zorder=3)
    if i == 0:
        e, (lo, hi) = clac["est"], clac["ci"]
        ax.errorbar(i - 0.2, e, yerr=[[e - lo], [hi - e]], fmt="D", color="#2e7d32", ms=5.5, capsize=3, zorder=4)
        ax.plot([i - 0.38, i - 0.02], [clac["resolution"]] * 2, color="0.6", lw=3, alpha=0.5, solid_capstyle="butt")
    if note:
        ax.text(i, -0.112, note, ha="center", va="top", fontsize=6.5, color="0.35")
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([c[0] for c in cats], fontsize=7.5)
ax.set_xlim(-0.6, len(cats) - 0.4)
ax.set_ylim(-0.125, 0.095)
ax.set_ylabel("Δ within-environment Spearman")
ax.plot([], [], "o", color="#1f5f99", label="information added")
ax.plot([], [], "o", mfc="white", color="#b3541e", label="model complexity added")
ax.plot([], [], "D", color="#2e7d32", label="published pipeline (CLAC, G2F; 97.5 % Holm)")
ax.plot([], [], color="0.6", lw=3, alpha=0.5, label="resolution 3/√N")
ax.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.1))
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/fig_info_kind.pdf")
fig.savefig("figures/fig_info_kind.png", dpi=200)
print("written figures/fig_info_kind.{pdf,png}")
