"""SX headroom check (docs/prereg_headroom_secondary_2026-09-30.md). Descriptive, not confirmatory.

  python scripts/headroom_secondary.py panels            # one panel per target year (no y stored)
  SCRAMBLE=1 python scripts/headroom_secondary.py panels # same with every yield of years >= Y replaced by junk
  python scripts/headroom_secondary.py analyze           # gates (consistency, scramble identity, a > 0), then scoring

Environment: BENCH, OUT, COUNT_DIR (results/count_secondary), HEADROOM_COMMIT, dartgxe.forward.data variables."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as tdist

from dartgxe.forward import secondary
from dartgxe.forward.data import LOADERS
from dartgxe.forward.pool import dersimonian_laird, floor_se

BENCH, OUT, COUNT = Path(os.environ["BENCH"]), Path(os.environ["OUT"]), Path(os.environ["COUNT_DIR"])
SPL = json.load(open(BENCH / "splits.json"))["G2F"]
TARGETS = [2016, 2018, 2020, 2022]
MIN_N, B, SEED = 25, 2000, 20260930
N_H1 = int(json.load(open(COUNT / "summary.json"))["N"])
MODELS = ["s0", "s1", "s1h", "s1g", "s1n", "s1_lor"]


def run_meta():
    import torch
    return {"commit": os.environ.get("HEADROOM_COMMIT", "unknown"), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version()}


def scrambled(ds, Y):
    c = ds.cells.copy()
    late = (c["year"] >= Y).to_numpy()
    c.loc[late, "y"] = 1e6 * np.random.default_rng([Y, 7]).normal(size=int(late.sum()))
    return type(ds)(ds.name, c, ds.markers, ds.min_n)


# ------------------------------------------------------------------------------------------------ panels
def panels():
    scr = os.environ.get("SCRAMBLE") == "1"
    pdir = OUT / ("panels_scramble" if scr else "panels")
    pdir.mkdir(parents=True, exist_ok=True)
    ds0 = LOADERS["G2F"]()
    plots = secondary.load_notes()
    notes = {m: secondary.note_cells(plots, m) for m in secondary.TRAITS_H}
    groups = json.load(open(COUNT / "exclusion_groups.json"))
    ny = pd.read_csv(COUNT / "ny.csv")
    for Y in TARGETS:
        fp = pdir / f"G2F_{Y}.parquet"
        if fp.exists():
            continue
        ds = scrambled(ds0, Y) if scr else ds0
        # NY environments: Y+1 envs with >= 25 genotyped cells of Y's scored hybrids (count §6.6; keys only)
        envs = SPL["scorable_environments"][str(Y)]
        hy = set(ds0.cells[(ds0.cells["year"] == Y) & ds0.cells["env"].isin(envs)]["genotype"])
        c = ds0.cells[(ds0.cells["year"] == Y + 1) & ds0.cells["genotype"].isin(hy)]
        per = c.groupby("env").size()
        ny_envs = sorted(per[per >= MIN_N].index)
        assert len(ny_envs) == int(ny.loc[ny["year"] == Y, "envs"].iloc[0])
        t0, start = time.time(), dt.datetime.now().isoformat(timespec="seconds")
        T, N, meta = secondary.sx_year(ds, Y, envs, notes, ny_envs=ny_envs, groups=groups)
        assert "y" not in T and "y" not in N
        T.to_parquet(fp, index=False)
        N.to_parquet(pdir / f"G2F_{Y}_ny.parquet", index=False)
        meta.update(run_meta(), scramble=scr, start=start, end=dt.datetime.now().isoformat(timespec="seconds"),
                    seconds=round(time.time() - t0, 1))
        json.dump(meta, open(fp.with_suffix(".json"), "w"), indent=1, default=str)
        print(Y, f"{meta['seconds']}s", {k: (round(v["a"], 4), [round(b, 4) for b in v["b"]]) for k, v in meta["fits"].items()},
              "calib", meta["calib_cells"], flush=True)


# ------------------------------------------------------------------------------------------------ scoring helpers
def spearman(y, p):
    if len(y) < 3 or np.std(y) == 0:
        return np.nan
    if np.std(p) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def pearson(y, p):
    if len(y) < 3 or np.std(y) == 0:
        return np.nan
    return 0.0 if np.std(p) == 0 else float(np.corrcoef(y, p)[0, 1])


def sel_diff(y, p):
    k = max(1, int(round(0.1 * len(y))))
    top = np.argsort(-p, kind="mergesort")[:k]
    return float(((y - y.mean()) / y.std())[top].mean()) if y.std() > 0 else np.nan


def tertile_sp(g, m):
    q = pd.qcut(g["x_silk"].rank(method="first"), 3, labels=False)
    v = [spearman(gg["y"].to_numpy(float), gg[m].to_numpy(float)) for _, gg in g.groupby(q) if len(gg) >= 8]
    return float(np.nanmean(v)) if v else np.nan


def two_way_se(D: pd.DataFrame, a: str, b: str, rng, B=B) -> float:
    """Year SE of mean env Spearman(a) - Spearman(b): resample environments and hybrids (one hybrid draw for all
    environments of the year), duplicates as ties."""
    envs = sorted(D["env"].unique())
    hyb = np.array(sorted(D["genotype"].unique()))
    hid = {h: i for i, h in enumerate(hyb)}
    per = [(g["genotype"].map(hid).to_numpy(), g["y"].to_numpy(float), g[a].to_numpy(float), g[b].to_numpy(float))
           for _, g in D.groupby("env")]
    out = np.empty(B)
    for r in range(B):
        ew = np.bincount(rng.integers(0, len(envs), len(envs)), minlength=len(envs))
        hw = np.bincount(rng.integers(0, len(hyb), len(hyb)), minlength=len(hyb))
        num = den = 0.0
        for k in np.nonzero(ew)[0]:
            h, y, pa, pb = per[k]
            rep = hw[h]
            if rep.sum() < 3:
                continue
            yy, aa, bb = np.repeat(y, rep), np.repeat(pa, rep), np.repeat(pb, rep)
            d = spearman(yy, aa) - spearman(yy, bb)
            if np.isfinite(d):
                num += ew[k] * d
                den += ew[k]
        out[r] = num / den if den else np.nan
    return float(np.nanstd(out, ddof=1))


def hk(ye):
    d, v = ye["d"].to_numpy(float), ye["se"].to_numpy(float) ** 2
    r = dersimonian_laird(floor_se(ye))
    ws = 1 / (v + r["tau2"])
    mu = (ws * d).sum() / ws.sum()
    q = (ws * (d - mu) ** 2).sum() / (len(d) - 1)
    se = np.sqrt(q / ws.sum())
    tq = tdist.ppf(0.975, len(d) - 1)
    w = 1 / v
    Q = float((w * (d - (w * d).sum() / w.sum()) ** 2).sum())
    return {"hk_lo": float(mu - tq * se), "hk_hi": float(mu + tq * se), "Q": Q,
            "I2": float(max(0.0, (Q - (len(d) - 1)) / Q)) if Q > 0 else 0.0}


def pooled(ye, level=0.95):
    r = dersimonian_laird(floor_se(ye[["dataset", "d", "se"]]), level=level)
    return {k: r[k] for k in ("est", "se", "lo", "hi", "tau2", "k")}


def arr_hash(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ analyze
def analyze():
    need = [OUT / d / f"G2F_{Y}{s}" for d in ("panels", "panels_scramble") for Y in TARGETS for s in (".parquet", "_ny.parquet", ".json")]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        sys.exit(f"panels missing: {missing}")
    # gate 1: scramble identity
    worst, hashes = 0.0, {}
    for Y in TARGETS:
        for suf, cols in ((".parquet", MODELS), ("_ny.parquet", ["s0", "s1"])):
            a = pd.read_parquet(OUT / "panels" / f"G2F_{Y}{suf}")
            b = pd.read_parquet(OUT / "panels_scramble" / f"G2F_{Y}{suf}")
            assert len(a) == len(b) and (a[["env", "genotype"]].values == b[["env", "genotype"]].values).all()
            for c in cols:
                x, z = a[c].to_numpy(float), b[c].to_numpy(float)
                worst = max(worst, float(np.max(np.abs(x - z)) / max(np.max(np.abs(x)), 1e-300)))
                hashes[f"{Y}{suf}:{c}"] = [arr_hash(x), arr_hash(z)]
        ma, mb = (json.load(open(OUT / d / f"G2F_{Y}.json")) for d in ("panels", "panels_scramble"))
        for k in ma["fits"]:
            fa = np.r_[ma["fits"][k]["a"], ma["fits"][k]["b"]]
            fb = np.r_[mb["fits"][k]["a"], mb["fits"][k]["b"]]
            worst = max(worst, float(np.max(np.abs(fa - fb)) / np.max(np.abs(fa))))
        worst = max(worst, abs(ma["delta_Y"] - mb["delta_Y"]) / ma["delta_Y"])
    metas = {Y: json.load(open(OUT / "panels" / f"G2F_{Y}.json")) for Y in TARGETS}
    a_pos = {Y: metas[Y]["fits"]["s1"]["a"] > 0 for Y in TARGETS}
    # gate 2: consistency with benchmark cell_reml; observed y joined only here
    ds = LOADERS["G2F"]()
    yv = ds.cells.set_index(["env", "genotype"])["y"]
    bench = pd.read_parquet(BENCH / "predictions_G2F.parquet")
    bench = bench[bench["method"].isin(["cell_reml", "rn_ridge"])].pivot_table(index=["env", "genotype"], columns="method", values="pred")
    Ts, Ns, cons = [], [], []
    for Y in TARGETS:
        T = pd.read_parquet(OUT / "panels" / f"G2F_{Y}.parquet")
        key = pd.MultiIndex.from_frame(T[["env", "genotype"]])
        T["y"] = yv.reindex(key).to_numpy()
        T["bench_cell_reml"] = bench["cell_reml"].reindex(key).to_numpy()
        T["rn_ridge"] = bench["rn_ridge"].reindex(key).to_numpy()
        for e, g in T.groupby("env"):
            cons.append(spearman(g["bench_cell_reml"].to_numpy(float), g["s0"].to_numpy(float)))
        N = pd.read_parquet(OUT / "panels" / f"G2F_{Y}_ny.parquet")
        N["y"] = yv.reindex(pd.MultiIndex.from_frame(N[["env", "genotype"]])).to_numpy()
        Ts.append(T.assign(target=Y)), Ns.append(N.assign(target=Y))
    T, NY = pd.concat(Ts, ignore_index=True), pd.concat(Ns, ignore_index=True)
    gates = {"scramble_max_rel_diff": worst, "scramble_pass": bool(worst <= 1e-9), "consistency_median": float(np.nanmedian(cons)),
             "consistency_min": float(np.nanmin(cons)), "bench_missing": int(T["bench_cell_reml"].isna().sum()),
             "a_positive": a_pos, "N": len(T), "N_count": N_H1}
    gates["consistency_pass"] = bool(gates["consistency_median"] >= 0.999 and gates["consistency_min"] >= 0.99 and gates["bench_missing"] == 0)
    gates["pass"] = bool(gates["scramble_pass"] and gates["consistency_pass"] and len(T) == N_H1)
    json.dump({**gates, "hashes": hashes}, open(OUT / "gates.json", "w"), indent=1, default=str)
    if not gates["pass"]:
        sys.exit(f"gate failed, no contrast written: {gates}")
    # per-environment metrics
    rows = []
    for (Y, e), g in T.groupby(["target", "env"]):
        y = g["y"].to_numpy(float)
        r = {"target": Y, "env": e, "n": len(g), "n_new": int(g["new"].sum())}
        for m in MODELS + ["rn_ridge"]:
            p = g[m].to_numpy(float)
            r[f"sp_{m}"], r[f"pe_{m}"], r[f"sd_{m}"] = spearman(y, p), pearson(y, p), sel_diff(y, p)
        for m in ("s0", "s1"):
            r[f"tert_{m}"] = tertile_sp(g, m)
        gn = g[g["new"]]
        for m in ("s0", "s1"):
            r[f"new_{m}"] = spearman(gn["y"].to_numpy(float), gn[m].to_numpy(float)) if len(gn) >= MIN_N else np.nan
        rows.append(r)
    E = pd.DataFrame(rows)
    E.to_parquet(OUT / "per_env.parquet", index=False)
    nrows = []
    for (Y, e), g in NY.groupby(["target", "env"]):
        y = g["y"].to_numpy(float)
        nrows.append({"target": Y, "env": e, "n": len(g), "sp_s0": spearman(y, g["s0"].to_numpy(float)),
                      "sp_s1": spearman(y, g["s1"].to_numpy(float))})
    EN = pd.DataFrame(nrows)
    EN.to_parquet(OUT / "per_env_ny.parquet", index=False)
    # year effects (env bootstrap for all; two-way bootstrap for the decision contrasts)
    rng = np.random.default_rng(SEED)
    contrasts = {"H1": ("sp_s1", "sp_s0"), "s1-s1g": ("sp_s1", "sp_s1g"), "s1-rn_ridge": ("sp_s1", "sp_rn_ridge"),
                 "lor": ("sp_s1_lor", "sp_s0"), "tertile": ("tert_s1", "tert_s0"), "new": ("new_s1", "new_s0"),
                 "s1g-s0": ("sp_s1g", "sp_s0"), "s1h-s0": ("sp_s1h", "sp_s0"), "s1h-s1": ("sp_s1h", "sp_s1"),
                 "s1n-s0": ("sp_s1n", "sp_s0"), "s1n-s1": ("sp_s1n", "sp_s1"),
                 "H1_pearson": ("pe_s1", "pe_s0"), "H1_seldiff": ("sd_s1", "sd_s0"), "s1h-s0_pearson": ("pe_s1h", "pe_s0")}
    ye = []
    for name, (a, b) in contrasts.items():
        for Y, g in E.groupby("target"):
            v = (g[a] - g[b]).dropna().to_numpy()
            bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
            ye.append({"contrast": name, "dataset": "G2F", "target": Y, "d": float(v.mean()), "se_env": float(bs.std(ddof=1)),
                       "n_env": len(v)})
    for Y, g in EN.groupby("target"):
        v = (g["sp_s1"] - g["sp_s0"]).to_numpy()
        bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        ye.append({"contrast": "NY", "dataset": "G2F", "target": Y, "d": float(v.mean()), "se_env": float(bs.std(ddof=1)), "n_env": len(v)})
    YE = pd.DataFrame(ye)
    tw = {}
    for Y in TARGETS:
        tw[("H1", Y)] = two_way_se(T[T["target"] == Y], "s1", "s0", np.random.default_rng([SEED, Y, 1]))
        tw[("NY", Y)] = two_way_se(NY[NY["target"] == Y], "s1", "s0", np.random.default_rng([SEED, Y, 2]))
    YE["se_twoway"] = [tw.get((c, t), np.nan) for c, t in zip(YE["contrast"], YE["target"])]
    YE["se"] = np.where(YE["se_twoway"].notna(), YE["se_twoway"], YE["se_env"])
    YE.to_csv(OUT / "year_effects.csv", index=False)
    PO = []
    for name, g in YE.groupby("contrast"):
        r = pooled(g)
        r_env = pooled(g.assign(se=g["se_env"]))
        PO.append({"contrast": name, **r, "lo_envboot": r_env["lo"], "hi_envboot": r_env["hi"], **hk(g),
                   "loyo": [pooled(g[g["target"] != Y]).get("est") for Y in TARGETS]})
    PO = pd.DataFrame(PO).set_index("contrast")
    PO.to_csv(OUT / "pooled.csv")
    res = 3 / np.sqrt(N_H1)
    h1, nyp = PO.loc["H1"], PO.loc["NY"]
    R = nyp["est"] / h1["est"] if h1["est"] != 0 else np.nan
    cond = {
        "a_resolution": bool(h1["est"] >= res),
        "b_ci": bool(h1["lo"] > 0),
        "c_loyo": bool(all(x > 0 for x in h1["loyo"])),
        "d_new": bool(PO.loc["new", "est"] > 0),
        "e_same_season": bool(PO.loc["s1-s1g", "est"] > 0 and PO.loc["s1-s1g", "est"] >= 0.5 * h1["est"]),
        "f_transfer": bool(nyp["est"] - norm.ppf(0.95) * nyp["se"] > 0 and R >= 0.5),
        "g_rn_ridge": bool(PO.loc["s1-rn_ridge", "est"] > 0),
        "h_maturity": bool(PO.loc["tertile", "est"] > 0),
        "i_gates_a": bool(gates["pass"] and all(a_pos.values())),
        "j_lor": bool(PO.loc["lor", "est"] > 0),
    }
    go = all(cond.values())
    verdict = {"GO": go, "conditions": cond, "H1": h1.to_dict(), "resolution": res, "N": N_H1, "NY": nyp.to_dict(),
               "transfer_ratio": R, "hk_includes_0": bool(h1["hk_lo"] <= 0 <= h1["hk_hi"]),
               "fits": {Y: metas[Y]["fits"] for Y in TARGETS}, "gates": gates,
               "note": "SX descriptive headroom check on previously seen outcomes; a GO is exploratory"}
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(YE.pivot_table(index="contrast", columns="target", values="d").round(4).to_string())
    print(PO[["est", "se", "lo", "hi", "lo_envboot", "hk_lo", "hk_hi", "tau2", "loyo"]].round(4).to_string())
    print(json.dumps({k: verdict[k] for k in ("GO", "conditions", "resolution", "transfer_ratio", "hk_includes_0")}, indent=1, default=float))
    print(json.dumps(verdict["fits"], indent=1, default=float))
    print("SX_ANALYZE_DONE")


if __name__ == "__main__":
    {"panels": panels, "analyze": analyze}[sys.argv[1]]()
