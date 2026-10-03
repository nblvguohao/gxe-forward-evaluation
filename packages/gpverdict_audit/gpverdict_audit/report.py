"""Plain-language report of a gpverdict-audit result (Markdown, HTML with an inline SVG plot, or JSON).
Same layout and styling as GPverdict 1.1.1."""
import html
import json
import math
import re

import numpy as np

CHECKS = [("A1_overlap", "Was the choice made on the reported data?"),
          ("A2_criterion", "Is the selection criterion admissible for within-environment selection?"),
          ("A3_source", "Where does the pooled score come from?"),
          ("A4_optimism", "How much does choosing on the reported data inflate the reported value?"),
          ("A5_baseline", "Is the lead over the baseline resolvable?")]
BADGE = {"FAIL": "#b2182b", "WARN": "#d6861a", "PASS": "#3c8c4a", "UNKNOWN": "#777777", "better": "#2166ac",
         "worse": "#b2182b", "tied": "#777777", "detectable, below resolution": "#777777"}


def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def render_markdown(res):
    out = ["# Tuning audit", "", f"**{res['summary']}**", ""]
    for key, question in CHECKS:
        if key not in res:
            continue
        r = res[key]
        out += [f"## {key.split('_')[0]} · {question}", "", f"- [{r['verdict']}] {r['message']}", ""]
        if key == "A2_criterion":
            out += ["| Criterion | Candidate it picks |", "|---|---|"]
            out += [f"| {c} | {p} |" for c, p in r["picks"].items()]
            out += ["", f"Author's criterion: `{r['criterion']}`; author's pick: `{r['chosen']}`, rank "
                        f"{r['chosen_rank_within_spearman']} of {r['candidates']} on within-environment Spearman."]
        if key == "A3_source":
            out += [f"phi = {_f(r['phi'], 2)} (median over candidates {_f(r['median_phi_all_candidates'], 2)}); "
                    f"within-environment scale u = {_f(r['u'])}, pooled-Pearson optimum u* = {_f(r['u_star'])}."]
        if key == "A4_optimism" and "optimism" in r:
            out += [f"Metric `{r['metric']}`: reported {_f(r['reported'])}; estimated optimism {_f(r['optimism'])} "
                    f"[{_f(r['optimism_interval'][0])}, {_f(r['optimism_interval'][1])}] from {r['bootstrap_reps']} environment "
                    f"bootstrap resamples; deployable estimate {_f(r['deploy_estimate'])}; the same candidate is picked in "
                    f"{_f(100 * r['pick_stability'], 0)} % of resamples."]
        if key == "A5_baseline" and "cells" in r:
            out += [f"{r['cells']:,} cells in {r['environments']} environments; top-10 % selection differential difference "
                    f"{_f(r['d_sel_diff'])} [{_f(r['interval_sel_diff'][0])}, {_f(r['interval_sel_diff'][1])}]."]
        out.append("")
    C = sorted(res["candidates"], key=lambda c: -c["within_spearman"] if math.isfinite(c["within_spearman"]) else 1e9)
    out += ["## Candidates (top 20 by within-environment Spearman)", "",
            "| Candidate | Within Spearman | Pooled Pearson | Pooled MSE | phi | log(u/u*) |", "|---|---:|---:|---:|---:|---:|"]
    out += [f"| {c['cand']} | {_f(c['within_spearman'])} | {_f(c['pooled_pearson'])} | {_f(c['pooled_mse'], 2)} | "
            f"{_f(c['phi'], 2)} | {_f(c['log_u_over_ustar'], 2)} |" for c in C[:20]]
    out += ["", "---", "gpverdict-audit " + __import__("gpverdict_audit").__version__ + " · Lv & Gu · computations run locally."]
    return "\n".join(out)


def scatter_svg(res, w=560, h=360):
    """Pooled Pearson (x) against within-environment Spearman (y) for every candidate; the author's pick in red."""
    C = [c for c in res["candidates"] if math.isfinite(c["pooled_pearson"]) and math.isfinite(c["within_spearman"])]
    if len(C) < 2:
        return ""
    chosen = res["A2_criterion"]["chosen"]
    x = np.array([c["pooled_pearson"] for c in C])
    y = np.array([c["within_spearman"] for c in C])
    pad = 50
    sx = lambda v: pad + (v - x.min()) / (np.ptp(x) or 1) * (w - 2 * pad)
    sy = lambda v: h - pad - (v - y.min()) / (np.ptp(y) or 1) * (h - 2 * pad)
    pts = []
    for c, xv, yv in zip(C, x, y):
        col, r = ("#b2182b", 6) if c["cand"] == chosen else ("#4a6fa5", 3.5)
        pts.append(f"<circle cx='{sx(xv):.1f}' cy='{sy(yv):.1f}' r='{r}' fill='{col}' fill-opacity='0.8'>"
                   f"<title>{html.escape(str(c['cand']))}: pooled r {xv:.3f}, within Spearman {yv:.3f}</title></circle>")
    axes = (f"<line x1='{pad}' y1='{h-pad}' x2='{w-pad}' y2='{h-pad}' stroke='#555'/>"
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{h-pad}' stroke='#555'/>"
            f"<text x='{w/2}' y='{h-12}' text-anchor='middle' font-size='12'>pooled Pearson</text>"
            f"<text x='14' y='{h/2}' text-anchor='middle' font-size='12' transform='rotate(-90 14 {h/2})'>within-environment Spearman</text>"
            f"<text x='{pad}' y='{h-pad+16}' font-size='10'>{x.min():.3f}</text><text x='{w-pad}' y='{h-pad+16}' font-size='10' text-anchor='end'>{x.max():.3f}</text>"
            f"<text x='{pad-4}' y='{h-pad}' font-size='10' text-anchor='end'>{y.min():.3f}</text><text x='{pad-4}' y='{pad+4}' font-size='10' text-anchor='end'>{y.max():.3f}</text>")
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px' role='img' aria-label='candidates'>{axes}{''.join(pts)}</svg>"
            "<p style='font-size:13px;color:#555'>Each point is a candidate; red is the author's pick. A pick far right but "
            "low is chosen for its environment means, not for ranking within environments.</p>")


def render_html(res):
    md = render_markdown(res)
    body, in_table = [], False
    for line in md.split("\n"):
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        esc = re.sub(r"`(.+?)`", r"<code>\1</code>", esc)
        if line.startswith("|"):
            cells = [c.strip() for c in esc.strip("|").split("|")]
            if set(line.replace("|", "").strip()) <= set("-: "):
                continue
            if not in_table:
                body.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</tbody></table>")
            in_table = False
        if line.startswith("# "):
            body.append(f"<h1>{esc[2:]}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{esc[3:]}</h2>")
            if line.startswith("## A3"):
                body.append(scatter_svg(res))
        elif line.startswith("- ["):
            v = line[3:line.index("]")]
            col = BADGE.get(v, "#777777")
            body.append(f"<p class='v' style='border-left-color:{col}'><span class='b' style='background:{col}'>"
                        f"{html.escape(v)}</span> {esc[esc.index(']') + 2:]}</p>")
        elif line == "---":
            body.append("<hr>")
        elif line.strip():
            body.append(f"<p>{esc}</p>")
    if in_table:
        body.append("</tbody></table>")
    css = ("body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1d2430;background:#fff;line-height:1.5}html{background:#fff;color-scheme:light}"
           "table{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:14px}th,td{border-bottom:1px solid #dde2ea;padding:5px 8px;text-align:right}"
           "th:first-child,td:first-child{text-align:left}p.v{background:#f6f7f9;border-left:4px solid #3c8c4a;padding:8px 12px;margin:6px 0}"
           "span.b{color:#fff;border-radius:3px;padding:1px 6px;font-size:12px;font-weight:600;margin-right:6px}"
           "code{background:#f1f3f6;padding:1px 4px;border-radius:3px}h2{margin-top:28px;border-bottom:2px solid #e8ebf0;padding-bottom:4px;font-size:18px}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Tuning audit</title><style>{css}</style></head><body>{''.join(body)}</body></html>"


def to_json(res):
    def conv(x):
        if isinstance(x, (np.floating, float)):
            return None if not math.isfinite(float(x)) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(i) for i in x]
        return x
    return json.dumps(conv(res), indent=1)
