"""Table bodies (LaTeX rows) for the revised supplementary material.

Every value is read from a stored result file; nothing is transcribed by hand.
Output: submission_rev/supplementary/tables_rev/*.tex
Usage: python revision/scripts/make_rev_supplement_tables.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import make_figures as MF  # noqa: E402

POD = os.path.join(ROOT, "results_rev_pod")
LOC = os.path.join(ROOT, "results_rev_local")
OUT = os.path.join(ROOT, "submission_rev", "supplementary", "tables_rev")
os.makedirs(OUT, exist_ok=True)
PROBES = MF.AUDIT_PROBES
SHORT = {"ambient_translation_diag": "transl.", "ambient_translation": "transl.",
         "rotation": "rot.", "coordinate_permutation": "perm.", "domain_scaling_up": "dil.",
         "domain_scaling_down": "contr.", "fitness_rescale": "$f$ resc.",
         "monotone_log": "$f$ log", "monotone_spline": "$f$ spline",
         "translate_5": "transl. 5", "translate_50": "transl. 50",
         "translate_500": "transl. 500", "scale_0.1": "scale 0.1"}


def name(a):
    return a.replace("_", " ").replace("random search", "random search")


def write(fname, rows):
    with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{fname}: {len(rows)} rows")


def pts(x, nd=2):
    return f"{100 * x:.{nd}f}"


def bold(s, b):
    return f"$\\mathbf{{{s}}}$" if b else f"${s}$"


# ---------------------------------------------------------------- audit (20 seeds)
def audit():
    ag = pd.read_csv(os.path.join(POD, "e2_audit_inference.csv"))
    an = pd.read_csv(os.path.join(POD, "e2_anchors_e1b_inference.csv"))
    d = pd.concat([an, ag], ignore_index=True)
    d["dist"] = (d.q_V_u <= .05) | (d.q_CvM <= .05)
    d["pair"] = d.q_signflip <= .05
    rows = []
    for a in MF._order(d.agent.unique()):
        cells = []
        for p in PROBES:
            r = d[(d.agent == a) & (d.probe == p)].iloc[0]
            s = bold(pts(r.V), r.dist) + ("$^\\circ$" if r.pair and not r.dist else "")
            if r.crashed_runs > 0:
                s += "$^\\dagger$"
            cells.append(s)
        rows.append(f"{name(a)} & " + " & ".join(cells) + " \\\\")
    write("audit_V.tex", rows)
    rows = []
    for a in MF._order(d.agent.unique()):
        cells = [f"${pts(d[(d.agent == a) & (d.probe == p)].iloc[0].loss)}$" for p in PROBES]
        rows.append(f"{name(a)} & " + " & ".join(cells) + " \\\\")
    write("audit_loss.tex", rows)
    sig = d[d.dist | d.pair].copy()
    rows = []
    for _, r in sig.iterrows():
        rows.append(f"{name(r.agent)} & {SHORT[r.probe]} & ${pts(r.V)}$ [{pts(r.V_lo)}, "
                    f"{pts(r.V_hi)}] & ${pts(r.loss)}$ [{pts(r.loss_lo)}, {pts(r.loss_hi)}] & "
                    f"{r.q_signflip:.3f} & {r.qBY_signflip:.3f} & {r.q_V_u:.3f} & "
                    f"{r.q_CvM:.3f} \\\\")
    write("audit_sig.tex", rows)


def calibration():
    c = pd.read_csv(os.path.join(POD, "e2_calibration.csv"))
    lab = {"null_continuous": "null, continuous scores", "null_ties": "null, tied scores",
           "spread_only": "spread changes, mean equal", "mean_shift": "mean shift",
           "reviewer_example": "exchangeability counterexample"}
    rows = []
    for _, r in c.iterrows():
        f = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
        rows.append(f"{lab.get(r.scenario, r.scenario)} & {f(r.signflip)} & {f(r.V_u)} & "
                    f"{f(r.CvM)} \\\\")
    write("calibration.tex", rows)


# ---------------------------------------------------------------- dimension and budget
def dimension():
    ch = pd.read_csv(os.path.join(POD, "anchor_pod_checks.csv"))
    rows = []
    for D in (30, 50, 100):
        s = ch[ch.set == f"E3 anchors {D}-D"]
        inv = s[s.expect_inv]
        det = s[~s.expect_inv]
        rows.append(f"{D} & {int(inv.significant.sum())}/{len(inv)} & "
                    f"{int(det.significant.sum())}/{len(det)} & "
                    f"{100 * det.V.min():.2f}--{100 * det.V.max():.2f} \\\\")
    write("dim_anchors.tex", rows)
    e = pd.read_csv(os.path.join(POD, "e3_agents_violations.csv"))
    e["sig"] = e.q <= .05
    keys = e[e.sig][["agent", "probe"]].drop_duplicates()
    rows = []
    for _, k in keys.sort_values(["agent", "probe"]).iterrows():
        v = {D: e[(e.D == D) & (e.agent == k.agent) & (e.probe == k.probe)].iloc[0]
             for D in (10, 30)}
        rows.append(f"{k.agent} & {SHORT[k.probe]} & " + " & ".join(
            bold(pts(v[D].V), v[D].sig) for D in (10, 30)) + " \\\\")
    write("dim_agents.tex", rows)
    a = e.groupby(["agent", "D"]).A_id.first().unstack()
    rows = [f"{ag} & ${pts(r[10])}$ & ${pts(r[30])}$ \\\\" for ag, r in a.iterrows()]
    write("dim_agents_perf.tex", rows)


def budget():
    b = pd.read_csv(os.path.join(POD, "e4_budget.csv"))
    b["sig"] = b.q <= .05
    keys = b[b.sig][["agent", "probe"]].drop_duplicates()
    rows = []
    for _, k in keys.sort_values(["agent", "probe"]).iterrows():
        cells = []
        for B in (2000, 5000, 20000):
            r = b[(b.budget == B) & (b.agent == k.agent) & (b.probe == k.probe)].iloc[0]
            cells.append(bold(pts(r.V), r.sig))
        rows.append(f"{k.agent} & {SHORT[k.probe]} & " + " & ".join(cells) + " \\\\")
    write("budget.tex", rows)


# ---------------------------------------------------------------- attribution
KO = {"none": "as distributed", "les_sigma": "step size box-relative",
      "les_mu": "mean update box-relative", "les_shell": "both",
      "symbol_init": "initial points box-relative", "rldas_bound": "bounds in repair",
      "rldas_rw": "random-walk feature", "rldas_feat": "dispersion feature",
      "rldas_madde": "MadDE repair", "rldas_search": "resampling",
      "rldas_all": "all four", "rlhpsde_dfdc": "relative $\\epsilon$, no raise"}


def attribution():
    e = pd.read_csv(os.path.join(POD, "e5_attribution.csv"))
    rows = []
    for agent, g in e.groupby("agent", sort=True):
        kos = sorted(g.ko.unique(), key=lambda k: (k != "none", k != g.ko.iloc[-1], k))
        for ko in kos:
            h = g[g.ko == ko]
            if ko.startswith(("untrained", "fixed")):
                continue
            cells = ", ".join(f"{SHORT.get(p, p)} {bold(pts(r.V), r.q <= .05)}"
                              for p, r in h.set_index("probe").iterrows())
            rows.append(f"{agent} & {KO.get(ko, ko)} & {cells} \\\\")
        for kind, lab in (("untrained", "untrained weights"), ("fixed", "fixed action")):
            h = g[g.ko.str.startswith(kind)]
            if len(h):
                cells = ", ".join(
                    f"{SHORT.get(p, p)} ${100 * s.V.min():.2f}$--${100 * s.V.max():.2f}$"
                    for p, s in h.groupby("probe"))
                rows.append(f"{agent} & {lab} ({h.ko.nunique()} draws) & {cells} \\\\")
    write("attribution.tex", rows)


# ---------------------------------------------------------------- rotation control
def rotation():
    c = pd.read_csv(os.path.join(POD, "e12_control_table.csv"))
    lab = {"de_cr1_ball_free": "arithmetic, ball init., unbounded",
           "de_cr1_ball_clip": "arithmetic, ball init., clipping",
           "de_cr1_box_free": "arithmetic, box init., unbounded",
           "de_cr1_box_clip": "arithmetic, box init., clipping",
           "de_bin_ball_free": "binomial, ball init., unbounded",
           "de_bin_box_clip": "binomial, box init., clipping"}
    rows = []
    for v in lab:
        r = {s: c[(c.variant == v) & (c.split == s)].iloc[0] for s in ("train", "heldout")}
        rows.append(f"{lab[v]} & " + " & ".join(
            f"${pts(r[s].V)}$ ({r[s].p:.3f}) & {100 * r[s].x_out_box_rot:.0f}\\%"
            for s in ("train", "heldout")) + " \\\\")
    write("rotation_control.tex", rows)
    a = pd.read_csv(os.path.join(POD, "e12_agents_table.csv"))
    rows = []
    for ag in sorted(a.agent.unique()):
        r = {s: a[(a.agent == ag) & (a.split == s)].iloc[0] for s in ("train", "heldout")}
        rows.append(f"{ag} & " + " & ".join(
            f"{bold(pts(r[s].V), r[s].q <= .05)} & {100 * r[s].x_out_box_rot:.0f}\\%"
            for s in ("train", "heldout")) + " \\\\")
    write("rotation_agents.tex", rows)


# ---------------------------------------------------------------- held-out x probes
def heldout():
    d = pd.read_csv(os.path.join(POD, "e8_decomposition.csv"))
    h = d[d.split == "heldout"].copy()
    tr = pd.read_csv(os.path.join(POD, "e8_violations.csv")) if os.path.exists(
        os.path.join(POD, "e8_violations.csv")) else None
    h["sig"] = h.q <= .05
    rows = []
    for _, r in h[h.sig].sort_values(["agent", "probe"]).iterrows():
        vt = ""
        if tr is not None and "split" in tr.columns:
            t = tr[(tr.split == "train") & (tr.agent == r.agent) & (tr.probe == r.probe)]
            if len(t):
                vt = bold(pts(t.iloc[0].V), t.iloc[0].q <= .05)
        rows.append(f"{r.agent} & {SHORT[r.probe]} & {vt} & ${pts(r.V)}$ & "
                    f"${pts(r['shift'])}$ & ${pts(r.pres)}$ \\\\")
    write("heldout.tex", rows)


def law():
    l = pd.read_csv(os.path.join(POD, "law_rev.csv"), index_col=0)
    rows = []
    for a in MF._order(l.index):
        r = l.loc[a]
        rows.append(f"{name(a)} & ${pts(r.V)}$ & ${r.rank_train:.2f}$ & ${r.rank_heldout:.2f}$ & "
                    f"${r.G:+.2f}$ & ${pts(r.A_train)}$ & ${pts(r.A_heldout)}$ \\\\")
    write("law_full.tex", rows)
    t = pd.read_csv(os.path.join(POD, "law_rev_tests.csv"))
    lab = {"G": "$V$ vs.\\ $G$", "A_heldout": "$V$ vs.\\ held-out $\\perf$",
           "A_heldout_norm": "$V$ vs.\\ normalized held-out $\\perf$",
           "A_train": "$V$ vs.\\ training $\\perf$", "rank_train": "$V$ vs.\\ training rank",
           "G | rank_train (partial)": "$V$ vs.\\ $G$, partial on training rank"}
    rows = [f"{lab[r.target]} & {r['sample']} ($n={r.n}$) & ${r.rho:+.2f}$ & {r.p:.3f} \\\\"
            for _, r in t.iterrows()]
    write("law_tests.tex", rows)


# ---------------------------------------------------------------- objective layer (E6)
def layer():
    v = pd.read_csv(os.path.join(POD, "e6_violations.csv"))
    c = pd.read_csv(os.path.join(POD, "e6_cost.csv"))
    rows = []
    w = v.pivot_table(index=["agent", "probe"], columns="cond", values=["V", "q"])
    for (ag, p), r in w.iterrows():
        if r[("q", "audit")] <= .05 or r[("q", "audit_layer")] <= .05:
            rows.append(f"{ag} & {SHORT[p]} & {bold(pts(r[('V', 'audit')]), r[('q', 'audit')] <= .05)}"
                        f" & {bold(pts(r[('V', 'audit_layer')]), r[('q', 'audit_layer')] <= .05)} \\\\")
    write("layer_violations.tex", rows)
    rows = []
    for _, r in c.iterrows():
        rows.append(f"{r.agent} & ${pts(r.A_train_none)}$ & ${pts(r.d_train)}$ "
                    f"[{pts(r.lo_train)}, {pts(r.hi_train)}] & ${pts(r.A_heldout_none)}$ & "
                    f"${pts(r.d_heldout)}$ [{pts(r.lo_heldout)}, {pts(r.hi_heldout)}] \\\\")
    write("layer_cost.tex", rows)


# ---------------------------------------------------------------- retraining (E7)
def retrain():
    rows = []
    for tag, lab in (("", "RLDAS, 10-D"), ("_30D", "RLDAS, 30-D"), ("_deddqn", "DEDDQN, 10-D")):
        a = pd.read_csv(os.path.join(POD, f"e7_arms{tag}.csv"))
        a["cond"] = a.cond.str.replace(r"@\d+", "", regex=True)
        m = a.groupby(["cond", "split"])[["A_id", "V_monotone_log", "V_monotone_spline",
                                          "V_ambient_translation_diag"]].mean()
        for cond in ("pretrained", "control", "control+L", "layer", "layer_rawreward"):
            if (cond, "train") not in m.index:
                continue
            r, h = m.loc[(cond, "train")], m.loc[(cond, "heldout")]
            n = int((a.cond == cond).sum() / 2)
            rows.append(f"{lab} & {cond.replace('_', ' ')} & {n} & ${pts(r.A_id)}$ & "
                        f"${pts(h.A_id)}$ & ${pts(r.V_monotone_log)}$ / ${pts(r.V_monotone_spline)}$"
                        f" & ${pts(r.V_ambient_translation_diag)}$ \\\\")
        rows.append("\\midrule")
    write("retrain_arms.tex", rows[:-1])
    rows = []
    for tag, lab in (("", "RLDAS, 10-D"), ("_30D", "RLDAS, 30-D"), ("_deddqn", "DEDDQN, 10-D")):
        c = pd.read_csv(os.path.join(POD, f"e7_contrasts{tag}.csv"))
        for con, g in c.groupby("contrast", sort=False):
            t, h = g[g.split == "train"].iloc[0], g[g.split == "heldout"].iloc[0]
            # e7_stats.py already reports points
            rows.append(f"{lab} & {con.replace('_', ' ')} & ${t['mean']:.2f}$ [{t.lo:.2f}, "
                        f"{t.hi:.2f}] & ${h['mean']:.2f}$ [{h.lo:.2f}, {h.hi:.2f}] \\\\")
        rows.append("\\midrule")
    write("retrain_contrasts.tex", rows[:-1])


# ---------------------------------------------------------------- harness cost, embedding
def hcost():
    e = pd.read_csv(os.path.join(POD, "exp6_harness_10D_v2.csv.gz"), low_memory=False)
    o = pd.read_csv(os.path.join(POD, "exp1_anchors_10D.csv.gz"), low_memory=False)
    keys = ["optimizer", "function", "instance", "run"]
    b0 = o.drop_duplicates(keys)[keys + ["A_base"]]
    b1 = e.drop_duplicates(keys)[keys + ["A_base", "ties_base", "first_tie_base"]]
    m = b0.merge(b1, on=keys, suffixes=("_off", "_on"))
    from scipy import stats
    rows = []
    for opt, g in m.groupby("optimizer", sort=False):
        diff = g.A_base_on - g.A_base_off
        p = stats.wilcoxon(g.A_base_on, g.A_base_off).pvalue if (diff != 0).any() else np.nan
        ft = pd.to_numeric(g.first_tie_base, errors="coerce")
        ft = ft[ft > 0]
        rows.append(f"{name(opt)} & ${g.A_base_off.mean():.4f}$ & ${g.A_base_on.mean():.4f}$ & "
                    f"${100 * diff.mean():+.3f}$ & ${100 * diff.abs().max():.2f}$ & "
                    f"{'--' if np.isnan(p) else ('$<10^{-4}$' if p < 1e-4 else f'{p:.3f}')} & "
                    f"{100 * (pd.to_numeric(g.first_tie_base, errors='coerce') > 0).mean():.1f}\\% & "
                    f"{'--' if ft.empty else f'{ft.median():.0f}'} \\\\")
    write("hcost.tex", rows)


def embedding():
    n = pd.read_csv(os.path.join(LOC, "e10_natural.csv"))
    rows = []
    for (emb, B), g in n.groupby(["embedding", "budget"]):
        rows.append(f"{emb.replace('_', ' ')} & {B:,} & {int((g.ties > 0).sum())}/{len(g)} \\\\")
    write("embedding_natural.tex", rows)
    a = pd.read_csv(os.path.join(LOC, "e10_adversarial.csv"))
    a = a[a.stream == "adaptive_adversary"]
    rows = [f"{r.embedding.replace('_', ' ')} & {r.batch} & {r.insertions_until_tie} \\\\"
            for _, r in a.iterrows()]
    write("embedding_adversarial.tex", rows)


def overhead():
    d = pd.read_csv(os.path.join(LOC, "e13_overhead.csv"))
    w = d.pivot_table(index=["optimizer", "D", "budget"], columns="layer",
                      values="us_per_eval", sort=False)
    def dlt(r, layer):  # avoids printing a negative zero
        return round(r[layer] - r["none"], 1) + 0.0

    rows = []
    for (opt, D, B), r in w.iterrows():
        rows.append(f"{name(opt)} & {D} & {B:,} & ${r['none']:.1f}$ & ${dlt(r, 'frame'):+.1f}$ & "
                    f"${dlt(r, 'objective'):+.1f}$ & ${dlt(r, 'both'):+.1f}$ \\\\")
    write("overhead.tex", rows)


if __name__ == "__main__":
    audit()
    calibration()
    dimension()
    budget()
    attribution()
    rotation()
    heldout()
    law()
    layer()
    retrain()
    hcost()
    embedding()
    overhead()
