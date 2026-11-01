"""Generate the LaTeX tables of the supplementary material from the result files.

Every number in the supplement comes from here, so that no value is typed by hand.
Run inside the MetaBox venv (scipy, pandas):

    python experiments/make_supplement_tables.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "submission", "supplementary", "tables")
os.makedirs(OUT, exist_ok=True)

ANCHORS = ["random_search", "DE/rand/1/bin", "DE/rand/1/exp", "DE/rand/1/arith",
           "DE/best/1/bin", "jDE", "SHADE", "LSHADE", "PSO", "CMA-ES"]
AGENTS = ["DEDDQN", "DEDQN", "LDE", "RLDEAFL", "RLHPSDE", "GLEET", "NRLPSO", "QLPSO",
          "RLEPSO", "RLPSO", "RLDAS", "B2OPT", "GLHF", "LES", "RNNOPT", "SYMBOL"]
PARADIGM = {**{a: "hand-designed" for a in ANCHORS},
            **{a: "RL-controlled DE" for a in ["DEDDQN", "DEDQN", "LDE", "RLDEAFL", "RLHPSDE"]},
            **{a: "RL-controlled PSO" for a in ["GLEET", "NRLPSO", "QLPSO", "RLEPSO", "RLPSO"]},
            "RLDAS": "RL algorithm selection",
            **{a: "learned update rule" for a in ["B2OPT", "GLHF", "LES", "RNNOPT"]},
            "SYMBOL": "symbolic rule discovery"}
CITE = {"DEDDQN": "sharma2019deddqn", "DEDQN": "tan2021dedqn", "LDE": "sun2021lde",
        "RLDEAFL": "guo2025rldeafl", "RLHPSDE": "tan2022rlhpsde", "GLEET": "ma2024gleet",
        "NRLPSO": "li2023nrlpso", "QLPSO": "xu2020qlpso", "RLEPSO": "yin2021rlepso",
        "RLPSO": "wu2022rlpso", "RLDAS": "guo2024rldas", "B2OPT": "li2025b2opt",
        "GLHF": "li2024pom", "LES": "lange2023les", "RNNOPT": "tv2019rnnopt",
        "SYMBOL": "chen2024symbol"}
LABEL = {"ambient_translation": "transl.", "ambient_translation_diag": "transl.",
         "rotation": "rot.", "coordinate_permutation": "perm.",
         "domain_scaling_up": "dil.", "domain_scaling_down": "contr.",
         "fitness_rescale": "$f$ resc.", "monotone_pow3": "$f$ cube",
         "monotone_log": "$f$ log", "monotone_spline": "$f$ spline"}


def tex_name(a):
    return a.replace("_", "\\_")


def cell(v, q):
    txt = f"{100 * v:.2f}"
    return f"\\textbf{{{txt}}}" if q <= 0.05 else txt


def write(name, body):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(body)
    print("wrote", name)


# --- S-II: agents and evaluation accounting --------------------------------------
a = pd.read_csv(os.path.join(RES, "exp2_audit.csv"))
a = a[a.error.isna() | (a.error == "")]
ident = a[a.probe == "identity"]
fe = ident.groupby("agent").real_fe.agg(["median", "min", "max"])
sec = a.groupby("agent").seconds.median()
rows = []
for ag in AGENTS:
    rows.append(f"{ag}~\\cite{{{CITE[ag]}}} & {PARADIGM[ag]} & {int(fe.loc[ag, 'median']):,} & "
                f"{int(fe.loc[ag, 'min']):,}--{int(fe.loc[ag, 'max']):,} & {sec.loc[ag]:.1f} \\\\")
write("agents.tex", "\n".join(rows).replace(",", "{,}") + "\n")

# --- S-II: instances ---------------------------------------------------------------
ins = pd.read_csv(os.path.join(RES, "table_instances.csv"))
rows = []
for fid, g in ins.groupby("fid"):
    g = g.sort_values("instance")
    adm = "yes" if g.audit_admissible.all() else "no"
    rows.append(f"F{fid} & {g.split.iloc[0]} & {g.radius.iloc[0]:.2f} & {g.radius.iloc[1]:.2f} & "
                f"{int(g.bias.iloc[0])} & {int(g.bias.iloc[1])} & {adm} \\\\")
write("instances.tex", "\n".join(rows) + "\n")

# --- S-III: dyadic diagnostic ----------------------------------------------------------
d = pd.read_csv(os.path.join(RES, "exp1b_dyadic_scaling.csv"))
d["absd"] = d["delta"].abs()
piv = d.pivot_table(index="optimizer", columns="probe", values="absd", aggfunc="mean")
rows = []
for o in ANCHORS:
    r = piv.loc[o]

    def fmt(v):
        return "0" if v == 0 else f"${v:.1e}".replace("e-0", "\\times10^{-").replace("e+0", "\\times10^{") + "}$"
    rows.append(f"{tex_name(o)} & {fmt(r['domain_scaling_up'])} & {fmt(r['domain_scaling_dyadic_up'])} & "
                f"{fmt(r['domain_scaling_dyadic_down'])} \\\\")
write("dyadic.tex", "\n".join(rows) + "\n")

# --- S-IV: validation scores in full -----------------------------------------------------
v = pd.read_csv(os.path.join(RES, "table_validation_scores.csv"))
vprobes = ["ambient_translation", "rotation", "coordinate_permutation", "domain_scaling_up",
           "domain_scaling_down", "fitness_rescale", "monotone_pow3", "monotone_log",
           "monotone_spline"]
rows = []
for o in ANCHORS:
    cells = []
    for pr in vprobes:
        r = v[(v.agent == o) & (v.probe == pr)].iloc[0]
        cells.append(cell(r.V, r.q))
    rows.append(tex_name(o) + " & " + " & ".join(cells) + " \\\\")
write("validation_full.tex", "\n".join(rows) + "\n")
write("validation_header.tex", " & ".join(LABEL[p] for p in vprobes) + "\n")

# --- S-IV: audit scores in full, V and signed shift ---------------------------------------
s = pd.read_csv(os.path.join(RES, "table_audit_scores.csv"))
aprobes = ["ambient_translation_diag", "rotation", "coordinate_permutation", "domain_scaling_up",
           "domain_scaling_down", "fitness_rescale", "monotone_log", "monotone_spline"]
for kind in ["V", "V_signed"]:
    rows = []
    for o in ANCHORS + AGENTS:
        cells = []
        for pr in aprobes:
            r = s[(s.agent == o) & (s.probe == pr)].iloc[0]
            if kind == "V":
                cells.append(cell(r.V, r.q))
            else:
                cells.append(f"{100 * r.V_signed:+.2f}")
        if o == AGENTS[0]:
            rows.append("\\midrule")
        rows.append(tex_name(o) + " & " + " & ".join(cells) + " \\\\")
    write(f"audit_{kind}.tex", "\n".join(rows) + "\n")
write("audit_header.tex", " & ".join(LABEL[p] for p in aprobes) + "\n")

# --- S-V: out-of-distribution analysis ----------------------------------------------------
t = pd.read_csv(os.path.join(RES, "table_law.csv"), index_col=0)
rows = []
for o in ANCHORS + AGENTS:
    r = t.loc[o]
    if o == AGENTS[0]:
        rows.append("\\midrule")
    rows.append(f"{tex_name(o)} & {100 * r.V:.2f} & {r.rank_id:.2f} & {r.rank_ood:.2f} & "
                f"{r.G_rank:+.2f} & {r.A_id:.3f} & {r.A_ood:.3f} \\\\")
write("law_full.tex", "\n".join(rows) + "\n")

L = t[t.paradigm != "hand-designed"]


def spearman(x, y):
    r = stats.spearmanr(x, y)
    return r.correlation, r.pvalue


def partial(df):
    rk = df[["V", "G_rank", "rank_id"]].rank()
    X = np.c_[np.ones(len(rk)), rk.rank_id]

    def res(col):
        b = np.linalg.lstsq(X, rk[col], rcond=None)[0]
        return rk[col] - X @ b
    return stats.pearsonr(res("V"), res("G_rank"))


t["logratio"] = np.log(t.A_ood / t.A_id)
t["rel_drop"] = -(t.logratio - t.loc[t.paradigm == "hand-designed", "logratio"].median())
L = t[t.paradigm != "hand-designed"]
loo = [stats.spearmanr(L.drop(x).V, L.drop(x).G_rank).correlation for x in L.index]
rho_all, p_all = spearman(t.V, t.G_rank)
rho_l, p_l = spearman(L.V, L.G_rank)
pc = partial(L)
ar = spearman(L.V, L.rel_drop)
conf = spearman(t.V, t.rank_id)
lines = [
    f"All methods, $V$ vs.\\ $G$ & $n=26$ & $\\rho={rho_all:+.2f}$ & $p={p_all:.3f}$ \\\\",
    f"Learned optimizers, $V$ vs.\\ $G$ & $n=16$ & $\\rho={rho_l:+.2f}$ & $p={p_l:.3f}$ \\\\",
    f"Learned, partial on training rank & $n=16$ & $r={pc[0]:+.2f}$ & $p={pc[1]:.3f}$ \\\\",
    f"Learned, $V$ vs.\\ area-ratio drop & $n=16$ & $\\rho={ar[0]:+.2f}$ & $p={ar[1]:.3f}$ \\\\",
    f"Learned, leave-one-out range & $n=15$ & \\multicolumn{{2}}{{l}}{{$\\rho\\in[{min(loo):+.2f},{max(loo):+.2f}]$}} \\\\",
    f"All, $V$ vs.\\ training rank (confound check) & $n=26$ & $\\rho={conf[0]:+.2f}$ & $p={conf[1]:.3f}$ \\\\",
]
write("law_robustness.tex", "\n".join(lines) + "\n")
par = L.groupby("paradigm")[["V", "G_rank"]].mean()
order = ["RL-controlled DE", "RL-controlled PSO", "RL algorithm selection",
         "learned update rule", "symbolic rule discovery"]
rows = [f"{pp} & {int((L.paradigm == pp).sum())} & {100 * par.loc[pp, 'V']:.2f} & "
        f"{par.loc[pp, 'G_rank']:+.2f} \\\\" for pp in order]
write("law_paradigm.tex", "\n".join(rows) + "\n")
print("note: all-method p here is the scipy asymptotic value; the paper reports the permutation value")

# --- S-VI: ablation of the objective layer ----------------------------------------------------
ab_path = os.path.join(RES, "exp7_rank_ablation.csv")
if os.path.exists(ab_path):
    ab = pd.read_csv(ab_path)
    modes = ["raw", "window", "worse_count", "order"]
    names = {"raw": "raw values", "window": "window quantile", "worse_count": "worse-count",
             "order": "order embedding"}
    opts = ["DE/rand/1/bin", "jDE", "PSO", "SHADE", "LSHADE", "CMA-ES"]
    mean = ab.groupby(["optimizer", "mode"]).A.mean().unstack("mode")
    n_cells = ab[["function", "instance"]].drop_duplicates().shape[0]
    n_runs = ab.run.nunique()
    rows = [f"{tex_name(o)} & " + " & ".join(f"{mean.loc[o, m]:.4f}" for m in modes) + " \\\\"
            for o in opts]
    text = (
        "Four presentations of the objective were compared on in-distribution performance "
        f"(six canonical functions, two instances, {n_runs} runs, {n_cells} cells), with the "
        "frame layer active in every harnessed variant (Table~\\ref{tab:s-ablation}). "
        "A quantile against a sliding window of recent values costs every algorithm that "
        "compares against stored values roughly half of its performance or more, because a "
        "stored value's quantile drifts as the run improves; CMA-ES, which ranks only the "
        "current generation, is unaffected by all three transforms. The worse-count transform "
        "of Proposition~2 recovers part of the loss. The order embedding of Proposition~3 "
        "reproduces raw-value performance exactly for DE/rand/1/bin, jDE and particle swarm "
        "optimization, slightly improves SHADE and L-SHADE, and is the variant used in the "
        "manuscript.\n\n"
        "\\begin{table}[!h]\n\\centering\n"
        "\\caption{Objective-layer ablation: mean performance measure}\n"
        "\\label{tab:s-ablation}\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Algorithm & " + " & ".join(names[m] for m in modes) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("ablation_text.tex", text)

# --- S-VI: harness on anchors -----------------------------------------------------------------
h = pd.read_csv(os.path.join(RES, "table_harness_anchors.csv"))
hprobes = ["ambient_translation", "domain_scaling_up", "domain_scaling_down", "fitness_rescale",
           "monotone_pow3", "monotone_log", "monotone_spline"]
rows = []
for o in ANCHORS:
    cells = []
    for pr in hprobes:
        r = h[(h.agent == o) & (h.probe == pr)].iloc[0]
        b = f"\\textbf{{{100 * r.V_b:.2f}}}" if r.q_b <= 0.05 else f"{100 * r.V_b:.2f}"
        cells.append(f"{b}/{100 * r.V_a:.2f}")
    rows.append(tex_name(o) + " & " + " & ".join(cells) + " \\\\")
write("harness_violation.tex", "\n".join(rows) + "\n")
write("harness_header.tex", " & ".join(LABEL[p] for p in hprobes) + "\n")
c = pd.read_csv(os.path.join(RES, "table_harness_cost.csv"), index_col=0)
rows = []
for o in ANCHORS:
    r = c.loc[o]
    pv = "--" if r.p >= 0.999 else (f"$<10^{{-4}}$" if r.p < 1e-4 else f"{r.p:.3f}")
    rows.append(f"{tex_name(o)} & {r.A_off:.4f} & {r.A_on:.4f} & {100 * r.delta:+.3f} & "
                f"{100 * r.max_abs:.2f} & {pv} \\\\")
write("harness_cost.tex", "\n".join(rows) + "\n")
