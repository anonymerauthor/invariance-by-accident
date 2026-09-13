"""Figures and summary tables of the audit.

Works on whatever results exist so far, so it can be re-run as experiments
finish.  All figures are vector PDF at two-column journal widths (3.5 in single, 7.16 in
double), 8 pt type, and survive grayscale printing:

* heatmaps use a single-hue sequential ramp, so darkness alone carries magnitude;
* the scatter encodes paradigm by marker shape *and* colour, and labels points
  directly, so identity never depends on colour;
* the categorical colours are the first three slots of the validated reference
  palette (all-pairs CVD separation >= 9).

Usage
-----
    python experiments/make_figures.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from invaudit.analysis import (aggregate_violation, law_test, ood_gap,  # noqa: E402
                               violation_matrix, violation_scores,
                               violation_significance)

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SINGLE, DOUBLE = 3.5, 7.16
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

PROBE_LABEL = {
    "ambient_translation": "translation", "ambient_translation_diag": "translation",
    "rotation": "rotation", "coordinate_permutation": "permutation",
    "domain_scaling_up": "dilation", "domain_scaling_down": "contraction",
    "fitness_rescale": "f rescale", "fitness_shift": "f shift",
    "monotone_pow3": "f cube", "monotone_log": "f log", "monotone_spline": "f spline",
    "identity_recomposed": "identity",
}
AUDIT_PROBES = ["ambient_translation_diag", "rotation", "coordinate_permutation",
                "domain_scaling_up", "domain_scaling_down", "fitness_rescale",
                "monotone_log", "monotone_spline"]

ANCHORS = ["random_search", "DE/rand/1/bin", "DE/rand/1/exp", "DE/rand/1/arith",
           "DE/best/1/bin", "jDE", "SHADE", "LSHADE", "PSO", "CMA-ES"]
PARADIGM = {a: "hand-designed" for a in ANCHORS}
PARADIGM.update({
    "DEDDQN": "RL-controlled DE", "DEDQN": "RL-controlled DE", "LDE": "RL-controlled DE",
    "RLHPSDE": "RL-controlled DE", "RLDEAFL": "RL-controlled DE",
    "RLDAS": "RL algorithm selection",
    "GLEET": "RL-controlled PSO", "NRLPSO": "RL-controlled PSO", "QLPSO": "RL-controlled PSO",
    "RLEPSO": "RL-controlled PSO", "RLPSO": "RL-controlled PSO",
    "B2OPT": "learned update rule", "GLHF": "learned update rule",
    "LES": "learned update rule", "RNNOPT": "learned update rule",
    "SYMBOL": "symbolic rule discovery",
})
# colour = design family (3 validated slots), marker = paradigm
STYLE = {
    "hand-designed":           dict(color=AQUA,   marker="o"),
    "RL-controlled DE":        dict(color=BLUE,   marker="s"),
    "RL-controlled PSO":       dict(color=BLUE,   marker="^"),
    "RL algorithm selection":  dict(color=BLUE,   marker="D"),
    "learned update rule":     dict(color=ORANGE, marker="v"),
    "symbolic rule discovery": dict(color=ORANGE, marker="P"),
}
PARADIGM_ORDER = list(STYLE)


CRASHES = {}


def _load(name):
    path = os.path.join(RES, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _combined(family):
    parts = []
    for name in (f"exp2_{family}.csv", f"exp3_{family}.csv"):
        d = _load(name)
        if d is not None:
            if "error" in d.columns:
                crashed = d["error"].notna() & (d["error"] != "")
                if crashed.any():
                    # A run that raises returns no usable solution: it scores
                    # zero, and the crash is reported alongside the scores.
                    d = d.copy()
                    d.loc[crashed, "A"] = 0.0
                    CRASHES.update(d.loc[crashed].groupby(["agent", "probe"]).size()
                                   .to_dict())
            parts.append(d[["family", "agent", "fid", "instance", "probe", "seed", "A"]])
    return pd.concat(parts, ignore_index=True) if parts else None


def _order(agents):
    key = {p: i for i, p in enumerate(PARADIGM_ORDER)}
    anchor_pos = {a: i for i, a in enumerate(ANCHORS)}
    return sorted(agents, key=lambda a: (key.get(PARADIGM.get(a, ""), 99),
                                         anchor_pos.get(a, 99), a))


def _heatmap(ax, M, row_labels, col_labels, vmax, sig=None, cell_font=8, xrot=40):
    """Cells show V in percentage points of ECDF area.

    An exact zero prints as a centred dot; a value whose permutation test does
    not survive FDR control prints in muted ink, so that the eye lands only on
    violations the data actually support.
    """
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=xrot,
                  ha="right" if xrot else "center")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            if v == 0.0:
                ax.text(j, i, "·", ha="center", va="center", fontsize=cell_font, color=INK2)
                continue
            strong = sig is None or bool(sig[i, j])
            if v > 0.55 * vmax:
                ink = "#ffffff" if strong else "#cde2fb"
            else:
                ink = INK if strong else "#a3a29d"
            ax.text(j, i, f"{100 * v:.1f}", ha="center", va="center",
                    fontsize=cell_font, color=ink,
                    fontweight="bold" if strong and sig is not None else "normal")
    return im


# --------------------------------------------------------------------------
# Figure: instrument validation (experiment 1)
# --------------------------------------------------------------------------

def fig_validation():
    df = _load("exp1_anchor_validation.csv")
    if df is None:
        print("validation: no exp1 results yet")
        return
    from invaudit.optimizers import DISTRIBUTIONAL_INVARIANCE
    probes = ["ambient_translation", "rotation", "coordinate_permutation",
              "domain_scaling_up", "domain_scaling_down", "fitness_rescale",
              "monotone_pow3", "monotone_log", "monotone_spline"]
    # probes that share a ground-truth entry
    truth_key = {"domain_scaling_down": "domain_scaling_up",
                 "monotone_pow3": "monotone_spline", "monotone_log": "monotone_spline"}
    base = (df.drop_duplicates(["optimizer", "function", "instance", "run"])
            [["optimizer", "function", "instance", "run", "A_base"]]
            .rename(columns={"A_base": "A"}).assign(probe="identity"))
    pr = df[df.probe.isin(probes)][["optimizer", "function", "instance", "run",
                                    "probe", "A_probe"]].rename(columns={"A_probe": "A"})
    long = (pd.concat([base, pr], ignore_index=True)
            .rename(columns={"optimizer": "agent", "function": "fid", "run": "seed"}))
    sc = violation_scores(long, probes)
    sig = violation_significance(long, probes)
    sc = sc.merge(sig[["agent", "probe", "p", "q", "significant"]],
                  on=["agent", "probe"], how="left")
    order = [a for a in ANCHORS if a in set(sc.agent)]
    M = violation_matrix(sc, order=order).reindex(columns=probes).to_numpy()
    S = (sc.pivot(index="agent", columns="probe", values="significant")
         .reindex(index=order, columns=probes).fillna(False).to_numpy())
    fig, ax = plt.subplots(figsize=(SINGLE, 3.3))
    vmax = float(np.nanmax(M)) or 1e-6
    im = _heatmap(ax, M, order, [PROBE_LABEL[p] for p in probes], vmax, sig=S)
    tally = {"hit": 0, "miss": 0, "false_alarm": 0, "correct_rejection": 0}
    misses = []
    for i, a in enumerate(order):
        for j, p in enumerate(probes):
            expect_violation = DISTRIBUTIONAL_INVARIANCE[a].get(truth_key.get(p, p), 1) == 0
            if expect_violation:
                ax.add_patch(plt.Rectangle((j - .44, i - .44), .88, .88, fill=False,
                                           edgecolor=ORANGE, linewidth=1.6, zorder=5))
            found = bool(S[i, j])
            key = ("hit" if found else "miss") if expect_violation else                   ("false_alarm" if found else "correct_rejection")
            tally[key] += 1
            if key in ("miss", "false_alarm"):
                misses.append(f"{key}: {a} / {p} V={M[i, j]:.2e}")
    from matplotlib.ticker import FuncFormatter
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02,
                      format=FuncFormatter(lambda v, _: f"{100 * v:.1f}"))
    cb.set_label("violation (x100)")
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIG, "fig_validation.pdf"))
    fig.savefig(os.path.join(FIG, "fig_validation.png"), dpi=220)
    plt.close(fig)
    sc.to_csv(os.path.join(RES, "table_validation_scores.csv"), index=False)
    print(f"validation: {len(order)} anchors x {len(probes)} probes -> fig_validation.pdf")
    print("  agreement with theory:", tally)
    for m in misses:
        print("   ", m)


# --------------------------------------------------------------------------
# Figure: audit heatmap (experiments 2 + 3)
# --------------------------------------------------------------------------

def fig_audit():
    df = _combined("audit")
    if df is None:
        print("audit: no results yet")
        return None
    probes = [p for p in AUDIT_PROBES if p in set(df.probe)]
    sc = violation_scores(df, probes)
    sig = violation_significance(df, probes)
    sc = sc.merge(sig[["agent", "probe", "p", "q", "significant"]],
                  on=["agent", "probe"], how="left")
    order = _order(sc.agent.unique())
    M = violation_matrix(sc, order=order).reindex(columns=probes)
    S = (sc.pivot(index="agent", columns="probe", values="significant")
         .reindex(index=order, columns=probes).fillna(False).to_numpy())
    # double-column layout: methods as columns, probes as rows
    MT = M.to_numpy().T
    ST = S.T
    labels_m = [a.replace("random_search", "RS").replace("DE/rand/1/", "DE/r/1/")
                .replace("DE/best/1/", "DE/b/1/") for a in order]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.55))
    vmax = float(np.nanquantile(MT, 0.98)) or 1e-6
    im = _heatmap(ax, MT, [PROBE_LABEL[p] for p in probes], labels_m, vmax, sig=ST,
                  cell_font=7.5, xrot=90)
    for (agent, probe), _n in CRASHES.items():
        if agent in order and probe in probes:
            j, i = order.index(agent), probes.index(probe)
            ax.text(j + .45, i - .42, "†", ha="right", va="top", fontsize=8,
                    color="#ffffff" if MT[i, j] > 0.55 * vmax else INK)
    prev, start_j = None, 0
    groups = []
    for j, a in enumerate(order + [None]):
        par = PARADIGM.get(a) if a else None
        if prev is not None and par != prev:
            groups.append((prev, start_j, j - 1))
            if a is not None:
                ax.axvline(j - .5, color=INK2, linewidth=0.9)
            start_j = j
        prev = par
    short = {"hand-designed": "hand-designed", "RL-controlled DE": "RL-controlled DE",
             "RL-controlled PSO": "RL-controlled PSO", "RL algorithm selection": "RL sel.",
             "learned update rule": "learned update rule",
             "symbolic rule discovery": "symb."}
    for par, a0, a1 in groups:
        if a1 > a0:   # single-agent groups are named in the caption instead
            ax.text((a0 + a1) / 2, -0.75, short.get(par, par), ha="center",
                    va="bottom", fontsize=8, color=INK)
    from matplotlib.ticker import FuncFormatter
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, extend="max",
                      format=FuncFormatter(lambda v, _: f"{100 * v:.1f}"))
    cb.set_label("violation (x100)")
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIG, "fig_audit_heatmap.pdf"))
    fig.savefig(os.path.join(FIG, "fig_audit_heatmap.png"), dpi=220)
    plt.close(fig)
    sc.to_csv(os.path.join(RES, "table_audit_scores.csv"), index=False)
    print(f"audit: {len(order)} methods x {len(probes)} probes -> fig_audit_heatmap.pdf")
    return sc


# --------------------------------------------------------------------------
# Figure: the law (violation vs out-of-distribution rank gap)
# --------------------------------------------------------------------------

def fig_law(sc):
    df_id, df_ood = _combined("audit"), _combined("ood")
    if sc is None or df_ood is None:
        print("law: need audit and ood results")
        return
    probes = [p for p in AUDIT_PROBES if p in set(sc.probe)]
    V = aggregate_violation(sc, probes)
    G = ood_gap(df_id, df_ood)
    common = V.index.intersection(G.index)
    if len(common) < 5:
        print(f"law: only {len(common)} methods with both measurements")
        return
    res_all = law_test(V, G["G_rank"])
    learned = [a for a in common if PARADIGM.get(a) != "hand-designed"]
    res_learned = law_test(V.loc[learned], G.loc[learned, "G_rank"]) if len(learned) >= 5 else None

    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
    for par in PARADIGM_ORDER:
        members = [a for a in common if PARADIGM.get(a) == par]
        if not members:
            continue
        st = STYLE[par]
        ax.scatter(100 * V.loc[members], G.loc[members, "G_rank"], s=26, marker=st["marker"],
                   facecolor=st["color"], edgecolor="#fcfcfb", linewidth=0.8,
                   label=par, zorder=3)
    # manual offsets where labels would collide
    offset = {"NRLPSO": (5, -10), "LDE": (5, 2), "GLHF": (-12, -12), "RLDEAFL": (-22, 6),
              "LES": (-14, -11), "B2OPT": (3, 3), "RLHPSDE": (3, 3), "DEDQN": (3, -9),
              "SYMBOL": (4, 5), "RLPSO": (3, 4), "QLPSO": (4, 2), "GLEET": (4, 2),
              "RLEPSO": (5, -11), "RLDAS": (4, 3), "DEDDQN": (4, 3), "RNNOPT": (4, 2)}
    leader = {"NRLPSO": (0.62, 1.22), "RLEPSO": (0.62, 0.48)}   # crowded corner: label in free space
    for a in common:
        if PARADIGM.get(a) == "hand-designed":
            continue
        xy = (100 * V.loc[a], G.loc[a, "G_rank"])
        if a in leader:
            ax.annotate(a, xy, xytext=leader[a], textcoords="data", fontsize=7.5,
                        color=INK2, va="center",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5,
                                        shrinkA=1, shrinkB=3))
        else:
            ax.annotate(a, xy, xytext=offset.get(a, (3, 2)),
                        textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("aggregate violation on training functions (x100)")
    ax.set_ylabel("rank lost on held-out functions G")
    ax.grid(True, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    txt = f"all: $\\rho$={res_all['rho']:.2f}, p={res_all['p_two_sided']:.3f} (n={res_all['n']})"
    if res_learned:
        txt += (f"\nlearned: $\\rho$={res_learned['rho']:.2f}, "
                f"p={res_learned['p_two_sided']:.3f} (n={res_learned['n']})")
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, va="top", ha="right",
            fontsize=8, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
              handletextpad=0.2, columnspacing=1.0)
    fig.savefig(os.path.join(FIG, "fig_law.pdf"))
    fig.savefig(os.path.join(FIG, "fig_law.png"), dpi=220)
    plt.close(fig)
    out = G.join(V.rename("V"), how="inner")
    out["paradigm"] = [PARADIGM.get(a, "") for a in out.index]
    out.to_csv(os.path.join(RES, "table_law.csv"))
    print("law:", res_all, "\nlearned only:", res_learned)


if __name__ == "__main__":
    fig_validation()
    sc = fig_audit()
    fig_law(sc)
