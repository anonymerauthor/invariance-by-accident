"""Figures of the revised manuscript (R3.7).

fig_audit_rev.pdf  26 methods x 8 probes on the MetaBox training instances, 20 seeds.
                   Colour = signed mean loss (points; diverging, loss warm / gain cool);
                   text = violation V (points).  Bold = significant under the
                   distribution-level tests (disjoint seeds, V_u or CvM, BH q <= .05);
                   a small circle marks pairs significant only under the paired
                   sign-flip test (pathwise).  Dagger = every run raises.
fig_law_rev.pdf    aggregate V against (a) the held-out rank gap G and (b) the absolute
                   held-out performance, all 26 methods.

Style follows experiments/make_figures.py (IEEE widths, 8 pt serif, validated palette).
Usage: python revision/scripts/make_rev_figures.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import make_figures as MF  # noqa: E402  (shared style, labels, paradigms)

DIR = os.path.join(ROOT, "results_rev_pod")
OUT = os.path.join(ROOT, "submission_rev", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)
# diverging: cool (gain) - neutral grey - warm (loss)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#6da7ec", "#cde2fb", "#f0efec", "#fbd3bf", "#f0956a", "#b8461b"])


def fig_audit():
    ag = pd.read_csv(os.path.join(DIR, "e2_audit_inference.csv"))
    an = pd.read_csv(os.path.join(DIR, "e2_anchors_e1b_inference.csv"))
    d = pd.concat([an, ag], ignore_index=True)
    d["dist"] = (d.q_V_u <= .05) | (d.q_CvM <= .05)
    d["paired_only"] = (d.q_signflip <= .05) & ~d.dist
    order = MF._order(d.agent.unique())
    probes = MF.AUDIT_PROBES
    piv = lambda c: d.pivot(index="agent", columns="probe", values=c).reindex(
        index=order, columns=probes).T
    L, V, S, P = (piv(c).to_numpy() for c in ("loss", "V", "dist", "paired_only"))
    crashed = piv("crashed_runs").to_numpy()
    lim = 4.0
    fig, ax = plt.subplots(figsize=(MF.DOUBLE, 2.7))
    im = ax.imshow(100 * L, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto",
                   interpolation="nearest")
    labels = [a.replace("random_search", "RS").replace("DE/rand/1/", "DE/r/1/")
              .replace("DE/best/1/", "DE/b/1/") for a in order]
    ax.set_xticks(range(len(order)), labels, rotation=90)
    ax.set_yticks(range(len(probes)), [MF.PROBE_LABEL[p] for p in probes])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(probes), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(probes)):
        for j in range(len(order)):
            v = 100 * V[i, j]
            if not np.isfinite(v):
                continue
            dark = abs(100 * L[i, j]) > 0.6 * lim
            if v < 0.005:
                ax.text(j, i, "·", ha="center", va="center", fontsize=7, color=MF.INK2)
            else:
                strong = bool(S[i, j])
                ink = ("#ffffff" if dark else MF.INK) if strong else (
                    "#e8e7e3" if dark else "#9a9994")
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.8,
                        color=ink, fontweight="bold" if strong else "normal")
            if P[i, j]:
                ax.plot(j + .36, i - .30, "o", ms=2.4, mfc="none", mec=MF.INK, mew=.6)
            if crashed[i, j] > 0:
                ax.text(j - .45, i - .42, "†", ha="left", va="top", fontsize=7,
                        color="#ffffff" if dark else MF.INK)
    prev, start = None, 0
    for j, a in enumerate(order + [None]):
        par = MF.PARADIGM.get(a) if a else None
        if prev is not None and par != prev:
            if a is not None:
                ax.axvline(j - .5, color=MF.INK2, linewidth=0.9)
            if j - 1 > start:
                ax.text((start + j - 1) / 2, -0.75, prev, ha="center", va="bottom",
                        fontsize=7.5)
            start = j
        prev = par
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, extend="both")
    cb.set_label("mean loss (points)")
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(OUT, "fig_audit_rev.pdf"))
    fig.savefig(os.path.join(OUT, "fig_audit_rev.png"), dpi=220)
    plt.close(fig)
    print("audit figure:", len(order), "methods; distribution-level significant:",
          int(d.dist.sum()), "; paired only:", int(d.paired_only.sum()))


def fig_law():
    law = pd.read_csv(os.path.join(DIR, "law_rev.csv"), index_col=0)
    tests = pd.read_csv(os.path.join(DIR, "law_rev_tests.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(MF.SINGLE, 2.6), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.86, bottom=0.36, wspace=0.42)
    for ax, col, ylab, tag in ((axes[0], "G", "held-out rank gap $G$", "a"),
                               (axes[1], "A_heldout", "held-out perf. (points)", "b")):
        y = law[col] * (100 if col == "A_heldout" else 1)
        if col == "G":
            ax.axhline(0, color=MF.GRID, linewidth=0.8, zorder=0)
        for par, st in MF.STYLE.items():
            m = [a for a in law.index if MF.PARADIGM.get(a) == par]
            if m:
                ax.scatter(100 * law.loc[m, "V"], y.loc[m], s=16, marker=st["marker"],
                           facecolor=st["color"], edgecolor="#fcfcfb", linewidth=0.6,
                           label=par, zorder=3)
        t = tests[(tests["sample"] == "learned") & (tests.target == col)].iloc[0]
        ax.set_title(f"({tag}) learned: $\\rho$={t.rho:.2f}, $p$={t.p:.3f}", fontsize=7.2)
        ax.set_xlabel("aggregate $V$ (points)")
        ax.set_ylabel(ylab)
        ax.grid(True, color=MF.GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, handletextpad=0.1,
               columnspacing=0.8, fontsize=6.8)
    fig.savefig(os.path.join(OUT, "fig_law_rev.pdf"), bbox_inches=None)
    fig.savefig(os.path.join(OUT, "fig_law_rev.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_audit()
    fig_law()
