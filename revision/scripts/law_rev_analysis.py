"""Violation vs held-out generalization on the revision data (R1.2, R3.5).

Inputs (results_rev_pod/, all on the pod, 20 seeds):
  e2_audit_inference.csv   per agent x probe V on the training functions (16 agents)
  anchor_pod_checks.csv    per anchor x probe V on the MetaBox instances (set "E1b ...")
  e1_audit_20seeds / e6_ood_20seeds            identity runs of the agents, train / held-out
  e1b_anchors_audit / e1b_anchors_ood          identity runs of the anchors, train / held-out

For each method: aggregate V (RMS over the eight probes), mean rank on training and
held-out functions (rank among all 26 methods per function), the rank gap G, the absolute
held-out performance A_ood, and A_ood normalized per function by the best method.
Spearman correlations of V with G, A_ood and normalized A_ood, with permutation p-values,
for all methods and for the 16 learned ones; partial correlation of V and G given the
training rank.

Usage: python revision/scripts/law_rev_analysis.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "results_rev_pod")
PROBES = ["ambient_translation_diag", "rotation", "coordinate_permutation", "domain_scaling_up",
          "domain_scaling_down", "fitness_rescale", "monotone_log", "monotone_spline"]
PARADIGM = {"DEDDQN": "RL-DE", "DEDQN": "RL-DE", "LDE": "RL-DE", "RLDEAFL": "RL-DE",
            "RLHPSDE": "RL-DE", "GLEET": "RL-PSO", "NRLPSO": "RL-PSO", "QLPSO": "RL-PSO",
            "RLEPSO": "RL-PSO", "RLPSO": "RL-PSO", "RLDAS": "RL-AS", "B2OPT": "update rule",
            "GLHF": "update rule", "LES": "update rule", "RNNOPT": "update rule",
            "SYMBOL": "symbolic"}


def identity(name):
    d = pd.read_csv(os.path.join(DIR, name), low_memory=False)
    d = d[d.probe == "identity"].copy()
    if "error" in d.columns:
        d = d[~d["error"].astype(str).str.startswith("UnsupportedInstance")]
        crashed = d["error"].notna() & (d["error"].astype(str) != "")
        d.loc[crashed, "A"] = 0.0
    return d.groupby(["agent", "fid"]).A.mean().unstack("fid")


def perm_spearman(x, y, n=20000, seed=0):
    rho = stats.spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(x, rng.permutation(y)).correlation for _ in range(n)])
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n + 1))


def partial(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return stats.pearsonr(res(rx), res(ry))


def main():
    e2 = pd.read_csv(os.path.join(DIR, "e2_audit_inference.csv"))
    an = pd.read_csv(os.path.join(DIR, "anchor_pod_checks.csv"))
    an = an[an.set.str.startswith("E1b")]
    v = pd.concat([e2[["agent", "probe", "V"]], an[["agent", "probe", "V"]]])
    v = v[v.probe.isin(PROBES)]
    V = v.groupby("agent").V.apply(lambda s: float(np.sqrt(np.mean(s ** 2))))

    tr = pd.concat([identity("e1_audit_20seeds.csv.gz"), identity("e1b_anchors_audit.csv.gz")])
    te = pd.concat([identity("e6_ood_20seeds.csv.gz"), identity("e1b_anchors_ood.csv.gz")])
    rtr = tr.rank(ascending=False, method="average").mean(axis=1)
    rte = te.rank(ascending=False, method="average").mean(axis=1)
    out = pd.DataFrame({"V": V, "rank_train": rtr, "rank_heldout": rte,
                        "A_train": tr.mean(axis=1), "A_heldout": te.mean(axis=1),
                        "A_heldout_norm": (te / te.max()).mean(axis=1)}).dropna()
    out["G"] = out.rank_heldout - out.rank_train
    out["paradigm"] = [PARADIGM.get(a, "hand-designed") for a in out.index]
    out.sort_values("V", ascending=False).to_csv(os.path.join(DIR, "law_rev.csv"))

    rows = []
    learned = out[out.paradigm != "hand-designed"]
    for label, d in (("all", out), ("learned", learned)):
        for target in ("G", "A_heldout", "A_heldout_norm", "A_train", "rank_train"):
            rho, p = perm_spearman(d.V.to_numpy(), d[target].to_numpy())
            rows.append(dict(sample=label, n=len(d), target=target, rho=rho, p=p))
        r, p = partial(d.V.to_numpy(), d.G.to_numpy(), d.rank_train.to_numpy())
        rows.append(dict(sample=label, n=len(d), target="G | rank_train (partial)", rho=r, p=p))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DIR, "law_rev_tests.csv"), index=False)
    pd.set_option("display.width", 200)
    print((out.assign(V=100 * out.V, A_train=100 * out.A_train, A_heldout=100 * out.A_heldout)
           .sort_values("V", ascending=False).round(3).to_string()))
    print()
    print(res.round(4).to_string(index=False))
    print("\nby paradigm (V and G means):")
    print(out.groupby("paradigm")[["V", "G", "A_heldout"]].mean()
          .assign(V=lambda x: 100 * x.V, A_heldout=lambda x: 100 * x.A_heldout).round(3))


if __name__ == "__main__":
    main()
