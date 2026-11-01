"""E4 analysis - does the audit verdict depend on the budget? (R2.5)

The audit records the running best at 30 log-spaced checkpoints up to 20,000
evaluations (b0..b29).  The performance measure at a smaller budget B is the ECDF
area over the checkpoints <= B, i.e. exactly what a run stopped at B would have
scored.  For B in {2,000, 5,000, 20,000} this gives V, sign-flip p and BH q per
agent x probe (BH within each budget), with no new runs.

Usage: python revision/scripts/e4_analysis.py --dir results_rev_pod [--inp e1_audit_20seeds.csv]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invaudit.analysis import _paired_cells, _v_from_diffs  # noqa: E402
from invaudit.inference import bh  # noqa: E402
from invaudit.metrics import default_checkpoints, default_targets, ecdf_area  # noqa: E402
from e5_analysis import signflip_p  # noqa: E402
from e12_analysis import load  # noqa: E402

FULL = 20000


def area_at(d, budget):
    cps = default_checkpoints(FULL, 31)
    cols = [f"b{j}" for j, c in enumerate(cps) if c <= budget]
    best = torch.from_numpy(d[cols].to_numpy(dtype=float))
    A = ecdf_area(best, default_targets()).numpy()
    A[d.crashed.to_numpy()] = 0.0
    return A, int(cps[len(cols) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rev_pod")
    ap.add_argument("--inp", default="e1_audit_20seeds.csv")
    ap.add_argument("--budgets", default="2000,5000,20000")
    ap.add_argument("--n-perm", type=int, default=4999)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    d = load(os.path.join(args.dir, args.inp))
    full = d.A.to_numpy().copy()
    rows = []
    for B in map(int, args.budgets.split(",")):
        A, last = area_at(d, B)
        if B == FULL:
            assert np.allclose(A[~d.crashed.to_numpy()], full[~d.crashed.to_numpy()]), \
                "recomputed A differs from the recorded A at the full budget"
        g0 = d.assign(A=A)
        for agent, g in g0.groupby("agent"):
            for probe in sorted(set(g.probe) - {"identity"}):
                cells = _paired_cells(g, agent, probe)
                rows.append(dict(budget=B, last_checkpoint=last, agent=agent, probe=probe,
                                 A_id=g[g.probe == "identity"].A.mean(),
                                 V=_v_from_diffs(cells), p=signflip_p(cells, args.n_perm, rng)))
    v = pd.DataFrame(rows)
    v["q"] = v.groupby("budget").p.transform(lambda p: bh(p.to_numpy()))
    v.to_csv(os.path.join(args.dir, "e4_budget.csv"), index=False)
    v["sig"] = v.q <= .05
    print("checkpoint used per budget:", v.groupby("budget").last_checkpoint.first().to_dict())
    print("significant pairs per budget:", v.groupby("budget").sig.sum().to_dict())
    w = v.pivot_table(index=["agent", "probe"], columns="budget", values="sig", aggfunc="first")
    any_sig = w[w.any(axis=1)]
    Vt = v.pivot_table(index=["agent", "probe"], columns="budget", values="V") * 100
    print("\n== pairs significant at any budget: V (points), * = q <= .05")
    out = Vt.loc[any_sig.index].round(2).astype(str) + any_sig.replace({True: "*", False: ""})
    print(out.to_string())
    B = sorted(v.budget.unique())
    full_sig = w[B[-1]]
    for b in B[:-1]:
        print(f"budget {b}: {int((w[b] & full_sig).sum())}/{int(full_sig.sum())} of the full-budget "
              f"violations detected; {int((w[b] & ~full_sig).sum())} only at {b}")


if __name__ == "__main__":
    main()
