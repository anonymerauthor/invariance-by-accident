"""E6 analysis - the objective layer switched on at test time for the 16 distributed agents
(R1.3, R3.5, AE.4).

Two tables:

1. ``e6_violations.csv``: per agent x probe, V and sign-flip p without the layer (E1 audit)
   and with it (E6 audit_layer); BH within each condition.
2. ``e6_cost.csv``: per agent, mean identity A on the training functions (audit grid)
   and on the held-out functions (ood), without and with the layer, and the paired
   difference layer - none with a cell bootstrap 95% CI.  Runs are paired by
   (fid, instance, seed): both conditions use the same seeds and instances.

Infeasible instances (UnsupportedInstance) are excluded; other errors count as crashes
with A = 0, as in the audit, and are reported.

Usage: python revision/scripts/e6_analysis.py --dir results_rev_pod
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invaudit.analysis import _paired_cells, _v_from_diffs  # noqa: E402
from invaudit.inference import bh  # noqa: E402
from e5_analysis import signflip_p  # noqa: E402

CELL = ["fid", "instance"]


def load(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            d = pd.read_csv(p, low_memory=False)
            break
    else:
        return None
    err = d["error"].fillna("").astype(str)
    unsup = err.str.startswith("UnsupportedInstance")
    d = d[~unsup].copy()
    d["crashed"] = err[~unsup] != ""
    d.loc[d.crashed, "A"] = 0.0
    return d


def paired_diff(a, b, n_boot, rng):
    """Mean of b - a over runs paired by (fid, instance, seed); CI by resampling cells."""
    m = a.merge(b, on=CELL + ["seed"], suffixes=("_a", "_b"))
    if m.empty:
        return np.nan, np.nan, np.nan, 0
    cell = (m.A_b - m.A_a).groupby([m.fid, m.instance]).mean().to_numpy()
    idx = rng.integers(0, len(cell), size=(n_boot, len(cell)))
    boot = cell[idx].mean(1)
    return float(cell.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rev_pod")
    ap.add_argument("--e1", default="e1_audit_20seeds.csv")
    ap.add_argument("--n-perm", type=int, default=4999)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    D = {k: load(os.path.join(args.dir, f)) for k, f in
         dict(audit=args.e1, audit_layer="e6_audit_layer_20seeds.csv",
              ood="e6_ood_20seeds.csv", ood_layer="e6_ood_layer_20seeds.csv").items()}
    for k, d in D.items():
        print(f"{k:12s}: " + ("missing" if d is None else
                              f"{len(d)} rows, {int(d.crashed.sum())} crashed, "
                              f"{d.agent.nunique()} agents"))
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)

    # 1. violations without / with the layer
    rows = []
    for cond in ("audit", "audit_layer"):
        d = D[cond]
        if d is None:
            continue
        for agent, g in d.groupby("agent"):
            for probe in sorted(set(g.probe) - {"identity"}):
                cells = _paired_cells(g, agent, probe)
                rows.append(dict(cond=cond, agent=agent, probe=probe, V=_v_from_diffs(cells),
                                 p=signflip_p(cells, args.n_perm, rng),
                                 crashed=int(g[g.probe == probe].crashed.sum())))
    if rows:
        v = pd.DataFrame(rows)
        v["q"] = v.groupby("cond").p.transform(lambda p: bh(p.to_numpy()))
        v.to_csv(os.path.join(args.dir, "e6_violations.csv"), index=False)
        w = v.pivot_table(index=["agent", "probe"], columns="cond", values=["V", "q"])
        w[("V", "audit")] *= 100
        if ("V", "audit_layer") in w:
            w[("V", "audit_layer")] *= 100
        print("\n== V (points) and BH q without / with the layer")
        print(w.round(3).to_string())
        sig = v[v.q <= .05].groupby("cond").size()
        print("\nsignificant pairs (q <= .05):", sig.to_dict())

    # 2. performance cost of the layer, training and held-out functions
    rows = []
    agents = sorted(set().union(*[set(d.agent) for d in D.values() if d is not None]))
    for agent in agents:
        r = dict(agent=agent)
        for split, a_key, b_key in (("train", "audit", "audit_layer"), ("heldout", "ood", "ood_layer")):
            a, b = D[a_key], D[b_key]
            ia = None if a is None else a[(a.agent == agent) & (a.probe == "identity")]
            ib = None if b is None else b[(b.agent == agent) & (b.probe == "identity")]
            if ia is not None and len(ia):
                r[f"A_{split}_none"] = ia.A.mean()
            if ib is not None and len(ib):
                r[f"A_{split}_layer"] = ib.A.mean()
            if ia is not None and ib is not None and len(ia) and len(ib):
                r[f"d_{split}"], r[f"lo_{split}"], r[f"hi_{split}"], r[f"n_{split}"] = \
                    paired_diff(ia, ib, args.n_boot, rng)
                r[f"crash_{split}"] = int(ia.crashed.sum()), int(ib.crashed.sum())
        rows.append(r)
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(args.dir, "e6_cost.csv"), index=False)
    num = [x for x in c.columns if x.startswith(("A_", "d_", "lo_", "hi_"))]
    show = c.copy()
    show[num] = show[num] * 100
    print("\n== identity A (points) without / with the layer; d = layer - none, 95% CI")
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
