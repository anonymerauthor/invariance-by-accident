"""E8 analysis - presentation probes on the held-out functions (AE.3, R1.2, R1.3, R3.5).

1. ``e8_violations.csv``: per agent x probe, V, sign-flip p and BH q on the training
   functions (E1 audit) and on the held-out functions (E8); BH within each split.
2. ``e8_decomposition.csv``: the two terms of the decomposition (T2), per agent x probe:
     shift  = A_train(identity) - A_heldout(identity)      (new functions)
     pres   = A_heldout(identity) - A_heldout(probe)        (new presentation, signed)
   with V_heldout as the bound on |pres|.
3. Agreement of train and held-out V across agent x probe pairs (Spearman), and
   whether a violation detected on the training functions is also detected held-out.

Infeasible instances (UnsupportedInstance) are excluded; other errors are crashes with
A = 0, as in the audit.

Usage: python revision/scripts/e8_analysis.py --dir results_rev_pod
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invaudit.analysis import _paired_cells, _v_from_diffs  # noqa: E402
from invaudit.inference import bh  # noqa: E402
from e5_analysis import signflip_p  # noqa: E402
from e12_analysis import load  # noqa: E402


def violations(d, split, rng, n_perm):
    rows = []
    for agent, g in d.groupby("agent"):
        for probe in sorted(set(g.probe) - {"identity"}):
            cells = _paired_cells(g, agent, probe)
            rows.append(dict(split=split, agent=agent, probe=probe, V=_v_from_diffs(cells),
                             p=signflip_p(cells, n_perm, rng), n_cells=len(cells),
                             A_id=g[g.probe == "identity"].A.mean(),
                             A_probe=g[g.probe == probe].A.mean(),
                             crashed=int(g[g.probe == probe].crashed.sum())))
    v = pd.DataFrame(rows)
    v["q"] = bh(v.p.to_numpy())
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rev_pod")
    ap.add_argument("--n-perm", type=int, default=4999)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    tr = load(os.path.join(args.dir, "e1_audit_20seeds.csv"))
    ho = load(os.path.join(args.dir, "e8_ood_probes_20seeds.csv"))
    for k, d in (("train", tr), ("heldout", ho)):
        print(f"{k}: {len(d)} rows, {int(d.crashed.sum())} crashed, {d.agent.nunique()} agents")

    v = pd.concat([violations(tr, "train", rng, args.n_perm),
                   violations(ho, "heldout", rng, args.n_perm)], ignore_index=True)
    v.to_csv(os.path.join(args.dir, "e8_violations.csv"), index=False)
    w = v.pivot_table(index=["agent", "probe"], columns="split", values=["V", "q"])
    w["V"] = w["V"] * 100
    print("\n== V (points) and BH q, training vs held-out functions")
    print(w.round(3).to_string())

    sig = v.assign(sig=v.q <= .05).pivot_table(index=["agent", "probe"], columns="split",
                                                values="sig", aggfunc="first")
    print("\nsignificant pairs:", v[v.q <= .05].groupby("split").size().to_dict())
    print("detected train -> also held-out:",
          int((sig.train & sig.heldout).sum()), "/", int(sig.train.sum()),
          "; held-out only:", int((~sig.train & sig.heldout).sum()))
    print("per probe (train / held-out significant):")
    print(v[v.q <= .05].groupby(["probe", "split"]).size().unstack(fill_value=0).to_string())
    m = w["V"].dropna()
    rho, p = spearmanr(m["train"], m["heldout"])
    print(f"Spearman(V_train, V_heldout) over {len(m)} pairs: rho = {rho:.3f}, p = {p:.2g}")

    # decomposition (T2): shift + presentation term; V_heldout bounds |pres|
    a_tr = tr[tr.probe == "identity"].groupby("agent").A.mean()
    h = v[v.split == "heldout"].copy()
    h["shift"] = h.agent.map(a_tr) - h.A_id
    h["pres"] = h.A_id - h.A_probe
    h["bound_ok"] = h.pres.abs() <= h.V + 1e-12
    h.to_csv(os.path.join(args.dir, "e8_decomposition.csv"), index=False)
    print("\n|pres| <= V_heldout in", int(h.bound_ok.sum()), "/", len(h), "pairs")
    s = h.groupby("probe")[["shift", "pres", "V"]].mean() * 100
    print("\n== mean over agents (points): shift (new functions), pres (new presentation), V")
    print(s.round(3).to_string())
    big = h[h.pres.abs() > h["shift"].abs()]
    print(f"\npairs where the presentation term exceeds the function-shift term: {len(big)}/{len(h)}")
    print((big[["agent", "probe", "shift", "pres", "V", "q"]]
           .assign(shift=lambda x: 100 * x["shift"], pres=lambda x: 100 * x.pres,
                   V=lambda x: 100 * x.V).round(3).to_string(index=False)))


if __name__ == "__main__":
    main()
