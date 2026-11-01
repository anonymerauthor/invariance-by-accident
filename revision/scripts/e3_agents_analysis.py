"""E3 agents - zero-shot audit of the pretrained 10-D agents at 30-D (R3.2).

Twelve agents whose networks accept any dimension (B2OPT, RLDAS, RLPSO and RNNOPT have
input layers sized for D = 10 and cannot run at 30-D) on the training functions,
2 instances, budget 50,000, under identity and one probe per group action.

Per agent x probe: V, sign-flip p and BH q at 30-D, next to the same quantities at
10-D from the E1 audit.  Both dimensions use seeds 0-9 so that the two columns have
the same power.  Also reports the zero-shot performance A at both dimensions.

Usage: python revision/scripts/e3_agents_analysis.py --dir results_rev_pod
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


def violations(d, D, rng, n_perm):
    rows = []
    for agent, g in d.groupby("agent"):
        for probe in sorted(set(g.probe) - {"identity"}):
            cells = _paired_cells(g, agent, probe)
            rows.append(dict(D=D, agent=agent, probe=probe, V=_v_from_diffs(cells),
                             p=signflip_p(cells, n_perm, rng), n_cells=len(cells),
                             A_id=g[g.probe == "identity"].A.mean(),
                             crashed=int(g.crashed.sum())))
    v = pd.DataFrame(rows)
    v["q"] = bh(v.p.to_numpy())
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rev_pod")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=4999)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    d30 = load(os.path.join(args.dir, "e3_agents_30D.csv"))
    d10 = load(os.path.join(args.dir, "e1_audit_20seeds.csv"))
    d30 = d30[d30.seed < args.seeds]
    d10 = d10[(d10.seed < args.seeds) & d10.agent.isin(set(d30.agent))
              & d10.probe.isin(set(d30.probe))]
    for k, d in (("10-D", d10), ("30-D", d30)):
        print(f"{k}: {len(d)} runs, {int(d.crashed.sum())} crashed, {d.agent.nunique()} agents, "
              f"probes {sorted(set(d.probe) - {'identity'})}")

    v = pd.concat([violations(d10, 10, rng, args.n_perm),
                   violations(d30, 30, rng, args.n_perm)], ignore_index=True)
    v.to_csv(os.path.join(args.dir, "e3_agents_violations.csv"), index=False)
    v["sig"] = v.q <= .05
    w = v.pivot_table(index=["agent", "probe"], columns="D", values="V") * 100
    s = v.pivot_table(index=["agent", "probe"], columns="D", values="sig", aggfunc="first")
    out = w.round(2).astype(str) + s.replace({True: "*", False: ""})
    print("\n== V (points), * = BH q <= .05 within dimension")
    print(out[s.any(axis=1)].to_string())
    print("\nsignificant pairs:", v[v.sig].groupby("D").size().to_dict())
    print("detected at 10-D -> also at 30-D:", int((s[10] & s[30]).sum()), "/", int(s[10].sum()),
          "; 30-D only:", int((~s[10] & s[30]).sum()))
    print("per probe (10-D / 30-D significant):")
    print(v[v.sig].groupby(["probe", "D"]).size().unstack(fill_value=0).to_string())
    rho, p = spearmanr(w[10], w[30])
    print(f"Spearman(V_10, V_30) over {len(w)} pairs: rho = {rho:.3f}, p = {p:.2g}")
    a = v.groupby(["agent", "D"]).A_id.first().unstack()
    print("\nzero-shot performance A (identity), 10-D vs 30-D:")
    print(a.round(3).to_string())


if __name__ == "__main__":
    main()
