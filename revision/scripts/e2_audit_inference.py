"""E2 - revised inference on the 20-seed agent audit (E1).  R2.1, R2.2, R3.3, R2.5.

For every agent x probe:
  * paired analysis on all 20 seeds (as in the submitted paper): V, signed loss,
    sign-flip p (exact under H0_path / exchangeable pairs), bootstrap 95% CIs;
  * disjoint-seed analysis (identity seeds 0-9, probe seeds 10-19, pre-specified):
    V_u and Cramer-von Mises with within-cell label permutation, exact under
    H0_dist (Definition 1);
  * Benjamini-Hochberg q-values over all agent-probe pairs for each test, and
    Benjamini-Yekutieli as a dependence-free sensitivity check.
Crashed runs score A = 0 (the method failed); the number of crashed runs is
reported in its own column.

Usage:  python revision/scripts/e2_audit_inference.py --inp results_rev/e1_audit_20seeds.csv \
            --out results_rev/e2_audit_inference.csv --workers 30
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

PROBES = ["ambient_translation_diag", "rotation", "coordinate_permutation",
          "domain_scaling_up", "domain_scaling_down", "fitness_rescale",
          "monotone_log", "monotone_spline"]
ID_SEEDS, PR_SEEDS = list(range(10)), list(range(10, 20))
_DF = None


def _load(path):
    import pandas as pd
    d = pd.read_csv(path)
    crashed = d["error"].notna() & (d["error"].astype(str) != "")
    d["crashed"] = crashed
    d.loc[crashed, "A"] = 0.0
    return d


def _init(path):
    global _DF
    _DF = _load(path)


def _one(args):
    import numpy as np
    from invaudit.analysis import _paired_cells, _v_from_diffs
    from invaudit.inference import bootstrap_paired, permutation_tests, split_cells
    agent, probe, n_perm = args
    d = _DF[_DF.agent == agent]
    cells = _paired_cells(d, agent, probe)
    V = _v_from_diffs(cells)
    # paired sign-flip, as in the paper
    rng = np.random.default_rng(0)
    acc = np.zeros(n_perm + 1)
    for D in cells:
        n = len(D)
        s = rng.choice([-1.0, 1.0], size=(n_perm + 1, n))
        s[0] = 1.0
        X = s * D
        acc += X.mean(1) ** 2 - X.var(1, ddof=1) / n
    Vs = np.sqrt(np.maximum(0.0, acc / len(cells)))
    p_sf = 1.0 if Vs[0] == 0 else (np.sum(Vs[1:] >= Vs[0]) + 1) / (n_perm + 1)
    ci = bootstrap_paired(cells, seed=1)
    pt = permutation_tests(split_cells(d, agent, probe, ID_SEEDS, PR_SEEDS),
                           n_perm=n_perm, seed=2)
    n_crash = int(d[(d.probe == probe)].crashed.sum())
    return dict(agent=agent, probe=probe, V=V, loss=float(np.mean([D.mean() for D in cells])),
                p_signflip=float(p_sf), **ci, **pt, crashed_runs=n_crash,
                n_seeds=int(d[d.probe == probe].seed.nunique()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results_rev/e1_audit_20seeds.csv")
    ap.add_argument("--out", default="results_rev/e2_audit_inference.csv")
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    import pandas as pd
    from invaudit.inference import bh, by
    d = _load(args.inp)
    tasks = [(a, p, args.n_perm) for a in sorted(d.agent.unique()) for p in PROBES
             if ((d.agent == a) & (d.probe == p)).any()]
    with mp.Pool(args.workers, initializer=_init, initargs=(args.inp,)) as pool:
        rows = pool.map(_one, tasks)
    r = pd.DataFrame(rows)
    for col in ("p_signflip", "p_V_u", "p_CvM"):
        r["q_" + col[2:]] = bh(r[col].to_numpy())
        r["qBY_" + col[2:]] = by(r[col].to_numpy())
    r.to_csv(args.out, index=False)
    show = r.assign(V100=100 * r.V, loss100=100 * r.loss)
    sig = show[(show.q_signflip <= 0.05) | (show.q_V_u <= 0.05) | (show.q_CvM <= 0.05)]
    cols = ["agent", "probe", "V100", "loss100", "q_signflip", "q_V_u", "q_CvM",
            "qBY_signflip", "crashed_runs"]
    print(f"{len(r)} pairs; significant at q<=0.05: signflip {int((r.q_signflip <= .05).sum())}, "
          f"V_u {int((r.q_V_u <= .05).sum())}, CvM {int((r.q_CvM <= .05).sum())}; "
          f"BY signflip {int((r.qBY_signflip <= .05).sum())}")
    print(sig[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
