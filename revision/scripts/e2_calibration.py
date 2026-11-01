"""E2 calibration - level and power of the audit's tests by simulation (R2.1, R2.2, R3.3).

Scenarios (14 cells, as in the agent audit; 20 seeds -> 10 + 10 in the disjoint-seed tests):
  null_continuous   X, Y ~ N(0, 1)                      (H0_dist true)
  null_ties         X, Y ~ uniform{0, .5, 1}            (H0_dist true, heavy ties like A = 0)
  spread_only       X ~ N(0,1), Y ~ N(0, 3^2)           (H0_mean true, H0_dist false)
  mean_shift        X ~ N(0,1), Y ~ N(0.5, 1)           (both false)
  reviewer_example  pairs uniform on {(0,.5),(.5,1),(1,0)} (equal marginals, coupled arms)

Output: results_rev/e2_calibration.csv with rejection rates at 5% and their
binomial standard errors.

Usage:  python revision/scripts/e2_calibration.py --reps 5000 --workers 30
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

N_CELLS, N_SEEDS, N_PERM = 14, 20, 999


def _one(args):
    import numpy as np
    from invaudit.inference import permutation_tests, signflip_counterexample
    scen, rep = args
    rng = np.random.default_rng(10_000 + rep)
    h = N_SEEDS // 2
    if scen == "reviewer_example":
        r = signflip_counterexample(n_cells=N_CELLS, n_seeds=N_SEEDS, n_rep=1,
                                    n_perm=N_PERM, seed=10_000 + rep)
        return scen, r["signflip"], r["V_u"], r["CvM"]
    gens = {
        "null_continuous": lambda: (rng.normal(size=h), rng.normal(size=h)),
        "null_ties": lambda: (rng.integers(0, 3, h) / 2.0, rng.integers(0, 3, h) / 2.0),
        "spread_only": lambda: (rng.normal(size=h), 3.0 * rng.normal(size=h)),
        "mean_shift": lambda: (rng.normal(size=h), 0.5 + rng.normal(size=h)),
    }
    cells = [gens[scen]() for _ in range(N_CELLS)]
    o = permutation_tests(cells, n_perm=N_PERM, seed=rep)
    return scen, float("nan"), float(o["p_V_u"] <= 0.05), float(o["p_CvM"] <= 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default="results_rev/e2_calibration.csv")
    args = ap.parse_args()
    import numpy as np
    import pandas as pd
    scens = ["null_continuous", "null_ties", "spread_only", "mean_shift", "reviewer_example"]
    tasks = [(s, r) for s in scens for r in range(args.reps)]
    with mp.Pool(args.workers) as pool:
        res = pool.map(_one, tasks, chunksize=20)
    df = pd.DataFrame(res, columns=["scenario", "signflip", "V_u", "CvM"])
    out = df.groupby("scenario", sort=False).agg(["mean", "count"])
    rows = []
    for s in scens:
        row = dict(scenario=s)
        for t in ("signflip", "V_u", "CvM"):
            p, n = out.loc[s, (t, "mean")], out.loc[s, (t, "count")]
            row[t] = p
            row[t + "_se"] = float(np.sqrt(p * (1 - p) / n)) if np.isfinite(p) else float("nan")
        rows.append(row)
    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
