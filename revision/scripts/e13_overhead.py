"""E13 - wall-clock overhead of the invariance harness (R3.6).

Runs anchors on one canonical function with and without the harness (frame layer, objective
layer, both) at D = 10, 30, 100, budget 20,000 or 100,000 evaluations, 15 runs per batch, and
reports the time per evaluation added by the harness.  Repeated three times; the median is
reported.  Timings depend on the machine; the ratio to the optimizer's own cost is the
transferable quantity.

Usage: python revision/scripts/e13_overhead.py [--out results_rev_local/e13_overhead.csv]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from invaudit.harness import HarnessedProblem  # noqa: E402
from invaudit.optimizers import get_optimizer  # noqa: E402
from invaudit.problem import make_instance  # noqa: E402

LAYERS = {"none": None, "frame": dict(box_frame=True, rank_fitness=False),
          "objective": dict(box_frame=False, rank_fitness=True),
          "both": dict(box_frame=True, rank_fitness=True)}


def one(opt, fname, D, budget, layer, n_runs=15):
    torch.set_default_dtype(torch.float64)
    p = make_instance(fname, D, 0, device="cpu", dtype=torch.float64, bounded=False)
    view = p if LAYERS[layer] is None else HarnessedProblem(p, n_runs=n_runs, **LAYERS[layer])
    t0 = time.perf_counter()
    get_optimizer(opt).run(view, budget, seed=1, n_runs=n_runs)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results_rev_local", "e13_overhead.csv"))
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for opt in ("DE/rand/1/bin", "CMA-ES"):
        for D in (10, 30, 100):
            for budget in (20000, 100000):
                for layer in LAYERS:
                    t = float(np.median([one(opt, "rastrigin", D, budget, layer)
                                         for _ in range(args.reps)]))
                    rows.append(dict(optimizer=opt, D=D, budget=budget, layer=layer,
                                     seconds=t, us_per_eval=1e6 * t / (15 * budget)))
                    print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    base = d[d.layer == "none"].set_index(["optimizer", "D", "budget"]).us_per_eval
    d["overhead_us_per_eval"] = d.us_per_eval - d.set_index(
        ["optimizer", "D", "budget"]).index.map(base).to_numpy()
    d.to_csv(args.out, index=False)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
