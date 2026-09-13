"""Experiment 6 - the invariance harness on the anchors, full validation grid.

Same grid as experiment 1 (fourteen functions, three instances, fifteen runs),
with the frame layer and the order-embedding objective layer active.  Two
quantities are read off it:

* the violation left under the probes the harness targets, which must vanish;
* the in-distribution cost, obtained by comparing the untransformed runs with
  those of experiment 1 under the same seeds - zero is predicted for every
  comparison-based optimiser, because the order embedding preserves every
  comparison the optimiser can make.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invaudit.audit import AuditConfig, run_audit  # noqa: E402
from invaudit.functions import FUNCTIONS  # noqa: E402
from invaudit.optimizers import optimizer_names  # noqa: E402

PROBES = ["ambient_translation", "domain_scaling_up", "domain_scaling_down",
          "fitness_rescale", "monotone_pow3", "monotone_log", "monotone_spline"]

def _cell(args):
    import torch
    torch.set_num_threads(1)
    from invaudit.audit import AuditConfig, run_cell
    opt, fname, inst = args
    cfg = AuditConfig(D=10, budget=20000, n_runs=15, n_instances=3, device="cpu",
                      harness_layers=dict(box_frame=True, rank_fitness=True,
                                          rank_mode="order"))
    return run_cell(cfg, opt, fname, inst, PROBES)


if __name__ == "__main__":
    # One cell per worker: the order embedding keeps per-run state in Python, so a
    # serial run of the full grid takes more than a day; cells are independent.
    import argparse
    import csv
    import itertools
    import multiprocessing as mp
    import time

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = "results/exp6_harness_anchors.csv"
    cells = list(itertools.product(optimizer_names(), list(FUNCTIONS), range(3)))
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = {(o, f, int(i)) for o, f, i in
                prev[["optimizer", "function", "instance"]].drop_duplicates()
                .itertuples(index=False)}
    todo = [c for c in cells if c not in done]
    print(f"{len(cells)} cells, {len(done)} done, {len(todo)} to run", flush=True)
    t0 = time.time()
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh, mp.Pool(args.workers) as pool:
        w = None
        for k, rows in enumerate(pool.imap_unordered(_cell, todo), 1):
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                if new:
                    w.writeheader()
            w.writerows(rows)
            fh.flush()
            el = time.time() - t0
            print(f"[{k:4d}/{len(todo)}] {rows[0]['optimizer']:16s} {rows[0]['function']:18s} "
                  f"elapsed={el / 60:6.1f}m eta={el / k * (len(todo) - k) / 60:6.1f}m",
                  flush=True)
