"""Parallel anchor grid (revision): experiment 1 / 6 cells at any dimension and budget.

One cell = one optimiser on one instance of one of the fourteen canonical functions,
untransformed and under every probe, ``--runs`` runs with common random numbers
(``invaudit.audit.run_cell``).  Cells are independent and run in a process pool;
finished cells are appended at once, so the grid resumes where it stopped.

Uses:
  * E3 (R3.2): the anchors at 30 / 50 / 100-D, budget ``--budget-per-dim`` x D.
  * Pod reruns of experiment 1 (``--probes exp1``) and experiment 6
    (``--harness --probes exp6``) at 10-D, so that the in-distribution cost of the
    harness is measured on one platform, with the order-embedding instrumentation
    (ties, first tie per run, non-finite values; R2.4).

Usage:
    python experiments/anchor_grid.py --dim 30 --probes e3 --out results_rev/e3_anchors_30D.csv
    python experiments/anchor_grid.py --dim 10 --probes exp6 --harness --out results_rev/exp6.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBE_SETS = {
    "exp1": ["identity_recomposed", "ambient_translation", "rotation", "coordinate_permutation",
             "domain_scaling_up", "domain_scaling_down", "fitness_rescale", "fitness_shift",
             "monotone_pow3", "monotone_log", "monotone_spline"],
    "exp6": ["ambient_translation", "domain_scaling_up", "domain_scaling_down",
             "fitness_rescale", "monotone_pow3", "monotone_log", "monotone_spline"],
    # reduced set for the dimension study: one probe per group action
    "e3": ["ambient_translation", "rotation", "coordinate_permutation", "domain_scaling_up",
           "fitness_rescale", "monotone_log"],
}
KEY = ("optimizer", "function", "instance")


def _cell(task):
    import torch
    torch.set_num_threads(1)
    from invaudit.audit import AuditConfig, run_cell
    opt, fname, inst, a = task
    t0 = time.time()
    layers = (dict(box_frame=True, rank_fitness=True, rank_mode="order") if a["harness"]
              else None)
    try:
        cfg = AuditConfig(D=a["dim"], budget=a["budget"], n_runs=a["runs"],
                          n_instances=a["instances"], device="cpu", harness_layers=layers)
        rows = run_cell(cfg, opt, fname, inst, PROBE_SETS[a["probes"]])
        err = ""
    except Exception as e:                         # recorded, the grid goes on
        rows, err = [], f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2)[-300:]}"
    return task, rows, time.time() - t0, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--budget", type=int, default=0, help="fixed budget (overrides per-dim)")
    ap.add_argument("--budget-per-dim", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--probes", choices=list(PROBE_SETS), default="exp1")
    ap.add_argument("--harness", action="store_true")
    ap.add_argument("--optimizers", default="all")
    ap.add_argument("--functions", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from invaudit.functions import FUNCTIONS
    from invaudit.optimizers import optimizer_names
    opts = optimizer_names() if args.optimizers == "all" else args.optimizers.split(",")
    funcs = list(FUNCTIONS) if args.functions == "all" else args.functions.split(",")
    a = dict(dim=args.dim, budget=args.budget or args.budget_per_dim * args.dim,
             runs=args.runs, instances=args.instances, probes=args.probes,
             harness=args.harness)
    cells = list(itertools.product(opts, funcs, range(args.instances)))
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {tuple(r[k] for k in KEY) for r in csv.DictReader(f)}
    todo = [c for c in cells if tuple(str(x) for x in c) not in done]
    if args.limit:
        todo = todo[::max(1, len(todo) // args.limit)][:args.limit]
    print(f"D={a['dim']} budget={a['budget']} probes={a['probes']} harness={a['harness']}: "
          f"{len(cells)} cells, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    import multiprocessing as mp
    t0, n_err = time.time(), 0
    with open(args.out, "a", newline="") as fh, mp.Pool(args.workers) as pool:
        w = None
        for k, (task, rows, sec, err) in enumerate(
                pool.imap_unordered(_cell, [c + (a,) for c in todo]), 1):
            if err:
                n_err += 1
                print(f"ERROR {task[:3]}: {err}", flush=True)
            else:
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    if new:
                        w.writeheader()
                w.writerows(rows)
                fh.flush()
            el = time.time() - t0
            print(f"[{k:4d}/{len(todo)}] {task[0]:16s} {task[1]:18s} {sec:7.1f}s "
                  f"elapsed={el / 3600:5.2f}h eta={el / k * (len(todo) - k) / 3600:5.2f}h "
                  f"errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
