"""Experiment 7 - ablation of the harness's objective layer.

Compares four ways of presenting objective values to the optimizer - raw values,
a quantile against a sliding window, the worse-count transform
and the persistent order embedding - on in-distribution
performance, with the frame layer active in all harnessed variants.  This is the
design ablation that selected the order embedding; it uses stable seeds so that
it can be reproduced exactly.
"""

import csv
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invaudit.harness import harness  # noqa: E402
from invaudit.metrics import Recorder, default_checkpoints, default_targets, ecdf_area  # noqa: E402
from invaudit.optimizers import get_optimizer  # noqa: E402
from invaudit.problem import make_instance, stable_seed  # noqa: E402

FUNCS = ["ellipsoid", "rastrigin", "schwefel12", "rosenbrock", "ackley", "sharp_ridge"]
OPTS = ["DE/rand/1/bin", "jDE", "PSO", "SHADE", "LSHADE", "CMA-ES"]
MODES = {"raw": None,
         "window": dict(box_frame=True, rank_fitness=True, rank_mode="window"),
         "worse_count": dict(box_frame=True, rank_fitness=True, rank_mode="worse_count"),
         "order": dict(box_frame=True, rank_fitness=True, rank_mode="order")}
D, R, B, INSTANCES = 10, 10, 20000, 2


def run(opt, fname, inst, layers):
    torch.set_default_dtype(torch.float64)
    p = make_instance(fname, D, inst)
    rec = Recorder(default_checkpoints(B, 31), R, "cpu", torch.float64)
    p.attach_recorder(rec)
    view, budget = p, B
    if layers:
        view = harness(p, R, seed=1, **layers)
        budget = B - view.warmup_cost
    get_optimizer(opt).run(view, budget, seed=stable_seed("ablation", opt, fname, inst), n_runs=R)
    return ecdf_area(rec.finalize(), default_targets()).numpy()


def cell(args):
    torch.set_num_threads(1)
    opt, fname, inst = args
    out = []
    for mode, layers in MODES.items():
        A = run(opt, fname, inst, layers)
        out += [dict(optimizer=opt, function=fname, instance=inst, mode=mode, run=r, A=float(a))
                for r, a in enumerate(A)]
    return out


if __name__ == "__main__":
    import multiprocessing as mp
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "exp7_rank_ablation.csv")
    cells = list(itertools.product(OPTS, FUNCS, range(INSTANCES)))
    with open(path, "w", newline="") as f, mp.Pool(16) as pool:
        w = None
        for k, rows in enumerate(pool.imap_unordered(cell, cells), 1):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
            w.writerows(rows)
            f.flush()
            print(f"[{k}/{len(cells)}] {rows[0]['optimizer']} {rows[0]['function']}", flush=True)
