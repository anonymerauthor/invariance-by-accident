"""E12 - how much of a rotation violation can the box explain? (R3.4)

The rotation probe rotates the function but keeps the presented box
axis-aligned (the box is the agent's interface and MetaBox treats it as a
scalar box).  The rotated presentation therefore differs from the original one
not only in orientation but also in the region of the function that lies in
the box.  A method that is perfectly rotation-equivariant can still show a
rotation violation if it initialises uniformly in the box or clips to it.

This script measures that box effect with reference DE variants (name
``de_<crossover>_<init>_<repair>``) that switch one design element at a time,
evaluated through exactly the same MetaBox view (boundary penalty, bias, box)
and instances as the agents:

  crossover  cr1: DE/rand/1 without crossover (CR = 1), rotation-equivariant;
             bin: binomial crossover with CR = 0.9, coordinate-wise.
  init       ball: initial population in the ball of radius 4 (the problem's
             own initialisation, rotation-invariant);
             box: uniform in the presented box (the MetaBox convention).
  repair     free: no bound handling (the boundary penalty, which rotates with
             the function, is the only constraint);
             clip: clip trial vectors to the presented box (the MetaBox convention).

de_cr1_ball_free is rotation-equivariant at every step: identity and rotated
runs are the same path in rotated coordinates, so its V must be 0 (sanity
check).  The other variants attribute a violation to crossover, box
initialisation and clipping, alone and together.

NP = 50, F = 0.5, budget 20,000 evaluations, 2 instances x 20 seeds on the
training and the held-out functions, identity and rotation, common random
numbers within a pair.  Exposure (fraction of evaluations outside the base
box) is recorded as in the audit.

Usage:
    python experiments/e12_rotation_control.py --out results_rev/e12_rotation_control.csv --workers 8
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

VARIANTS = ["de_cr1_ball_free", "de_cr1_ball_clip", "de_cr1_box_free", "de_cr1_box_clip",
            "de_bin_ball_free", "de_bin_box_clip"]
PROBES = ["identity", "rotation"]
KEY = ("variant", "split", "fid", "instance", "probe", "seed")
NP, F_W, BUDGET = 50, 0.5, 20000


def de_run(view, problem, variant, seed):
    """DE/rand/1 on a MetaBoxView; minimises view.func."""
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    D = problem.D
    lb = np.broadcast_to(np.asarray(view.lb, dtype=float), (D,))
    ub = np.broadcast_to(np.asarray(view.ub, dtype=float), (D,))
    _, cx, init, repair = variant.split("_")
    if init == "ball":
        # sample in base coordinates, then present: the same base points for
        # identity and rotation (the problem's own initialisation)
        gen = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
        X = problem.present(problem.init_population(NP, gen)).numpy()
    else:
        X = lb + (ub - lb) * rng.random((NP, D))
    clip = repair == "clip"
    cr = 1.0 if cx == "cr1" else 0.9
    fx = np.asarray(view.func(X), dtype=float)
    fes = NP
    while fes + NP <= BUDGET:
        idx = np.array([rng.choice(np.delete(np.arange(NP), i), 3, replace=False)
                        for i in range(NP)])
        V = X[idx[:, 0]] + F_W * (X[idx[:, 1]] - X[idx[:, 2]])
        if cr < 1.0:
            mask = rng.random((NP, D)) < cr
            mask[np.arange(NP), rng.integers(0, D, NP)] = True
            V = np.where(mask, V, X)
        else:
            rng.random((NP, D)); rng.integers(0, D, NP)      # keep the stream aligned
        if clip:
            V = np.clip(V, lb, ub)
        fv = np.asarray(view.func(V), dtype=float)
        fes += NP
        better = fv <= fx
        X[better], fx[better] = V[better], fv[better]


def run_task(t):
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    import exp2_metabox_audit as E2
    from invaudit.external import metabox as MB
    from invaudit.metrics import Recorder, default_checkpoints, default_targets, ecdf_area
    from invaudit.transforms import get_probe
    t0 = time.time()
    try:
        cps = default_checkpoints(BUDGET, E2.N_CHECKPOINTS)
        p0, bias = MB.bbob_instance(t["fid"], t["instance"])
        p = get_probe(t["probe"]).apply(p0, 777 + 31 * t["instance"])
        if p is p0:
            p = p0.with_action(label="identity")
        rec = Recorder(cps, 1, "cpu", torch.float64)
        p.attach_recorder(rec)
        expo = {}
        view = MB.MetaBoxView(p, bias=bias, max_fes=BUDGET, exposure=expo)
        seed = 1000 + 97 * t["seed"] + 13 * t["instance"] + t["fid"]
        de_run(view, p, t["variant"], seed)
        best = rec.finalize()
        n = max(expo["n"], 1)
        row = dict(t, A=float(ecdf_area(best, default_targets())[0]),
                   best_final=float(best[0, -1]), real_fe=int(rec.fe),
                   x_out_box=expo["out_box"] / n, x_on_face=expo["on_face"] / n,
                   seconds=round(time.time() - t0, 2), error="")
        for j in range(best.shape[1]):
            row[f"b{j}"] = float(best[0, j])
        return row
    except Exception as e:
        return dict(t, A=float("nan"), seconds=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {str(e)[:300]} | "
                          f"{traceback.format_exc(limit=3)[-400:]}".replace("\n", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_rev/e12_rotation_control.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import exp2_metabox_audit as E2
    tasks = [dict(variant=v, split=sp, fid=f, instance=i, probe=p, seed=s)
             for v in args.variants.split(",")
             for sp, fids in (("train", E2.TRAIN_FIDS), ("heldout", E2.TEST_FIDS))
             for f, i, p, s in itertools.product(fids, range(args.instances), PROBES,
                                                 range(args.seeds))]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {tuple(r[k] for k in KEY) for r in csv.DictReader(f)}
    todo = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    if args.limit:
        todo = todo[::max(1, len(todo) // args.limit)][:args.limit]
    print(f"{len(tasks)} tasks, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    fields = list(KEY) + ["A", "best_final", "real_fe", "x_out_box", "x_on_face", "seconds",
                          "error"] + [f"b{j}" for j in range(E2.N_CHECKPOINTS)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    import multiprocessing as mp
    t0 = time.time()
    with open(args.out, "a", newline="") as f, mp.Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_task, todo, chunksize=4), 1):
            w.writerow(row)
            f.flush()
            if i % 200 == 0 or i == len(todo):
                print(f"[{i:5d}/{len(todo)}] elapsed={(time.time() - t0) / 3600:5.2f}h", flush=True)


if __name__ == "__main__":
    main()
