"""Experiment 2 - audit the sixteen pretrained MetaBox-v2 agents.

Runs in the MetaBox environment (see README).

Two task families share one runner:

``audit``  every agent on its *training* functions (the strictest in-distribution
           setting), untransformed and under each probe, with the seed shared
           across a probe pair so that the comparison is paired.
``ood``    every agent, untransformed, on the functions held out from its
           training.  Used later for the out-of-distribution side of the law; it
           is never mixed with the probes, which keeps the law non-circular.
``rot``    (revision, E12) identity and rotation on the training functions, with
           the boundary exposure of every run recorded (see below).
``ood_probes`` (revision, E8) every probe on the held-out functions.  Held-out
           instances whose optimum lies outside the probe-safe ball (F20, F24 for
           some instances) are excluded, as in the audit.

Families ending in ``_layer`` run with the objective layer on.  The revision
families also record *exposure*: the fraction of evaluations outside the ball
of radius 5 (``x_out_ball``), outside the base box (``x_out_box``) and on the
presented box faces (``x_on_face``).  Inside the ball a rotated presentation
coincides with the original one, so a run that never leaves it cannot owe a
rotation violation to the box.

Every task is one independent episode, executed in a process pool with one
thread per worker.  Results are appended to a CSV as they complete and the
runner skips tasks already present, so an interrupted run resumes where it
stopped.  Besides the ECDF area the best value at every budget checkpoint and
the number of evaluations actually spent are stored, which lets the anytime
analysis be redone with different targets without re-running anything.

Usage
-----
    python experiments/exp2_metabox_audit.py \
        --family audit --workers 12 --out results/exp2_audit.csv
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

TRAIN_FIDS_ALL = [1, 2, 3, 5, 15, 16, 17, 21]      # MetaBox "difficult" training split
# F5 places its optimum on the boundary by definition, so no search-space probe
# other than a permutation preserves feasibility; it is excluded from the audit
# (see external.metabox.bbob_instance).  The held-out functions are used only
# untransformed, so none of them needs excluding.
TRAIN_FIDS = [f for f in TRAIN_FIDS_ALL if f != 5]
TEST_FIDS = [f for f in range(1, 25) if f not in TRAIN_FIDS_ALL]
# the invariance harness's frame layer removes these by construction
FRAME_PROBES = ["identity", "ambient_translation_diag", "domain_scaling_up",
                "domain_scaling_down"]
DOSE_PROBES = ["identity", "translate_5", "translate_50", "translate_500",
               "scale_0.1", "scale_20", "scale_100"]
AUDIT_PROBES = ["identity", "ambient_translation_diag", "rotation",
                "coordinate_permutation", "domain_scaling_up", "domain_scaling_down",
                "fitness_rescale", "monotone_log", "monotone_spline"]
N_CHECKPOINTS = 31
BUDGET = 20000
KEY = ("family", "agent", "fid", "instance", "probe", "seed")
EXPOSURE_FAMILIES = ("rot", "ood_probes")          # base families that record exposure
EXPOSURE_FIELDS = ["n_eval", "x_out_ball", "x_out_box", "x_on_face"]


def _task_key(t):
    return tuple(str(t[k]) for k in KEY)


def build_tasks(family, agents, n_instances, n_seeds):
    tasks = []
    base = family[:-len("_layer")] if family.endswith("_layer") else family
    if base == "audit":
        grid = itertools.product(agents, TRAIN_FIDS, range(n_instances), AUDIT_PROBES,
                                 range(n_seeds))
    elif base == "frame":
        grid = itertools.product(agents, TRAIN_FIDS, range(n_instances), FRAME_PROBES,
                                 range(n_seeds))
    elif base == "dose":
        grid = itertools.product(agents, TRAIN_FIDS, range(n_instances), DOSE_PROBES,
                                 range(n_seeds))
    elif base == "rot":
        grid = itertools.product(agents, TRAIN_FIDS, range(n_instances),
                                 ["identity", "rotation"], range(n_seeds))
    elif base == "ood_probes":
        grid = itertools.product(agents, TEST_FIDS, range(n_instances), AUDIT_PROBES,
                                 range(n_seeds))
    elif base == "ood":
        grid = itertools.product(agents, TEST_FIDS, range(n_instances), ["identity"],
                                 range(n_seeds))
    else:
        raise ValueError(family)
    for agent, fid, inst, probe, s in grid:
        tasks.append(dict(family=family, agent=agent, fid=fid, instance=inst,
                          probe=probe, seed=s))
    return tasks


def run_task(t):
    """Worker: one episode.  Imports happen here so each process is self-contained."""
    warnings.filterwarnings("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from invaudit.external.metabox import bbob_instance, run_agent
    from invaudit.metrics import Recorder, default_checkpoints, default_targets, ecdf_area
    from invaudit.transforms import get_probe

    t0 = time.time()
    try:
        dim, budget = int(t.get("dim", 10)), int(t.get("budget", BUDGET))
        cps = default_checkpoints(budget, N_CHECKPOINTS)
        base = t["family"][:-len("_layer")] if t["family"].endswith("_layer") else t["family"]
        p0, bias = bbob_instance(t["fid"], t["instance"], dim=dim,
                                 require_ball=(base != "ood"))
        probe = get_probe(t["probe"])
        p = probe.apply(p0, 777 + 31 * t["instance"])
        if p is p0:
            p = p0.with_action(label="identity")
        rec = Recorder(cps, 1, "cpu", torch.float64)
        p.attach_recorder(rec)
        seed = 1000 + 97 * t["seed"] + 13 * t["instance"] + t["fid"]
        expo = {} if base in EXPOSURE_FAMILIES else None
        run_agent(t["agent"], p, bias, seed=seed, maxFEs=budget,
                  frame=5.0 if t["family"] == "frame" else None,
                  objective_layer=t["family"].endswith("_layer"), exposure=expo)
        best = rec.finalize()
        A = float(ecdf_area(best, default_targets())[0])
        row = dict(t, A=A, best_final=float(best[0, -1]), real_fe=int(rec.fe),
                   seconds=round(time.time() - t0, 2), error="")
        for j in range(best.shape[1]):
            row[f"b{j}"] = float(best[0, j])
        if expo is not None:
            n = max(expo["n"], 1)
            row.update(n_eval=expo["n"], x_out_ball=expo["out_ball"] / n,
                       x_out_box=expo["out_box"] / n, x_on_face=expo["on_face"] / n)
        return row
    except Exception as e:  # keep the pool alive; the failure is recorded
        return dict(t, A=float("nan"), best_final=float("nan"), real_fe=-1,
                    seconds=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {str(e)[:300]} | "
                          f"{traceback.format_exc(limit=2)[-300:]}".replace("\n", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["audit", "ood", "frame", "dose", "audit_layer", "ood_layer",
                                         "rot", "ood_probes", "ood_probes_layer"], required=True)
    ap.add_argument("--agents", default="all")
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    ap.add_argument("--dim", type=int, default=10,
                    help="(revision, E3) zero-shot at another dimension with the pretrained "
                         "10-D models; write to a separate --out")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--probes", default="all",
                    help="comma list to keep (identity is always kept), e.g. for E3")
    args = ap.parse_args()

    from invaudit.external.metabox import METABOX_AGENTS
    agents = METABOX_AGENTS if args.agents == "all" else args.agents.split(",")
    tasks = build_tasks(args.family, agents, args.instances, args.seeds)
    for t in tasks:
        t.update(dim=args.dim, budget=args.budget)
    if args.probes != "all":
        keep = set(args.probes.split(",")) | {"identity"}
        tasks = [t for t in tasks if t["probe"] in keep]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    done.add(_task_key(r))
    todo = [t for t in tasks if _task_key(t) not in done]
    # slow agents first: the pool then drains the long tail on many cores
    slow = {"B2OPT": 0, "DEDDQN": 1, "NRLPSO": 2, "RLPSO": 3, "RLDEAFL": 4, "QLPSO": 5}
    todo.sort(key=lambda t: slow.get(t["agent"], 99))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(tasks)} tasks, {len(done)} already done, {len(todo)} to run "
          f"on {args.workers} workers", flush=True)
    if not todo:
        return

    fields = list(KEY) + ["dim", "budget", "A", "best_final", "real_fe", "seconds", "error"] + \
        [f"b{j}" for j in range(N_CHECKPOINTS)]
    if args.family.replace("_layer", "") in EXPOSURE_FAMILIES:
        fields += EXPOSURE_FIELDS
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    import multiprocessing as mp
    t0 = time.time()
    with open(args.out, "a", newline="") as f, mp.Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        n_err = 0
        for i, row in enumerate(pool.imap_unordered(run_task, todo, chunksize=1), 1):
            w.writerow(row)
            f.flush()
            n_err += bool(row["error"])
            if i % 25 == 0 or i == len(todo) or row["error"]:
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                msg = (f"[{i:5d}/{len(todo)}] {row['agent']:8s} F{row['fid']:<2d} "
                       f"{row['probe']:24s} A={row['A']:.4f} {row['seconds']:6.1f}s "
                       f"elapsed={el / 3600:5.2f}h eta={eta / 3600:5.2f}h errors={n_err}")
                if row["error"]:
                    msg += f"\n    ERROR {row['error'][:240]}"
                print(msg, flush=True)


if __name__ == "__main__":
    main()
