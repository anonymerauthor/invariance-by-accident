"""Experiment 2 - audit the sixteen pretrained MetaBox-v2 agents.

Runs in the MetaBox environment (see README).

Two task families share one runner:

``audit``  every agent on its *training* functions (the strictest in-distribution
           setting), untransformed and under each probe, with the seed shared
           across a probe pair so that the comparison is paired.
``ood``    every agent, untransformed, on the functions held out from its
           training.  Used later for the out-of-distribution side of the law; it
           is never mixed with the probes, which keeps the law non-circular.

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


def _task_key(t):
    return tuple(str(t[k]) for k in KEY)


def build_tasks(family, agents, n_instances, n_seeds):
    tasks = []
    if family == "audit":
        grid = itertools.product(agents, TRAIN_FIDS, range(n_instances), AUDIT_PROBES,
                                 range(n_seeds))
    elif family == "frame":
        grid = itertools.product(agents, TRAIN_FIDS, range(n_instances), FRAME_PROBES,
                                 range(n_seeds))
    elif family == "dose":
        grid = itertools.product(agents, TRAIN_FIDS, range(n_instances), DOSE_PROBES,
                                 range(n_seeds))
    elif family == "ood":
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
        cps = default_checkpoints(BUDGET, N_CHECKPOINTS)
        p0, bias = bbob_instance(t["fid"], t["instance"],
                                 require_ball=(t["family"] != "ood"))
        probe = get_probe(t["probe"])
        p = probe.apply(p0, 777 + 31 * t["instance"])
        if p is p0:
            p = p0.with_action(label="identity")
        rec = Recorder(cps, 1, "cpu", torch.float64)
        p.attach_recorder(rec)
        seed = 1000 + 97 * t["seed"] + 13 * t["instance"] + t["fid"]
        run_agent(t["agent"], p, bias, seed=seed, maxFEs=BUDGET,
                  frame=5.0 if t["family"] == "frame" else None)
        best = rec.finalize()
        A = float(ecdf_area(best, default_targets())[0])
        row = dict(t, A=A, best_final=float(best[0, -1]), real_fe=int(rec.fe),
                   seconds=round(time.time() - t0, 2), error="")
        for j in range(best.shape[1]):
            row[f"b{j}"] = float(best[0, j])
        return row
    except Exception as e:  # keep the pool alive; the failure is recorded
        return dict(t, A=float("nan"), best_final=float("nan"), real_fe=-1,
                    seconds=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {str(e)[:300]} | "
                          f"{traceback.format_exc(limit=2)[-300:]}".replace("\n", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["audit", "ood", "frame", "dose"], required=True)
    ap.add_argument("--agents", default="all")
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    args = ap.parse_args()

    from invaudit.external.metabox import METABOX_AGENTS
    agents = METABOX_AGENTS if args.agents == "all" else args.agents.split(",")
    tasks = build_tasks(args.family, agents, args.instances, args.seeds)

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

    fields = list(KEY) + ["A", "best_final", "real_fe", "seconds", "error"] + \
        [f"b{j}" for j in range(N_CHECKPOINTS)]
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
