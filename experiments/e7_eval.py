"""E7 evaluation - audit the retrained RLDAS / DEDDQN models (AE.4, R1.3, R1.4, R3.6).

Every model is run on the audit grid (training functions, 2 instances) and on
the held-out functions (MetaBox difficult test split, 2 instances), under
identity and the eight audit probes, with 20 seeds - the protocol of E1.

Arms (column ``arm``):
  pretrained          the distributed MetaBox model, no layer (E1 reference)
  <cond>_<seed>       a retrained model (cond = control | layer | layer_rawreward);
                      layer models are evaluated with the objective layer on,
                      as they were trained, control models without it
  control_<seed>+L    a control model with the objective layer switched on at
                      test time only (zero-shot layer, as E6)

The model is the last checkpoint of each run (``--checkpoint last``) or a given
checkpoint index (``--checkpoint 18``: the index MetaBox selected for the
distributed DEDDQN); a fixed index is appended to the arm name (``@18``).  Raw objective values are recorded through the
problem's recorder whether or not the layer is on, so A is always computed on
raw values.

Usage:
    python experiments/e7_eval.py --runs results_rev/e7 --out results_rev/e7_eval.csv --workers 30
    add --limit 40 for a smoke test
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re
import sys
import time
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

PROBES = ["identity", "ambient_translation_diag", "rotation", "coordinate_permutation",
          "domain_scaling_up", "domain_scaling_down", "fitness_rescale",
          "monotone_log", "monotone_spline"]
KEY = ("agent", "arm", "split", "fid", "instance", "probe", "seed")


def pick_checkpoint(run_dir, which="last"):
    cps = glob.glob(os.path.join(run_dir, "model", "**", "checkpoint-*.pkl"), recursive=True)
    idx = lambda p: int(re.search(r"checkpoint-(\d+)\.pkl$", p).group(1))
    if which != "last":
        cps = [p for p in cps if idx(p) == int(which)]
    if not cps:
        return None, -1
    p = max(cps, key=idx)
    return p, idx(p)


def arms(runs_root, agent="RLDAS", suite="10D", which="last", allow_unfinished=False):
    # the distributed models are fixed at 10-D (state size depends on D)
    out = [dict(agent=agent, arm="pretrained", model="", layer=False, ckpt=-1)]         if suite == "10D" else []
    tag = "" if which == "last" else f"@{which}"
    for d in sorted(glob.glob(os.path.join(runs_root, f"{agent}_{suite}_*"))):
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj):
            continue
        if "end" not in json.load(open(rj)) and not (allow_unfinished and which != "last"):
            continue                                 # unfinished run
        m = re.match(rf"{agent}_{suite}_(control|layer_rawreward|layer)_(\d+)$", os.path.basename(d))
        if not m:
            continue
        cond, seed = m.groups()
        path, k = pick_checkpoint(d, which)
        if path is None:
            continue
        out.append(dict(agent=agent, arm=f"{cond}_{seed}{tag}", model=path,
                        layer=cond != "control", ckpt=k))
        if cond == "control":
            out.append(dict(agent=agent, arm=f"{cond}_{seed}{tag}+L", model=path, layer=True, ckpt=k))
    return out


def build_tasks(arm_list, n_instances=2, n_seeds=20, dim=10, budget=20000):
    import exp2_metabox_audit as E2
    tasks = []
    for a in arm_list:
        for split, fids in (("train", E2.TRAIN_FIDS), ("heldout", E2.TEST_FIDS)):
            for fid, inst, probe, s in itertools.product(fids, range(n_instances), PROBES,
                                                         range(n_seeds)):
                tasks.append(dict(a, split=split, fid=fid, instance=inst, probe=probe, seed=s,
                                  dim=dim, budget=budget))
    return tasks


def run_task(t):
    warnings.filterwarnings("ignore")
    import pickle
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import exp2_metabox_audit as E2
    from invaudit.external import metabox as MB
    from invaudit.metrics import Recorder, default_checkpoints, default_targets, ecdf_area
    from invaudit.transforms import get_probe
    t0 = time.time()
    try:
        budget = int(t.get("budget", E2.BUDGET))
        cps = default_checkpoints(budget, E2.N_CHECKPOINTS)
        p0, bias = MB.bbob_instance(t["fid"], t["instance"], dim=int(t.get("dim", 10)))
        p = get_probe(t["probe"]).apply(p0, 777 + 31 * t["instance"])
        if p is p0:
            p = p0.with_action(label="identity")
        rec = Recorder(cps, 1, "cpu", torch.float64)
        p.attach_recorder(rec)
        seed = 1000 + 97 * t["seed"] + 13 * t["instance"] + t["fid"]
        from metaevobox.environment.basic_environment import PBO_Env
        agent, opt_cls = MB.load_agent(t["agent"])
        if t["model"]:
            with open(t["model"], "rb") as f:
                agent = pickle.load(f)
            agent.config.full_meta_data = False
        cfg = MB.make_config(maxFEs=budget, dim=p.D, device="cpu")
        env = PBO_Env(MB.MetaBoxView(p, bias=bias, objective_layer=bool(t["layer"]),
                                     max_fes=budget), opt_cls(cfg))
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float64)
        agent.rollout_episode(env, seed, {})
        best = rec.finalize()
        A = float(ecdf_area(best, default_targets())[0])
        row = dict(t, A=A, best_final=float(best[0, -1]), real_fe=int(rec.fe),
                   seconds=round(time.time() - t0, 2), error="")
        for j in range(best.shape[1]):
            row[f"b{j}"] = float(best[0, j])
        return row
    except Exception as e:
        return dict(t, A=float("nan"), best_final=float("nan"), real_fe=-1,
                    seconds=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {str(e)[:300]} | "
                          f"{traceback.format_exc(limit=3)[-400:]}".replace("\n", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results_rev/e7")
    ap.add_argument("--agent", default="RLDAS", choices=["RLDAS", "DEDDQN"])
    ap.add_argument("--checkpoint", default="last", help="'last' or a checkpoint index")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--budget", type=int, default=20000, help="maxFEs (MetaBox: 50000 at 30-D)")
    ap.add_argument("--out", default="results_rev/e7_eval.csv")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="", help="comma-separated arm names to keep")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="with a fixed --checkpoint, also use runs stopped before the last epoch")
    args = ap.parse_args()
    arm_list = arms(args.runs, args.agent, suite=f"{args.dim}D", which=args.checkpoint,
                    allow_unfinished=args.allow_unfinished)
    if args.arms:
        keep = set(args.arms.split(","))
        arm_list = [a for a in arm_list if a["arm"] in keep]
    for a in arm_list:
        print(f"  {a['arm']:24s} layer={a['layer']!s:5s} ckpt={a['ckpt']:3d} {a['model']}")
    tasks = build_tasks(arm_list, n_seeds=args.seeds, dim=args.dim, budget=args.budget)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    # files written before the agent column existed hold RLDAS only
                    done.add(tuple(str(r.get(k) or ("RLDAS" if k == "agent" else "")) for k in KEY))
    todo = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    if args.limit:
        todo = todo[::max(1, len(todo) // args.limit)][:args.limit]
    print(f"{len(tasks)} tasks, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    import exp2_metabox_audit as E2
    fields = list(KEY) + ["model", "layer", "ckpt", "A", "best_final", "real_fe", "seconds",
                          "error"] + [f"b{j}" for j in range(E2.N_CHECKPOINTS)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    if not new_file:                       # append with the existing file's columns
        with open(args.out, newline="") as f:
            fields = next(csv.reader(f))
    import multiprocessing as mp
    t0 = time.time()
    with open(args.out, "a", newline="") as f, mp.Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_task, todo, chunksize=4), 1):
            w.writerow(row)
            f.flush()
            if i % 500 == 0 or i == len(todo) or row["error"]:
                msg = (f"[{i:6d}/{len(todo)}] {row['arm']:22s} {row['split']:7s} F{row['fid']:<2d} "
                       f"{row['probe']:24s} A={row['A']:.4f} elapsed={(time.time() - t0) / 3600:5.2f}h")
                if row["error"]:
                    msg += f"\n    ERROR {row['error'][:300]}"
                print(msg, flush=True)


if __name__ == "__main__":
    main()
