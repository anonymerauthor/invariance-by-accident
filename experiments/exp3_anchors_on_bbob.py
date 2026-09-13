"""Experiment 3 - the hand-designed anchors on exactly the MetaBox instances.

Runs in the MetaBox environment (see README), because the instances are built by
MetaBox's own BBOB implementation (see ``external.metabox.bbob_instance``).

The anchors go through the same instances, probes, budget and performance
measure as the sixteen pretrained agents of experiment 2, so that the analysis
can place hand-designed and automatically designed optimisers on one axis.  Two
differences are deliberate and harmless: the anchors run with common random
numbers inside a batch of runs (they support it; the MetaBox agents do not), and
they are not shown MetaBox's additive bias, to which every anchor is invariant by
construction.

Output is written in the long format shared with experiment 2:
``family, agent, fid, instance, probe, seed, A``.

Usage
-----
    python experiments/exp3_anchors_on_bbob.py \
        --family audit --workers 3 --out results/exp3_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from exp2_metabox_audit import AUDIT_PROBES, TEST_FIDS, TRAIN_FIDS  # noqa: E402

BUDGET = 20000
KEY = ("family", "agent", "fid", "instance")


def run_cell_task(t):
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from invaudit.audit import AuditConfig, run_cell
    from invaudit.external.metabox import bbob_instance

    family = t["family"]
    probes = AUDIT_PROBES if family == "audit" else ["identity"]

    def factory(fname, D, instance):
        return bbob_instance(int(fname[1:]), instance, dim=D,
                             require_ball=(family == "audit"))[0]

    cfg = AuditConfig(D=10, budget=BUDGET, n_runs=t["seeds"], n_instances=1,
                      device="cpu", problem_factory=factory)
    t0 = time.time()
    rows = run_cell(cfg, t["agent"], f"F{t['fid']}", t["instance"], probes)
    out = []
    seen_identity = set()
    for r in rows:
        base = dict(family=family, agent=t["agent"], fid=t["fid"],
                    instance=t["instance"], seed=r["run"])
        if r["run"] not in seen_identity:
            out.append(dict(base, probe="identity", A=r["A_base"]))
            seen_identity.add(r["run"])
        if r["probe"] != "identity":
            out.append(dict(base, probe=r["probe"], A=r["A_probe"]))
    return t, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["audit", "ood"], required=True)
    ap.add_argument("--agents", default="all")
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from invaudit.optimizers import optimizer_names
    agents = optimizer_names() if args.agents == "all" else args.agents.split(",")
    fids = TRAIN_FIDS if args.family == "audit" else TEST_FIDS
    cells = [dict(family=args.family, agent=a, fid=f, instance=i, seeds=args.seeds)
             for a, f, i in itertools.product(agents, fids, range(args.instances))]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add(tuple(str(r[k]) for k in KEY))
    todo = [c for c in cells if tuple(str(c[k]) for k in KEY) not in done]
    print(f"{len(cells)} cells, {len(cells) - len(todo)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    fields = ["family", "agent", "fid", "instance", "probe", "seed", "A"]
    new_file = not os.path.exists(args.out)
    import multiprocessing as mp
    t0 = time.time()
    with open(args.out, "a", newline="") as f, mp.Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for i, (t, rows, sec) in enumerate(pool.imap_unordered(run_cell_task, todo), 1):
            w.writerows(rows)
            f.flush()
            el = time.time() - t0
            print(f"[{i:4d}/{len(todo)}] {t['agent']:16s} F{t['fid']:<2d} inst={t['instance']} "
                  f"{sec:6.1f}s elapsed={el / 60:6.1f}m eta={el / i * (len(todo) - i) / 60:6.1f}m",
                  flush=True)


if __name__ == "__main__":
    main()
