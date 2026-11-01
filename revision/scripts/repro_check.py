"""Cross-platform reproduction check (Windows results -> RunPod Linux).

Re-runs a sample of stored tasks and compares them with the stored values:

* MetaBox agents: for every agent, F1 instance 0 seed 0 under ``identity`` and
  under ``ambient_translation_diag`` (exp2 audit family).  Compared: ECDF area A,
  final best value, evaluations spent and all 31 best-so-far checkpoints.
* Anchors: one validation cell (sphere, instance 0, 15 runs) per optimizer
  under ``rotation`` (exp1).  Compared: A_base and A_probe of every run.

Prints one line per task with the largest absolute difference and a final
summary.  Exact equality is reported separately from agreement to 1e-12.

Usage:  python revision/scripts/repro_check.py --workers 32
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../invaudit
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))


def _agent_task(t):
    import exp2_metabox_audit as E
    return t, E.run_task(dict(family="audit", **t))


def _anchor_task(opt):
    import torch
    torch.set_num_threads(1)
    from invaudit.audit import AuditConfig, run_cell
    cfg = AuditConfig(D=10, budget=20000, n_runs=15, n_instances=3, device="cpu")
    return opt, run_cell(cfg, opt, "sphere", 0, ["rotation"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    import pandas as pd
    from invaudit.external.metabox import METABOX_AGENTS
    from invaudit.optimizers import optimizer_names

    res = os.path.join(ROOT, "results")
    old2 = pd.read_csv(os.path.join(res, "exp2_audit.csv"))
    old1 = pd.read_csv(os.path.join(res, "exp1_anchor_validation.csv"))

    agent_tasks = [dict(agent=a, fid=1, instance=0, probe=p, seed=0)
                   for a in METABOX_AGENTS
                   for p in ("identity", "ambient_translation_diag")]
    cols = ["A", "best_final", "real_fe"] + [f"b{j}" for j in range(31)]

    n_exact = n_close = n_total = 0
    with mp.Pool(args.workers) as pool:
        for t, row in pool.imap_unordered(_agent_task, agent_tasks):
            m = old2[(old2.agent == t["agent"]) & (old2.fid == 1) & (old2.instance == 0)
                     & (old2.probe == t["probe"]) & (old2.seed == 0)]
            n_total += 1
            if row.get("error"):
                print(f"AGENT {t['agent']:8s} {t['probe']:26s} ERROR {row['error'][:200]}")
                continue
            if m.empty:
                print(f"AGENT {t['agent']:8s} {t['probe']:26s} no stored row")
                continue
            o = m.iloc[0]
            # compare only the columns present and finite in both records
            both = [c for c in cols if c in row and pd.notna(o[c])
                    and row[c] == row[c]]
            missing = [c for c in cols if c not in both]
            diff = max(abs(float(row[c]) - float(o[c])) for c in both)
            exact = all(float(row[c]) == float(o[c]) for c in both)
            dA = abs(float(row["A"]) - float(o["A"]))
            n_exact += exact
            n_close += diff <= 1e-12
            print(f"AGENT {t['agent']:8s} {t['probe']:26s} A_new={row['A']:.6f} "
                  f"A_old={o['A']:.6f} dA={dA:.1e} maxdiff={diff:.3e} exact={exact}"
                  + (f" uncompared={missing}" if missing else ""), flush=True)

        for opt, rows in pool.imap_unordered(_anchor_task, optimizer_names()):
            new = pd.DataFrame(rows)
            o = old1[(old1.optimizer == opt) & (old1.function == "sphere")
                     & (old1.instance == 0) & (old1.probe == "rotation")].sort_values("run")
            n_total += 1
            if len(o) != len(new):
                print(f"ANCHOR {opt:16s} stored rows {len(o)} != {len(new)}")
                continue
            d = max((new.A_base.values - o.A_base.values).__abs__().max(),
                    (new.A_probe.values - o.A_probe.values).__abs__().max())
            exact = bool(d == 0.0)
            n_exact += exact
            n_close += d <= 1e-12
            print(f"ANCHOR {opt:16s} rotation maxdiff={d:.3e} exact={exact}", flush=True)

    print(f"SUMMARY exact={n_exact}/{n_total} within1e-12={n_close}/{n_total}")


if __name__ == "__main__":
    main()
