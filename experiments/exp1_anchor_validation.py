"""Experiment 1 - instrument validation on optimisers of known invariance.

Before the audit can say anything about automatically designed optimisers it has
to reproduce what is already provable about hand-designed ones.  Every algorithm
in the anchor set has invariances that can be read off its update rule, so this
experiment is a falsifiable test of the measurement, declared in advance:

  * DE/rand/1/arith and CMA-ES must show no systematic loss under rotation,
    while DE/rand/1/bin, DE/best/1/bin, jDE, SHADE, LSHADE and PSO must.
  * SHADE and LSHADE must lose performance under a monotone re-parameterisation
    of the objective, because their memory update is weighted by fitness
    differences.  Every purely comparison-based algorithm must not.
  * No algorithm may react to a translation, a dilation or a rescaling of the
    objective.
  * A coordinate permutation must leave every algorithm's *performance* intact,
    while the pathwise reading separates coordinate-tied randomness from the
    rest.

Usage
-----
    python experiments/exp1_anchor_validation.py --quick
    python experiments/exp1_anchor_validation.py --budget 20000 --runs 15 \
        --instances 3 --out results/exp1.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invaudit.audit import AuditConfig, run_audit  # noqa: E402
from invaudit.functions import FUNCTIONS  # noqa: E402
from invaudit.optimizers import (DISTRIBUTIONAL_INVARIANCE,  # noqa: E402
                                 STRONG_EQUIVARIANCE, optimizer_names)

PROBES = ["identity_recomposed", "ambient_translation", "rotation",
          "coordinate_permutation", "domain_scaling_up", "domain_scaling_down",
          "fitness_rescale", "fitness_shift", "monotone_pow3", "monotone_log",
          "monotone_spline"]

QUICK_FUNCS = ["sphere", "ellipsoid", "rastrigin", "schwefel12"]


def scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per (optimiser, probe): distributional and pathwise readings."""
    rows = []
    for (opt, probe), g in df.groupby(["optimizer", "probe"]):
        d = g["delta"].to_numpy()
        n = len(d)
        rows.append(dict(
            optimizer=opt, probe=probe, n=n,
            V_dist=d.mean(),                       # signed: performance actually lost
            se=d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            V_path=np.abs(d).mean(),               # pathwise equivariance defect
            A_base=g["A_base"].mean(),
        ))
    out = pd.DataFrame(rows)
    out["t"] = out["V_dist"] / out["se"].replace(0.0, np.nan)
    return out


def _fmt_table(piv, cols, title, fmt="{:12.3e}"):
    print(f"\n{title}\n")
    print("optimizer".ljust(17) + "".join(f"{c[:12]:>14s}" for c in cols))
    for opt in piv.index:
        cells = "".join(f"{fmt.format(piv.loc[opt, c]):>14s}" for c in cols)
        print(opt.ljust(17) + cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bounded", action="store_true")
    ap.add_argument("--out", default="results/exp1_anchor_validation.csv")
    args = ap.parse_args()

    if args.quick:
        args.budget, args.runs, args.instances = 10000, 8, 1

    funcs = QUICK_FUNCS if args.quick else list(FUNCTIONS)
    cfg = AuditConfig(D=args.dim, budget=args.budget, n_runs=args.runs,
                      n_instances=args.instances, device=args.device,
                      bounded=args.bounded)
    print(f"anchor validation: {len(optimizer_names())} optimisers x {len(funcs)} "
          f"functions x {cfg.n_instances} instances x {len(PROBES)} probes, "
          f"D={cfg.D}, budget={cfg.budget}, runs={cfg.n_runs}, "
          f"bounded={cfg.bounded}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = run_audit(cfg, optimizer_names(), funcs, PROBES, verbose=True,
                   checkpoint=args.out)

    s = scores(df)
    order = optimizer_names()
    pv_dist = s.pivot(index="optimizer", columns="probe", values="V_dist").loc[order, PROBES]
    pv_path = s.pivot(index="optimizer", columns="probe", values="V_path").loc[order, PROBES]

    _fmt_table(pv_dist, PROBES,
               "V_dist  signed loss of performance under the action "
               "(positive = the canonical presentation was being exploited)")
    _fmt_table(pv_path, PROBES,
               "V_path  pathwise equivariance defect under common random numbers")

    # --- falsifiable check against the declared ground truth -----------------
    print("\ncheck against analytically known invariance")
    checks = [("distributional", DISTRIBUTIONAL_INVARIANCE, pv_dist, "V_dist"),
              ("pathwise", STRONG_EQUIVARIANCE, pv_path, "V_path")]
    for label, table, piv, col in checks:
        ok = bad = 0
        msgs = []
        for opt in order:
            for probe, expect in table[opt].items():
                if probe not in piv.columns:
                    continue
                v = piv.loc[opt, probe]
                if label == "pathwise":
                    measured_inv = abs(v) < 1e-12
                else:
                    row = s[(s.optimizer == opt) & (s.probe == probe)].iloc[0]
                    measured_inv = (abs(row["t"]) < 3.0) if row["se"] > 0 else True
                if bool(expect) == bool(measured_inv):
                    ok += 1
                else:
                    bad += 1
                    msgs.append(f"    {opt:16s} {probe:24s} expected_invariant="
                                f"{bool(expect)}  {col}={v:.3e}")
        print(f"  {label:16s} {ok} agree, {bad} disagree")
        for m in msgs:
            print(m)

    noise = pv_path["identity_recomposed"].abs().max()
    print(f"\nnumerical noise floor (identity through the action machinery): "
          f"{noise:.3e}")
    print(f"raw results written to {args.out}")


if __name__ == "__main__":
    main()
