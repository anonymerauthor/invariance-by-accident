"""E7 analysis - retrained models with / without the objective layer.

Per arm: mean A under identity on the training functions and on the held-out
functions, and the violation score V (paired over seeds, per cell) of each
probe on both splits.  Arms are then summarised per condition (mean and range
over training seeds).

Usage: python revision/scripts/e7_analysis.py --inp results_rev_pod/e7_eval.csv.gz
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from invaudit.analysis import _paired_cells, _v_from_diffs  # noqa: E402

PROBES = ["ambient_translation_diag", "rotation", "coordinate_permutation",
          "domain_scaling_up", "domain_scaling_down", "fitness_rescale",
          "monotone_log", "monotone_spline"]


def condition(arm):
    if arm == "pretrained":
        return "pretrained"
    m = re.match(r"(control|layer_rawreward|layer)_\d+(@\d+)?(\+L)?$", arm)
    return m.group(1) + (m.group(2) or "") + (m.group(3) or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results_rev_pod/e7_eval.csv.gz")
    ap.add_argument("--out", default="results_rev_pod/e7_arms.csv")
    args = ap.parse_args()
    d = pd.read_csv(args.inp, low_memory=False)
    # infeasible instances (e.g. held-out F20: optimum outside the probe-safe ball) are
    # excluded by design, as in the audit; they are not crashes of the optimizer
    unsup = d["error"].astype(str).str.startswith("UnsupportedInstance")
    print(f"excluded {int(unsup.sum())} rows on unsupported instances "
          f"({sorted(d[unsup].fid.unique())})")
    d = d[~unsup].copy()
    d["crashed"] = d["error"].notna() & (d["error"].astype(str) != "")
    d.loc[d.crashed, "A"] = 0.0
    rows = []
    for (arm, split), g in d.groupby(["arm", "split"]):
        g = g.assign(agent=arm)
        r = dict(arm=arm, cond=condition(arm), split=split,
                 A_id=g[g.probe == "identity"].A.mean(), crashed=int(g.crashed.sum()))
        for p in PROBES:
            if (g.probe == p).any():
                r["V_" + p] = _v_from_diffs(_paired_cells(g, arm, p))
        rows.append(r)
    r = pd.DataFrame(rows)
    r.to_csv(args.out, index=False)
    vcols = [c for c in r.columns if c.startswith("V_")]
    r["V_max"] = r[vcols].max(axis=1)
    pd.set_option("display.width", 250)
    for split in ("train", "heldout"):
        s = r[r.split == split]
        agg = s.groupby("cond").agg(n=("arm", "size"), A_id=("A_id", "mean"),
                                    A_min=("A_id", "min"), A_max=("A_id", "max"),
                                    **{c[2:]: (c, "mean") for c in vcols + ["V_max"]})
        num = agg.columns.drop("n")
        agg[num] = agg[num] * 100
        print(f"\n== {split} (A and V in points, x100; mean over training seeds)")
        print(agg.round(2).to_string())


if __name__ == "__main__":
    main()
