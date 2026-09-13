"""Power check of the anchor result on the MetaBox instances.

The audit of Fig. 3 finds no significant violation in the hand-designed anchors
with five seeds per cell.  This script asks whether that is a matter of power:
it analyses a repetition of experiment 3 with fifteen seeds per cell (the seed
count of the validation experiment) and compares the anchors' rotation effect
on the axis-aligned canonical functions with that on the MetaBox instances,
which are randomly rotated by construction.

Usage
-----
    python experiments/exp3_anchors_on_bbob.py \
        --family audit --seeds 15 --workers 10 --out results/exp3_audit_15seeds.csv
    python experiments/anchor_power_check.py
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from invaudit.analysis import audit_to_long, violation_scores, violation_significance  # noqa: E402

AUDIT_PROBES = ["ambient_translation_diag", "rotation", "coordinate_permutation",
                "domain_scaling_up", "domain_scaling_down", "fitness_rescale",
                "monotone_log", "monotone_spline"]


def scores(df, probes):
    sc = violation_scores(df, probes)
    sig = violation_significance(df, probes)
    return sc.merge(sig[["agent", "probe", "p", "q", "significant"]], on=["agent", "probe"])


def main():
    d15 = pd.read_csv(os.path.join(RES, "exp3_audit_15seeds.csv"))
    probes = [p for p in AUDIT_PROBES if p in set(d15.probe)]
    sc15 = scores(d15, probes)
    sc15.to_csv(os.path.join(RES, "table_anchor_power.csv"), index=False)

    d5 = pd.read_csv(os.path.join(RES, "exp3_audit.csv"))
    sc5 = scores(d5, probes)

    print(f"anchors, 5 seeds : {int(sc5.significant.sum())} of {len(sc5)} pairs significant "
          f"(FDR over the anchor pairs)")
    print(f"anchors, 15 seeds: {int(sc15.significant.sum())} of {len(sc15)} pairs significant")
    print(f"largest V x100 with 15 seeds: {100 * sc15.V.max():.2f}")
    print(sc15.sort_values("V", ascending=False).head(12)
          .assign(V=lambda x: (100 * x.V).round(2), V_signed=lambda x: (100 * x.V_signed).round(2))
          [["agent", "probe", "V", "V_signed", "p", "q", "significant"]].to_string(index=False))

    # rotation: axis-aligned canonical functions versus randomly rotated MetaBox instances
    e = pd.read_csv(os.path.join(RES, "exp1_anchor_validation.csv"))
    canon = (100 * e[e.probe == "rotation"].groupby(["optimizer", "function"]).delta.mean()
             ).groupby("optimizer").max()
    b = d15[d15.probe == "identity"].set_index(["agent", "fid", "instance", "seed"]).A
    r = d15[d15.probe == "rotation"].set_index(["agent", "fid", "instance", "seed"]).A
    mb = (100 * (b - r)).groupby(["agent", "fid"]).mean().groupby("agent").max()
    print("\nlargest mean loss under rotation (points), canonical vs MetaBox (15 seeds):")
    print(pd.DataFrame({"canonical": canon.round(2), "metabox": mb.round(2)}).to_string())


if __name__ == "__main__":
    main()
