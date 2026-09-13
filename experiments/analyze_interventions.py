"""Derived tables for the realistic-domain, frame-layer and harness experiments.

Reads the per-run result files and writes

    results/table_dose.csv               violation under realistic domains
    results/table_frame_intervention.csv agents before / after the frame layer
    results/table_harness_anchors.csv    anchors: violation without / with the harness
    results/table_harness_cost.csv       anchors: in-distribution cost of the harness

Runs that raise an exception are scored as zero, as in the rest of the analysis.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")

from invaudit.analysis import violation_significance  # noqa: E402

COLS = ["family", "agent", "fid", "instance", "probe", "seed", "A", "error"]


def load_agents(name):
    d = pd.read_csv(os.path.join(RES, name))[COLS].copy()
    crashed = d.error.notna() & (d.error != "")
    d.loc[crashed, "A"] = 0.0
    return d


def anchors_long(df, probes):
    """Experiment-1-style rows (one untransformed score per run) to long format."""
    keys = ["optimizer", "function", "instance", "run"]
    base = (df.drop_duplicates(keys)[keys + ["A_base"]]
            .rename(columns={"A_base": "A"}).assign(probe="identity"))
    pr = df[df.probe.isin(probes)][keys + ["probe", "A_probe"]].rename(columns={"A_probe": "A"})
    return (pd.concat([base, pr], ignore_index=True)
            .rename(columns={"optimizer": "agent", "function": "fid", "run": "seed"}))


def main():
    # realistic domains
    dose = load_agents("exp5_dose.csv")
    dprobes = ["translate_5", "translate_50", "translate_500", "scale_0.1", "scale_20", "scale_100"]
    violation_significance(dose, dprobes).to_csv(os.path.join(RES, "table_dose.csv"), index=False)

    # frame layer on the agents: same seeds before and after
    fprobes = ["ambient_translation_diag", "domain_scaling_up", "domain_scaling_down"]
    after = load_agents("exp4_frame.csv")
    agents = sorted(after.agent.unique())
    before = load_agents("exp2_audit.csv")
    before = before[before.agent.isin(agents) & before.probe.isin(["identity"] + fprobes)]
    k = ["agent", "fid", "instance", "seed"]
    m = (before[before.probe == "identity"]
         .merge(after[after.probe == "identity"], on=k, suffixes=("_b", "_a")))
    print(f"frame layer: {len(m)} untransformed runs, "
          f"max |A_before - A_after| = {np.abs(m.A_b - m.A_a).max():.2e}")
    t = (violation_significance(before, fprobes)
         .merge(violation_significance(after, fprobes), on=["agent", "probe"],
                suffixes=("_before", "_after")))
    t[["agent", "probe", "V_before", "q_before", "V_after", "q_after"]].to_csv(
        os.path.join(RES, "table_frame_intervention.csv"), index=False)

    # harness on the anchors
    hprobes = ["ambient_translation", "domain_scaling_up", "domain_scaling_down",
               "fitness_rescale", "monotone_pow3", "monotone_log", "monotone_spline"]
    without = pd.read_csv(os.path.join(RES, "exp1_anchor_validation.csv"))
    with_h = pd.read_csv(os.path.join(RES, "exp6_harness_anchors.csv"))
    with_h = with_h.drop_duplicates(["optimizer", "function", "instance", "probe", "run"],
                                    keep="last")
    sb = violation_significance(anchors_long(without, hprobes), hprobes)
    sa = violation_significance(anchors_long(with_h, hprobes), hprobes)
    sb.merge(sa, on=["agent", "probe"], suffixes=("_b", "_a")).to_csv(
        os.path.join(RES, "table_harness_anchors.csv"), index=False)

    keys = ["optimizer", "function", "instance", "run"]
    b0 = without.drop_duplicates(keys)[keys + ["A_base"]]
    b1 = with_h.drop_duplicates(keys)[keys + ["A_base"]]
    mm = b0.merge(b1, on=keys, suffixes=("_off", "_on"))
    rows = []
    for opt, g in mm.groupby("optimizer"):
        diff = g.A_base_on - g.A_base_off
        p = stats.wilcoxon(g.A_base_on, g.A_base_off).pvalue if (diff != 0).any() else 1.0
        rows.append(dict(optimizer=opt, A_off=g.A_base_off.mean(), A_on=g.A_base_on.mean(),
                         delta=diff.mean(), max_abs=diff.abs().max(), p=p))
    pd.DataFrame(rows).set_index("optimizer").to_csv(os.path.join(RES, "table_harness_cost.csv"))
    print("wrote table_dose, table_frame_intervention, table_harness_anchors, table_harness_cost")


if __name__ == "__main__":
    main()
