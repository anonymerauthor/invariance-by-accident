"""Anchor reruns on the pod (queue 9): validation, harness cost, dimension, MetaBox instances.

1. Experiment 1 at 10-D (``exp1_anchors_10D.csv``): sign-flip test per optimiser x probe,
   checked against the analytically known distributional invariance; agreement with the
   original (local) experiment 1.
2. Experiment 6 at 10-D (``exp6_harness_10D.csv``): anchors with the harness.  Violations
   with / without the harness, in-distribution cost (A_base on vs off, same seeds), and
   the order-embedding instrumentation (ties, first tie, non-finite values; R2.4).
3. E3 (R3.2): the same check at 30 / 50 / 100-D (``e3_anchors_{D}D.csv``).
4. E1b: anchors on the MetaBox BBOB instances, 20 seeds (``e1b_anchors_audit.csv``).

A probe expected to be invariant that is significant (BH q <= .05) is a false alarm.
A probe expected to break invariance that is not significant is only a miss of power;
the known non-invariances are crossover x rotation and SHADE-type weights x spline.

Usage: python revision/scripts/anchor_pod_analysis.py --dir results_rev_pod
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from invaudit.analysis import audit_to_long, violation_significance  # noqa: E402
from invaudit.optimizers import DISTRIBUTIONAL_INVARIANCE  # noqa: E402

ALIAS = {"ambient_translation_diag": "ambient_translation"}


def read(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            return pd.read_csv(p, low_memory=False)
    return None


def check(sig, label):
    """Agreement of the sign-flip verdicts with the known invariance."""
    s = sig.copy()
    s["expect_inv"] = [bool(DISTRIBUTIONAL_INVARIANCE[a].get(ALIAS.get(p, p), 1))
                       for a, p in zip(s.agent, s.probe)]
    fa = s[s.expect_inv & s.significant]
    det = s[~s.expect_inv]
    print(f"\n== {label}: {len(s)} tests, {int(s.significant.sum())} significant; "
          f"false alarms {len(fa)}/{int(s.expect_inv.sum())}; "
          f"known non-invariances detected {int(det.significant.sum())}/{len(det)}")
    if len(fa):
        print("  false alarms:")
        print((fa[["agent", "probe", "V", "p", "q"]].assign(V=lambda x: 100 * x.V))
              .round(4).to_string(index=False))
    if len(det):
        print("  known non-invariances (V in points):")
        print((det[["agent", "probe", "V", "q", "significant"]].assign(V=lambda x: 100 * x.V))
              .round(3).to_string(index=False))
    return s.assign(set=label)


def anchors(d, label):
    probes = sorted(set(d.probe) - {"identity", "identity_recomposed"})
    return check(violation_significance(audit_to_long(d), probes), label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rev_pod")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    out = []

    # 1. experiment 1, pod rerun
    e1 = read(os.path.join(args.dir, "exp1_anchors_10D.csv"))
    s1 = anchors(e1, "exp1 10-D (pod)")
    out.append(s1)
    old = read(os.path.join(ROOT, "results", "exp1_anchor_validation.csv"))
    if old is not None:
        probes = sorted(set(old.probe) - {"identity", "identity_recomposed"})
        so = violation_significance(audit_to_long(old), probes)
        m = so.merge(s1, on=["agent", "probe"], suffixes=("_old", "_pod"))
        print(f"\nexp1 original vs pod: verdict agreement "
              f"{int((m.significant_old == m.significant_pod).sum())}/{len(m)}, "
              f"Spearman(V) = {stats.spearmanr(m.V_old, m.V_pod)[0]:.3f}")
        dis = m[m.significant_old != m.significant_pod]
        if len(dis):
            print((dis[["agent", "probe", "V_old", "q_old", "V_pod", "q_pod"]]
                   .assign(V_old=lambda x: 100 * x.V_old, V_pod=lambda x: 100 * x.V_pod))
                  .round(4).to_string(index=False))

    # 2. experiment 6: harness
    e6 = read(os.path.join(args.dir, "exp6_harness_10D.csv"))
    s6 = anchors(e6, "exp6 10-D with harness (pod)")
    out.append(s6)
    hp = sorted(set(e6.probe))
    m = (s1[s1.probe.isin(hp)].merge(s6, on=["agent", "probe"], suffixes=("_off", "_on")))
    print("\nharness on the anchors, probes of experiment 6: significant without / with:",
          int(m.significant_off.sum()), "/", int(m.significant_on.sum()))
    keys = ["optimizer", "function", "instance", "run"]
    b0 = e1.drop_duplicates(keys)[keys + ["A_base"]]
    b1 = e6.drop_duplicates(keys)[keys + ["A_base", "ties_base", "nonfinite_base",
                                         "first_tie_base"]]
    mm = b0.merge(b1, on=keys, suffixes=("_off", "_on"))
    rows = []
    for opt, g in mm.groupby("optimizer", sort=False):
        diff = g.A_base_on - g.A_base_off
        p = stats.wilcoxon(g.A_base_on, g.A_base_off).pvalue if (diff != 0).any() else 1.0
        ft = pd.to_numeric(g.first_tie_base, errors="coerce")
        rows.append(dict(optimizer=opt, n=len(g), A_off=g.A_base_off.mean(),
                         A_on=g.A_base_on.mean(), delta_pts=100 * diff.mean(),
                         max_abs_pts=100 * diff.abs().max(), p=p,
                         runs_with_ties=float((g.ties_base > 0).mean()),
                         median_first_tie=ft.median(), nonfinite=int(g.nonfinite_base.sum())))
    cost = pd.DataFrame(rows)
    print("\n== in-distribution cost of the harness (untransformed runs, same seeds) "
          "and order-embedding ties")
    print(cost.round(4).to_string(index=False))
    cost.to_csv(os.path.join(args.dir, "harness_cost_pod.csv"), index=False)
    fp = e6.drop_duplicates(keys + ["probe"])
    t = fp.groupby("probe")[["ties_probe", "nonfinite_probe"]].agg(
        lambda x: float((x > 0).mean()))
    print("\nfraction of probed runs with ties / non-finite values, per probe:")
    print(t.round(3).to_string())
    fn = fp.groupby("function").ties_probe.agg(lambda x: float((x > 0).mean()))
    print("\nfraction of probed runs with ties, per function (top 6):")
    print(fn.sort_values(ascending=False).head(6).round(3).to_string())

    # 3. dimension
    for D in (30, 50, 100):
        d = read(os.path.join(args.dir, f"e3_anchors_{D}D.csv"))
        if d is None:
            print(f"\n(e3_anchors_{D}D.csv not yet available)")
            continue
        out.append(anchors(d, f"E3 anchors {D}-D"))

    # 4. MetaBox instances, 20 seeds
    b = read(os.path.join(args.dir, "e1b_anchors_audit.csv"))
    if b is not None:
        b = b.assign(fid=b.fid.astype(str))
        probes = sorted(set(b.probe) - {"identity"})
        out.append(check(violation_significance(b, probes), "E1b anchors, MetaBox instances"))

    pd.concat(out, ignore_index=True).to_csv(os.path.join(args.dir, "anchor_pod_checks.csv"),
                                             index=False)


if __name__ == "__main__":
    main()
