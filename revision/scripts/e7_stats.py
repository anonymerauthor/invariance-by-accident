"""E7 statistics.

(1) Per arm x split x probe: sign-flip p-value of V (as in the audit), BH over all tests.
(2) Condition contrasts in identity performance: per episode cell (split, fid, instance,
    seed) average A over the training seeds of each condition, take the paired
    difference between two conditions, and bootstrap the mean over cells (95% CI).

Usage: python revision/scripts/e7_stats.py --inp results_rev_pod/e7_eval.csv.gz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from invaudit.analysis import _paired_cells, _v_from_diffs  # noqa: E402
from invaudit.inference import bh  # noqa: E402
from e5_analysis import signflip_p  # noqa: E402
from e7_analysis import PROBES, condition  # noqa: E402

CONTRASTS = [("layer", "control"), ("layer_rawreward", "control"),
             ("layer_rawreward", "layer"), ("control", "pretrained"),
             ("layer_rawreward", "pretrained"), ("control+L", "control")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results_rev_pod/e7_eval.csv.gz")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _30D")
    ap.add_argument("--n-perm", type=int, default=4999)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    d = pd.read_csv(args.inp, low_memory=False)
    d = d[~d["error"].astype(str).str.startswith("UnsupportedInstance")].copy()
    crashed = d["error"].notna() & (d["error"].astype(str) != "")
    d.loc[crashed, "A"] = 0.0
    # a fixed checkpoint index (DEDDQN: "@14") is the same for every arm; drop it so
    # that the contrasts below match
    d["cond"] = d.arm.map(condition).str.replace(r"@\d+", "", regex=True)
    rng = np.random.default_rng(0)

    rows = []
    for (arm, split), g in d.groupby(["arm", "split"]):
        g = g.assign(agent=arm)
        for p in PROBES:
            cells = _paired_cells(g, arm, p)
            rows.append(dict(arm=arm, cond=condition(arm), split=split, probe=p,
                             V=_v_from_diffs(cells), p=signflip_p(cells, args.n_perm, rng)))
    v = pd.DataFrame(rows)
    v["q"] = bh(v.p.to_numpy())
    v.to_csv(f"results_rev_pod/e7_violation_tests{args.tag}.csv", index=False)
    sig = v.assign(sig=v.q <= 0.05).groupby(["cond", "split", "probe"]).sig.agg(["sum", "size"])
    print("== arms with significant V (q<=0.05) / arms, per condition")
    print(sig.unstack("probe")["sum"].astype(str) + "/" + sig.unstack("probe")["size"].astype(str))

    idn = d[d.probe == "identity"]
    m = idn.groupby(["cond", "split", "fid", "instance", "seed"]).A.mean().unstack("cond")
    out = []
    for a, b in CONTRASTS:
        if a not in m.columns or b not in m.columns:     # e.g. no pretrained model at 30-D
            continue
        for split in ("train", "heldout"):
            diff = (m.loc[split, a] - m.loc[split, b]).dropna().to_numpy()
            idx = rng.integers(0, len(diff), size=(args.n_boot, len(diff)))
            bs = diff[idx].mean(1)
            out.append(dict(contrast=f"{a} - {b}", split=split, n_cells=len(diff),
                            mean=100 * diff.mean(), lo=100 * np.quantile(bs, 0.025),
                            hi=100 * np.quantile(bs, 0.975)))
    o = pd.DataFrame(out)
    o.to_csv(f"results_rev_pod/e7_contrasts{args.tag}.csv", index=False)
    print("\n== identity A contrasts (points, mean over cells, bootstrap 95% CI)")
    print(o.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
