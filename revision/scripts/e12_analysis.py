"""E12 analysis - the box effect of the rotation probe (R3.4).

1. Reference DE variants (e12_rotation_control.csv): rotation V and sign-flip p per
   variant and split; mean exposure.  de_cr1_ball_free must give V = 0.
2. Agents (e12_rot_20seeds.csv: training functions; e8_ood_probes_20seeds.csv:
   held-out functions, if present): rotation V, p, and exposure (fraction of
   evaluations outside the base box under rotation, on the presented box faces).

Usage: python revision/scripts/e12_analysis.py --dir results_rev_pod
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invaudit.analysis import _paired_cells, _v_from_diffs  # noqa: E402
from invaudit.inference import bh  # noqa: E402
from e5_analysis import signflip_p  # noqa: E402


def load(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            d = pd.read_csv(p, low_memory=False, encoding="latin-1")
            err = d["error"].fillna("").astype(str)
            d = d[~err.str.startswith("UnsupportedInstance")].copy()
            d["crashed"] = d["error"].fillna("").astype(str) != ""
            d.loc[d.crashed, "A"] = 0.0
            return d
    return None


def rotation_table(d, unit_cols, rng, n_perm):
    d = d.assign(unit=d[unit_cols].astype(str).agg("|".join, axis=1))
    rows = []
    for unit, g in d.groupby("unit"):
        cells = _paired_cells(g, unit, "rotation", agent_col="unit")
        if not cells:
            continue
        r = dict(zip(unit_cols, unit.split("|")))
        rot, idn = g[g.probe == "rotation"], g[g.probe == "identity"]
        r.update(A_id=idn.A.mean(), V=_v_from_diffs(cells), p=signflip_p(cells, n_perm, rng),
                 n_cells=len(cells), crashed=int(g.crashed.sum()))
        for c in ("x_out_box", "x_on_face"):
            if c in g:
                r[c + "_id"], r[c + "_rot"] = idn[c].mean(), rot[c].mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    t["q"] = bh(t.p.to_numpy())
    return t


def show(t, title):
    s = t.copy()
    for c in ("A_id", "V"):
        s[c] = s[c] * 100
    print(f"\n== {title} (A_id, V in points; x_* = fraction of evaluations)")
    print(s.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rev_pod")
    ap.add_argument("--n-perm", type=int, default=4999)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    c = load(os.path.join(args.dir, "e12_rotation_control.csv"))
    if c is not None:
        t = rotation_table(c, ["variant", "split"], rng, args.n_perm)
        t.to_csv(os.path.join(args.dir, "e12_control_table.csv"), index=False)
        show(t, "reference DE variants")
    parts = []
    a = load(os.path.join(args.dir, "e12_rot_20seeds.csv"))
    if a is not None:
        parts.append(a.assign(split="train"))
    o = load(os.path.join(args.dir, "e8_ood_probes_20seeds.csv"))
    if o is not None:
        parts.append(o[o.probe.isin(["identity", "rotation"])].assign(split="heldout"))
    if parts:
        t = rotation_table(pd.concat(parts, ignore_index=True), ["agent", "split"], rng,
                           args.n_perm)
        t.to_csv(os.path.join(args.dir, "e12_agents_table.csv"), index=False)
        show(t.sort_values(["split", "V"], ascending=[True, False]), "agents, rotation")


if __name__ == "__main__":
    main()
