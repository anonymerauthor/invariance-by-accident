"""E5b - knock-out attribution table (AE.1, R1.1).

For every agent x knock-out x probe: violation score V (paired over seeds, as in
the audit), sign-flip p-value, mean identity performance, and crashed runs.
Each knock-out is compared with its own identity runs, so a knock-out that
changes performance on the original box does not create a spurious shift.

Usage: python revision/scripts/e5_analysis.py --inp results_rev_pod/e5_knockouts.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from invaudit.analysis import _paired_cells, _v_from_diffs  # noqa: E402
from invaudit.inference import bh  # noqa: E402


def signflip_p(cells, n_perm, rng):
    acc = np.zeros(n_perm + 1)
    for D in cells:
        n = len(D)
        s = rng.choice([-1.0, 1.0], size=(n_perm + 1, n))
        s[0] = 1.0
        X = s * D
        acc += X.mean(1) ** 2 - X.var(1, ddof=1) / n
    Vs = np.sqrt(np.maximum(0.0, acc / len(cells)))
    return 1.0 if Vs[0] == 0 else (np.sum(Vs[1:] >= Vs[0]) + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results_rev_pod/e5_knockouts.csv")
    ap.add_argument("--out", default="results_rev_pod/e5_attribution.csv")
    ap.add_argument("--n-perm", type=int, default=4999)
    args = ap.parse_args()
    d = pd.read_csv(args.inp, low_memory=False)
    d["crashed"] = d["error"].notna() & (d["error"].astype(str) != "")
    d.loc[d.crashed, "A"] = 0.0
    d["unit"] = d.agent + "|" + d.ko.astype(str)
    rng = np.random.default_rng(0)
    rows = []
    for unit, g in d.groupby("unit"):
        a_id = g[g.probe == "identity"].A.mean()
        for probe in sorted(set(g.probe) - {"identity"}):
            cells = _paired_cells(g, unit, probe, agent_col="unit")
            agent, ko = unit.split("|")
            rows.append(dict(agent=agent, ko=ko, probe=probe, A_identity=a_id,
                             A_probe=g[g.probe == probe].A.mean(),
                             V=_v_from_diffs(cells), p=signflip_p(cells, args.n_perm, rng),
                             crashed=int(g[g.probe == probe].crashed.sum()),
                             n_cells=len(cells)))
    r = pd.DataFrame(rows)
    r["q"] = bh(r.p.to_numpy())
    r.to_csv(args.out, index=False)
    show = r.assign(A_id100=100 * r.A_identity, A_pr100=100 * r.A_probe, V100=100 * r.V)
    pd.set_option("display.width", 200)
    print(show[["agent", "ko", "probe", "A_id100", "A_pr100", "V100", "q", "crashed"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
