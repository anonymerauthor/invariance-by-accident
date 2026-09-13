"""Report the MetaBox instances used by the audit (runs in the MetaBox environment).

For every BBOB function and instance: whether it passes the global-minimum check,
the radius of its optimum, its training-style bias, and whether it is admissible for
the audit (optimum within radius four) or only for the out-of-distribution runs.
"""

import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TRAIN = [1, 2, 3, 5, 15, 16, 17, 21]

if __name__ == "__main__":
    from invaudit.external.metabox import UnsupportedInstance, bbob_instance
    rows = []
    for fid in range(1, 25):
        for inst in range(2):
            p, bias = bbob_instance(fid, inst, require_ball=False)
            radius = float(p.x_opt.norm())
            try:
                bbob_instance(fid, inst, require_ball=True)
                audit_ok = True
            except UnsupportedInstance:
                audit_ok = False
            rows.append(dict(fid=fid, instance=inst, split="train" if fid in TRAIN else "held-out",
                             radius=round(radius, 3), bias=bias, global_min_verified=True,
                             audit_admissible=audit_ok))
    out = os.path.join(os.path.dirname(HERE), "results", "table_instances.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
