"""Experiment 1b - are the small pathwise translation/scaling readings arithmetic?

Experiment 1 found pathwise defects of order 1e-5 to 1e-3 under translation and
dilation by 1e3 for algorithms that are algebraically equivariant to both.  If
they are amplified rounding error, a dilation by a power of two - exact in
binary floating point - must read exactly zero, while the 1e3 dilation keeps
its small non-zero reading.  A residual under the dyadic dilation would instead
point to a violation in the algorithm's logic.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invaudit.audit import AuditConfig, run_audit  # noqa: E402
from invaudit.functions import FUNCTIONS  # noqa: E402
from invaudit.optimizers import optimizer_names  # noqa: E402

PROBES = ["domain_scaling_up", "domain_scaling_dyadic_up", "domain_scaling_dyadic_down"]

if __name__ == "__main__":
    out = "results/exp1b_dyadic_scaling.csv"
    cfg = AuditConfig(D=10, budget=20000, n_runs=5, n_instances=3, device="cpu")
    df = run_audit(cfg, optimizer_names(), list(FUNCTIONS), PROBES, checkpoint=out)
    df["abs_delta"] = df["delta"].abs()
    print(df.pivot_table(index="optimizer", columns="probe", values="abs_delta",
                         aggfunc="mean").to_string(float_format=lambda v: f"{v:.2e}"))
