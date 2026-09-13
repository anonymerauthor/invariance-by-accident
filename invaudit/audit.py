"""The audit driver.

For one cell - an (optimiser, function, instance, seed block) combination - the
driver runs the algorithm on the untransformed instance and on every probed view
of it, under *common random numbers*: the same generator seed, and an initial
population drawn once in base coordinates and mapped through the action.  An
exactly equivariant algorithm therefore produces a bit-identical transformed
trajectory, so the measured defect is the algorithm's own and the numerical
noise floor - reported separately by the ``identity_recomposed`` probe - is many
orders of magnitude below any real violation.

Two readings are recorded per probe:

``V_signed``  mean(A_base - A_probe).  Positive means the algorithm performs
              worse once the symmetry is applied, i.e. it was exploiting the
              canonical presentation.
``V_abs``     mean |A_base - A_probe|.  Catches equivariance breaking that has no
              systematic direction, which is the regime a rotation probe lands in
              when the instance distribution is already orientation-symmetric.

Both are expressed in units of the bounded performance measure, and the
run-to-run standard deviation of ``A_base`` is stored alongside so that a
violation can be reported in units of ordinary stochastic variability.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .metrics import Recorder, default_checkpoints, default_targets, ecdf_area
from .harness import harness
from .optimizers import get_optimizer
from .problem import make_instance, stable_seed
from .transforms import get_probe, probe_names

__all__ = ["AuditConfig", "run_cell", "run_audit"]


@dataclass
class AuditConfig:
    D: int = 10
    budget: int = 20000
    n_runs: int = 15
    n_instances: int = 3
    n_targets: int = 51
    n_checkpoints: int = 31
    bounded: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    base_seed: int = 20260912
    probe_seed: int = 777
    # invariance harness: None, or e.g. dict(box_frame=True, rank_fitness=True)
    harness_layers: Optional[dict] = None
    # (function name, D, instance) -> Problem; defaults to the canonical suite.
    # Lets the anchors be audited on exactly the instances an external library
    # uses, so that every point in the analysis is measured the same way.
    problem_factory: Optional[Callable] = None


def _cell_seed(cfg: AuditConfig, opt_name: str, fname: str, instance: int) -> int:
    return stable_seed("cell", cfg.base_seed, opt_name, fname, instance)


def run_cell(cfg: AuditConfig, opt_name: str, fname: str, instance: int,
             probes: Sequence[str]) -> List[dict]:
    """Run one optimiser on one instance, untransformed and under every probe."""
    torch.set_default_dtype(cfg.dtype)
    cps = default_checkpoints(cfg.budget, cfg.n_checkpoints)
    tg = default_targets(cfg.n_targets).to(cfg.device)
    if cfg.problem_factory is None:
        p0 = make_instance(fname, cfg.D, instance, device=cfg.device,
                           dtype=cfg.dtype, bounded=cfg.bounded)
    else:
        p0 = cfg.problem_factory(fname, cfg.D, instance)
    seed = _cell_seed(cfg, opt_name, fname, instance)

    def _score(p):
        """Run the optimiser on view ``p`` and return per-run ECDF areas.

        A fresh recorder is attached to the innermost problem, so every
        evaluation - including any harness warm-up, which is charged to the same
        budget - is counted exactly once and scored in the original objective.
        """
        rec = Recorder(cps, cfg.n_runs, cfg.device, cfg.dtype)
        p.attach_recorder(rec)
        view, budget = p, cfg.budget
        if cfg.harness_layers:
            view = harness(p, cfg.n_runs, seed=cfg.base_seed, **cfg.harness_layers)
            budget = cfg.budget - view.warmup_cost
        get_optimizer(opt_name).run(view, budget, seed=seed, n_runs=cfg.n_runs)
        return ecdf_area(rec.finalize(), tg).cpu().numpy()

    A_base = _score(p0)

    rows = []
    for probe_name in probes:
        probe = get_probe(probe_name)
        p1 = probe.apply(p0, cfg.probe_seed + 31 * instance)
        if p1 is p0:
            # the identity probe returns the very same object; give it its own
            # view so that recorders are never shared
            p1 = p0.with_action(label="identity")
        A_probe = _score(p1)
        for r in range(cfg.n_runs):
            rows.append(dict(
                optimizer=opt_name, function=fname, D=cfg.D, instance=instance,
                probe=probe_name, group_action=probe.group_action,
                space=probe.space, run=r,
                A_base=float(A_base[r]), A_probe=float(A_probe[r]),
                delta=float(A_base[r] - A_probe[r]),
                bounded=cfg.bounded,
                harnessed=bool(cfg.harness_layers),
            ))
    return rows


def run_audit(cfg: AuditConfig, optimizers: Iterable[str],
              functions: Iterable[str], probes: Optional[Sequence[str]] = None,
              verbose: bool = True, checkpoint: Optional[str] = None) -> pd.DataFrame:
    """Run the full grid.

    With ``checkpoint`` set, every finished cell is appended to that CSV at once
    and cells already present are skipped, so a long run survives interruption
    and resumes where it stopped.
    """
    import csv
    import os

    probes = list(probes) if probes is not None else probe_names()
    optimizers, functions = list(optimizers), list(functions)
    cells = list(itertools.product(optimizers, functions, range(cfg.n_instances)))

    done_cells = set()
    if checkpoint and os.path.exists(checkpoint):
        prev = pd.read_csv(checkpoint)
        done_cells = {(o, f, int(i)) for o, f, i in
                      prev[["optimizer", "function", "instance"]].drop_duplicates()
                      .itertuples(index=False)}
    todo = [c for c in cells if c not in done_cells]
    if verbose and done_cells:
        print(f"resuming: {len(cells) - len(todo)} of {len(cells)} cells already done",
              flush=True)

    rows, t0 = [], time.time()
    writer, fh = None, None
    for k, (opt_name, fname, inst) in enumerate(todo, 1):
        cell_rows = run_cell(cfg, opt_name, fname, inst, probes)
        rows.extend(cell_rows)
        if checkpoint:
            if writer is None:
                new = not os.path.exists(checkpoint)
                fh = open(checkpoint, "a", newline="")
                writer = csv.DictWriter(fh, fieldnames=list(cell_rows[0].keys()))
                if new:
                    writer.writeheader()
            writer.writerows(cell_rows)
            fh.flush()
        if verbose:
            el = time.time() - t0
            eta = el / k * (len(todo) - k)
            print(f"[{k:4d}/{len(todo)}] {opt_name:16s} {fname:18s} inst={inst} "
                  f"elapsed={el/60:6.1f}m eta={eta/60:6.1f}m", flush=True)
    if fh is not None:
        fh.close()
    if checkpoint:
        return pd.read_csv(checkpoint)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (optimiser, probe) violation scores aggregated over the problem set."""
    base_sd = (df[df.probe == "identity"]
               .groupby(["optimizer", "function", "instance"])["A_base"]
               .std().groupby("optimizer").mean().rename("run_sd"))
    g = df.groupby(["optimizer", "probe"])
    out = g.agg(V_signed=("delta", "mean"),
                V_abs=("delta", lambda s: s.abs().mean()),
                V_max=("delta", lambda s: s.abs().max()),
                A_base=("A_base", "mean"),
                A_probe=("A_probe", "mean"),
                n=("delta", "size")).reset_index()
    return out.merge(base_sd, on="optimizer", how="left")
