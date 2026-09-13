"""Bounded, anytime performance measure.

The audit needs a score that (i) lies in [0, 1] so that differences across
probes and algorithms are commensurable, (ii) is anytime, so that an algorithm
merely slowed down by a transformation is distinguished from one broken by it,
and (iii) is computed in the *original* objective, never in the transformed one.

We use the area under the target-hitting ECDF, the standard measure of the
COCO/BBOB protocol: for a grid of precision targets and a grid of evaluation
budgets, the score is the fraction of (target, budget) pairs the run has solved.
Because every canonical function has ``f_opt = 0`` exactly, a target is simply an
absolute objective value.

What is measured is the best *evaluated* point, following COCO, not the point
the algorithm nominates as its incumbent.  The distinction matters here in a way
it does not in ordinary benchmarking.  The invariance harness replaces the
objective an algorithm sees with a sliding quantile, which is order preserving
within a generation but not across generations; an algorithm that tracks its own
incumbent by comparing stored objective values would then nominate the wrong
point, and the audit would record a performance collapse that the search itself
never suffered.  Scoring the best evaluated point removes that artefact, and it
is invariant to any strictly increasing re-parameterisation of the objective by
construction, since such a map cannot change which point is best.

Recording happens inside the problem, so it also counts the evaluations any
wrapper spends on its own account - a harness warm-up is charged to the budget
automatically rather than by bookkeeping that can drift out of step.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["default_targets", "default_checkpoints", "ecdf_area", "Recorder"]


def default_targets(n: int = 51, hi: float = 1e2, lo: float = 1e-8) -> torch.Tensor:
    return torch.from_numpy(np.logspace(np.log10(hi), np.log10(lo), n))


def default_checkpoints(budget: int, n: int = 31, start: int = 1) -> np.ndarray:
    """Log-spaced evaluation budgets at which the running best is recorded."""
    return np.unique(np.round(np.logspace(np.log10(max(start, 1)),
                                          np.log10(budget), n)).astype(np.int64))


def ecdf_area(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Area under the target-hitting ECDF.

    Parameters
    ----------
    values : (R, J) best untransformed objective value reached by each run at
        each of the J budget checkpoints.
    targets : (K,) precision targets.

    Returns
    -------
    (R,) scores in [0, 1].
    """
    v = values.unsqueeze(-1)                      # (R, J, 1)
    t = targets.to(values).view(1, 1, -1)         # (1, 1, K)
    return (v <= t).to(values.dtype).mean(dim=(1, 2))


class Recorder:
    """Tracks the best untransformed objective value against the budget.

    Lives on the problem, so every evaluation - whoever makes it - is counted
    once and exactly once.
    """

    def __init__(self, checkpoints, n_runs: int, device, dtype):
        self.checkpoints = np.asarray(checkpoints, dtype=np.int64)
        self.J = len(self.checkpoints)
        self.R = n_runs
        self.best = torch.full((n_runs, self.J), float("inf"),
                               device=device, dtype=dtype)
        self.running = torch.full((n_runs,), float("inf"), device=device, dtype=dtype)
        self.fe = 0
        self._next = 0

    def record(self, y_base: torch.Tensor) -> None:
        """Absorb one batch of untransformed objective values.

        ``y_base`` is (R, N) for a population step, or (N,) for a bookkeeping
        evaluation made outside the per-run structure (a harness warm-up), in
        which case the same points are charged to every run.
        """
        if y_base.dim() == 1:
            n = y_base.shape[0]
            m = y_base.min().expand(self.R)
        else:
            n = y_base.shape[-1]
            m = y_base.reshape(self.R, -1).min(dim=-1).values
        self.running = torch.minimum(self.running, m)
        self.fe += n
        while self._next < self.J and self.checkpoints[self._next] <= self.fe:
            self.best[:, self._next] = self.running
            self._next += 1

    def finalize(self) -> torch.Tensor:
        while self._next < self.J:
            self.best[:, self._next] = self.running
            self._next += 1
        return self.best

    @property
    def exhausted(self) -> bool:
        return self._next >= self.J
