"""Problem instances and the group actions used by the audit.

A *base instance* is a canonical function (see :mod:`invaudit.functions`) whose
optimum has been relocated to ``x_opt``.  The audit then presents the optimiser
with a *transformed* view of that instance,

    phi(x) = s * (Q @ x) + t          (search-space action)
    psi(y) = g(y)                     (objective-space action)

so that the presented objective is ``g(f_base(phi^{-1}(x)))``.  Both actions are
invertible, which is what lets us score every run in the *original* space: the
optimiser's incumbent is mapped back with ``phi^{-1}`` and re-evaluated under the
untransformed objective.  Comparing raw objective values across a monotone
re-parameterisation would measure the transformation rather than the algorithm.

Two experimental controls are baked in.

*Initialisation.*  The initial population is drawn once in base coordinates and
then mapped through ``phi``.  Sampling independently inside the presented box
would make the *initialisation* non-equivariant under rotation and would
contaminate the measured defect of the algorithm.  For this mapping to stay
feasible under rotation the initial sample is drawn from a ball rather than a
box.

*Boundary handling.*  Clipping to an axis-aligned box is itself not
rotation-equivariant, for every algorithm including fully invariant ones.  The
audit therefore supports an unbounded regime (the box defines initialisation
only) used for instrument validation, and a bounded regime that exposes the
extra violation induced by constraint handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import torch

from .functions import FunctionSpec, get_function

__all__ = [
    "Problem",
    "make_instance",
    "sample_in_ball",
    "random_orthogonal",
    "random_permutation_matrix",
    "stable_seed",
]


def stable_seed(*parts) -> int:
    """A seed that is identical in every interpreter session.

    Python's built-in ``hash`` is salted per process for strings, so seeds derived
    from it silently change between runs - instances would move and no result
    could be reproduced.  CRC32 of the textual key is stable everywhere.
    """
    import zlib
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return (zlib.crc32(key) % (2 ** 31 - 2)) + 1


# --------------------------------------------------------------------------
# sampling utilities
# --------------------------------------------------------------------------

def sample_in_ball(n, D, radius, generator, device, dtype):
    """Uniform sample inside the ball of given radius (rotation-safe)."""
    g = torch.randn(n, D, generator=generator, device=device, dtype=dtype)
    g = g / g.norm(dim=-1, keepdim=True)
    u = torch.rand(n, 1, generator=generator, device=device, dtype=dtype)
    return g * radius * u ** (1.0 / D)


def random_orthogonal(D, generator, device, dtype):
    """Haar-distributed orthogonal matrix."""
    A = torch.randn(D, D, generator=generator, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R)
    return Q * torch.sign(d).unsqueeze(0)


def random_permutation_matrix(D, generator, device, dtype):
    """Coordinate permutation, expressed as an orthogonal matrix."""
    perm = torch.randperm(D, generator=generator, device=device)
    P = torch.zeros(D, D, device=device, dtype=dtype)
    P[torch.arange(D, device=device), perm] = 1.0
    return P


# --------------------------------------------------------------------------
# problem
# --------------------------------------------------------------------------

@dataclass
class Problem:
    """A (possibly transformed) view of one benchmark instance."""

    spec: FunctionSpec
    D: int
    x_opt: torch.Tensor                      # optimum in base coordinates
    half_width: float = 5.0                  # canonical box is [-hw, hw]^D
    r_init: float = 4.0                      # initial population ball radius
    bounded: bool = False                    # clip to the presented box?

    # search-space action phi(x) = s * (Q @ x) + t
    Q: Optional[torch.Tensor] = None
    s: float = 1.0
    t: Optional[torch.Tensor] = None
    # objective-space action
    g: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    label: str = "identity"
    # performance bookkeeping (see metrics.Recorder); never shared between views
    recorder: Optional[object] = field(default=None, repr=False, compare=False)

    # ---------------- geometry ----------------

    @property
    def device(self):
        return self.x_opt.device

    @property
    def dtype(self):
        return self.x_opt.dtype

    @property
    def lb(self) -> torch.Tensor:
        base = torch.full((self.D,), -self.half_width, device=self.device, dtype=self.dtype)
        return self._map_box(base)

    @property
    def ub(self) -> torch.Tensor:
        base = torch.full((self.D,), self.half_width, device=self.device, dtype=self.dtype)
        return self._map_box(base)

    def _map_box(self, corner: torch.Tensor) -> torch.Tensor:
        # Q maps the symmetric box to a set that is no longer a box; by design the
        # optimum and the initial ball stay inside it, so the presented box is kept
        # as-is under rotation and only translation/scaling move it.
        out = self.s * corner
        if self.t is not None:
            out = out + self.t
        return out

    # ---------------- actions ----------------

    def to_base(self, X: torch.Tensor) -> torch.Tensor:
        """Map presented coordinates back to base coordinates (phi^{-1})."""
        Y = X
        if self.t is not None:
            Y = Y - self.t
        if self.s != 1.0:
            Y = Y / self.s
        if self.Q is not None:
            Y = Y @ self.Q          # Q^T y  ==  y @ Q  for orthogonal Q
        return Y

    def from_base(self, X: torch.Tensor) -> torch.Tensor:
        """Map base coordinates into presented coordinates (phi)."""
        Y = X
        if self.Q is not None:
            Y = Y @ self.Q.T
        if self.s != 1.0:
            Y = Y * self.s
        if self.t is not None:
            Y = Y + self.t
        return Y

    # ---------------- objectives ----------------

    def base_value(self, X: torch.Tensor) -> torch.Tensor:
        """Untransformed objective value of presented points.  Used for scoring."""
        return self.spec.fn(self.to_base(X) - self.x_opt)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Objective as seen by the optimiser.

        The untransformed value is recorded before the objective-space action is
        applied, so performance is always measured in the original objective.
        """
        y = self.base_value(X)
        if self.recorder is not None:
            self.recorder.record(y)
        if self.g is not None:
            y = self.g(y)
        return y

    def attach_recorder(self, recorder) -> "Problem":
        self.recorder = recorder
        return self

    # ---------------- initialisation ----------------

    def init_population(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """Initial population in *base* coordinates (identical across a CRN pair)."""
        return sample_in_ball(n, self.D, self.r_init, generator, self.device, self.dtype)

    def present(self, X_base: torch.Tensor) -> torch.Tensor:
        return self.from_base(X_base)

    # ---------------- transformed views ----------------

    def with_action(self, *, Q=None, s=None, t=None, g=None, label=None) -> "Problem":
        """Return a transformed view.  Actions replace rather than compose."""
        return replace(
            self,
            Q=self.Q if Q is None else Q,
            s=self.s if s is None else s,
            t=self.t if t is None else t,
            g=self.g if g is None else g,
            label=label or self.label,
            recorder=None,
        )

    def relocated(self, x_opt_new: torch.Tensor, label: str) -> "Problem":
        """A different instance of the same function (not a group action)."""
        return replace(self, x_opt=x_opt_new, label=label, recorder=None)

    def __repr__(self):  # pragma: no cover
        return f"Problem({self.spec.name}, D={self.D}, {self.label})"


def make_instance(fname: str, D: int, instance: int, *, device="cpu",
                  dtype=torch.float64, half_width=5.0, r_init=4.0,
                  bounded=False) -> Problem:
    """Deterministically build instance ``instance`` of function ``fname``.

    The optimum is drawn from a ball so that it stays feasible under every
    search-space action the audit applies.
    """
    spec = get_function(fname)
    gen = torch.Generator(device=device)
    gen.manual_seed(stable_seed("instance", fname, D, instance))
    x_opt = sample_in_ball(1, D, r_init, gen, device, dtype)[0]
    return Problem(spec=spec, D=D, x_opt=x_opt, half_width=half_width,
                   r_init=r_init, bounded=bounded)
