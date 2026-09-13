"""Canonical benchmark functions.

Every function in this module is defined in *canonical form*:

    f(0) = 0   and   f(z) >= 0  for all z,

i.e. the optimum sits at the origin with objective value zero.  Instance
generation (optimum relocation, optional instance rotation) and the audit
transformations are applied on top of this canonical form, never inside it.
Keeping the base functions canonical is what makes the group actions in
:mod:`invaudit.transforms` well defined and exactly invertible.

All functions are batched: they accept ``Z`` of shape ``(..., D)`` and return
``(...)``.  Computations run in the dtype/device of the input; the audit uses
float64 throughout (see the note on finite precision in :mod:`invaudit.audit`).
"""

from __future__ import annotations

import math
import torch

__all__ = ["FUNCTIONS", "get_function", "FunctionSpec"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _condition(Z: torch.Tensor, alpha: float) -> torch.Tensor:
    """Coordinate-wise conditioning with condition number ``alpha`` (BBOB style)."""
    D = Z.shape[-1]
    if D == 1:
        expo = torch.zeros(1, dtype=Z.dtype, device=Z.device)
    else:
        i = torch.arange(D, dtype=Z.dtype, device=Z.device)
        expo = 0.5 * i / (D - 1)
    return Z * (alpha ** expo)


# --------------------------------------------------------------------------
# separable, unimodal
# --------------------------------------------------------------------------

def sphere(Z):
    return (Z ** 2).sum(-1)


def ellipsoid(Z):
    """Separable, ill-conditioned (cond 1e6)."""
    D = Z.shape[-1]
    i = torch.arange(D, dtype=Z.dtype, device=Z.device)
    w = 1e6 ** (i / max(D - 1, 1))
    return (w * Z ** 2).sum(-1)


def discus(Z):
    return 1e6 * Z[..., 0] ** 2 + (Z[..., 1:] ** 2).sum(-1)


def bent_cigar(Z):
    return Z[..., 0] ** 2 + 1e6 * (Z[..., 1:] ** 2).sum(-1)


def different_powers(Z):
    D = Z.shape[-1]
    i = torch.arange(D, dtype=Z.dtype, device=Z.device)
    expo = 2 + 4 * i / max(D - 1, 1)
    return (Z.abs() ** expo).sum(-1)


# --------------------------------------------------------------------------
# non-separable, unimodal
# --------------------------------------------------------------------------

def schwefel12(Z):
    """Sum of squared partial sums.  Non-separable in canonical orientation."""
    return (torch.cumsum(Z, dim=-1) ** 2).sum(-1)


def rosenbrock(Z):
    """Shifted so that the optimum is at the origin (classic optimum is all-ones)."""
    X = Z + 1.0
    a = X[..., :-1]
    b = X[..., 1:]
    return (100.0 * (a ** 2 - b) ** 2 + (a - 1.0) ** 2).sum(-1)


def sharp_ridge(Z):
    return Z[..., 0] ** 2 + 100.0 * torch.sqrt((Z[..., 1:] ** 2).sum(-1) + 1e-300)


# --------------------------------------------------------------------------
# multimodal
# --------------------------------------------------------------------------

def rastrigin(Z):
    return 10.0 * Z.shape[-1] + (Z ** 2 - 10.0 * torch.cos(2 * math.pi * Z)).sum(-1)


def ackley(Z):
    D = Z.shape[-1]
    s1 = (Z ** 2).sum(-1) / D
    s2 = torch.cos(2 * math.pi * Z).sum(-1) / D
    return -20.0 * torch.exp(-0.2 * torch.sqrt(s1)) - torch.exp(s2) + 20.0 + math.e


def griewank(Z):
    D = Z.shape[-1]
    i = torch.arange(1, D + 1, dtype=Z.dtype, device=Z.device)
    return (Z ** 2).sum(-1) / 4000.0 - torch.prod(torch.cos(Z / torch.sqrt(i)), dim=-1) + 1.0


def levy(Z):
    W = 1.0 + Z / 4.0
    t1 = torch.sin(math.pi * W[..., 0]) ** 2
    w = W[..., :-1]
    t2 = ((w - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * w + 1.0) ** 2)).sum(-1)
    wl = W[..., -1]
    t3 = (wl - 1.0) ** 2 * (1.0 + torch.sin(2 * math.pi * wl) ** 2)
    return t1 + t2 + t3


def schaffer_f7(Z):
    """Non-separable multimodal with a moderate global structure."""
    s = torch.sqrt(Z[..., :-1] ** 2 + Z[..., 1:] ** 2)
    v = torch.sqrt(s + 1e-300) * (1.0 + torch.sin(50.0 * s ** 0.2) ** 2)
    return (v.mean(-1)) ** 2


def hgbat(Z):
    """Highly non-separable; optimum shifted to the origin."""
    X = Z - 1.0
    s1 = (X ** 2).sum(-1)
    s2 = X.sum(-1)
    D = Z.shape[-1]
    return torch.sqrt((s1 ** 2 - s2 ** 2).abs() + 1e-300) + (0.5 * s1 + s2) / D + 0.5


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

class FunctionSpec:
    """Metadata that the audit needs about a base function."""

    def __init__(self, name, fn, separable, unimodal, ill_conditioned=False):
        self.name = name
        self.fn = fn
        self.separable = separable
        self.unimodal = unimodal
        self.ill_conditioned = ill_conditioned

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"FunctionSpec({self.name})"


_SPECS = [
    FunctionSpec("sphere", sphere, separable=True, unimodal=True),
    FunctionSpec("ellipsoid", ellipsoid, separable=True, unimodal=True, ill_conditioned=True),
    FunctionSpec("discus", discus, separable=True, unimodal=True, ill_conditioned=True),
    FunctionSpec("bent_cigar", bent_cigar, separable=True, unimodal=True, ill_conditioned=True),
    FunctionSpec("different_powers", different_powers, separable=True, unimodal=True),
    FunctionSpec("schwefel12", schwefel12, separable=False, unimodal=True),
    FunctionSpec("rosenbrock", rosenbrock, separable=False, unimodal=False),
    FunctionSpec("sharp_ridge", sharp_ridge, separable=False, unimodal=True),
    FunctionSpec("rastrigin", rastrigin, separable=True, unimodal=False),
    FunctionSpec("ackley", ackley, separable=False, unimodal=False),
    FunctionSpec("griewank", griewank, separable=False, unimodal=False),
    FunctionSpec("levy", levy, separable=False, unimodal=False),
    FunctionSpec("schaffer_f7", schaffer_f7, separable=False, unimodal=False),
    FunctionSpec("hgbat", hgbat, separable=False, unimodal=False),
]

FUNCTIONS = {s.name: s for s in _SPECS}


def get_function(name: str) -> FunctionSpec:
    if name not in FUNCTIONS:
        raise KeyError(f"unknown function {name!r}; available: {sorted(FUNCTIONS)}")
    return FUNCTIONS[name]
