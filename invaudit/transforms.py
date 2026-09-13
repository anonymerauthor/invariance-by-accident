"""The symmetry catalogue.

Each entry is a *probe*: a named family of actions, a sampler that draws one
concrete action for a given seed, and a flag saying whether the action is a
genuine group action on the problem (relocation of the optimum is not - it
produces a different instance of the same function, and is reported separately).

Design notes:

* ``ambient_translation`` moves the function *and* its box.  It detects absolute
  coordinate bias (hard-coded constants, initialisation centred at the origin).
* ``optimum_relocation`` keeps the box and moves the optimum inside it.  It is
  the classical centre-bias probe and is *not* a group action.
* ``affine_fitness`` and ``monotone_fitness`` are deliberately separated.  Many
  adaptive DE variants weight their memory updates by ratios of fitness
  differences, which is invariant to ``a*y + b`` but not to a general monotone
  re-parameterisation.  Keeping the two probes apart isolates that distinction.
* ``coordinate_permutation`` is close to a positive control: most optimisers'
  performance is distributionally unchanged by it.  The exception is instructive
  - exponential crossover copies contiguous coordinate blocks, so it depends on
  which variables are neighbours, and the audit flagged it before the ground
  truth table did.

Magnitudes are chosen large enough to expose scale-dependent constants
(``1e3``) yet small enough that float64 keeps the 1e-8 target resolution used by
the performance measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch

from .problem import (Problem, random_orthogonal, random_permutation_matrix,
                      sample_in_ball)

__all__ = ["Probe", "PROBES", "get_probe", "probe_names"]


# --------------------------------------------------------------------------
# monotone objective maps
# --------------------------------------------------------------------------

def _g_pow3(y):
    return y ** 3


def _g_log1p(y):
    # strongly compresses the range; destroys the meaning of fitness differences
    return torch.log1p(y)


def _g_sqrt(y):
    return torch.sqrt(y + 1e-300)


class MonotoneSpline:
    """Random strictly increasing piecewise-linear map applied in log space.

    Acting on ``u = log1p(y)`` keeps the map well conditioned over the many
    orders of magnitude an optimisation trajectory covers, while the positive
    slopes guarantee that the ranking of any two objective values is preserved
    exactly.
    """

    def __init__(self, seed: int, n_knots: int = 12, u_max: float = 40.0,
                 dtype=torch.float64, device="cpu"):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        self.knots = torch.linspace(0.0, u_max, n_knots, dtype=dtype, device=device)
        slopes = torch.exp(2.0 * torch.randn(n_knots - 1, generator=gen,
                                             dtype=dtype, device=device))
        widths = self.knots[1:] - self.knots[:-1]
        heights = slopes * widths
        self.values = torch.cat([
            torch.zeros(1, dtype=dtype, device=device),
            torch.cumsum(heights, dim=0),
        ])
        self.slopes = slopes
        self.tail_slope = slopes[-1]

    def __call__(self, y):
        u = torch.log1p(y.clamp_min(0.0))
        idx = torch.searchsorted(self.knots, u.contiguous(), right=True) - 1
        idx = idx.clamp(0, self.knots.numel() - 2)
        base_u = self.knots[idx]
        base_v = self.values[idx]
        slope = self.slopes[idx]
        out = base_v + slope * (u - base_u)
        # linear extrapolation beyond the last knot keeps the map increasing
        over = u > self.knots[-1]
        if over.any():
            out = torch.where(over,
                              self.values[-1] + self.tail_slope * (u - self.knots[-1]),
                              out)
        return out


class AffineFitness:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def __call__(self, y):
        return self.a * y + self.b


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

@dataclass
class Probe:
    name: str
    apply: Callable[[Problem, int], Problem]
    group_action: bool
    space: str            # "search" or "objective"
    description: str


def _ambient_translation(p: Problem, seed: int) -> Problem:
    gen = torch.Generator(device=p.device)
    gen.manual_seed(seed)
    direction = torch.randn(p.D, generator=gen, device=p.device, dtype=p.dtype)
    direction = direction / direction.norm()
    t = direction * 1.0e3
    return p.with_action(t=t, label="ambient_translation")


def _ambient_translation_diag(p: Problem, seed: int) -> Problem:
    """Translation along the main diagonal, norm 1e3.

    Same group action as ``ambient_translation`` up to the choice of direction,
    which no translation-invariant algorithm can distinguish.  The diagonal keeps
    a box with equal bounds in every coordinate, so libraries that store bounds
    as scalars can be audited without being handed a vector box they were never
    written for.
    """
    t = torch.full((p.D,), 1.0e3 / p.D ** 0.5, device=p.device, dtype=p.dtype)
    return p.with_action(t=t, label="ambient_translation_diag")


def _rotation(p: Problem, seed: int) -> Problem:
    gen = torch.Generator(device=p.device)
    gen.manual_seed(seed)
    Q = random_orthogonal(p.D, gen, p.device, p.dtype)
    return p.with_action(Q=Q, label="rotation")


def _coordinate_permutation(p: Problem, seed: int) -> Problem:
    gen = torch.Generator(device=p.device)
    gen.manual_seed(seed)
    P = random_permutation_matrix(p.D, gen, p.device, p.dtype)
    return p.with_action(Q=P, label="coordinate_permutation")


def _scale_up(p: Problem, seed: int) -> Problem:
    return p.with_action(s=1.0e3, label="domain_scaling_up")


def _scale_down(p: Problem, seed: int) -> Problem:
    return p.with_action(s=1.0e-3, label="domain_scaling_down")


def _scale_dyadic_up(p: Problem, seed: int) -> Problem:
    """Dilation by 2**10.

    Multiplying by a power of two is exact in binary floating point, so an
    algorithm that is algebraically scale equivariant must read exactly zero
    here.  Comparing with ``domain_scaling_up`` (factor 1e3, which rounds)
    separates violations of logic from amplified rounding error.
    """
    return p.with_action(s=2.0 ** 10, label="domain_scaling_dyadic_up")


def _scale_dyadic_down(p: Problem, seed: int) -> Problem:
    return p.with_action(s=2.0 ** -10, label="domain_scaling_dyadic_down")


def _fitness_rescale(p: Problem, seed: int) -> Problem:
    return p.with_action(g=AffineFitness(1.0e3, 0.0), label="fitness_rescale")


def _fitness_shift(p: Problem, seed: int) -> Problem:
    # A pure additive shift is algebraically harmless to every difference-based
    # rule, so any reading here is finite precision: adding 1e6 to values of
    # order 1e-8 annihilates them.  Separating shift from rescaling is what lets
    # the audit attribute a violation to logic rather than to arithmetic.
    return p.with_action(g=AffineFitness(1.0, 1.0e6), label="fitness_shift")


def _monotone_pow3(p: Problem, seed: int) -> Problem:
    return p.with_action(g=_g_pow3, label="monotone_pow3")


def _monotone_log(p: Problem, seed: int) -> Problem:
    return p.with_action(g=_g_log1p, label="monotone_log")


def _monotone_spline(p: Problem, seed: int) -> Problem:
    g = MonotoneSpline(seed=seed, dtype=p.dtype, device=p.device)
    return p.with_action(g=g, label="monotone_spline")


def _optimum_relocation(p: Problem, seed: int) -> Problem:
    gen = torch.Generator(device=p.device)
    gen.manual_seed(seed)
    x_new = sample_in_ball(1, p.D, p.r_init, gen, p.device, p.dtype)[0]
    return p.relocated(x_new, label="optimum_relocation")


def _identity(p: Problem, seed: int) -> Problem:
    return p


def _identity_recomposed(p: Problem, seed: int) -> Problem:
    """Identity written as an explicit action.

    Exercises the same code path as a real probe (a Q, an s and a t are applied
    and inverted) so that any residual reading is attributable to the numerics of
    the transformation rather than to the algorithm.  This is the noise floor
    that every other probe is judged against.
    """
    eye = torch.eye(p.D, device=p.device, dtype=p.dtype)
    zero = torch.zeros(p.D, device=p.device, dtype=p.dtype)
    return p.with_action(Q=eye, s=1.0, t=zero, label="identity_recomposed")


_PROBES: List[Probe] = [
    Probe("identity", _identity, True, "none",
          "null probe; with common random numbers it must read exactly zero"),
    Probe("identity_recomposed", _identity_recomposed, True, "none",
          "identity applied through the full action machinery; numerical noise floor"),
    Probe("ambient_translation", _ambient_translation, True, "search",
          "function and box translated by 1e3 in a random direction"),
    Probe("ambient_translation_diag", _ambient_translation_diag, True, "search",
          "function and box translated by 1e3 along the diagonal (scalar bounds)"),
    Probe("rotation", _rotation, True, "search",
          "random Haar orthogonal change of coordinates"),
    Probe("coordinate_permutation", _coordinate_permutation, True, "search",
          "positive control: random coordinate permutation"),
    Probe("domain_scaling_up", _scale_up, True, "search",
          "search space dilated by 1e3"),
    Probe("domain_scaling_down", _scale_down, True, "search",
          "search space contracted by 1e-3"),
    Probe("domain_scaling_dyadic_up", _scale_dyadic_up, True, "search",
          "search space dilated by 2**10 (exact in binary floating point)"),
    Probe("domain_scaling_dyadic_down", _scale_dyadic_down, True, "search",
          "search space contracted by 2**-10 (exact in binary floating point)"),
    Probe("fitness_rescale", _fitness_rescale, True, "objective",
          "objective multiplied by 1e3"),
    Probe("fitness_shift", _fitness_shift, True, "objective",
          "objective shifted by +1e6; isolates finite-precision effects"),
    Probe("monotone_pow3", _monotone_pow3, True, "objective",
          "objective mapped by y^3"),
    Probe("monotone_log", _monotone_log, True, "objective",
          "objective mapped by log(1+y); annihilates fitness differences"),
    Probe("monotone_spline", _monotone_spline, True, "objective",
          "random strictly increasing spline in log space"),
    Probe("optimum_relocation", _optimum_relocation, False, "instance",
          "different optimum location; not a group action"),
]

# ---- dose-response probes: realistic unit and domain changes -----------------
#
# The main probes use magnitudes (1e3, 1e-3) chosen to expose a dependence
# unambiguously.  These levels ask the practical question instead: does the
# dependence already bite for the changes of domain a user actually makes -
# a box [0, 10] instead of [-5, 5], the CEC box [-100, 100], a unit box?


def _make_translate(c):
    def apply(p: Problem, seed: int) -> Problem:
        t = torch.full((p.D,), float(c), device=p.device, dtype=p.dtype)
        return p.with_action(t=t, label=f"translate_{c:g}")
    return apply


def _make_scale(s):
    def apply(p: Problem, seed: int) -> Problem:
        return p.with_action(s=float(s), label=f"scale_{s:g}")
    return apply


DOSE_TRANSLATIONS = [5, 50, 500]          # box [c-5, c+5] per coordinate
DOSE_SCALES = [0.1, 20, 100]              # box [-5s, 5s]; 20 is the CEC box
for _c in DOSE_TRANSLATIONS:
    _PROBES.append(Probe(f"translate_{_c:g}", _make_translate(_c), True, "search",
                         f"box moved to [{_c - 5:g}, {_c + 5:g}] per coordinate"))
for _s in DOSE_SCALES:
    _PROBES.append(Probe(f"scale_{_s:g}", _make_scale(_s), True, "search",
                         f"box scaled to [{-5 * _s:g}, {5 * _s:g}]"))


PROBES = {p.name: p for p in _PROBES}


def probe_names(include_instance=True):
    return [p.name for p in _PROBES if include_instance or p.group_action]


def get_probe(name: str) -> Probe:
    if name not in PROBES:
        raise KeyError(f"unknown probe {name!r}; available: {list(PROBES)}")
    return PROBES[name]
