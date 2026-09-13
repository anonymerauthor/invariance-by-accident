"""MetaBox-v2 adapter: audit the pretrained MetaBBO agents it ships.

MetaBox-v2 (the benchmark platform of Gong and colleagues) distributes sixteen
agents trained on the ten-dimensional COCO-BBOB suite.  They span every current
design paradigm - reinforcement-learned control of DE and PSO, learned update
rules, and symbolic rule discovery - and none of them was trained or evaluated
with invariance in mind, which makes the set the natural subject of the audit.

Three decisions keep the comparison faithful to how the agents were trained and
tested.

*The untransformed view is the agent's own problem.*  Training instances carry
a random additive bias and a boundary penalty.  The adapter reproduces both: the
penalty is part of the objective and therefore moves with every search-space
action, and the bias is added *before* the objective-space action, so that the
identity probe hands an agent exactly the objective it was trained on while
performance is still recorded on the canonical, zero-optimum objective.

*The known optimum is exposed, as MetaBox does.*  Most agents build their state
from ``problem.eval(x) - problem.optimum``.  That is privileged information and
the audit keeps the convention rather than silently changing the method; the
exposed optimum is the value the *presented* objective takes at the optimum, so
it stays consistent under every probe.  A consequence worth stating plainly: an
additive fitness shift is invisible to such agents by construction, which is why
rescaling and monotone probes, not the shift probe, are the informative ones.

*Bounds are scalars whenever possible.*  MetaBox treats ``lb``/``ub`` as scalars.
Translations are therefore applied along the diagonal (see
``ambient_translation_diag``) so that a moved box is still a scalar box; a
vector box is passed only when unavoidable.

Instances are drawn with the optimum inside a ball of radius 4 rather than in
MetaBox's cube ``[-4, 4]^D``.  The ball is a subset of the training support, and
it is the largest region that keeps the optimum feasible under rotation.

Evaluations are counted as they actually happen, not as a method's own counter
reports them.  The two disagree for DEDQN, whose landscape-feature routine
charges ``rwsteps`` evaluations while evaluating the population instead; it
terminates after roughly half of its nominal budget.  The audit measures the
method as shipped and records that fact rather than correcting it.
"""

from __future__ import annotations

import importlib
import os
import pickle
from typing import Optional

import numpy as np
import torch

from ..functions import FunctionSpec
from ..problem import Problem, random_orthogonal, sample_in_ball

__all__ = ["METABOX_AGENTS", "MetaBoxView", "bbob_instance", "make_config",
           "load_agent", "run_agent", "UnsupportedInstance"]

METABOX_AGENTS = ["B2OPT", "DEDDQN", "DEDQN", "GLEET", "GLHF", "LDE", "LES",
                  "NRLPSO", "QLPSO", "RLDAS", "RLDEAFL", "RLEPSO", "RLHPSDE",
                  "RLPSO", "RNNOPT", "SYMBOL"]

# design paradigm, used to group results
PARADIGM = {
    "DEDDQN": "RL-controlled DE", "DEDQN": "RL-controlled DE", "LDE": "RL-controlled DE",
    "RLHPSDE": "RL-controlled DE", "RLDEAFL": "RL-controlled DE",
    "RLDAS": "RL algorithm selection",
    "GLEET": "RL-controlled PSO", "NRLPSO": "RL-controlled PSO", "QLPSO": "RL-controlled PSO",
    "RLEPSO": "RL-controlled PSO", "RLPSO": "RL-controlled PSO",
    "B2OPT": "learned update rule", "GLHF": "learned update rule",
    "LES": "learned update rule", "RNNOPT": "learned update rule",
    "SYMBOL": "symbolic rule discovery",
}


def _mb():
    return importlib.import_module("metaevobox")


def make_config(maxFEs: int = 20000, dim: int = 10, device: str = "cpu"):
    from metaevobox.config import get_config
    cfg = get_config(["--train_problem", "bbob-10D", "--test_problem", "bbob-10D",
                      "--test_difficulty", "difficult", "--device", device])
    cfg.dim = dim
    cfg.maxFEs = maxFEs
    cfg.n_logpoint = 50
    cfg.log_interval = max(1, maxFEs // cfg.n_logpoint)
    cfg.full_meta_data = False
    return cfg


def load_agent(name: str, difficulty: str = "difficult"):
    root = os.path.dirname(_mb().__file__)
    path = os.path.join(root, "model", "bbob-10D", difficulty, f"{name}.pkl")
    with open(path, "rb") as f:
        agent = pickle.load(f)
    agent.config.full_meta_data = False
    opt_cls = getattr(importlib.import_module("metaevobox.environment.optimizer"),
                      f"{name}_Optimizer")
    return agent, opt_cls


# --------------------------------------------------------------------------
# problems
# --------------------------------------------------------------------------

class UnsupportedInstance(ValueError):
    """The instance cannot be audited without confounding feasibility."""


def bbob_instance(fid: int, instance: int, dim: int = 10, r_opt: float = 4.0,
                  half_width: float = 5.0, require_ball: bool = True):
    """A MetaBox BBOB instance in canonical form, plus its training-style bias.

    Returns ``(problem, bias)``.  ``problem`` evaluates
    ``F_fid(x) - F_fid(x_opt)`` (boundary penalty included, bias excluded) so that
    its optimum is exactly zero, as the audit's performance measure requires.

    Several MetaBox constructors overwrite the shift they are given - F5 moves
    the optimum to a corner of the box, F20 and F24 pick their own sign pattern,
    the Gallagher functions rescale it.  The optimum is therefore read back from
    the constructed object, never assumed, and two checks are enforced:

    * it is the global minimum, verified by dense sampling of the box and of
      shells around it;
    * with ``require_ball`` it lies within ``r_opt`` of the centre, which is what
      keeps it feasible under every search-space probe.  Functions that fail
      (F5 always does: its optimum sits on the boundary by definition) raise
      :class:`UnsupportedInstance` and are excluded from that task family.
    """
    from metaevobox.environment.problem.SOO.COCO_BBOB import bbob_numpy as B
    from metaevobox.environment.problem.SOO.COCO_BBOB.bbob_numpy import rotate_gen

    seed = 104729 * fid + 7919 * instance + 17
    gen = torch.Generator().manual_seed(seed)
    shift = sample_in_ball(1, dim, r_opt, gen, "cpu", torch.float64)[0]
    np.random.seed(seed % (2 ** 32 - 1))
    H = rotate_gen(dim)
    bias = float(np.random.randint(1, 26) * 100)

    mb = getattr(B, f"F{fid}")(dim=dim, shift=shift.numpy(), rotate=H, bias=0.0,
                               lb=-half_width, ub=half_width)
    x_opt_np = np.asarray(mb.get_optimal(), dtype=np.float64).reshape(-1)
    f_star = float(mb.func(x_opt_np.reshape(1, -1))[0])

    if require_ball and np.linalg.norm(x_opt_np) > r_opt + 1e-9:
        raise UnsupportedInstance(
            f"F{fid}: optimum at radius {np.linalg.norm(x_opt_np):.2f} > {r_opt}; "
            f"not feasible under every search-space probe")
    rng = np.random.RandomState(seed + 1)
    probes = [rng.uniform(-half_width, half_width, size=(20000, dim))]
    for scale in (1.0, 1e-1, 1e-2, 1e-4):
        probes.append(x_opt_np + scale * rng.standard_normal((500, dim)))
    y_min = min(float(np.min(mb.func(P) - f_star)) for P in probes)
    if y_min < -1e-8 * max(1.0, abs(f_star)):
        raise UnsupportedInstance(
            f"F{fid}: reported optimum is not the global minimum "
            f"(found a point {y_min:.3e} below it)")

    def fn(Z: torch.Tensor) -> torch.Tensor:
        shape = Z.shape[:-1]
        Xb = Z.reshape(-1, dim).cpu().numpy() + x_opt_np
        y = mb.func(Xb) - f_star
        return torch.as_tensor(np.asarray(y, dtype=np.float64),
                               dtype=Z.dtype, device=Z.device).reshape(shape)

    spec = FunctionSpec(f"bbob_f{fid}", fn, separable=fid in (1, 2, 3, 4, 5),
                        unimodal=fid in (1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
    p = Problem(spec=spec, D=dim, x_opt=torch.from_numpy(x_opt_np.copy()),
                half_width=half_width, r_init=r_opt, bounded=False)
    return p, bias


class MetaBoxView:
    """Presents a (probed) invaudit problem through MetaBox's problem interface."""

    def __init__(self, problem: Problem, bias: float = 0.0, label: str = "",
                 frame: Optional[float] = None):
        """``frame``: if given, the invariance harness's frame layer is active and
        the agent always sees the box ``[-frame, frame]^D`` - its training box -
        whatever box the problem is presented in.  On an untransformed problem
        whose box already is the training box the map is the identity, so the
        layer changes nothing in distribution."""
        from metaevobox.environment.problem.basic_problem import Basic_Problem
        self._Basic = Basic_Problem
        self.p = problem
        self.bias = float(bias)
        self.dim = problem.D
        lb, ub = problem.lb, problem.ub
        self.frame = frame
        if frame is not None:
            self._c = (0.5 * (lb + ub)).numpy()
            self._w = (0.5 * (ub - lb)).numpy() / float(frame)
            self.lb, self.ub = -float(frame), float(frame)
        else:
            self._c, self._w = None, None
            self.lb = float(lb[0]) if bool(torch.all(lb == lb[0])) else lb.numpy()
            self.ub = float(ub[0]) if bool(torch.all(ub == ub[0])) else ub.numpy()
        y0 = torch.tensor([self.bias], dtype=torch.float64)
        self.optimum = float(problem.g(y0)[0]) if problem.g is not None else self.bias
        self.label = label or f"{problem.spec.name}:{problem.label}"
        self.T1 = 0.0

    def __deepcopy__(self, memo):
        # Several optimisers (SYMBOL, and the MetaBox tester itself) deep-copy the
        # problem and evaluate the copy.  Evaluations on a copy are still
        # evaluations spent by the method, so every copy must write to the same
        # recorder; the underlying problem is shared rather than duplicated.
        clone = MetaBoxView.__new__(MetaBoxView)
        clone.__dict__.update(self.__dict__)
        memo[id(self)] = clone
        return clone

    # MetaBox interface ----------------------------------------------------

    def reset(self):
        self.T1 = 0.0

    def eval(self, x):
        return self._Basic.eval(self, x)

    def func(self, x):
        is_torch = isinstance(x, torch.Tensor)
        X = torch.as_tensor(np.asarray(x.detach().cpu() if is_torch else x),
                            dtype=torch.float64)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if self.frame is not None:
            X = torch.from_numpy(self._c) + torch.from_numpy(self._w) * X
        y = self.p.base_value(X.unsqueeze(0))            # (1, n) canonical
        if self.p.recorder is not None:
            self.p.recorder.record(y)
        y = y + self.bias
        if self.p.g is not None:
            y = self.p.g(y)
        y = y[0]
        return y if is_torch else y.numpy()

    def __str__(self):
        return self.label


def run_agent(name: str, problem: Problem, bias: float, seed: int,
              maxFEs: int = 20000, device: str = "cpu", frame: Optional[float] = None):
    """One test episode of a pretrained agent on a probed problem."""
    from metaevobox.environment.basic_environment import PBO_Env
    agent, opt_cls = load_agent(name)
    cfg = make_config(maxFEs=maxFEs, dim=problem.D, device=device)
    view = MetaBoxView(problem, bias=bias, frame=frame)
    env = PBO_Env(view, opt_cls(cfg))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)
    return agent.rollout_episode(env, seed, {})
