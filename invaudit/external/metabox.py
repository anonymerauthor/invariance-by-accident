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
                 frame: Optional[float] = None, objective_layer: bool = False,
                 max_fes: int = 20000, exposure: Optional[dict] = None):
        """``frame``: if given, the invariance harness's frame layer is active and
        the agent always sees the box ``[-frame, frame]^D`` - its training box -
        whatever box the problem is presented in.  On an untransformed problem
        whose box already is the training box the map is the identity, so the
        layer changes nothing in distribution.

        ``objective_layer``: the harness's objective layer.  The agent sees the
        persistent order code of every value it evaluates instead of the value,
        and the exposed optimum is a constant below every code the run can issue
        (see :data:`OPT_CODE_OFFSET`), so ``eval(x) - optimum`` stays positive.
        Performance is still recorded on the canonical objective.

        ``exposure``: if a dict is given, every evaluation is counted in it
        (instrumentation only; no effect on the run).  Keys: ``n`` evaluations,
        ``out_ball`` points outside the ball of radius ``half_width`` in base
        coordinates (the largest region on which a rotated presentation and the
        original one coincide), ``out_box`` points outside the base box (where
        the rotated presentation shows the function, or its boundary penalty, on
        a region the original presentation never offers), and ``on_face`` points
        on the faces of the presented box (clipped or boundary-repaired points).
        Copies of the view share the dict."""
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
        self.objective_layer = objective_layer
        self.max_fes = int(max_fes)
        self._emb = None
        self.exposure = exposure
        if exposure is not None:
            for k in ("n", "out_ball", "out_box", "on_face"):
                exposure.setdefault(k, 0)
        if objective_layer:
            self.optimum = optimum_code(self.max_fes)
            self._new_embedding()

    def _new_embedding(self):
        from ..harness import _OrderEmbeddingState
        self._emb = _OrderEmbeddingState(1, "cpu", torch.float64)

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
        if self.objective_layer:
            self._new_embedding()

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
        if self.exposure is not None:
            self._count_exposure(X)
        y = self.p.base_value(X.unsqueeze(0))            # (1, n) canonical
        if self.p.recorder is not None:
            self.p.recorder.record(y)
        y = y + self.bias
        if self.p.g is not None:
            y = self.p.g(y)
        if self.objective_layer:
            y = self._emb.push_and_rank(y)           # (1, n) codes
        y = y[0]
        return y if is_torch else y.numpy()

    def _count_exposure(self, X: torch.Tensor):
        hw = float(self.p.half_width)
        Xb = self.p.to_base(X)                       # base coordinates, box centred at 0
        lb, ub = self.p.lb, self.p.ub                # presented box
        tol = 1e-9 * float(torch.max(ub - lb))
        e = self.exposure
        e["n"] += int(X.shape[0])
        e["out_ball"] += int((Xb.norm(dim=1) > hw * (1 + 1e-12)).sum())
        e["out_box"] += int((Xb.abs().amax(dim=1) > hw * (1 + 1e-12)).sum())
        e["on_face"] += int(((X <= lb + tol) | (X >= ub - tol)).any(dim=1).sum())

    def __str__(self):
        return self.label


def optimum_code(max_fes: int) -> float:
    """Exposed optimum under the objective layer.

    The order embedding gives the first batch codes 0, 1, ... and every later
    record-breaking value a code at most one below the current minimum, so no
    run of at most ``4 * max_fes`` evaluations (SYMBOL, the largest spender,
    performs about 1.9 * max_fes) issues a code below ``-4 * max_fes``.  The
    optimum is placed one further below, which keeps ``eval(x) - optimum >= 1``
    for agents that build costs from it."""
    return -4.0 * float(max_fes) - 1.0


class ObjectiveLayerProblem:
    """Objective layer around a MetaBox training problem (for retraining, E7).

    Delegates everything to the wrapped problem except ``func``/``eval``, which
    return persistent order codes of the wrapped problem's values, and
    ``optimum``, which is :func:`optimum_code`.  ``reset`` starts a new
    embedding, so every episode is coded from scratch, exactly as at test time.

    ``raw_best`` tracks the best raw value seen in the episode and ``last_raw``
    the raw values of the last evaluation (both minus the raw optimum when known).  It is never shown to the optimizer; it exists only so
    that a training reward can be computed from raw values (E7, condition
    ``layer_rawreward``).
    """

    def __init__(self, inner, max_fes: int = 20000):
        from metaevobox.environment.problem.basic_problem import Basic_Problem
        self._Basic = Basic_Problem
        self.inner = inner
        self.max_fes = int(max_fes)
        self.optimum = optimum_code(self.max_fes)
        self.T1 = 0.0
        self._emb = None
        self.reset()

    def __getattr__(self, name):          # dim, lb, ub, ... of the wrapped problem
        inner = self.__dict__.get("inner")
        if inner is None or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(inner, name)

    def reset(self):
        from ..harness import _OrderEmbeddingState
        self.T1 = 0.0
        if hasattr(self.inner, "reset"):
            self.inner.reset()
        self._emb = _OrderEmbeddingState(1, "cpu", torch.float64)
        self.raw_best = float("inf")
        self.last_raw = None

    def func(self, x):
        y = np.asarray(self.inner.func(x), dtype=np.float64).reshape(-1)
        opt = getattr(self.inner, "optimum", None)
        raw = y - (float(opt) if opt is not None else 0.0)
        self.last_raw = raw
        self.raw_best = min(self.raw_best, float(np.min(raw)))
        c = self._emb.push_and_rank(torch.from_numpy(y.copy())[None, :])[0]
        return c.numpy()

    def eval(self, x):
        return self._Basic.eval(self, x)

    def __str__(self):
        return f"objlayer({self.inner})"


def run_agent(name: str, problem: Problem, bias: float, seed: int,
              maxFEs: int = 20000, device: str = "cpu", frame: Optional[float] = None,
              objective_layer: bool = False, agent_path: Optional[str] = None,
              exposure: Optional[dict] = None):
    """One test episode of a pretrained (or retrained: ``agent_path``) agent.
    ``exposure``: optional dict filled with evaluation counts (see MetaBoxView)."""
    from metaevobox.environment.basic_environment import PBO_Env
    agent, opt_cls = load_agent(name)
    if agent_path is not None:
        with open(agent_path, "rb") as f:
            agent = pickle.load(f)
        agent.config.full_meta_data = False
    cfg = make_config(maxFEs=maxFEs, dim=problem.D, device=device)
    view = MetaBoxView(problem, bias=bias, frame=frame, objective_layer=objective_layer,
                       max_fes=maxFEs, exposure=exposure)
    env = PBO_Env(view, opt_cls(cfg))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)
    return agent.rollout_episode(env, seed, {})
