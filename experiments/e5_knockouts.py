"""E5b - knock-out attribution of the symmetry violations (reviewer 1, comment 1).

Each significant violation of the main audit is traced to a component of the
agent and confirmed by changing exactly that component.  Two kinds of
intervention:

*Shell knock-outs* rewrite one hand-written constant or expression of the
optimizer so that it is stated relative to the presented box.  Every rewrite
is chosen to evaluate to exactly the original value on the training box
[-5, 5]^D, so it makes two falsifiable predictions: the untransformed runs are
bit-identical to the agent as distributed, and the violation it targets
vanishes.  Rewrites are applied as text substitutions on the method source
(``SourcePatch``); the patched source is hashed and the hash is stored with
every result row.

*Policy swaps* keep the code and replace the learned part: re-initialized
weights of the same architecture (``untrained:k``, k = seed of the draw) or a
constant action for reinforcement-learned controllers (``fixed:k``).  A
violation that persists with untrained weights belongs to the architecture or
its inputs; one that disappears with a fixed action enters through the policy.

Runs inside the MetaBox environment; same task protocol, seeds, instances and
scoring as experiment 2 (``exp2_metabox_audit.run_task``), so the ``none``
rows must reproduce the E1 audit exactly.

Usage
-----
    python experiments/e5_knockouts.py --workers 30 --out results_rev/e5_knockouts.csv
    python experiments/e5_knockouts.py --limit 40 ...      # smoke test
    python experiments/e5_knockouts.py --list              # print the patch table
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import itertools
import os
import sys
import textwrap
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

OPT = "metaevobox.environment.optimizer."


# --------------------------------------------------------------------------
# source patches
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SourcePatch:
    module: str
    cls: str
    method: str
    old: str
    new: str          # may span several lines; continuation lines get the indent of `old`
    count: int = 1    # exact number of occurrences that must be replaced


def _attr_name(cls: str, method: str) -> str:
    # private methods are name-mangled
    if method.startswith("__") and not method.endswith("__"):
        return f"_{cls.lstrip('_')}{method}"
    return method


_ORIGINAL_SRC: Dict[Tuple[str, str, str], str] = {}


def apply_patches(patches: List[SourcePatch]) -> str:
    """Apply a set of patches in this process; return the sha256 of the patched sources.

    Patches on the same method are applied in order to that method's *original*
    source and compiled once, so a set may contain several edits of one method."""
    by_method: Dict[Tuple[str, str, str], List[SourcePatch]] = {}
    for p in patches:
        by_method.setdefault((p.module, p.cls, p.method), []).append(p)
    digests = []
    for (module, cls, method), ps in by_method.items():
        mod = importlib.import_module(module)
        C = getattr(mod, cls)
        attr = _attr_name(cls, method)
        key = (module, cls, method)
        if key not in _ORIGINAL_SRC:
            _ORIGINAL_SRC[key] = textwrap.dedent(inspect.getsource(C.__dict__[attr]))
        src = _ORIGINAL_SRC[key]
        if "super()" in src:
            raise RuntimeError(f"{cls}.{method}: zero-argument super() cannot be re-compiled")
        for p in ps:
            if src.count(p.old) != p.count:
                raise RuntimeError(f"{cls}.{method}: patch text found {src.count(p.old)} "
                                   f"times, expected {p.count}")
            line = next(l for l in src.split("\n") if p.old in l)
            indent = line[:len(line) - len(line.lstrip())]
            src = src.replace(p.old, p.new.replace("\n", "\n" + indent))
        # compile inside a class body of the same name so that name mangling of
        # self.__x attributes matches the original class
        wrapped = f"class {cls}:\n" + textwrap.indent(src, "    ")
        ns: Dict[str, object] = {}
        exec(compile(wrapped, f"<knockout {cls}.{method}>", "exec"), mod.__dict__, ns)
        setattr(C, attr, ns[cls].__dict__[attr])
        digests.append(hashlib.sha256(src.encode()).hexdigest()[:16])
    return ",".join(digests)


_C = "(problem.ub + problem.lb) / 2"          # centre of the presented box
_W = "(problem.ub - problem.lb) / 2"          # half-width of the presented box

SHELL_KNOCKOUTS: Dict[str, List[SourcePatch]] = {
    # LES: initial step size proportional to the absolute upper bound
    "les_sigma": [SourcePatch(
        OPT + "les_optimizer", "LES_Optimizer", "init_population",
        "sigma = np.ones(problem.dim)*self.ub*self.sigma_ratio",
        f"sigma = np.ones(problem.dim)*({_W})*self.sigma_ratio")],
    # LES: mean update mu' = (1-a) mu + a sum W (x - mu) shrinks the mean towards the
    # origin; rewritten relative to the box centre (identical when the centre is 0)
    "les_mu": [SourcePatch(
        OPT + "les_optimizer", "LES_Optimizer", "update",
        "alpha_mu * np.sum((self.evolution_info['parents'] - self.evolution_info['mu'])*W[:,None],axis=0)",
        "alpha_mu * (np.sum((self.evolution_info['parents'] - self.evolution_info['mu'])*W[:,None],axis=0)"
        f" + {_C})")],
    # SYMBOL: student population drawn in [-ub, ub] instead of [lb, ub]
    "symbol_init": [SourcePatch(
        OPT + "symbol_optimizer", "Population", "reset",
        "rand_pos = self.rng.uniform(low = -self.max_x, high = self.max_x,",
        "rand_pos = self.rng.uniform(low = self.min_x, high = self.max_x,")],
    # RLDAS: hard-coded [-100, 100] reinitialization range -> centre +- 20 half-widths
    "rldas_bound": [SourcePatch(
        OPT + "rldas_optimizer", "NL_SHADE_RSP", "step",
        "us = us * ((-100 <= us) * (us <= 100)) + ((us > 100) + (us < -100)) * (self.rng.rand(NP, dim) * 200 - 100)",
        f"_c = {_C}; _w = {_W}; _lo = _c - 20 * _w; _hi = _c + 20 * _w\n"
        "us = us * ((_lo <= us) * (us <= _hi)) + ((us > _hi) + (us < _lo)) * (self.rng.rand(NP, dim) * (_hi - _lo) + _lo)")],
    # RLDAS: random walk in the absolute unit cube -> same walk relative to the box
    "rldas_rw": [SourcePatch(
        OPT + "rldas_optimizer", "Population", "get_feature",
        "random_walk_samples = rw_sampling(self.group, self.rng)",
        f"random_walk_samples = {_C} + ({_W}) / 5 * rw_sampling(self.group, self.rng)")],
    # RLDAS: dispersion and FDC features assume the [-100, 100] / [-5, 5] box
    "rldas_feat": [
        SourcePatch(OPT + "rldas_optimizer", "Population", "get_feature",
                    "disp, disp_ratio = dispersion(self.group, self.cost)",
                    f"disp, disp_ratio = dispersion((self.group - {_C}) * (5 / ({_W})), self.cost)"),
        SourcePatch(OPT + "rldas_optimizer", "Population", "get_feature",
                    "fdc = cal_fdc(self.group / 100,",
                    f"fdc = cal_fdc((self.group - {_C}) * (5 / ({_W})) / 100,")],
    # RLDAS/MadDE: mutants outside the hard-coded [-100, 100] are repaired towards +-100
    "rldas_madde": [
        SourcePatch(OPT + "rldas_optimizer", "MadDE", "step",
                    "v[v < -100] = (population.group[v < -100] - 100) / 2",
                    f"_c = {_C}; _w = {_W}; _lo = _c - 20 * _w; _hi = _c + 20 * _w\n"
                    "v[v < _lo] = (population.group[v < _lo] + _lo) / 2"),
        SourcePatch(OPT + "rldas_optimizer", "MadDE", "step",
                    "v[v > 100] = (population.group[v > 100] + 100) / 2",
                    "v[v > _hi] = (population.group[v > _hi] + _hi) / 2")],
    # RLHPSDE: the DFDC feature raises when rounding pushes r outside [-1, 1]
    "rlhpsde_dfdc": [SourcePatch(
        OPT + "rlhpsde_optimizer", "RLHPSDE_Optimizer", "__DFDC",
        'raise ValueError(f"DFDC error: {r}, {cost.std()}, {dist.std()}")',
        "return bool(r > 0.15)")],
}
SHELL_KNOCKOUTS["rldas_all"] = (SHELL_KNOCKOUTS["rldas_bound"] + SHELL_KNOCKOUTS["rldas_rw"]
                                + SHELL_KNOCKOUTS["rldas_feat"] + SHELL_KNOCKOUTS["rldas_madde"])
SHELL_KNOCKOUTS["les_shell"] = SHELL_KNOCKOUTS["les_sigma"] + SHELL_KNOCKOUTS["les_mu"]
SHELL_KNOCKOUTS["rldas_search"] = (SHELL_KNOCKOUTS["rldas_bound"] + SHELL_KNOCKOUTS["rldas_rw"]
                                   + SHELL_KNOCKOUTS["rldas_feat"])

# policy attribute of each agent, and number of actions of RL controllers
POLICY_ATTR = {"B2OPT": "Opt", "GLHF": "Pom", "LES": "best_x", "DEDDQN": "model",
               "RLDAS": "actor", "RNNOPT": "net"}
N_ACTIONS = {"DEDDQN": 4, "RLDAS": 3}

# (agent, knock-out, probes).  "none" is the agent as distributed.
SEARCH_T = ["identity", "ambient_translation_diag", "translate_5", "translate_50", "translate_500"]
GRID: List[Tuple[str, str, List[str]]] = [
    ("LES", "none", SEARCH_T), ("LES", "les_sigma", SEARCH_T),
    ("LES", "les_mu", SEARCH_T), ("LES", "les_shell", SEARCH_T),
    ("SYMBOL", "none", SEARCH_T), ("SYMBOL", "symbol_init", SEARCH_T),
]
_RLDAS_S = ["identity", "ambient_translation_diag", "translate_50",
            "domain_scaling_up", "domain_scaling_down"]
GRID += [("RLDAS", k, _RLDAS_S) for k in
         ("none", "rldas_bound", "rldas_rw", "rldas_feat", "rldas_search",
          "rldas_madde", "rldas_all")]
_RLHPSDE_S = ["identity", "domain_scaling_up", "domain_scaling_down", "scale_0.1"]
GRID += [("RLHPSDE", k, _RLHPSDE_S) for k in ("none", "rlhpsde_dfdc")]
_MONO = ["identity", "monotone_log", "monotone_spline"]
GRID += [("DEDDQN", k, _MONO) for k in
         ["none"] + [f"fixed:{a}" for a in range(4)] + [f"untrained:{s}" for s in range(3)]]
GRID += [("RLDAS", k, _MONO) for k in
         ["none"] + [f"fixed:{a}" for a in range(3)] + [f"untrained:{s}" for s in range(3)]]
_B2_S = ["identity", "ambient_translation_diag", "domain_scaling_up", "domain_scaling_down"]
GRID += [("B2OPT", k, _B2_S) for k in ("none", "untrained:0", "untrained:1")]
GRID += [("GLHF", k, ["identity", "ambient_translation_diag"]) for k in
         ("none", "untrained:0", "untrained:1", "untrained:2")]
GRID += [("RNNOPT", k, ["identity", "rotation", "coordinate_permutation", "monotone_spline"])
         for k in ("none", "untrained:0", "untrained:1", "untrained:2")]


# --------------------------------------------------------------------------
# policy swaps
# --------------------------------------------------------------------------

def _reinit_module(m, seed: int):
    """Fresh weights for a torch module: default layer init, N(0,1) for bare parameters."""
    import torch
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    owned = set()
    for sub in m.modules():
        if hasattr(sub, "reset_parameters") and sub is not m:
            sub.reset_parameters()
            owned.update(id(p) for p in sub.parameters(recurse=False))
    with torch.no_grad():
        for p in m.parameters():
            if id(p) not in owned:
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype))


def swap_policy(agent_name: str, agent, spec: str):
    import numpy as np
    import torch
    kind, arg = spec.split(":")
    k = int(arg)
    attr = POLICY_ATTR[agent_name]
    if kind == "untrained":
        if agent_name == "LES":
            # the meta-optimizer (CMA-ES) starts at mean 0 with sigma 0.1
            rng = np.random.RandomState(1000 + k)
            agent.best_x = rng.normal(0.0, 0.1, size=np.shape(agent.best_x))
        else:
            _reinit_module(getattr(agent, attr), 1000 + k)
            if agent_name == "DEDDQN" and hasattr(agent, "target_model"):
                agent.target_model.load_state_dict(agent.model.state_dict())
    elif kind == "fixed":
        if agent_name == "DEDDQN":
            agent.get_action = lambda state, *a, **kw: np.array([k])
        elif agent_name == "RLDAS":
            agent.actor = lambda obs, *a, **kw: (torch.tensor([k]), None, None)
        else:
            raise ValueError(f"no fixed action for {agent_name}")
    else:
        raise ValueError(spec)


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------

_APPLIED: Dict[str, str] = {}


def run_task(t):
    warnings.filterwarnings("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import exp2_metabox_audit as E2
    from invaudit.external import metabox as MB
    from invaudit.metrics import Recorder, default_checkpoints, default_targets, ecdf_area
    from invaudit.transforms import get_probe

    t0 = time.time()
    ko = t["ko"]
    try:
        # shell patches are process-wide: apply each set once, and never mix sets
        sha = ""
        if ko in SHELL_KNOCKOUTS:
            if _APPLIED and ko not in _APPLIED:
                raise RuntimeError("worker already patched with another knock-out")
            if ko not in _APPLIED:
                _APPLIED[ko] = apply_patches(SHELL_KNOCKOUTS[ko])
            sha = _APPLIED[ko]
        elif _APPLIED:
            raise RuntimeError("unpatched task on a patched worker")

        cps = default_checkpoints(E2.BUDGET, E2.N_CHECKPOINTS)
        p0, bias = MB.bbob_instance(t["fid"], t["instance"])
        p = get_probe(t["probe"]).apply(p0, 777 + 31 * t["instance"])
        if p is p0:
            p = p0.with_action(label="identity")
        rec = Recorder(cps, 1, "cpu", torch.float64)
        p.attach_recorder(rec)
        seed = 1000 + 97 * t["seed"] + 13 * t["instance"] + t["fid"]

        # as MB.run_agent, with the policy swap inserted after loading
        from metaevobox.environment.basic_environment import PBO_Env
        agent, opt_cls = MB.load_agent(t["agent"])
        if ":" in ko:
            swap_policy(t["agent"], agent, ko)
        cfg = MB.make_config(maxFEs=E2.BUDGET, dim=p.D, device="cpu")
        env = PBO_Env(MB.MetaBoxView(p, bias=bias), opt_cls(cfg))
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float64)
        agent.rollout_episode(env, seed, {})

        best = rec.finalize()
        A = float(ecdf_area(best, default_targets())[0])
        row = dict(t, A=A, best_final=float(best[0, -1]), real_fe=int(rec.fe),
                   seconds=round(time.time() - t0, 2), patch_sha=sha, error="")
        for j in range(best.shape[1]):
            row[f"b{j}"] = float(best[0, j])
        return row
    except Exception as e:
        return dict(t, A=float("nan"), best_final=float("nan"), real_fe=-1,
                    seconds=round(time.time() - t0, 2), patch_sha="",
                    error=f"{type(e).__name__}: {str(e)[:300]} | "
                          f"{traceback.format_exc(limit=3)[-400:]}".replace("\n", " "))


KEY = ("agent", "ko", "fid", "instance", "probe", "seed")


def build_tasks(n_instances=2, n_seeds=20):
    import exp2_metabox_audit as E2
    tasks = []
    for agent, ko, probes in GRID:
        for fid, inst, probe, s in itertools.product(E2.TRAIN_FIDS, range(n_instances),
                                                     probes, range(n_seeds)):
            tasks.append(dict(agent=agent, ko=ko, fid=fid, instance=inst, probe=probe, seed=s))
    return tasks


def _worker_init():
    # each worker serves exactly one knock-out (see run_task); nothing else to set up
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default="results_rev/e5_knockouts.csv")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated agent:ko filters")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, patches in SHELL_KNOCKOUTS.items():
            for p in patches:
                print(f"{name:14s} {p.cls}.{p.method}\n    - {p.old}\n    + {p.new}")
        return

    tasks = build_tasks(n_seeds=args.seeds)
    if args.only:
        keep = {tuple(x.split(":", 1)) for x in args.only.split(",")}
        tasks = [t for t in tasks if (t["agent"], t["ko"]) in keep]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    done.add(tuple(str(r[k]) for k in KEY))
    todo = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    if args.limit:
        # smoke test: a few tasks from every (agent, ko) group
        seen: Dict[Tuple[str, str], int] = {}
        small = []
        for t in todo:
            g = (t["agent"], t["ko"])
            if seen.get(g, 0) < max(1, args.limit // 40):
                small.append(t)
                seen[g] = seen.get(g, 0) + 1
        todo = small
    print(f"{len(tasks)} tasks, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    # one pool per knock-out group, so that a patched worker never serves another group
    groups: Dict[str, List[dict]] = {}
    for t in todo:
        groups.setdefault(t["ko"] if t["ko"] in SHELL_KNOCKOUTS else "_unpatched", []).append(t)
    slow = {"B2OPT": 0, "DEDDQN": 1}
    fields = list(KEY) + ["A", "best_final", "real_fe", "seconds", "patch_sha", "error"] + \
        [f"b{j}" for j in range(31)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    import multiprocessing as mp
    t0 = time.time()
    n_all, i = len(todo), 0
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for g, gt in sorted(groups.items()):
            gt.sort(key=lambda t: slow.get(t["agent"], 99))
            with mp.Pool(min(args.workers, len(gt)), maxtasksperchild=None) as pool:
                for row in pool.imap_unordered(run_task, gt, chunksize=1):
                    w.writerow(row)
                    f.flush()
                    i += 1
                    if i % 50 == 0 or i == n_all or row["error"]:
                        el = time.time() - t0
                        msg = (f"[{i:5d}/{n_all}] {g:14s} {row['agent']:7s} {row['ko']:12s} "
                               f"{row['probe']:24s} A={row['A']:.4f} {row['seconds']:6.1f}s "
                               f"elapsed={el / 3600:5.2f}h")
                        if row["error"]:
                            msg += f"\n    ERROR {row['error'][:300]}"
                        print(msg, flush=True)


if __name__ == "__main__":
    main()
