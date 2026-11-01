"""E7 - retrain a MetaBox agent with and without the objective layer (AE.4, R1.4, R2.5, R3.6).

Reproduces MetaBox-v2's training protocol for the pretrained agents (the
configuration stored inside the distributed pickles: bbob "difficult" split,
100 epochs, train_batch_size 16, train_mode "single", maxFEs from the suite)
and changes exactly one thing in the ``layer`` condition: every training
problem is wrapped in :class:`invaudit.external.metabox.ObjectiveLayerProblem`,
so the agent only ever sees persistent order codes of objective values.  The
``control`` condition retrains with the same code path and no wrapper, which
separates the effect of the layer from the variability of retraining itself.
Seeds: 3849 is MetaBox's default (the pretrained agents' seed); further seeds
add retraining replicates.

The run writes into its own directory (MetaBox writes logs relative to the
working directory); checkpoints are saved every 5 epochs as
``<outdir>/model/<agent>/<timestamp>_<suite>_difficult/checkpoint-<n>.pkl``.

Usage:
    python experiments/e7_train.py --agent RLDAS --condition layer --seed 3849 \
        --suite bbob-10D --outdir results_rev/e7/RLDAS_10D_layer_3849
    add --epochs 1 for a timing smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["RLDAS", "DEDDQN", "GLEET", "LDE"])
    ap.add_argument("--condition", required=True, choices=["control", "layer", "layer_rawreward"])
    ap.add_argument("--seed", type=int, default=3849)
    ap.add_argument("--suite", default="bbob-10D", choices=["bbob-10D", "bbob-30D"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--parallel", default="subproc", choices=["dummy", "subproc"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import importlib
    import numpy as np
    import torch
    from metaevobox.config import Config
    from metaevobox.environment.problem.utils import construct_problem_set
    from metaevobox.trainer import Trainer
    from invaudit.external.metabox import ObjectiveLayerProblem

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    # MetaBox's own entry point: defaults, then init_config (maxFEs, intervals).
    # Config() parses sys.argv for its defaults, so hide this script's arguments.
    argv, sys.argv = sys.argv, sys.argv[:1]
    cfg = Config(dict(train_problem=args.suite, test_problem=args.suite,
                      train_difficulty="difficult", test_difficulty="difficult",
                      train_batch_size=16, train_mode="single",
                      max_epoch=args.epochs, end_mode="epoch",
                      train_parallel_mode=args.parallel, seed=args.seed, device="cpu",
                      no_tb=True))
    sys.argv = argv
    cfg.agent_save_dir = os.path.join(outdir, "model") + os.sep
    cfg, (train_set, test_set) = construct_problem_set(cfg)
    if args.condition.startswith("layer"):
        train_set.data = [ObjectiveLayerProblem(p, max_fes=int(cfg.maxFEs))
                          for p in train_set.data]

    torch.set_num_threads(1)
    # Logging fix (no effect on learning): RLDAS stacks the per-environment cost
    # logs with np.array, which fails on current numpy when the logs differ in
    # length (an episode that overshoots maxFEs logs extra points; in the control
    # condition some logs end short).  Only the first and last entries are used
    # (normaliser, final best), so pad with the last value / truncate to n_logpoint+1.
    if args.agent == "RLDAS":
        sys.path.insert(0, HERE)
        from e5_knockouts import SourcePatch, apply_patches
        apply_patches([SourcePatch(
            "metaevobox.baseline.metabbo.rldas", "RLDAS", "train_episode",
            "env_cost = np.array(env.get_env_attr('cost'))",
            "env_cost = np.array([(list(c) + [c[-1]] * (self.config.n_logpoint + 1))"
            "[:self.config.n_logpoint + 1] for c in env.get_env_attr('cost')])",
            count=2)])
    if args.condition == "layer_rawreward":
        # The agent observes only order codes, but its training reward is computed
        # from raw values: RLDAS's reward (last best - best) / initial best uses
        # the episode's raw best instead of the coded best.  The reward is not
        # used at test time, so the deployed policy is invariant either way; this
        # condition separates the effect of coded observations from that of a
        # coded reward.
        sys.path.insert(0, HERE)
        from e5_knockouts import SourcePatch, apply_patches
        if args.agent == "DEDDQN":
            # DEDDQN's reward max(parent - trial, 0) uses raw values kept alongside
            # the coded costs; selection still compares codes (same order).
            opt_mod = "metaevobox.environment.optimizer.deddqn_optimizer"
            apply_patches([
                SourcePatch(opt_mod, "DEDDQN_Optimizer", "init_population",
                            "self.__cost = problem.eval(self.__X) - problem.optimum",
                            "self.__cost = problem.eval(self.__X) - problem.optimum; "
                            "self._rawcost = np.asarray(problem.last_raw, dtype=float).copy()"),
                SourcePatch(opt_mod, "DEDDQN_Optimizer", "update",
                            "trial_cost = problem.eval(trial) - problem.optimum",
                            "trial_cost = problem.eval(trial) - problem.optimum; "
                            "_raw_trial = float(np.asarray(problem.last_raw).reshape(-1)[0])"),
                SourcePatch(opt_mod, "DEDDQN_Optimizer", "update",
                            "reward = max(self.__cost[self.__pointer] - trial_cost, 0)",
                            "reward = max(self._rawcost[self.__pointer] - _raw_trial, 0)"),
                SourcePatch(opt_mod, "DEDDQN_Optimizer", "update",
                            "self.__cost[self.__pointer] = trial_cost",
                            "self.__cost[self.__pointer] = trial_cost; "
                            "self._rawcost[self.__pointer] = _raw_trial"),
            ])
        elif args.agent != "RLDAS":
            raise SystemExit("layer_rawreward is implemented for RLDAS and DEDDQN only")
        else:
          opt_mod = "metaevobox.environment.optimizer.rldas_optimizer"
          apply_patches([
            SourcePatch(opt_mod, "RLDAS_Optimizer", "init_population",
                        "self.cost_scale_factor = self.population.gbest",
                        "self.cost_scale_factor = self.population.gbest; "
                        "self._raw_scale = problem.raw_best"),
            SourcePatch(opt_mod, "RLDAS_Optimizer", "update",
                        "last_cost = self.population.gbest",
                        "last_cost = self.population.gbest; _raw_last = problem.raw_best"),
            SourcePatch(opt_mod, "RLDAS_Optimizer", "update",
                        "reward = max((last_cost - self.population.gbest) / self.cost_scale_factor, 0)",
                        "reward = max((_raw_last - problem.raw_best) / self._raw_scale, 0)"),
        ])
    agent_cls = getattr(importlib.import_module("metaevobox.baseline.metabbo"), args.agent)
    opt_cls = getattr(importlib.import_module("metaevobox.environment.optimizer"),
                      f"{args.agent}_Optimizer")
    meta = dict(vars(args), maxFEs=int(cfg.maxFEs), dim=int(cfg.dim),
                train_problems=[str(p) for p in train_set.data],
                start=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(outdir, "run.json"), "w") as f:
        json.dump(meta, f, indent=1)

    os.chdir(outdir)
    t0 = time.time()
    agent = agent_cls(cfg)
    Trainer(cfg, agent, opt_cls(cfg), (train_set, test_set)).train()
    meta.update(seconds=round(time.time() - t0, 1),
                checkpoints=sorted(os.path.relpath(p, outdir) for p in
                                   glob.glob(os.path.join(outdir, "model", "**", "*.pkl"),
                                             recursive=True)),
                end=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(outdir, "run.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
