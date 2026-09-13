# Invariance by Accident

**A symmetry audit and an invariance harness for black-box optimizers — hand-designed and automatically designed.**

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![status](https://img.shields.io/badge/status-under%20review-lightgrey)

An optimizer that is invariant to a transformation of the problem cannot tell the
transformed problem from the original, so its performance carries over unchanged.
Hand-designed evolutionary algorithms were built around this property.
Automatically designed optimizers — reinforcement-learned controllers, learned update
rules, symbolic or generated rules — inherit whatever invariances their design pipeline
happens to preserve, and their generalization is usually checked only on held-out
functions.

This repository provides the tools to **measure** those invariances and to **impose**
them from the outside:

- **Symmetry audit** — presents an optimizer with transformed views of the same
  instance under common random numbers, scores every run in the original objective,
  and tests invariance instance by instance with a calibrated permutation test.
- **Invariance harness** — a wrapper around the *problem* (not the optimizer) that fixes
  the search frame and replaces objective values by a persistent order embedding,
  making any optimizer invariant to translation, dilation and every strictly increasing
  transformation of the objective.
- **Ten reference optimizers** with analytically known invariances, used as ground truth.
- **An adapter for the sixteen pretrained agents of MetaBox-v2**, and every experiment
  script together with the stored per-run results.

---

## Contents

- [Key results](#key-results)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Repository layout](#repository-layout)
- [Methodological safeguards](#methodological-safeguards)
- [Extending the audit](#extending-the-audit)
- [Data](#data)

---

## Key results

| Experiment | Outcome |
|---|---|
| Validation on 10 hand-designed optimizers (90 algorithm–probe pairs with known answers) | No false alarm on 76 invariant pairs; the rotation and objective-shape dependences predicted by the update rules are recovered |
| Audit of the 16 pretrained MetaBox-v2 agents | 16 significant violations out of 208 pairs — all in learned optimizers, none in hand-designed ones; one agent raises an exception on every run when the domain is contracted |
| Realistic domains | Moving the box from `[-5,5]^D` to `[0,10]^D` already costs learned update rules up to 3.7 points of performance; control agents stay below 0.06 |
| Held-out functions | Violations on training functions are *not* associated with loss on held-out functions (the association is reversed), so held-out-function tests do not reveal these failures |
| Frame layer of the harness on the affected agents | All significant search-space violations and all crashes removed; untransformed runs bit-for-bit identical |
| Objective layer on comparison-based optimizers | Trajectories bit-for-bit identical to the unwrapped optimizer |

Performance is the area under the target-hitting ECDF (51 targets from `1e2` to `1e-8`,
31 budgets up to 20,000 evaluations), expressed in points (×100).

## Installation

Two environments are used, because the pretrained agents pin their own dependencies.

**Core library, anchors and analysis**

```bash
git clone https://github.com/anonymerauthor/invariance-by-accident.git
cd invariance-by-accident
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**Pretrained MetaBox-v2 agents** (only needed for the agent experiments)

```bash
python -m venv .venv-metabox && source .venv-metabox/bin/activate
pip install torch                                        # CPU build is sufficient
pip install -r requirements-metabox.txt
pip install -e .
```

The sixteen checkpoints ship with the `metaevobox` package; nothing is downloaded
separately. All experiments run on CPU.

Run the test suite with `python -m pytest tests`.

## Quick start

**Does an optimizer's performance survive a symmetry?**

```python
import torch
from invaudit.problem import make_instance
from invaudit.transforms import get_probe
from invaudit.optimizers import get_optimizer
from invaudit.metrics import Recorder, default_checkpoints, default_targets, ecdf_area

torch.set_default_dtype(torch.float64)
budget, runs = 10_000, 5
checkpoints, targets = default_checkpoints(budget), default_targets()

def score(problem, optimizer, seed=1):
    recorder = Recorder(checkpoints, runs, "cpu", torch.float64)
    problem.attach_recorder(recorder)          # performance is recorded in the original objective
    get_optimizer(optimizer).run(problem, budget, seed=seed, n_runs=runs)
    return ecdf_area(recorder.finalize(), targets)

original = make_instance("rastrigin", D=10, instance=0)
reshaped = get_probe("monotone_log").apply(make_instance("rastrigin", D=10, instance=0), seed=7)

for name in ["DE/rand/1/bin", "SHADE", "CMA-ES"]:
    shift = score(original, name) - score(reshaped, name)
    print(f"{name:14s} mean shift under log(1+f): {100 * shift.mean():+.3f} points")
```

**Audit a grid and test significance**

```python
from invaudit.audit import AuditConfig, run_audit
from invaudit.analysis import audit_to_long, violation_significance

cfg = AuditConfig(D=10, budget=10_000, n_runs=10, n_instances=2)
probes = ["rotation", "domain_scaling_up", "monotone_spline"]
df = run_audit(cfg, ["DE/rand/1/bin", "CMA-ES"], ["ellipsoid", "rastrigin"], probes)

print(violation_significance(audit_to_long(df), probes))   # V, p, q, significant
```

**Make any optimizer invariant**

```python
from invaudit.harness import harness

problem = make_instance("rastrigin", D=10, instance=0)
recorder = Recorder(checkpoints, runs, "cpu", torch.float64)
problem.attach_recorder(recorder)

view = harness(problem, n_runs=runs, box_frame=True, rank_fitness=True)   # order embedding
get_optimizer("SHADE").run(view, budget, seed=1, n_runs=runs)
print(ecdf_area(recorder.finalize(), targets).mean())
```

## Reproducing the experiments

All scripts write to `results/` and resume from partial output if interrupted.
Times are wall-clock on a 20-thread laptop CPU.

| Script | What it does | Environment | Time |
|---|---|---|---|
| `experiments/exp1_anchor_validation.py --budget 20000 --runs 15 --instances 3` | Validation on the 10 anchors, 14 canonical functions, 11 probes | core | ~3 h |
| `experiments/exp1b_dyadic_scaling.py` | Dilation by `1e3` vs. `2^±10` (floating-point diagnostic) | core | ~45 min |
| `experiments/exp2_metabox_audit.py --family audit --workers 12 --out results/exp2_audit.csv` | Audit of the 16 agents on their training functions | MetaBox | ~6 h |
| `experiments/exp2_metabox_audit.py --family ood --workers 12 --out results/exp2_ood.csv` | 16 agents, untransformed, on held-out functions | MetaBox | ~1 h |
| `experiments/exp2_metabox_audit.py --family frame --agents B2OPT,LES,SYMBOL,GLHF,RLHPSDE,RLDAS --workers 6 --out results/exp4_frame.csv` | Frame layer on the agents with search-space violations | MetaBox | ~1 h |
| `experiments/exp2_metabox_audit.py --family dose --agents B2OPT,LES,SYMBOL,GLHF,RLHPSDE,RLDAS,LDE,GLEET --workers 14 --out results/exp5_dose.csv` | Realistic domains | MetaBox | ~1.5 h |
| `experiments/exp3_anchors_on_bbob.py --family audit --out results/exp3_audit.csv` (and `--family ood`) | The anchors on the MetaBox instances | MetaBox | ~50 min |
| `experiments/exp3_anchors_on_bbob.py --family audit --seeds 15 --workers 10 --out results/exp3_audit_15seeds.csv` | Anchor audit repeated with fifteen seeds (power check) | MetaBox | ~15 min |
| `experiments/exp6_harness_anchors.py --workers 17` | Harness on the anchors, full validation grid | core | ~2 h |
| `experiments/exp7_rank_ablation.py` | Ablation of the objective layer (raw / window quantile / worse-count / order embedding) | core | ~15 min |

Analysis, which reads only the stored results:

```bash
python experiments/make_figures.py            # figures and violation / out-of-distribution tables
python experiments/analyze_interventions.py   # realistic domains, frame layer, harness tables
python experiments/anchor_power_check.py      # fifteen-seed power check of the anchor result
python experiments/instance_report.py         # verified optima and biases of the MetaBox instances
```

`make_figures.py` and `analyze_interventions.py` run in the core environment;
`instance_report.py` builds the MetaBox instances and needs the MetaBox environment.

## Repository layout

```
invariance-by-accident/
├── invaudit/                     library
│   ├── functions.py              14 canonical functions (f(0) = 0, f ≥ 0)
│   ├── problem.py                instances, presented problems, stable seeds
│   ├── transforms.py             symmetry probes (search space and objective)
│   ├── metrics.py                ECDF-area performance, evaluation recorder
│   ├── optimizers.py             10 reference optimizers with known invariances
│   ├── audit.py                  audit driver (common random numbers, checkpointing)
│   ├── analysis.py               violation score, sign-flip permutation test, FDR
│   ├── harness.py                frame layer, worse-count transform, order embedding
│   └── external/metabox.py       adapter for the pretrained MetaBox-v2 agents
├── experiments/                  one script per experiment + analysis
├── results/                      per-run results and derived tables (see results/README.md)
├── tests/                        unit tests of the core guarantees
├── requirements.txt
├── requirements-metabox.txt
└── pyproject.toml
```

## Methodological safeguards

Each of the following was introduced because the naive choice produced a wrong answer
during development.

- **Score the best evaluated point, in the original objective.** Comparing raw values
  across a monotone re-parameterization measures the transformation, not the algorithm,
  and an optimizer's self-reported incumbent is unreliable once values are rank-coded.
- **Pathwise vs. distributional invariance.** Per-coordinate randomness makes an
  algorithm pathwise permutation-dependent although its performance law is unchanged;
  both readings are reported.
- **Measure per instance.** On an instance distribution that is itself symmetric (e.g.
  randomly rotated suites), average performance is blind to any dependence on that
  symmetry.
- **Equivariant initialization and unbounded validation.** Box-uniform initialization and
  box clipping are themselves not rotation-equivariant.
- **Separate logic from arithmetic.** Objective rescaling and shifting are distinct probes,
  and dilations by powers of two (exact in floating point) confirm that residual pathwise
  readings are rounding.
- **Seed-matched noise correction and a permutation test.** The violation score is the
  noise-corrected RMS of per-instance shifts; significance uses sign-flip permutations with
  Benjamini–Hochberg control.
- **Read, don't assume, the optimum.** Several MetaBox constructors overwrite the shift they
  are given; optima are read back and verified to be global minima.
- **Count evaluations as they happen.** Some reference implementations spend budgets far
  from their nominal value; the recorder counts every evaluation, including those on copies
  of the problem.
- **Stable seeds.** All seeds derive from CRC32 hashes, never from Python's salted `hash`.

## Extending the audit

**Your own optimizer** — subclass `invaudit.optimizers.Optimizer` and implement
`run(problem, budget, seed, n_runs)`. Evaluate batches of shape `(n_runs, pop, D)` with
`problem.evaluate`, draw the initial population through `problem.init_population` and
`problem.present` so that it is mapped consistently under every probe, and consume random
numbers in a data-independent pattern. Performance is recorded by the problem; the
optimizer never reports an incumbent.

**Your own probe** — add a function `(problem, seed) -> problem` built with
`Problem.with_action(Q=..., s=..., t=..., g=...)` and register it in
`invaudit/transforms.py`.

**Another optimizer library** — follow `invaudit/external/metabox.py`: expose the presented
problem through the library's problem interface, share the recorder across copies, and
reproduce the setting the optimizer was trained or tuned in.

## Data

`results/` contains every per-run result used in the analysis (about 25 MB of CSV), so all
figures and tables can be regenerated without re-running the experiments. Column
definitions are in [`results/README.md`](results/README.md). Runs that raised an exception
are stored with their error message and scored as zero.

---

This repository is anonymized for double-blind review. Licensing and citation information
will be added upon publication.
