"""Unit tests of the guarantees the audit and the harness rely on."""

import pandas as pd
import pytest
import torch

from invaudit.analysis import audit_to_long, violation_significance
from invaudit.audit import AuditConfig, run_audit
from invaudit.functions import FUNCTIONS
from invaudit.harness import harness
from invaudit.metrics import Recorder, default_checkpoints, default_targets, ecdf_area
from invaudit.optimizers import get_optimizer
from invaudit.problem import make_instance, stable_seed
from invaudit.transforms import PROBES

torch.set_default_dtype(torch.float64)
BUDGET, RUNS = 2000, 3


def _score(problem, name, seed=11, layers=None):
    rec = Recorder(default_checkpoints(BUDGET), RUNS, "cpu", torch.float64)
    problem.attach_recorder(rec)
    view = harness(problem, RUNS, **layers) if layers else problem
    get_optimizer(name).run(view, BUDGET, seed=seed, n_runs=RUNS)
    return ecdf_area(rec.finalize(), default_targets())


def test_canonical_form():
    z = torch.zeros(2, 10)
    for spec in FUNCTIONS.values():
        assert torch.all(spec.fn(z).abs() < 1e-12), spec.name


def test_search_space_actions_are_invertible():
    p = make_instance("rastrigin", 10, 0)
    gen = torch.Generator().manual_seed(1)
    x = p.init_population(64, gen)
    v0 = p.base_value(p.present(x))
    for name, probe in PROBES.items():
        if probe.space != "search":
            continue
        q = probe.apply(make_instance("rastrigin", 10, 0), 3)
        v = q.base_value(q.present(x))
        assert torch.allclose(v, v0, rtol=1e-9, atol=1e-9), name


def test_objective_actions_preserve_ranking():
    p = make_instance("sphere", 10, 0)
    x = p.present(p.init_population(64, torch.Generator().manual_seed(2)))
    r0 = torch.argsort(torch.argsort(p.evaluate(x)))
    for name, probe in PROBES.items():
        if probe.space == "objective":
            q = probe.apply(make_instance("sphere", 10, 0), 3)
            assert torch.equal(torch.argsort(torch.argsort(q.evaluate(q.present(x)))), r0), name


def test_stable_seed_is_deterministic():
    assert stable_seed("cell", 1, "DE", "sphere", 0) == stable_seed("cell", 1, "DE", "sphere", 0)


@pytest.mark.parametrize("name", ["DE/rand/1/bin", "SHADE", "CMA-ES"])
def test_runs_are_deterministic(name):
    a = _score(make_instance("ellipsoid", 10, 0), name)
    b = _score(make_instance("ellipsoid", 10, 0), name)
    assert torch.equal(a, b)


@pytest.mark.parametrize("name", ["DE/rand/1/bin", "jDE", "PSO", "CMA-ES"])
def test_order_embedding_is_free_for_comparison_based_optimizers(name):
    raw = _score(make_instance("rastrigin", 10, 1), name)
    coded = _score(make_instance("rastrigin", 10, 1), name,
                   layers=dict(box_frame=False, rank_fitness=True, rank_mode="order"))
    assert torch.equal(raw, coded)


def test_harness_removes_monotone_violation_of_shade():
    probe = PROBES["monotone_log"]
    layers = dict(box_frame=True, rank_fitness=True, rank_mode="order")
    a = _score(make_instance("ellipsoid", 10, 0), "SHADE", layers=layers)
    b = _score(probe.apply(make_instance("ellipsoid", 10, 0), 5), "SHADE", layers=layers)
    assert torch.equal(a, b)


def test_scale_equivariant_optimizer_reads_exactly_zero():
    # a dilation by a power of two is exact in binary floating point
    probe = "domain_scaling_dyadic_up"
    cfg = AuditConfig(D=5, budget=BUDGET, n_runs=RUNS, n_instances=1)
    df = run_audit(cfg, ["DE/rand/1/bin"], ["sphere"], [probe], verbose=False)
    res = violation_significance(audit_to_long(df), [probe], n_perm=200)
    assert isinstance(res, pd.DataFrame)
    assert float(res.V.iloc[0]) == 0.0
