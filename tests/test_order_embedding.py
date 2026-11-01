"""Order embedding of the invariance harness: guarantees and finite precision (R2.4)."""
import math

import torch

from invaudit.harness import (_OrderEmbeddingState, count_code_inversions,
                              count_code_ties)

torch.set_default_dtype(torch.float64)


def _stream(seed=0, n_batches=300, batch=10):
    g = torch.Generator().manual_seed(seed)
    # coarse grid so that repeated values occur; slow drift reuses gaps
    return [torch.round(torch.randn(batch, generator=g) * (1 + 0.01 * b) * 100) / 100
            for b in range(n_batches)]


def _codes(stream, g=lambda y: y):
    st = _OrderEmbeddingState(1, "cpu", torch.float64)
    out = torch.cat([st.push_and_rank(g(y).reshape(1, -1))[0] for y in stream])
    return st, out


def test_codes_identical_under_monotone_maps():
    s = _stream()
    _, c0 = _codes(s)
    for g in (lambda y: y ** 3, torch.exp, lambda y: 7.0 * y - 3.0):
        _, c1 = _codes(s, g)
        assert torch.equal(c0, c1)


def test_order_and_equality_preserved_across_time():
    s = _stream(1)
    st, c = _codes(s)
    v = torch.cat(s)
    i = torch.argsort(v, stable=True)
    vs, cs = v[i], c[i]
    same = vs[1:] == vs[:-1]
    assert bool((cs[1:] >= cs[:-1]).all())             # no inversion, ever
    assert bool((cs[1:][same] == cs[:-1][same]).all())  # equal values, equal codes
    if st.exhausted == 0:
        assert bool((cs[1:][~same] > cs[:-1][~same]).all())  # strict without ties
    assert count_code_inversions(st) == 0


def test_nan_and_inf_are_worst_and_never_invert():
    st = _OrderEmbeddingState(1, "cpu", torch.float64)
    y = torch.tensor([[1.0, float("nan"), float("inf"), -2.0, float("nan"), 0.5]])
    c = st.push_and_rank(y)[0]
    assert st.n_nonfinite == 3
    assert c[1] == c[2] == c[4]                          # NaN and +inf share the worst code
    assert c[1] > c[0] > c[5] > c[3]
    st.push_and_rank(torch.tensor([[float("nan"), 0.75, float("-inf")]]))
    assert count_code_inversions(st) == 0


def test_adaptive_adversary_forces_a_counted_tie():
    # values in (1e-300, 1) split geometrically: float64 values outlast the codes,
    # so the first tie is forced by the code resolution and must be counted
    st = _OrderEmbeddingState(1, "cpu", torch.float64)
    st.push_and_rank(torch.tensor([[-1.0, 1e-300, 1.0]]))
    n = 0
    while st.exhausted == 0 and n < 500:
        V, C = st.V[0], st.C[0]
        rel = (C[1:] - C[:-1]) / C[1:].abs().clamp_min(1e-300)
        rel[V[:-1] <= 0] = math.inf
        k = int(torch.argmin(rel))
        st.push_and_rank(torch.sqrt(V[k] * V[k + 1]).reshape(1, 1))
        n += 1
    assert st.exhausted > 0
    assert count_code_ties(st) == st.exhausted
    assert st.first_tie_eval[0] is not None
    assert 20 < n < 80                                   # about the float64 mantissa
    assert count_code_inversions(st) == 0
