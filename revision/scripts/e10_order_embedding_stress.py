"""E10 - finite-precision behaviour of the order embedding (reviewer 2, comment 4).

Three parts, each written to its own CSV under the output directory.

1. ``e10_correctness.csv``  random value streams with repeats: (a) no code
   inversion between any two values ever issued, equal values get equal codes;
   (b) codes are bit-identical under three strictly increasing maps g;
   (c) number of ties (distinct values that share a code).
2. ``e10_natural.csv``  the anchors on the canonical functions, objective layer
   only (no frame), budgets 20,000 and 100,000, one run per cell: ties, first
   tie (evaluation index and raw best value at that time), deepest relative gap
   among distinct codes (bits), number of codes, wall time.
3. ``e10_adversarial.csv``  synthetic streams: an adaptive adversary that always
   inserts into the gap of smallest *relative* width; convergence from above
   towards a level with a better value below; convergence from below towards a
   level with a worse value above; batches of 1 and 10.  Reports the number of
   insertions until the first tie (-1: none within the cap).

Ties are counted from the stored codes for every embedding (the legacy
embedding's own counter misses collisions between new codes in one gap).

Usage:  python revision/scripts/e10_order_embedding_stress.py --workers 16 --out results_rev
        (add --quick for a smoke test)
"""
from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

# (name, kind, placement, eps)
EMBEDDINGS = [
    ("legacy", "legacy", None, None),
    ("even", "v2", "even", None),
    ("worse/16", "v2", "worse", 1 / 16),
    ("recent/2", "v2", "recent", 1 / 2),
    ("recent/16", "v2", "recent", 1 / 16),
    ("recent/256", "v2", "recent", 1 / 256),
    ("trend/16", "v2", "trend", 1 / 16),
    ("trend/4", "v2", "trend", 1 / 4),
]


def make_state(kind, placement, eps, R=1):
    import torch
    from invaudit.harness import _OrderEmbeddingState, _OrderEmbeddingV2State
    if kind == "legacy":
        return _OrderEmbeddingState(R, "cpu", torch.float64)
    return _OrderEmbeddingV2State(R, "cpu", torch.float64, eps=eps or 1 / 16,
                                  placement=placement)


def ties_of(state):
    from invaudit.harness import count_code_ties
    return count_code_ties(state, 0)


def inversions_of(state):
    from invaudit.harness import count_code_inversions
    return count_code_inversions(state, 0)


def depth_bits(state):
    """-log2 of the smallest relative gap between distinct consecutive codes."""
    import torch
    C = state.C[0]
    if C.numel() < 2:
        return 0.0
    d = C[1:] - C[:-1]
    pos = d > 0
    if not bool(pos.any()):
        return float("inf")
    g = d[pos] / C[1:][pos].abs().clamp_min(torch.finfo(C.dtype).tiny)
    return -math.log2(float(g.min()))


# ----------------------------------------------------------------- part 1
def correctness(n_streams=50, n_batches=400, batch=10, seed=0):
    import torch
    rows = []
    gen = torch.Generator().manual_seed(seed)
    maps = {"cube": lambda y: y ** 3, "exp": torch.exp, "affine": lambda y: 7.0 * y - 3.0}
    for s in range(n_streams):
        # values on a coarse grid so that repeats occur; slow drift reuses gaps
        ys = [torch.round(torch.randn(batch, generator=gen, dtype=torch.float64)
                          * (1 + 0.01 * b) * 100) / 100 for b in range(n_batches)]
        for name, kind, placement, eps in EMBEDDINGS:
            st = make_state(kind, placement, eps)
            c = torch.cat([st.push_and_rank(y.reshape(1, -1))[0] for y in ys])
            v = torch.cat(ys)
            iv = torch.argsort(v, stable=True)
            vs, cs = v[iv], c[iv]
            same = vs[1:] == vs[:-1]
            no_inversion = bool((cs[1:] >= cs[:-1]).all())
            equal_ok = bool((cs[1:][same] == cs[:-1][same]).all())
            inv_ok = True
            for g in maps.values():
                st2 = make_state(kind, placement, eps)
                c2 = torch.cat([st2.push_and_rank(g(y).reshape(1, -1))[0] for y in ys])
                inv_ok &= bool(torch.equal(c, c2))
            rows.append(dict(stream=s, embedding=name, ties=ties_of(st),
                             no_inversion=no_inversion, equal_values_equal_codes=equal_ok,
                             invariant_under_g=inv_ok))
    return rows


# ----------------------------------------------------------------- part 2
def _natural(args):
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    from invaudit.harness import harness
    from invaudit.optimizers import get_optimizer
    from invaudit.problem import make_instance
    opt, fn, budget, name, kind, placement, eps = args
    p = make_instance(fn, 10, 0)
    mode = "order" if kind == "legacy" else "order2"
    h = harness(p, 1, box_frame=False, rank_fitness=True, rank_mode=mode,
                order_eps=eps or 1 / 16, order_placement=placement or "recent")
    st = h._rank
    # record the first batch after which a tie exists, and the raw best then
    rec = {"best": float("inf"), "n": 0, "tie_eval": -1, "tie_best": float("nan")}
    orig = st._codes_for_run

    def wrapped(r, yr):
        rec["best"] = min(rec["best"], float(yr.min()))
        rec["n"] += yr.numel()
        out = orig(r, yr)
        if rec["tie_eval"] < 0 and ties_of(st) > 0:
            rec["tie_eval"], rec["tie_best"] = rec["n"], rec["best"]
        return out
    st._codes_for_run = wrapped
    t0 = time.time()
    get_optimizer(opt).run(h, budget, seed=1, n_runs=1)
    return dict(optimizer=opt, function=fn, budget=budget, embedding=name,
                ties=ties_of(st), inversions=inversions_of(st),
                first_tie_eval=rec["tie_eval"], raw_best_at_first_tie=rec["tie_best"],
                final_raw_best=rec["best"], depth_bits=depth_bits(st),
                n_codes=int(st.C[0].numel()), seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------- part 3
def adversarial(max_insert=5000):
    import torch
    rows = []
    for name, kind, placement, eps in EMBEDDINGS:
        # (a) adaptive adversary: always split the gap of smallest relative code
        # width.  Values live in (1e-300, 1), where float64 has ~2**62 distinct
        # values, and are split geometrically, so the adversary is limited by the
        # embedding's code resolution, not by the resolution of the values.
        st = make_state(kind, placement, eps)
        st.push_and_rank(torch.tensor([[-1.0, 1e-300, 1.0]], dtype=torch.float64))
        n, value_exhausted = 0, False
        while ties_of(st) == 0 and n < max_insert:
            V, C = st.V[0], st.C[0]
            rel = (C[1:] - C[:-1]) / C[1:].abs().clamp_min(1e-300)
            rel[V[:-1] <= 0] = float("inf")         # stay inside (1e-300, 1)
            k = int(torch.argmin(rel))
            y = torch.sqrt(V[k] * V[k + 1])
            if bool((y <= V[k]) | (y >= V[k + 1])):
                value_exhausted = True
                break
            st.push_and_rank(y.reshape(1, 1))
            n += 1
        rows.append(dict(stream="adaptive_adversary", batch=1, embedding=name,
                         insertions_until_tie=n if ties_of(st) else
                         (-2 if value_exhausted else -1)))
        # (b) convergence from above / (c) from below
        for sname, sign in (("converge_from_above", 1.0), ("converge_from_below", -1.0)):
            for batch in (1, 10):
                st = make_state(kind, placement, eps)
                # level 1.0; a better value at 0.5 (from above) or a worse one at 1.5
                st.push_and_rank(torch.tensor([[1.0 - 0.5 * sign, 1.0 + 0.5 * sign]],
                                              dtype=torch.float64))
                n, k = 0, 1
                while ties_of(st) == 0 and n < max_insert:
                    d = sign * 0.5 * 0.5 ** (k / 8.0)
                    y = 1.0 + d * (1.0 + 0.01 * torch.arange(batch, dtype=torch.float64))
                    st.push_and_rank(y.reshape(1, -1))
                    n += batch
                    k += 1
                rows.append(dict(stream=sname, batch=batch, embedding=name,
                                 insertions_until_tie=n if ties_of(st) else -1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results_rev")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    import pandas as pd
    from invaudit.functions import FUNCTIONS
    from invaudit.optimizers import optimizer_names
    os.makedirs(args.out, exist_ok=True)

    c = pd.DataFrame(correctness(n_streams=5 if args.quick else 50))
    c.to_csv(os.path.join(args.out, "e10_correctness.csv"), index=False)
    print(c.groupby("embedding", sort=False).agg(
        ties=("ties", "sum"), no_inversion=("no_inversion", "all"),
        equal_codes=("equal_values_equal_codes", "all"),
        invariant=("invariant_under_g", "all")).to_string(), flush=True)

    a = pd.DataFrame(adversarial())
    a.to_csv(os.path.join(args.out, "e10_adversarial.csv"), index=False)
    print(a.pivot_table(index=["stream", "batch"], columns="embedding",
                        values="insertions_until_tie", sort=False).to_string(), flush=True)

    opts = optimizer_names()
    fns = list(FUNCTIONS)
    budgets = [20000, 100000]
    if args.quick:
        opts, fns, budgets = ["CMA-ES", "DE/rand/1/bin"], ["schaffer_f7", "sphere"], [20000]
    tasks = [(o, f, b, *e) for o, f, b in itertools.product(opts, fns, budgets)
             for e in EMBEDDINGS]
    tasks.sort(key=lambda t: -t[2])            # long budgets first
    out = os.path.join(args.out, "e10_natural.csv")
    rows = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_natural, tasks), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"[{i}/{len(tasks)}]", flush=True)
    n = pd.DataFrame(rows)
    n.to_csv(out, index=False)
    n["tie_before_target"] = (n.ties > 0) & (n.raw_best_at_first_tie > 1e-8)
    print(n.groupby(["budget", "embedding"], sort=False).agg(
        runs=("ties", "size"), runs_with_ties=("ties", lambda s: int((s > 0).sum())),
        ties_before_final_target=("tie_before_target", "sum"),
        ties=("ties", "sum"), inversions=("inversions", "sum"),
        max_depth=("depth_bits", lambda s: s[s < float("inf")].max())).to_string(), flush=True)


if __name__ == "__main__":
    main()
