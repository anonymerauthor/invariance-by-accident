"""Inference for the audit: paired and distribution-level tests, multiplicity.

Notation: for a method, a probe h and a cell c (function, instance), X_cs is the
untransformed score of seed s and Y_cs the transformed one.

Three null hypotheses, from strong to weak:

H0_path   X_cs = Y_cs for every seed (pathwise equivariance under common random
          numbers).  Implies exchangeability of every pair (X_cs, Y_cs).
H0_dist   X_c and Y_c have the same law (Definition 1: invariance).
H0_mean   E X_c = E Y_c (mean invariance; what V measures).

The paired sign-flip test of :mod:`invaudit.analysis` is exact under
exchangeability of each pair, i.e. under H0_path or under H0_dist with
independent arms.  Under H0_dist with *coupled* arms it is not guaranteed: equal
marginals do not make the paired difference symmetric (the counterexample below,
reproduced in :func:`signflip_counterexample`).

The tests here avoid the coupling.  The two arms are taken from *disjoint* seed
sets (pre-specified: identity seeds 0..k-1, probe seeds k..2k-1), so they are
independent; under H0_dist the 2k scores of a cell are exchangeable and a
within-cell label permutation test is exact.  Two statistics:

* ``V_u``   noise-corrected RMS of the per-cell mean shift with the unpaired
            standard error (sensitive to mean shifts);
* ``CvM``   sum over cells of the two-sample Cramer-von Mises statistic computed
            on empirical distribution functions, ties handled exactly
            (sensitive to any difference in law, including spread and tails).

Each is a test of H0_dist; neither is exact for H0_mean alone.  Multiplicity is
controlled with Benjamini-Hochberg over all method-probe pairs and, as a
sensitivity check valid under arbitrary dependence, Benjamini-Yekutieli.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

CELL = ["fid", "instance"]


# ------------------------------------------------------------------ data
def split_cells(df: pd.DataFrame, agent: str, probe: str, id_seeds: Sequence[int],
                pr_seeds: Sequence[int], agent_col: str = "agent") -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per cell, (X from id_seeds under identity, Y from pr_seeds under the probe)."""
    if set(id_seeds) & set(pr_seeds):
        raise ValueError("seed sets must be disjoint for independent arms")
    a = df[(df[agent_col] == agent) & (df.probe == "identity") & df.seed.isin(id_seeds)]
    b = df[(df[agent_col] == agent) & (df.probe == probe) & df.seed.isin(pr_seeds)]
    out = []
    for key, ga in a.groupby(CELL):
        gb = b[(b.fid == key[0]) & (b.instance == key[1])]
        if len(ga) and len(gb):
            out.append((ga.A.to_numpy(float), gb.A.to_numpy(float)))
    return out


# ------------------------------------------------------------------ statistics
def _v_unpaired_batch(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per permutation row: d^2 - s^2 for one cell.  X: (P, n), Y: (P, m)."""
    n, m = X.shape[1], Y.shape[1]
    d = X.mean(1) - Y.mean(1)
    s2 = X.var(1, ddof=1) / n + Y.var(1, ddof=1) / m
    return d ** 2 - s2


def _cvm_batch(Z: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Two-sample Cramer-von Mises on ECDFs, for many label patterns at once.

    Z: pooled scores (N,); L: (P, N) boolean, True = first sample.  The statistic
    is (1/N) * sum over pooled points of (F_X - F_Y)^2, with both ECDFs evaluated
    after the last of any tied values, so ties are handled exactly."""
    order = np.argsort(Z, kind="stable")
    Zs = Z[order]
    Ls = L[:, order]
    n = Ls.sum(1, keepdims=True).astype(float)
    m = Ls.shape[1] - n
    cx = np.cumsum(Ls, 1) / n
    cy = np.cumsum(~Ls, 1) / m
    last = np.r_[Zs[1:] != Zs[:-1], True]          # end of each tie group
    w = np.diff(np.r_[0, np.nonzero(last)[0] + 1])  # multiplicity of each distinct value
    diff2 = (cx[:, last] - cy[:, last]) ** 2
    return (diff2 * w).sum(1) / Ls.shape[1]


def permutation_tests(cells: List[Tuple[np.ndarray, np.ndarray]], n_perm: int = 4000,
                      seed: int = 0) -> Dict[str, float]:
    """Within-cell label permutation tests of H0_dist with V_u and CvM."""
    rng = np.random.default_rng(seed)
    acc_v = np.zeros(n_perm + 1)
    acc_c = np.zeros(n_perm + 1)
    signed = []
    for X, Y in cells:
        Z = np.r_[X, Y]
        N, n = len(Z), len(X)
        L = np.zeros((n_perm + 1, N), dtype=bool)
        L[0, :n] = True
        for p in range(1, n_perm + 1):
            L[p, rng.permutation(N)[:n]] = True
        # gather the two samples of every pattern: first-sample positions first
        idx = np.argsort(~L, axis=1, kind="stable")
        Zp = Z[idx]
        acc_v += _v_unpaired_batch(Zp[:, :n], Zp[:, n:])
        acc_c += _cvm_batch(Z, L)
        signed.append(X.mean() - Y.mean())
    C = len(cells)
    V = np.sqrt(np.maximum(0.0, acc_v / C))
    T = acc_c
    pv = 1.0 if V[0] == 0.0 else (int(np.sum(V[1:] >= V[0])) + 1) / (n_perm + 1)
    pc = (int(np.sum(T[1:] >= T[0])) + 1) / (n_perm + 1)
    return dict(V_u=float(V[0]), p_V_u=pv, CvM=float(T[0]), p_CvM=pc,
                loss_signed=float(np.mean(signed)), n_cells=C)


# ------------------------------------------------------------------ multiplicity
def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    r = p[o] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


def by(p: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli: BH times the harmonic number, valid under any dependence."""
    m = len(p)
    return np.minimum(bh(p) * np.sum(1.0 / np.arange(1, m + 1)), 1.0)


# ------------------------------------------------------------------ intervals
def bootstrap_paired(cells_paired: List[np.ndarray], n_boot: int = 4000, seed: int = 0,
                     alpha: float = 0.05) -> Dict[str, float]:
    """Percentile intervals for V and for the signed mean loss.

    Seeds are resampled within each cell (cells are the fixed design); for V the
    noise correction is recomputed on every resample."""
    rng = np.random.default_rng(seed)
    C = len(cells_paired)
    accV = np.zeros(n_boot)
    accS = np.zeros(n_boot)
    for D in cells_paired:
        n = len(D)
        B = D[rng.integers(0, n, size=(n_boot, n))]
        s2 = B.var(1, ddof=1) / n if n > 1 else 0.0
        accV += B.mean(1) ** 2 - s2
        accS += B.mean(1)
    V = np.sqrt(np.maximum(0.0, accV / C))
    S = accS / C
    q = [alpha / 2, 1 - alpha / 2]
    return dict(V_lo=float(np.quantile(V, q[0])), V_hi=float(np.quantile(V, q[1])),
                loss_lo=float(np.quantile(S, q[0])), loss_hi=float(np.quantile(S, q[1])))


# ------------------------------------------------------------------ exchangeability counterexample
def signflip_counterexample(n_cells: int = 14, n_seeds: int = 20, n_rep: int = 2000,
                            n_perm: int = 999, seed: int = 0) -> Dict[str, float]:
    """Rejection rates at 5% when pairs are drawn uniformly from
    {(0, .5), (.5, 1), (1, 0)}: equal marginals, asymmetric paired difference.

    Returns the rate of the paired sign-flip test (not guaranteed under this
    null) and of the disjoint-seed permutation tests (exact under it)."""
    rng = np.random.default_rng(seed)
    pairs = np.array([[0.0, 0.5], [0.5, 1.0], [1.0, 0.0]])
    rej = dict(signflip=0, V_u=0, CvM=0)
    for _ in range(n_rep):
        draws = pairs[rng.integers(0, 3, size=(n_cells, n_seeds))]   # (C, S, 2)
        # paired sign-flip with the noise-corrected V
        acc = np.zeros(n_perm + 1)
        for c in range(n_cells):
            D = draws[c, :, 0] - draws[c, :, 1]
            s = rng.choice([-1.0, 1.0], size=(n_perm + 1, n_seeds))
            s[0] = 1.0
            Xs = s * D
            acc += Xs.mean(1) ** 2 - Xs.var(1, ddof=1) / n_seeds
        V = np.sqrt(np.maximum(0.0, acc / n_cells))
        p = 1.0 if V[0] == 0 else (np.sum(V[1:] >= V[0]) + 1) / (n_perm + 1)
        rej["signflip"] += p <= 0.05
        # disjoint seeds: X from the first half, Y from the second half
        h = n_seeds // 2
        cells = [(draws[c, :h, 0], draws[c, h:, 1]) for c in range(n_cells)]
        r = permutation_tests(cells, n_perm=n_perm, seed=int(rng.integers(1 << 31)))
        rej["V_u"] += r["p_V_u"] <= 0.05
        rej["CvM"] += r["p_CvM"] <= 0.05
    return {k: v / n_rep for k, v in rej.items()}
