"""Anchor optimisers with analytically known invariance properties.

These are not the subject of the study; they are the *instrument calibration*.
Every algorithm here has invariances that can be read off its update rule, so
the audit can be checked against ground truth before it is pointed at
automatically designed optimisers whose invariances nobody knows.

Distributional invariance of the performance measure (the notion that governs
generalisation; see the tables at the bottom of this module for the pathwise
counterpart):

    algorithm        translation  rotation  scaling  rescale f  monotone f
    random search         yes        yes      yes       yes        yes
    DE/rand/1/bin         yes        NO       yes       yes        yes
    DE/rand/1/exp         yes        NO       yes       yes        yes
    DE/rand/1/arith       yes        yes      yes       yes        yes
    DE/best/1/bin         yes        NO       yes       yes        yes
    jDE                   yes        NO       yes       yes        yes
    SHADE                 yes        NO       yes       yes        NO
    LSHADE                yes        NO       yes       yes        NO
    PSO                   yes        NO       yes       yes        yes
    CMA-ES                yes        yes      yes       yes        yes

The two entries that matter most are the ones nobody measures: SHADE and LSHADE
weight their memory updates by ratios of fitness *differences*, which cancels an
affine re-scaling of the objective but not a general monotone one.

Implementation constraints.

*Common random numbers.*  The audit compares a run on ``f`` with a run on
``h.f`` under the same random stream, so a pathwise-equivariant algorithm must
produce a bit-identical transformed trajectory.  That requires every algorithm
to consume the stream in a data-independent pattern.  Where canonical
pseudo-code uses rejection sampling (SHADE's Cauchy draw for F) it is replaced by
a deterministic clamp, which leaves the adaptation behaviour intact while keeping
the draw count fixed.

*No self-reported incumbent.*  Performance is recorded by the problem itself
from every evaluation (see :class:`invaudit.metrics.Recorder`).  Optimisers
therefore only search; they never nominate a best point.  Anything that tracks a
best value below does so because the algorithm's *own update* needs it (PSO's
global attractor), not for bookkeeping.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .problem import Problem, sample_in_ball

__all__ = ["OPTIMIZERS", "get_optimizer", "optimizer_names",
           "STRONG_EQUIVARIANCE", "DISTRIBUTIONAL_INVARIANCE"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _distinct(R, N, k, gen, device):
    """(R, N, k) indices, distinct within a row and different from the row index."""
    r = torch.rand(R, N, N, generator=gen, device=device)
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    r = r.masked_fill(eye, float("inf"))
    return torch.argsort(r, dim=-1)[..., :k]


def _clip(X, problem: Problem):
    if not problem.bounded:
        return X
    return torch.max(torch.min(X, problem.ub), problem.lb)


def _init(problem: Problem, R, N, gen):
    X0 = sample_in_ball(R * N, problem.D, problem.r_init, gen,
                        problem.device, problem.dtype).view(R, N, problem.D)
    return problem.present(X0)


def _gather(X, idx):
    """Rows of X selected by idx: X (R, N, D), idx (R, M) -> (R, M, D)."""
    return torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, X.shape[-1]))


class Optimizer:
    """Common interface: batched over ``R`` independent runs of one problem."""

    name = "optimizer"
    default_pop = 50

    def __init__(self, pop_size: Optional[int] = None, **kw):
        self.pop_size = pop_size or self.default_pop
        self.kw = kw

    def run(self, problem: Problem, budget: int, seed: int, n_runs: int) -> None:
        """Search until ``budget`` evaluations per run have been spent."""
        raise NotImplementedError

    def __repr__(self):  # pragma: no cover
        return f"{self.name}(N={self.pop_size})"


# --------------------------------------------------------------------------
# random search
# --------------------------------------------------------------------------

class RandomSearch(Optimizer):
    name = "random_search"

    def run(self, problem, budget, seed, n_runs):
        dev, dt = problem.device, problem.dtype
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        R, N, D = n_runs, self.pop_size, problem.D
        problem.evaluate(_clip(_init(problem, R, N, gen), problem))
        fe = N
        while fe < budget:
            Xb = sample_in_ball(R * N, D, problem.r_init, gen, dev, dt).view(R, N, D)
            problem.evaluate(_clip(problem.present(Xb), problem))
            fe += N


# --------------------------------------------------------------------------
# differential evolution family
# --------------------------------------------------------------------------

def _crossover(X, V, CR, scheme, gen, dev):
    """CR is (R, N, 1); returns the trial population."""
    R, N, D = X.shape
    if scheme == "arith":
        # u = x + K (v - x); rotation invariant (per-individual scalar blend)
        K = torch.rand(R, N, 1, generator=gen, device=dev, dtype=X.dtype)
        return X + K * (V - X)
    if scheme == "bin":
        mask = torch.rand(R, N, D, generator=gen, device=dev, dtype=X.dtype) < CR
        jrand = torch.randint(0, D, (R, N, 1), generator=gen, device=dev)
        mask = mask.scatter(2, jrand, True)
        return torch.where(mask, V, X)
    if scheme == "exp":
        start = torch.randint(0, D, (R, N, 1), generator=gen, device=dev)
        u = torch.rand(R, N, D, generator=gen, device=dev, dtype=X.dtype)
        # length of the copied segment: leading run of successes, at least 1
        run = torch.cumprod((u < CR).to(X.dtype), dim=2)
        L = 1 + run.sum(dim=2, keepdim=True)
        offs = torch.arange(D, device=dev).view(1, 1, D)
        mask = ((offs - start) % D) < L
        return torch.where(mask, V, X)
    raise ValueError(scheme)


class DE(Optimizer):
    """Classic DE with a fixed strategy and fixed control parameters."""

    def __init__(self, base="rand", scheme="bin", F=0.5, CR=0.9, pop_size=None):
        super().__init__(pop_size)
        self.base = base
        self.scheme = scheme
        self.F = F
        self.CR = CR
        self.name = f"DE/{base}/1/{scheme}"

    def run(self, problem, budget, seed, n_runs):
        dev, dt = problem.device, problem.dtype
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        R, N, D = n_runs, self.pop_size, problem.D
        ar = torch.arange(R, device=dev)
        X = _clip(_init(problem, R, N, gen), problem)
        f = problem.evaluate(X)
        fe = N
        CR = torch.full((R, N, 1), self.CR, device=dev, dtype=dt)
        while fe < budget:
            idx = _distinct(R, N, 3, gen, dev)
            g1, g2, g3 = (_gather(X, idx[..., k]) for k in range(3))
            if self.base == "rand":
                V = g1 + self.F * (g2 - g3)
            else:  # best
                V = X[ar, f.argmin(1)].unsqueeze(1) + self.F * (g2 - g3)
            U = _clip(_crossover(X, V, CR, self.scheme, gen, dev), problem)
            fu = problem.evaluate(U)
            fe += N
            better = fu <= f
            X = torch.where(better.unsqueeze(-1), U, X)
            f = torch.where(better, fu, f)


class JDE(Optimizer):
    """Brest et al. self-adaptive F and CR; purely comparison based."""

    name = "jDE"

    def run(self, problem, budget, seed, n_runs):
        dev, dt = problem.device, problem.dtype
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        R, N, D = n_runs, self.pop_size, problem.D
        tau1 = tau2 = 0.1
        F = torch.full((R, N, 1), 0.5, device=dev, dtype=dt)
        CR = torch.full((R, N, 1), 0.9, device=dev, dtype=dt)
        X = _clip(_init(problem, R, N, gen), problem)
        f = problem.evaluate(X)
        fe = N
        while fe < budget:
            u = torch.rand(R, N, 4, generator=gen, device=dev, dtype=dt)
            Fn = torch.where(u[..., 0:1] < tau1, 0.1 + 0.9 * u[..., 1:2], F)
            CRn = torch.where(u[..., 2:3] < tau2, u[..., 3:4], CR)
            idx = _distinct(R, N, 3, gen, dev)
            g = [_gather(X, idx[..., k]) for k in range(3)]
            V = g[0] + Fn * (g[1] - g[2])
            U = _clip(_crossover(X, V, CRn, "bin", gen, dev), problem)
            fu = problem.evaluate(U)
            fe += N
            better = fu <= f
            X = torch.where(better.unsqueeze(-1), U, X)
            f = torch.where(better, fu, f)
            F = torch.where(better.unsqueeze(-1), Fn, F)
            CR = torch.where(better.unsqueeze(-1), CRn, CR)


class SHADE(Optimizer):
    """Success-history adaptive DE (optionally with linear population reduction).

    The memory update uses the Lehmer mean of successful parameters weighted by
    ``delta f / sum(delta f)``.  Those weights are invariant to ``a*y + b`` but
    not to a general monotone re-parameterisation of the objective - the single
    most consequential invariance break in the modern DE literature, and the one
    this audit is built to expose.

    Deviation from the published pseudo-code: the Cauchy draw for F is clamped
    instead of resampled, so that the number of random draws does not depend on
    the data.  Without this the common-random-number pairing would drift and the
    measured defect would be contaminated by stream desynchronisation.
    """

    name = "SHADE"

    def __init__(self, pop_size=None, H=10, p=0.11, linear_reduction=False,
                 min_pop=4, name=None):
        super().__init__(pop_size)
        self.H = H
        self.p = p
        self.linear_reduction = linear_reduction
        self.min_pop = min_pop
        if name:
            self.name = name

    def run(self, problem, budget, seed, n_runs):
        dev, dt = problem.device, problem.dtype
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        R, N0, D = n_runs, self.pop_size, problem.D
        MF = torch.full((R, self.H), 0.5, device=dev, dtype=dt)
        MCR = torch.full((R, self.H), 0.5, device=dev, dtype=dt)
        k = torch.zeros(R, dtype=torch.long, device=dev)
        # Circular archive, filled strictly sequentially so that slots [0, a_len)
        # are exactly the written ones.  Sampling an unwritten slot would hand the
        # algorithm the origin of the coordinate system and silently destroy its
        # translation invariance.
        A = torch.zeros(R, N0, D, device=dev, dtype=dt)
        a_len = torch.zeros(R, dtype=torch.long, device=dev)
        a_ptr = torch.zeros(R, dtype=torch.long, device=dev)
        X = _clip(_init(problem, R, N0, gen), problem)
        f = problem.evaluate(X)
        fe = N = N0
        while fe < budget:
            ridx = torch.randint(0, self.H, (R, N), generator=gen, device=dev)
            mcr = torch.gather(MCR, 1, ridx).unsqueeze(-1)
            mf = torch.gather(MF, 1, ridx).unsqueeze(-1)
            CR = (mcr + 0.1 * torch.randn(R, N, 1, generator=gen, device=dev,
                                          dtype=dt)).clamp(0.0, 1.0)
            cau = torch.tan(math.pi * (torch.rand(R, N, 1, generator=gen, device=dev,
                                                  dtype=dt) - 0.5))
            F = (mf + 0.1 * cau).clamp(1e-6, 1.0)

            order = torch.argsort(f, dim=1)
            pnum = max(2, int(round(self.p * N)))
            psel = torch.randint(0, pnum, (R, N), generator=gen, device=dev)
            xpb = _gather(X, torch.gather(order, 1, psel))

            idx = _distinct(R, N, 2, gen, dev)
            xr1 = _gather(X, idx[..., 0])
            # r2 is drawn from the population united with the filled archive
            pool = torch.cat([X, A], dim=1)
            span = (N + a_len).clamp(min=1)
            ru = torch.rand(R, N, generator=gen, device=dev, dtype=dt)
            r2 = (ru * span.unsqueeze(1).to(dt)).long().clamp(max=pool.shape[1] - 1)
            xr2 = _gather(pool, r2)

            V = X + F * (xpb - X) + F * (xr1 - xr2)
            U = _clip(_crossover(X, V, CR, "bin", gen, dev), problem)
            fu = problem.evaluate(U)
            fe += N
            better = fu < f
            delta = (f - fu).clamp_min(0.0)

            nsucc = better.sum(1)
            size = A.shape[1]
            if int(nsucc.max()) > 0:
                for r in range(R):
                    sel = better[r].nonzero(as_tuple=True)[0]
                    kk = sel.numel()
                    if kk:
                        slots = (a_ptr[r] + torch.arange(kk, device=dev)) % size
                        A[r, slots] = X[r, sel]
                        a_ptr[r] = (a_ptr[r] + kk) % size
                a_len = torch.minimum(a_len + nsucc, torch.full_like(a_len, size))

            # the difference-weighted memory update: this is where monotone
            # invariance is lost
            wsum = (delta * better).sum(1, keepdim=True)
            has = wsum.squeeze(1) > 0
            w = torch.where(wsum > 0, delta * better / wsum.clamp_min(1e-300),
                            torch.zeros_like(delta))
            Fs, CRs = F.squeeze(-1), CR.squeeze(-1)
            lehmer_F = (w * Fs * Fs).sum(1) / (w * Fs).sum(1).clamp_min(1e-300)
            lehmer_CR = (w * CRs * CRs).sum(1) / (w * CRs).sum(1).clamp_min(1e-300)
            if bool(has.any()):
                rows = has.nonzero(as_tuple=True)[0]
                MF[rows, k[rows]] = lehmer_F[rows]
                MCR[rows, k[rows]] = lehmer_CR[rows]
                k[rows] = (k[rows] + 1) % self.H

            X = torch.where(better.unsqueeze(-1), U, X)
            f = torch.where(better, fu, f)

            if self.linear_reduction:
                Nnew = int(round(N0 + (self.min_pop - N0) * fe / budget))
                Nnew = max(self.min_pop, min(N, Nnew))
                if Nnew < N:
                    keep = torch.argsort(f, dim=1)[:, :Nnew]
                    X = _gather(X, keep)
                    f = torch.gather(f, 1, keep)
                    N = Nnew


# --------------------------------------------------------------------------
# swarm
# --------------------------------------------------------------------------

class PSO(Optimizer):
    """Constriction PSO.  Per-coordinate random weights break rotation invariance."""

    name = "PSO"

    def run(self, problem, budget, seed, n_runs):
        dev, dt = problem.device, problem.dtype
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        R, N, D = n_runs, self.pop_size, problem.D
        ar = torch.arange(R, device=dev)
        w, c1, c2 = 0.7298, 1.49618, 1.49618
        vmax = 0.2 * (problem.ub - problem.lb)
        X = _clip(_init(problem, R, N, gen), problem)
        V = torch.zeros_like(X)
        f = problem.evaluate(X)
        fe = N
        P, fp = X.clone(), f.clone()
        gi = fp.argmin(1)
        G = P[ar, gi].clone()
        fg = fp.gather(1, gi[:, None]).squeeze(1)
        while fe < budget:
            r1 = torch.rand(R, N, D, generator=gen, device=dev, dtype=dt)
            r2 = torch.rand(R, N, D, generator=gen, device=dev, dtype=dt)
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (G.unsqueeze(1) - X)
            V = torch.max(torch.min(V, vmax), -vmax)
            X = _clip(X + V, problem)
            f = problem.evaluate(X)
            fe += N
            imp = f < fp
            P = torch.where(imp.unsqueeze(-1), X, P)
            fp = torch.where(imp, f, fp)
            gi = fp.argmin(1)
            cand = fp.gather(1, gi[:, None]).squeeze(1)
            upd = cand < fg
            G = torch.where(upd[:, None], P[ar, gi], G)
            fg = torch.where(upd, cand, fg)


# --------------------------------------------------------------------------
# CMA-ES
# --------------------------------------------------------------------------

class CMAES(Optimizer):
    """Batched (mu/mu_w, lambda)-CMA-ES.  Rank based, hence monotone invariant."""

    name = "CMA-ES"

    def __init__(self, pop_size=None, sigma_frac=0.3):
        super().__init__(pop_size)
        self.sigma_frac = sigma_frac
        self._explicit_pop = pop_size

    def run(self, problem, budget, seed, n_runs):
        dev, dt = problem.device, problem.dtype
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        R, D = n_runs, problem.D
        lam = self._explicit_pop or (4 + int(3 * math.log(D)))
        mu = lam // 2
        wr = torch.tensor([math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)],
                          device=dev, dtype=dt)
        wr = wr / wr.sum()
        mueff = float(1.0 / (wr ** 2).sum())
        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2 / ((D + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (D + 1)) - 1) + cs
        chiN = math.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D * D))

        m = _init(problem, R, lam, gen).mean(dim=1)
        sigma = torch.full((R,), float(self.sigma_frac *
                                       (problem.ub - problem.lb).mean().item()),
                           device=dev, dtype=dt)
        C = torch.eye(D, device=dev, dtype=dt).expand(R, D, D).clone()
        pc = torch.zeros(R, D, device=dev, dtype=dt)
        ps = torch.zeros(R, D, device=dev, dtype=dt)
        B = torch.eye(D, device=dev, dtype=dt).expand(R, D, D).clone()
        Dg = torch.ones(R, D, device=dev, dtype=dt)

        fe, gcount = 0, 0
        while fe < budget:
            z = torch.randn(R, lam, D, generator=gen, device=dev, dtype=dt)
            y = torch.einsum("rij,rlj->rli", B * Dg.unsqueeze(1), z)
            X = _clip(m.unsqueeze(1) + sigma.view(R, 1, 1) * y, problem)
            f = problem.evaluate(X)
            fe += lam
            gcount += 1
            top = torch.argsort(f, dim=1)[:, :mu]
            ysel = _gather(y, top)
            yw = (wr.view(1, mu, 1) * ysel).sum(1)
            m = m + sigma.unsqueeze(-1) * yw

            Cinv_half = torch.einsum("rij,rj,rkj->rik", B, 1.0 / Dg, B)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * \
                torch.einsum("rij,rj->ri", Cinv_half, yw)
            psn = ps.norm(dim=-1)
            hsig = (psn / math.sqrt(1 - (1 - cs) ** (2 * gcount)) / chiN <
                    (1.4 + 2 / (D + 1))).to(dt)
            pc = (1 - cc) * pc + hsig.unsqueeze(-1) * math.sqrt(cc * (2 - cc) * mueff) * yw
            rank1 = torch.einsum("ri,rj->rij", pc, pc)
            rankmu = torch.einsum("l,rli,rlj->rij", wr, ysel, ysel)
            delta = (1 - hsig) * cc * (2 - cc)
            C = ((1 - c1 - cmu) * C
                 + c1 * (rank1 + delta.view(R, 1, 1) * C)
                 + cmu * rankmu)
            sigma = sigma * torch.exp((cs / damps) * (psn / chiN - 1)).clamp(0.5, 2.0)
            C = 0.5 * (C + C.transpose(1, 2))
            evals, B = torch.linalg.eigh(C)
            Dg = torch.sqrt(evals.clamp_min(1e-300))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def _factory():
    return {
        "random_search": lambda: RandomSearch(),
        "DE/rand/1/bin": lambda: DE("rand", "bin"),
        "DE/rand/1/exp": lambda: DE("rand", "exp"),
        "DE/rand/1/arith": lambda: DE("rand", "arith"),
        "DE/best/1/bin": lambda: DE("best", "bin"),
        "jDE": lambda: JDE(),
        "SHADE": lambda: SHADE(),
        "LSHADE": lambda: SHADE(linear_reduction=True, name="LSHADE"),
        "PSO": lambda: PSO(),
        "CMA-ES": lambda: CMAES(),
    }


OPTIMIZERS = _factory()


def optimizer_names():
    return list(OPTIMIZERS)


def get_optimizer(name: str) -> Optimizer:
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimiser {name!r}")
    return OPTIMIZERS[name]()


# --------------------------------------------------------------------------
# ground truth used to validate the instrument
# --------------------------------------------------------------------------
#
# Two notions have to be kept apart, and conflating them is the single easiest
# way to misread an audit.
#
# *Pathwise (strong) equivariance*: for a fixed random stream the algorithm's
# trajectory on ``h.f`` is exactly ``h`` applied to its trajectory on ``f``.
# This can only hold when the algorithm's randomness is coordinate-free.  Any
# rule that draws one random number per coordinate - binomial crossover, PSO's
# per-dimension weights - fails it even under a mere relabelling of coordinates,
# and CMA-ES fails it under rotation purely because an eigendecomposition fixes
# eigenvector signs arbitrarily.  A pathwise reading therefore localises *where*
# coordinate-tied randomness enters, and nothing more.
#
# *Distributional invariance*: the law of the performance measure is unchanged.
# This is the notion that governs generalisation, and it is the headline score.
#
# A coordinate-wise algorithm is distributionally permutation-invariant but
# pathwise permutation-dependent; the gap between the two tables below is
# exactly that effect.

_ALL = ("ambient_translation", "rotation", "coordinate_permutation",
        "domain_scaling_up", "fitness_rescale", "monotone_spline")


def _row(**overrides):
    row = {k: 1 for k in _ALL}
    row.update(overrides)
    return row


STRONG_EQUIVARIANCE = {
    # pathwise; 1 means a CRN pair must agree bit for bit
    "random_search":   _row(),
    "DE/rand/1/bin":   _row(rotation=0, coordinate_permutation=0),
    "DE/rand/1/exp":   _row(rotation=0, coordinate_permutation=0),
    "DE/rand/1/arith": _row(),
    "DE/best/1/bin":   _row(rotation=0, coordinate_permutation=0),
    "jDE":             _row(rotation=0, coordinate_permutation=0),
    "SHADE":           _row(rotation=0, coordinate_permutation=0, monotone_spline=0),
    "LSHADE":          _row(rotation=0, coordinate_permutation=0, monotone_spline=0),
    "PSO":             _row(rotation=0, coordinate_permutation=0),
    "CMA-ES":          _row(rotation=0, coordinate_permutation=0),
}

DISTRIBUTIONAL_INVARIANCE = {
    # the notion that governs generalisation; 1 means no systematic performance shift
    "random_search":   _row(),
    "DE/rand/1/bin":   _row(rotation=0),
    # exponential crossover copies *contiguous* coordinate blocks, so which
    # variables are neighbours matters: not permutation invariant.  The original
    # version of this table said otherwise; the audit caught it.
    "DE/rand/1/exp":   _row(rotation=0, coordinate_permutation=0),
    "DE/rand/1/arith": _row(),
    "DE/best/1/bin":   _row(rotation=0),
    "jDE":             _row(rotation=0),
    "SHADE":           _row(rotation=0, monotone_spline=0),
    "LSHADE":          _row(rotation=0, monotone_spline=0),
    "PSO":             _row(rotation=0),
    "CMA-ES":          _row(),
}
