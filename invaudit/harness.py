"""The invariance harness: canonicalisation as a wrapper.

An algorithm cannot be made invariant by asking it nicely.  It can be made
invariant by changing the coordinates it is allowed to see.  The harness does
exactly that, from the outside, without touching the optimiser - which is what
makes it applicable to a reinforcement-learned policy, a language-model-generated
heuristic or a meta-tuned configuration alike.

Three layers, in increasing order of how much they assume.

``BoxFrame``
    The optimiser is shown the unit box.  A query ``u`` is mapped to
    ``c + w * u`` where ``c`` and ``w`` are the centre and half-width of the
    presented box.  Because the audit's translation and dilation probes move the
    box together with the function, the harnessed problem is *literally
    identical* across such a pair, so the violation is driven to zero
    algebraically rather than statistically.  Cost: nothing.

``RankFitness``
    The optimiser is shown the quantile of each objective value within a sliding
    window of recent evaluations, instead of the value itself.  Strictly
    increasing maps preserve ranks exactly, so this removes every
    objective-space violation by construction - including the ones that a
    difference-weighted memory such as SHADE's is subject to.  It is a no-op for
    an algorithm that already uses only within-generation ranks (CMA-ES), which
    is a useful internal check.  Cost: one sort per generation.

``PCAFrame`` (the dial)
    Rotation has no counterpart in the box: there is no frame to read off the
    problem statement, so one has to be estimated from data.  A short warm-up
    phase samples the ball, the covariance of the better half fixes an
    orthogonal frame, and the optimiser then works in it.  This buys rotation
    invariance at the price of the very thing coordinate-wise search is good at,
    which is why it is offered as a dial rather than a default.

What the harness cannot do is recover information that was destroyed before it
arrived.  Adding 1e6 to objective values of order 1e-8 annihilates them in the
floating-point evaluation itself; ranking afterwards ranks noise.  The harness
removes violations caused by an algorithm's logic, never those caused by
arithmetic.
"""

from __future__ import annotations

from typing import Optional

import torch

from .problem import Problem, sample_in_ball

__all__ = ["HarnessedProblem", "harness"]


class _RankState:
    """Empirical CDF of recent objective values, per run.

    The reference set has to be a *sliding window*, not the whole history.  A
    cumulative reference looks appealing - it is the most stable estimate of the
    CDF - but it is useless in practice: once a run starts converging, almost
    every new point beats everything seen before, every quantile collapses to
    zero and selection loses its signal entirely.  A window keeps the resolution
    where the search currently is, at the cost of making a stored quantile go
    stale after a few generations.

    The window length is set from the first batch the optimiser evaluates, so it
    adapts to whatever population size the algorithm happens to use.
    """

    def __init__(self, R: int, device, dtype, window_mult: int = 20,
                 min_window: int = 256):
        self.R = R
        self.device = device
        self.dtype = dtype
        self.window_mult = window_mult
        self.min_window = min_window
        self.buf: Optional[torch.Tensor] = None
        self.filled = 0
        self.ptr = 0

    def _ensure(self, N: int):
        if self.buf is None:
            W = max(self.min_window, self.window_mult * N)
            self.buf = torch.empty(self.R, W, device=self.device, dtype=self.dtype)

    def push_and_rank(self, y: torch.Tensor) -> torch.Tensor:
        R, N = y.shape
        self._ensure(N)
        W = self.buf.shape[1]
        idx = (self.ptr + torch.arange(N, device=self.device)) % W
        self.buf[:, idx] = y
        self.ptr = int((self.ptr + N) % W)
        self.filled = min(self.filled + N, W)
        hist = self.buf[:, :self.filled]
        srt, _ = torch.sort(hist, dim=1)
        yc = y.contiguous()
        lo = torch.searchsorted(srt.contiguous(), yc, right=False)
        hi = torch.searchsorted(srt.contiguous(), yc, right=True)
        # midrank keeps the map well behaved in the presence of ties
        return 0.5 * (lo + hi).to(y.dtype) / float(self.filled)


class _WorseCountState:
    """Time-consistent rank transform: minus the number of worse values seen.

    ``T_t(y) = -|{h in H_t : h > y}| / C`` with ``H_t`` the full history of the
    run and ``C`` a fixed constant.

    Proposition.  (i) ``T`` is invariant under any strictly increasing map
    applied to every objective value.  (ii) If ``y1`` was transformed at time
    ``t1`` and ``y2 <= y1`` is transformed at a later time ``t2 >= t1``, then
    ``T_t2(y2) <= T_t1(y1)``.

    Proof of (ii).  ``|{h in H_t2 : h > y2}| >= |{h in H_t2 : h > y1}|`` because
    ``y2 <= y1``, and ``>= |{h in H_t1 : h > y1}|`` because ``H_t1`` is a subset of
    ``H_t2``.  Negate.

    Consequence: an elitist rule that accepts a candidate whose transformed value
    does not exceed a *stored* one - DE's greedy replacement, PSO's personal-best
    update - never rejects a genuine improvement, however stale the stored value.
    A quantile against a sliding window, or against the whole history normalised
    by its size, offers no such guarantee; both were tried first and both make
    elitist algorithms stagnate.  The guarantee is one-sided: a worse candidate
    can be accepted if many still-worse points arrived since the stored value was
    computed, which is why DE and jDE still lose performance under this
    transform.  The fix is to stop recomputing: :class:`_OrderEmbeddingState`
    memoises the code of every value and is consistent in both directions.
    """

    def __init__(self, R: int, device, dtype, scale: float = 1.0e4):
        self.R = R
        self.device = device
        self.dtype = dtype
        self.scale = scale
        self.buf = torch.empty(R, 4096, device=device, dtype=dtype)
        self.n = 0

    def push_and_rank(self, y: torch.Tensor) -> torch.Tensor:
        R, N = y.shape
        if self.n + N > self.buf.shape[1]:
            new = torch.empty(R, max(2 * self.buf.shape[1], self.n + N),
                              device=self.device, dtype=self.dtype)
            new[:, :self.n] = self.buf[:, :self.n]
            self.buf = new
        self.buf[:, self.n:self.n + N] = y
        self.n += N
        srt, _ = torch.sort(self.buf[:, :self.n], dim=1)
        not_worse = torch.searchsorted(srt.contiguous(), y.contiguous(), right=True)
        worse = self.n - not_worse
        return -worse.to(y.dtype) / self.scale


class _OrderEmbeddingState:
    """Two-sided consistent rank transform: a persistent order embedding.

    Every distinct objective value ever seen in a run receives a code the first
    time it appears, and that code is never changed.  A new value is placed
    strictly between the codes of its neighbours among all previously coded
    values (new extremes extend the range by a fixed gap), so for *every* pair of
    values ever issued, at any two times, code order equals value order.  This is
    the order-maintenance problem, solved here with dyadic floating-point codes.

    Consequences.

    * Monotone invariance is exact: codes depend only on the relative order of
      values and on the order in which they arrived, both of which a strictly
      increasing map leaves untouched.
    * An algorithm that only ever *compares* objective values - including stored
      ones from arbitrarily old generations - takes exactly the decisions it would
      take on the raw objective.  Its trajectory is bit-identical; the harness is
      free for it.  This is the property the sliding-window quantile and the
      worse-count transform could not deliver (the first rejects improvements,
      the second is consistent in one direction only).
    * An algorithm that uses magnitudes - SHADE's difference-weighted memory - is
      handed rank-consistent surrogate magnitudes instead.  That is precisely the
      information that has to be withheld for it to become invariant.

    Codes live on an integer lattice of spacing 2**30 inside the float64 exact
    integer range, so values inserted into the same gap can be separated about
    thirty times before the gap is exhausted.  An exhausted gap falls back to a
    tie, and every such event is counted so that the audit can report it.
    """

    GAP = float(2 ** 30)

    def __init__(self, R: int, device, dtype):
        self.R = R
        self.device = device
        self.dtype = dtype
        self.V = [torch.empty(0, device=device, dtype=dtype) for _ in range(R)]
        self.C = [torch.empty(0, device=device, dtype=dtype) for _ in range(R)]
        self.exhausted = 0

    def _codes_for_run(self, r: int, yr: torch.Tensor) -> torch.Tensor:
        V, C = self.V[r], self.C[r]
        uniq, inv = torch.unique(yr, sorted=True, return_inverse=True)
        if V.numel() == 0:
            codes_u = torch.arange(uniq.numel(), device=self.device,
                                   dtype=self.dtype) * self.GAP
            self.V[r], self.C[r] = uniq, codes_u
            return codes_u[inv]

        n_old = V.numel()
        pos = torch.searchsorted(V, uniq)                       # left insertion point
        hit = (pos < n_old) & (V[pos.clamp(max=n_old - 1)] == uniq)
        codes_u = torch.empty_like(uniq)
        codes_u[hit] = C[pos[hit]]

        new = ~hit
        if bool(new.any()):
            pn = pos[new]
            # group the new values by the gap they fall into; pn is sorted because
            # uniq is sorted, so groups are consecutive runs
            _, counts = torch.unique_consecutive(pn, return_counts=True)
            starts = torch.cumsum(counts, 0) - counts
            m = torch.repeat_interleave(counts, counts).to(self.dtype)
            j = (torch.arange(pn.numel(), device=self.device)
                 - torch.repeat_interleave(starts, counts)).to(self.dtype)

            has_lo = pn > 0
            has_hi = pn < n_old
            lo = torch.where(has_lo, C[(pn - 1).clamp(min=0)], torch.zeros_like(m))
            hi = torch.where(has_hi, C[pn.clamp(max=n_old - 1)], torch.zeros_like(m))

            inside = lo + (j + 1.0) / (m + 1.0) * (hi - lo)
            below = hi - (m - j) * self.GAP                     # new best values
            above = lo + (j + 1.0) * self.GAP                   # new worst values
            code = torch.where(has_lo & has_hi, inside,
                               torch.where(has_hi, below, above))
            both = has_lo & has_hi
            self.exhausted += int((both & ((code <= lo) | (code >= hi))).sum())
            codes_u[new] = code

            V2 = torch.cat([V, uniq[new]])
            C2 = torch.cat([C, code])
            order = torch.argsort(V2)
            self.V[r], self.C[r] = V2[order], C2[order]
        return codes_u[inv]

    def push_and_rank(self, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(y)
        for r in range(y.shape[0]):
            out[r] = self._codes_for_run(r, y[r]) / self.GAP
        return out


class HarnessedProblem:
    """A canonicalised view of a problem.  Duck-types :class:`Problem`."""

    def __init__(self, inner: Problem, box_frame=True, rank_fitness=True,
                 pca_frame=False, pca_warmup_per_dim=20, n_runs=1,
                 rank_mode="order", rank_window_mult=20, seed=0):
        self.inner = inner
        self.box_frame = box_frame
        self.rank_fitness = rank_fitness
        self.pca_frame = pca_frame
        self.pca_warmup_per_dim = pca_warmup_per_dim
        self.n_runs = n_runs
        self.seed = seed
        self.rank_mode = rank_mode
        self.rank_window_mult = rank_window_mult
        self._rank = None
        if rank_fitness:
            if rank_mode == "order":
                self._rank = _OrderEmbeddingState(n_runs, inner.device, inner.dtype)
            elif rank_mode == "worse_count":
                self._rank = _WorseCountState(n_runs, inner.device, inner.dtype)
            elif rank_mode == "window":
                self._rank = _RankState(n_runs, inner.device, inner.dtype,
                                        window_mult=rank_window_mult)
            else:
                raise ValueError(f"unknown rank_mode {rank_mode!r}")
        self._B = None
        self.warmup_cost = 0
        if pca_frame:
            self._fit_pca_frame()

    @property
    def rank_ties_forced(self) -> int:
        """Number of order-embedding gaps exhausted (ties forced by precision)."""
        return getattr(self._rank, "exhausted", 0)

    # ---------------- Problem interface ----------------

    @property
    def spec(self):
        return self.inner.spec

    @property
    def D(self):
        return self.inner.D

    @property
    def device(self):
        return self.inner.device

    @property
    def dtype(self):
        return self.inner.dtype

    @property
    def bounded(self):
        return self.inner.bounded

    @property
    def r_init(self):
        return self.inner.r_init

    @property
    def x_opt(self):
        return self.inner.x_opt

    @property
    def _center(self):
        return 0.5 * (self.inner.lb + self.inner.ub)

    @property
    def _halfwidth(self):
        return 0.5 * (self.inner.ub - self.inner.lb)

    @property
    def lb(self):
        if not self.box_frame:
            return self.inner.lb
        return torch.full((self.D,), -1.0, device=self.device, dtype=self.dtype)

    @property
    def ub(self):
        if not self.box_frame:
            return self.inner.ub
        return torch.full((self.D,), 1.0, device=self.device, dtype=self.dtype)

    # ---------------- frame ----------------

    def _fit_pca_frame(self):
        """Estimate an orthogonal frame from a short random warm-up."""
        n = self.pca_warmup_per_dim * self.D
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + 9901)
        Xb = sample_in_ball(n, self.D, self.inner.r_init, gen,
                            self.device, self.dtype)
        X = self.inner.present(Xb)
        y = self.inner.evaluate(X)
        self.warmup_cost = n
        k = max(self.D + 1, n // 4)
        idx = torch.argsort(y)[:k]
        Z = X[idx]
        Z = Z - Z.mean(0, keepdim=True)
        C = Z.T @ Z / max(k - 1, 1)
        C = 0.5 * (C + C.T)
        _, evecs = torch.linalg.eigh(C)
        self._B = evecs

    def to_inner(self, U: torch.Tensor) -> torch.Tensor:
        X = U
        if self._B is not None:
            X = X @ self._B.T
        if self.box_frame:
            X = self._center + self._halfwidth * X
        return X

    def from_inner(self, X: torch.Tensor) -> torch.Tensor:
        U = X
        if self.box_frame:
            U = (U - self._center) / self._halfwidth
        if self._B is not None:
            U = U @ self._B
        return U

    # ---------------- objectives ----------------

    def base_value(self, U: torch.Tensor) -> torch.Tensor:
        return self.inner.base_value(self.to_inner(U))

    def evaluate(self, U: torch.Tensor) -> torch.Tensor:
        y = self.inner.evaluate(self.to_inner(U))
        if self._rank is not None:
            shape = y.shape
            flat = y.reshape(shape[0], -1) if y.dim() > 1 else y.reshape(1, -1)
            r = self._rank.push_and_rank(flat)
            y = r.reshape(shape)
        return y

    # ---------------- initialisation ----------------

    def init_population(self, n: int, generator: torch.Generator) -> torch.Tensor:
        return self.inner.init_population(n, generator)

    def present(self, X_base: torch.Tensor) -> torch.Tensor:
        return self.from_inner(self.inner.present(X_base))

    def __repr__(self):  # pragma: no cover
        layers = [n for n, on in [("box", self.box_frame),
                                  ("rank", self.rank_fitness),
                                  ("pca", self.pca_frame)] if on]
        return f"Harnessed({self.inner!r}, layers={'+'.join(layers) or 'none'})"


def harness(problem: Problem, n_runs: int, *, box_frame=True, rank_fitness=True,
            pca_frame=False, rank_mode="order", rank_window_mult=20,
            seed=0) -> HarnessedProblem:
    return HarnessedProblem(problem, box_frame=box_frame, rank_fitness=rank_fitness,
                            pca_frame=pca_frame, n_runs=n_runs, rank_mode=rank_mode,
                            rank_window_mult=rank_window_mult, seed=seed)
