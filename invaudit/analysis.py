"""Statistics for the audit.

Everything here operates on the long format shared by the experiments:

    family, agent, fid, instance, probe, seed, A

Three quantities are defined, each chosen to survive a specific objection.

Violation ``V(m, g)`` - how much method ``m``'s performance moves under symmetry
``g``.  The signed mean difference is not enough: a method that gains on some
functions and loses on others under a rotation averages out to zero while being
anything but invariant.  The mean absolute run-level difference is not usable
either, because for methods without common random numbers it is dominated by
ordinary seed-to-seed noise.  The score is therefore the *noise-corrected RMS of
the per-cell mean shift*:

    D_cs     = A_identity(c, s) - A_probe(c, s)      seed-matched, for cell c and seed s
    d_c      = mean_s D_cs,   s_c^2 = var_s(D_cs) / n
    V        = sqrt(max(0, mean_c (d_c^2 - s_c^2)))

``E[d_c^2] = delta_c^2 + sigma_c^2``, so ``V`` estimates the RMS of the systematic
shifts ``delta_c``.  Pairing by seed matters: an unpaired variance, tried first,
overestimated the noise of runs coupled by common random numbers so strongly
that known violations of the anchors vanished.

OOD gap ``G(m)`` - how much a method's standing deteriorates from the functions
it was trained on to the functions held out from it.  ECDF areas differ by an
order of magnitude between functions, so the primary gap is rank based: methods
are ranked per function, and ``G = mean rank on held-out functions - mean rank on
training functions`` (positive = worse out of distribution).  It needs no
reference algorithm and is insensitive to per-function scale.

The law - association between ``V`` and ``G`` across methods, reported as a
Spearman correlation with a permutation p-value and a bootstrap interval over
methods, and additionally within the automatically designed methods alone, so
that the result cannot be an artefact of learned-versus-hand-designed contrast.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["audit_to_long", "violation_scores", "violation_significance", "violation_matrix",
           "aggregate_violation", "ood_gap", "law_test", "bootstrap_ci"]

CELL = ["fid", "instance"]


def audit_to_long(df: pd.DataFrame, probes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert the output of :func:`invaudit.audit.run_audit` to the long format.

    ``run_audit`` stores one row per (cell, probe, run) with the untransformed score
    repeated in ``A_base``; the analysis functions expect one row per score, with the
    untransformed runs labelled ``identity``.
    """
    keys = ["optimizer", "function", "instance", "run"]
    base = (df.drop_duplicates(keys)[keys + ["A_base"]]
            .rename(columns={"A_base": "A"}).assign(probe="identity"))
    pr = df if probes is None else df[df.probe.isin(probes)]
    pr = pr[pr.probe != "identity"][keys + ["probe", "A_probe"]].rename(columns={"A_probe": "A"})
    return (pd.concat([base, pr], ignore_index=True)
            .rename(columns={"optimizer": "agent", "function": "fid", "run": "seed"}))


def _paired_cells(df: pd.DataFrame, agent: str, probe: str, agent_col: str = "agent"):
    """Per cell, the seed-matched differences A_identity - A_probe.

    Runs of a cell are paired by seed.  For optimisers audited under common
    random numbers the pairing removes most of the run-to-run variance; for
    optimisers whose stream is not coupled to the transformation it costs
    nothing, because the paired variance then reduces to the unpaired one.
    """
    a = df[(df[agent_col] == agent) & (df.probe == "identity")]
    b = df[(df[agent_col] == agent) & (df.probe == probe)]
    m = a.merge(b, on=CELL + ["seed"], suffixes=("_id", "_pr"))
    return [g["A_id"].to_numpy(float) - g["A_pr"].to_numpy(float)
            for _, g in m.groupby(CELL)]


def _v_from_diffs(cells) -> float:
    acc = 0.0
    for D in cells:
        n = len(D)
        s2 = D.var(ddof=1) / n if n > 1 else 0.0
        acc += D.mean() ** 2 - s2
    return float(np.sqrt(max(0.0, acc / len(cells))))


def violation_scores(df: pd.DataFrame, probes: Sequence[str],
                     agent_col: str = "agent") -> pd.DataFrame:
    """Long table: agent, probe, V (noise-corrected RMS shift), V_signed, n_cells."""
    rows = []
    for probe in probes:
        if probe == "identity":
            continue
        for agent in sorted(set(df.loc[df.probe == probe, agent_col])):
            cells = _paired_cells(df, agent, probe, agent_col)
            if not cells:
                continue
            means = np.array([D.mean() for D in cells])
            rows.append(dict(agent=agent, probe=probe, V=_v_from_diffs(cells),
                             V_signed=float(means.mean()),
                             raw_rms=float(np.sqrt(np.mean(means ** 2))),
                             V_path=float(np.mean([np.abs(D).mean() for D in cells])),
                             n_cells=len(cells)))
    return pd.DataFrame(rows)


def violation_significance(df: pd.DataFrame, probes: Sequence[str],
                           n_perm: int = 4000, seed: int = 0,
                           fdr: float = 0.05, agent_col: str = "agent") -> pd.DataFrame:
    """Sign-flip permutation test of distributional invariance, FDR controlled.

    Under the null hypothesis the seed-matched difference of a cell is
    symmetric about zero, so flipping the sign of each difference at random
    leaves its distribution unchanged.  V is recomputed - noise correction
    included - on every flip pattern, and the p-value is the fraction of
    patterns whose V reaches the observed one.  Benjamini-Hochberg control of
    the false discovery rate is applied across all agent-probe tests.  Its
    calibration is checked empirically on the anchors, where invariance is
    known: see the validation experiment.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for probe in probes:
        if probe == "identity":
            continue
        for agent in sorted(set(df.loc[df.probe == probe, agent_col])):
            cells = _paired_cells(df, agent, probe, agent_col)
            if not cells:
                continue
            acc = np.zeros(n_perm + 1)
            for D in cells:
                n = len(D)
                signs = rng.choice([-1.0, 1.0], size=(n_perm + 1, n))
                signs[0] = 1.0
                X = signs * D
                s2 = X.var(1, ddof=1) / n if n > 1 else 0.0
                acc += X.mean(1) ** 2 - s2
            V_all = np.sqrt(np.maximum(0.0, acc / len(cells)))
            obs = float(V_all[0])
            p = 1.0 if obs == 0.0 else (int(np.sum(V_all[1:] >= obs)) + 1) / (n_perm + 1)
            rows.append(dict(agent=agent, probe=probe, V=obs, p=p))
    out = pd.DataFrame(rows)
    m = len(out)
    order = np.argsort(out["p"].to_numpy())
    ranked = out["p"].to_numpy()[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    qvals = np.empty(m)
    qvals[order] = np.minimum(q, 1.0)
    out["q"] = qvals
    out["significant"] = out["q"] <= fdr
    return out


def violation_matrix(scores: pd.DataFrame, value: str = "V",
                     order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    m = scores.pivot(index="agent", columns="probe", values=value)
    if order is not None:
        m = m.loc[[a for a in order if a in m.index]]
    return m


def aggregate_violation(scores: pd.DataFrame, probes: Sequence[str]) -> pd.Series:
    """Root mean square of V over a set of probes, per agent."""
    s = scores[scores.probe.isin(probes)]
    return s.groupby("agent")["V"].apply(lambda v: float(np.sqrt(np.mean(v ** 2))))


def ood_gap(df_id: pd.DataFrame, df_ood: pd.DataFrame,
            agents: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Rank-based out-of-distribution gap, plus the ECDF-area version.

    ``df_id`` and ``df_ood`` hold untransformed runs (probe == identity) on the
    training and held-out functions respectively.
    """
    def mean_perf(d):
        d = d[d.probe == "identity"]
        return d.groupby(["agent", "fid"])["A"].mean().unstack("fid")

    pid, pood = mean_perf(df_id), mean_perf(df_ood)
    if agents is not None:
        agents = [a for a in agents if a in pid.index and a in pood.index]
        pid, pood = pid.loc[agents], pood.loc[agents]
    common = pid.index.intersection(pood.index)
    pid, pood = pid.loc[common], pood.loc[common]
    # rank 1 = best on that function; ties share the average rank
    rid = pid.rank(axis=0, ascending=False, method="average")
    rood = pood.rank(axis=0, ascending=False, method="average")
    out = pd.DataFrame({
        "rank_id": rid.mean(axis=1),
        "rank_ood": rood.mean(axis=1),
        "A_id": pid.mean(axis=1),
        "A_ood": pood.mean(axis=1),
    })
    out["G_rank"] = out["rank_ood"] - out["rank_id"]
    out["G_area"] = out["A_id"] - out["A_ood"]
    return out


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = fn(x[idx], y[idx])
    vals = vals[np.isfinite(vals)]
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def law_test(V: pd.Series, G: pd.Series, n_perm: int = 20000, seed: int = 0) -> dict:
    """Spearman association between violation and OOD gap across methods."""
    common = V.index.intersection(G.index)
    x = V.loc[common].to_numpy(dtype=float)
    y = G.loc[common].to_numpy(dtype=float)
    rho = float(stats.spearmanr(x, y).correlation)
    rng = np.random.default_rng(seed)
    perm = np.array([stats.spearmanr(x, rng.permutation(y)).correlation
                     for _ in range(n_perm)])
    p_one = float((np.sum(perm >= rho) + 1) / (n_perm + 1))
    p_two = float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1))
    lo, hi = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b).correlation,
                          seed=seed)
    return dict(n=len(common), rho=rho, p_one_sided=p_one, p_two_sided=p_two,
                ci_low=lo, ci_high=hi)
