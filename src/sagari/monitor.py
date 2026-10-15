"""Monitoring a campaign and deciding when the search stops (Section 3.6 of the paper).

With model-to-model scatter the lowest model is a poor estimator, so the campaign is monitored with a
smoothed estimate: a GP with a pessimistic prior mean is fitted to the models, and its mean is turned into
tempered weights w = exp(-(mu - mu_min) / 2T) over 60000 uniform points of the unit cube, with T the
fitted scatter (at least 1). Each parameter's monitoring range is the 16-84% weighted quantile range; its
best value the surrogate minimum. The search stops when, for the parameters that matter, both ends of the
range move by less than a fraction of its width between rounds, after a minimum number of refinement rounds.

The weights are a convergence diagnostic, deliberately broader than a likelihood, not the posterior: the
reported intervals come from :mod:`sagari.posterior`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

from .gp import PessimisticGP, feasibility_gp, training_window
from .space import as_space


def smoothed_estimate(table: pd.DataFrame, space, seed: int = 0, n_train_cap: int = 300,
                      feasible_factor: float = 10.0, n_uniform: int = 60000, temperature: float | None = None,
                      quantiles: tuple = (0.16, 0.84), train_window: float | None = 30.0, kernel: str = "matern52",
                      feasibility: bool = False) -> dict | None:
    """The monitoring estimate after a round.

    Returns ``{param: [best, lo, hi], "chi2_min", "temperature", "ess"}`` in physical units (best: the
    surrogate minimum; lo, hi: weighted quantiles), or None with fewer than 2 x dim + 2 finite models.
    Use the ``kernel`` and ``train_window`` of the campaign's sampler. ``feasibility=True`` also multiplies
    the weights by the probability, from the feasibility GP, of lying inside the training window (off in the
    paper).
    """
    space = as_space(space)
    finite = table[np.isfinite(table["chi2"])]
    dim = space.dim
    if len(finite) < 2 * dim + 2:
        return None
    gpf, chi2_min, thr = feasibility_gp(table, space, seed, feasible_factor)
    usable = finite[finite["chi2"].values - chi2_min <= thr]
    if len(usable) < 2 * dim + 2:
        usable = finite
    best = usable.nsmallest(min(n_train_cap, len(usable)), "chi2")
    X = space.to_unit(best)
    keep, win_scatter = training_window(X, best["chi2"].values, dim, seed, train_window, kernel)
    X, best = X[keep], best[keep]
    limit = float(best["chi2"].max() - best["chi2"].min()) if win_scatter is not None else thr
    gp = PessimisticGP(X, best["chi2"].values, chi2_min + thr, dim, seed, kernel)
    T = max(1.0, float(np.sqrt(gp.kernel_.k2.noise_level) * gp.scale)) if temperature is None else temperature

    rng = np.random.default_rng([seed, 4242, len(finite)])
    U = rng.random((n_uniform, dim))
    mu = gp.predict(U)
    # the minimum itself: refine around the best training points
    top = X[np.argsort(gp.predict(X))[:20]]
    loc = np.clip(np.repeat(top, 200, axis=0) + 0.02 * rng.standard_normal((len(top) * 200, dim)), 0.0, 1.0)
    mu_loc = gp.predict(loc)
    i_best = int(np.argmin(mu_loc))
    best_phys = space.from_unit(loc[[i_best]]).iloc[0]

    p_ok, p_ok_mean = np.ones(len(U)), None
    if feasibility:
        mf, sf = gpf.predict(U, return_std=True)
        p_ok = norm.cdf((np.log10(limit + 20.0) - mf) / np.maximum(sf, 1e-12))
        p_ok_mean = float(p_ok.mean())
    # the reported minimum ignores box points the feasibility GP puts outside the window
    inside = p_ok >= 0.5
    mu_min = float(min(mu_loc[i_best], mu[inside].min() if inside.any() else np.inf))
    w = np.exp(np.minimum(-(mu - mu_min) / (2.0 * T), 600.0)) * p_ok
    w /= w.sum()
    P = space.from_unit(U)
    out = {"chi2_min": mu_min, "temperature": T, "ess": float(1.0 / np.sum(w ** 2))}
    if p_ok_mean is not None:
        out["p_ok_mean"] = p_ok_mean
    for p in space.free_params:
        x = P[p].values
        o = np.argsort(x)
        c = np.cumsum(w[o])
        lo, hi = (float(x[o][min(np.searchsorted(c, q), len(x) - 1)]) for q in quantiles)
        out[p] = [float(best_phys[p]), lo, hi]
    return out


def estimate_moved(prev: dict, cur: dict, space) -> dict:
    """For each free parameter of ``space``, the larger move of the two ends of its range between two
    estimates, in units of the current range width (log10 for logarithmic axes)."""
    space = as_space(space)
    moves = {}
    for p in space.free_params:
        f = np.log10 if p in space.log_params else (lambda x: x)
        lo0, hi0 = f(prev[p][1]), f(prev[p][2])
        lo1, hi1 = f(cur[p][1]), f(cur[p][2])
        width = max(hi1 - lo1, 1e-12)
        moves[p] = max(abs(lo1 - lo0), abs(hi1 - hi0)) / width
    return moves


def estimate_converged(history: list, space, frac: float = 0.2, patience: int = 1, min_rounds: int = 2) -> bool:
    """The stopping rule (Section 3.6): True when, in each of the last ``patience`` rounds, both ends of
    every range of ``space`` (pass :meth:`ParameterSpace.subspace` for the parameters that matter) moved
    by less than ``frac`` x its width, and ``history`` (round 0 plus the refinement rounds, one estimate
    each) holds at least ``min_rounds`` refinement rounds. Stability alone can come early, in the wrong
    basin, before a narrow deeper one has been found; ``min_rounds`` guards against that.

    The defaults are the adopted settings of the paper (threshold 0.2, patience 1, two refinement rounds).
    """
    space = as_space(space)
    if len(history) - 1 < min_rounds:
        return False
    if len(history) < patience + 1 or any(h is None for h in history[-(patience + 1):]):
        return False
    win = history[-(patience + 1):]
    return all(max(estimate_moved(a, b, space).values()) < frac for a, b in zip(win[:-1], win[1:]))


# ---------------------------------------------------------------- the trusted best model (a cross-check)
def trusted_best_point(table: pd.DataFrame, space, support_ok: float = 0.1) -> pd.Series | None:
    """The lowest-chi2 model that has at least one other model within ``support_ok`` (unit cube); an isolated
    model never qualifies. None if no model does yet."""
    space = as_space(space)
    finite = table[np.isfinite(table["chi2"])]
    if len(finite) < 2:
        return None
    Xs = space.to_unit(finite)
    dist, _ = cKDTree(Xs).query(Xs, k=2)
    corroborated = finite[dist[:, 1] < support_ok]
    if len(corroborated) == 0:
        return None
    return corroborated.loc[corroborated["chi2"].idxmin()]


def trusted_converged(history: list, space, tol: float = 0.01, patience: int = 2) -> bool:
    """True once the trusted best model moved by less than ``tol`` (unit cube, max over axes) in each of the
    last ``patience`` rounds. A None in that window (nothing corroborated yet) counts as still moving."""
    space = as_space(space)
    if len(history) < patience + 1:
        return False
    window = history[-(patience + 1):]
    for prev, cur in zip(window[:-1], window[1:]):
        if prev is None or cur is None:
            return False
        a = space.to_unit(prev[space.free_params].values.astype(float))[0]
        b = space.to_unit(cur[space.free_params].values.astype(float))[0]
        if np.max(np.abs(b - a)) >= tol:
            return False
    return True


__all__ = ["smoothed_estimate", "estimate_moved", "estimate_converged", "trusted_best_point", "trusted_converged"]
