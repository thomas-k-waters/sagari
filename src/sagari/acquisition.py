"""One refinement round: feasibility-weighted Expected Improvement on a noisy chi2 surface (Section 3.5).

The round fits the main GP to the models below the hopelessness threshold and inside the training window,
scores a two-scale candidate pool by Expected Improvement against the surrogate's own best prediction at an
evaluated model, multiplies it by the probability (from the feasibility GP) that a candidate lies inside the
training window, takes the batch greedily subject to a minimum separation, verifies up to two isolated low
models by their neighbors, and fills the rest of the round from the campaign's global Sobol sequence.

The defaults are the settings of the paper (Matern 5/2 kernel, 30 sigma window, 35% global share). The
other options reproduce the variants of its benchmark (Appendix B) and should not normally be changed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .gp import feasibility_gp, fit_gp, fitted_scatter, training_window
from .sobol import DEFAULT_SEED, SobolGlobalDesign
from .space import as_space

STAGE_VERIFY, STAGE_LOCAL, STAGE_GLOBAL = "gp_ei_verify", "gp_ei_local", "sobol_global_continuation"


def _greedy_diverse(order, points, n, min_sep):
    """Take points in ``order`` (best first), skipping any closer than ``min_sep`` to one already taken;
    top up from the skipped ones if fewer than ``n`` survive."""
    chosen = []
    for i in order:
        if len(chosen) == n:
            break
        if min_sep <= 0 or not chosen or np.min(np.linalg.norm(points[chosen] - points[i], axis=1)) >= min_sep:
            chosen.append(i)
    if len(chosen) < n:
        taken = set(chosen)
        chosen += [i for i in order if i not in taken][: n - len(chosen)]
    return np.array(chosen, dtype=int)


def _as_sobol(sobol, dim, seed) -> SobolGlobalDesign:
    if isinstance(sobol, SobolGlobalDesign):
        return sobol
    if sobol is None:
        raise ValueError("propose_round needs the campaign's Sobol state (a SobolGlobalDesign or its JSON path), so "
                         "that the global share continues the sequence of the initial design instead of repeating it")
    return SobolGlobalDesign(dim, seed, state_path=Path(sobol))


def propose_round(
    table: pd.DataFrame,
    space,
    n_new: int,
    sobol,
    seed: int = DEFAULT_SEED,
    *,
    kernel: str = "matern52",
    train_window: float | None = 30.0,
    global_fraction: float = 0.35,
    n_train_cap: int = 300,
    n_local_cap: int = 200,
    n_candidates: int = 5000,
    n_local_candidates: int = 4000,
    local_scales: tuple = (0.01, 0.03, 0.08),
    min_separation: float = 0.04,
    verify: bool = True,
    verify_step: float = 0.03,
    verify_max: int = 2,
    support_ok: float = 0.1,
    noise_diagnostic: bool = True,
    incumbent: str = "predicted",
    redraw_candidates: bool = True,
    feasibility: bool = True,
    feasible_factor: float = 10.0,
    max_train: int = 2000,
    target: str = "raw",
    log_offset: float = 20.0,
    include_rest: bool = False,
    exclude_bad_radius: float = 0.0,
    exclude_bad_min: int = 2,
) -> pd.DataFrame:
    """Propose the next ``n_new`` models.

    Parameters
    ----------
    table : DataFrame
        Every model evaluated so far: one column per free parameter and ``chi2`` (NaN for a failed model).
        Pass only the models inside the current box (:meth:`ParameterSpace.contains`).
    space : ParameterSpace, dict or YAML path
    n_new : int
        Models in the round.
    sobol : SobolGlobalDesign or path
        The campaign's global sequence (advanced by the global share of this round).
    seed : int
        Seed of every random choice in the round (candidates, GP optimizer restarts).
    kernel, train_window, global_fraction
        Matern 5/2 or squared-exponential kernel; training window in units of sigma_chi2 (None: the best
        ``n_train_cap`` models); share of the round that continues the Sobol sequence.
    n_train_cap, n_local_cap
        Most models the main GP is fitted to; most points placed by EI in one round.
    n_candidates, n_local_candidates, local_scales, redraw_candidates
        The candidate pool: uniform points, plus Gaussian perturbations of widths ``local_scales`` around
        the 20 best-predicted models; redrawn every round.
    min_separation
        Greedy batch selection skips candidates closer than this (unit cube) to one already taken.
    verify, verify_step, verify_max, support_ok, noise_diagnostic
        For up to ``verify_max`` models within 2 sigma_chi2 of the best with no other model within
        ``support_ok``, add the 2 x dim points +-``verify_step`` away along each axis. With
        ``noise_diagnostic=False`` the band is twice the sd of the 20 lowest chi2 instead of 2 sigma_chi2
        (as in the paper's replays and benchmark).
    incumbent
        ``"predicted"``: the lowest value the GP predicts at an evaluated model (Section 3.5);
        ``"raw"``: the lowest chi2 (correct only for noiseless evaluations).
    feasibility, feasible_factor, max_train
        Feasibility weighting of EI (:func:`sagari.gp.feasibility_gp`).
    target, log_offset, include_rest, exclude_bad_radius, exclude_bad_min
        Alternatives evaluated in the benchmark and not recommended: a log chi2 target, training on all
        models with chi2 capped, and excluding candidates near clusters of bad models.

    Returns
    -------
    DataFrame
        Free and fixed parameters and ``stage`` (``gp_ei_verify``, ``gp_ei_local``,
        ``sobol_global_continuation``). Diagnostics are in ``.attrs``: ``gp_noise_sd`` (sigma_chi2),
        ``raw_best``, ``n_verify``, ``n_train``, ``window_dchi2``, ``feasible_frac``, ``thr_dchi2``.
    """
    space = as_space(space)
    dim = space.dim
    sobol = _as_sobol(sobol, dim, seed)

    finite = table[np.isfinite(table["chi2"])]
    failed = table[~np.isfinite(table["chi2"])]
    if len(finite) < 2 * dim + 2:
        raise ValueError(f"only {len(finite)} finite models -- too few to fit a {dim}-D Gaussian process; "
                         "run (more of) the initial design first")

    # the main GP sees only models below the hopelessness threshold (at most n_train_cap), so the steep
    # walls cannot set its length scales; the feasibility GP covers the rest
    dchi2_all = finite["chi2"].values - finite["chi2"].min()
    if feasibility:
        gpf, _, thr_dchi2 = feasibility_gp(table, space, seed, feasible_factor, max_train)
        usable = finite[dchi2_all <= thr_dchi2]
    else:
        usable = finite
    if len(usable) < 2 * dim + 2:
        usable = finite
    best = usable.nsmallest(min(n_train_cap, len(usable)), "chi2")
    X_best = space.to_unit(best)
    chi2_best = best["chi2"].values
    keep, win_scatter = training_window(X_best, chi2_best, dim, seed, train_window, kernel)
    window_dchi2 = None
    if win_scatter is not None:
        window_dchi2 = float((chi2_best[keep] - chi2_best.min()).max())
    best, X_best, chi2_best = best[keep], X_best[keep], chi2_best[keep]
    chi2_min = chi2_best.min()
    X_train, chi2_train = X_best, chi2_best
    n_capped = 0
    if include_rest:
        y_cap = chi2_best.max()
        others = pd.concat([finite.drop(best.index), failed])
        if len(others) > max_train - len(best):
            others = others.sample(max_train - len(best), random_state=seed)
        if len(others):
            X_train = np.vstack([X_best, space.to_unit(others)])
            chi2_train = np.concatenate([chi2_best, np.full(len(others), y_cap)])
            n_capped = len(others)
    if target == "log":
        y_train = np.log(chi2_train - chi2_min + log_offset)
    elif target == "raw":
        y_train = chi2_train
    else:
        raise ValueError(f"target must be 'raw' or 'log', got {target!r}")
    gp = fit_gp(X_train, y_train, dim, seed, kernel)

    noise_sd = None
    if noise_diagnostic:
        gp_raw = gp if (target == "raw" and n_capped == 0) else fit_gp(X_best, chi2_best, dim, seed, kernel)
        noise_sd = fitted_scatter(gp_raw, chi2_best)

    n_global = int(round(n_new * global_fraction))
    n_local = min(n_new - n_global, n_local_cap)
    n_global = n_new - n_local

    mu_train = gp.predict(X_train)
    f_best = mu_train.min() if incumbent == "predicted" else y_train.min()

    # verification points around isolated low models
    verify_unit = np.empty((0, dim))
    if verify and n_local > 0:
        X_all = space.to_unit(finite)
        d_nn = np.array([np.sort(np.linalg.norm(X_all - x, axis=1))[1] for x in X_best])
        band = 2.0 * (noise_sd if noise_sd else np.std(np.sort(chi2_best)[:20]))
        lone = np.where((d_nn > support_ok) & (chi2_best <= chi2_min + band))[0]
        lone = lone[np.argsort(chi2_best[lone])][:verify_max]
        offs = np.vstack([s * verify_step * np.eye(dim)[k] for k in range(dim) for s in (+1, -1)])
        if len(lone):
            verify_unit = np.clip(np.vstack([X_best[i] + offs for i in lone]), 0.0, 1.0)[:n_local]

    # candidate pool: uniform, plus perturbations of the best-predicted models
    cand_rng = np.random.default_rng([seed, 999, len(finite)] if redraw_candidates else seed + 999)
    candidates = [cand_rng.random((n_candidates, dim))]
    if n_local_candidates > 0:
        n_cent = min(20, len(X_train))
        centres = X_train[np.argsort(mu_train)[:n_cent]]
        per = max(1, n_local_candidates // (n_cent * len(local_scales)))
        for sc in local_scales:
            for c in centres:
                candidates.append(c + sc * cand_rng.standard_normal((per, dim)))
    candidates = np.clip(np.vstack(candidates), 0.0, 1.0)

    n_excluded = 0
    if exclude_bad_radius > 0:
        from scipy.spatial import cKDTree
        X_all = space.to_unit(table)
        good = table.index.isin(best.index)
        neigh = cKDTree(X_all).query_ball_point(candidates, exclude_bad_radius)
        ok = np.array([len(nb) < exclude_bad_min or good[nb].any() for nb in neigh])
        n_excluded = int((~ok).sum())
        if ok.sum() >= n_local:
            candidates = candidates[ok]

    mu, sigma = gp.predict(candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    p_ok_mean = None
    if feasibility:
        # with a training window the main GP knows only the region inside it, so "feasible" means inside
        # the window as the feasibility GP (fitted to every model) predicts it
        thr = np.log10((window_dchi2 if window_dchi2 is not None else thr_dchi2) + 20.0)
        mf, sf = gpf.predict(candidates, return_std=True)
        p_ok = norm.cdf((thr - mf) / np.maximum(sf, 1e-12))
        ei = ei * p_ok
        p_ok_mean = float(p_ok.mean())
    n_ei = n_local - len(verify_unit)
    pick = _greedy_diverse(np.argsort(-ei), candidates, n_ei, min_separation)
    local_unit = np.vstack([verify_unit, candidates[pick]])

    global_unit = sobol.extend(n_global)
    out = space.from_unit(np.vstack([local_unit, global_unit]), with_fixed=True)
    out["stage"] = ([STAGE_VERIFY] * len(verify_unit) + [STAGE_LOCAL] * (n_local - len(verify_unit))
                    + [STAGE_GLOBAL] * n_global)
    out.attrs.update(gp_noise_sd=noise_sd, raw_best=float(chi2_min), n_verify=int(len(verify_unit)),
                     gp_target=target, n_train=int(len(chi2_train)), train_window=train_window, kernel=kernel,
                     window_dchi2=window_dchi2, n_capped=int(n_capped), n_failed_in_table=int(len(failed)),
                     n_candidates_excluded=n_excluded, feasible_frac=p_ok_mean,
                     thr_dchi2=float(thr_dchi2) if feasibility else None)
    return out


def initial_design(space, n: int, sobol) -> pd.DataFrame:
    """The first ``n`` points of the campaign's scrambled Sobol sequence (round 0), with ``stage``
    ``sobol_global``. ``sobol`` is the campaign's :class:`SobolGlobalDesign` (or its JSON path, read with
    the default seed)."""
    space = as_space(space)
    sobol = _as_sobol(sobol, space.dim, DEFAULT_SEED)
    out = space.from_unit(sobol.extend(n), with_fixed=True)
    out["stage"] = "sobol_global"
    return out


__all__ = ["propose_round", "initial_design"]
