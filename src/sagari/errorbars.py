"""Error-bar rounds and the inference surrogate (Section 3.7 of the paper).

The stopping rule says when the search has found the solution, not when the intervals are resolved:
Expected Improvement concentrates models at the minimum. After the search, error-bar rounds spread models
over the region the surrogate puts within Delta chi2_fill = 10 + 2 sigma_chi2 of its minimum (scrambled-Sobol
candidates kept in Sobol order) and along each parameter's profile. Each model carries the surrogate's
prediction made before it runs, so the round tests the surrogate on models it has not seen
(:func:`out_of_sample`). The rounds stop when the surrogate's latent sd at the ends of the Delta chi2 = 4 and
10 ranges is at most 3 and the ends moved by less than 0.2 of the width during the round
(:func:`profile_verdict`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .gp import PessimisticGP, feasibility_gp, fit_gp, noise_var, training_window
from .sobol import DEFAULT_SEED
from .space import as_space

WINDOW_FRACTIONS = (1.0, 2 / 3, 0.4, 0.27, 0.17)     # auto-window candidates, as fractions of train_window


# ---------------------------------------------------------------- the inference surrogate
def inference_gp(table: pd.DataFrame, space, seed: int = DEFAULT_SEED, kernel: str = "matern52",
                 train_window: float | None = 30.0, n_train_cap: int = 1000, feasible_factor: float = 10.0,
                 auto_window: bool = False):
    """The surrogate of the error-bar rounds and of the final inference: the monitoring GP (pessimistic prior
    at the feasibility threshold, nugget for the scatter) fitted to the models below the threshold and inside
    the training window, at most ``n_train_cap`` of them (raise the cap so that the window decides).

    ``auto_window=True`` instead takes the widest window whose fitted scatter matches the leave-one-out rms
    of the models near the floor (:func:`window_check`); the returned GP then carries ``.window`` and
    ``.window_checks``. Off in the paper (unstable on replayed sets).

    Returns ``(gp, chi2_min, threshold)``, or None with too few finite models.
    """
    space = as_space(space)
    finite = table[np.isfinite(table["chi2"])]
    dim = space.dim
    if len(finite) < 2 * dim + 2:
        return None
    _, chi2_min, thr = feasibility_gp(table, space, seed, feasible_factor)
    usable = finite[finite["chi2"].values - chi2_min <= thr]
    if len(usable) < 2 * dim + 2:
        usable = finite
    if auto_window and train_window:
        gp = _auto_window_gp(usable, space, seed, kernel, train_window, n_train_cap, chi2_min + thr)
        if gp is not None:
            return gp, chi2_min, thr
    best = usable.nsmallest(min(n_train_cap, len(usable)), "chi2")
    X = space.to_unit(best)
    keep, _ = training_window(X, best["chi2"].values, dim, seed, train_window, kernel)
    X, best = X[keep], best[keep]
    return PessimisticGP(X, best["chi2"].values, chi2_min + thr, dim, seed, kernel), chi2_min, thr


def window_check(gp: PessimisticGP, n_floor: int = 150) -> dict:
    """The fitted scatter against the leave-one-out rms of the ``n_floor`` training models with the lowest
    left-out prediction. A nugget set by a wide window over-smooths the floor where the scatter is smaller
    there, and the floor is where the intervals are read."""
    from scipy.linalg import cho_solve
    g = gp.gp
    kinv = np.diag(cho_solve((g.L_, True), np.eye(len(g.X_train_))))
    r = g.alpha_ / kinv * gp.scale                       # chi2 minus its prediction without that model
    m = g.y_train_ * gp.scale + gp.level - r
    near = np.argsort(m)[:min(n_floor, len(m))]
    rms = float(np.sqrt(np.mean(r[near] ** 2)))
    noise = float(np.sqrt(noise_var(gp)))
    return dict(noise=noise, loo_rms_floor=rms, ratio=noise / rms, n_train=len(m))


def _auto_window_gp(usable, space, seed, kernel, train_window, n_train_cap, level, n_low=300, tol=1.1, n_pass=3):
    """The widest training window whose GP passes :func:`window_check` (1/tol <= ratio <= tol). Candidates
    are ``train_window`` x the scatter, then :data:`WINDOW_FRACTIONS` of it; if none passes, the next pass
    measures them in the smallest leave-one-out rms near the floor found so far (at most ``n_pass`` passes).
    A window holding more than ``n_train_cap`` models is sampled as its ``n_low`` lowest plus a random draw
    of the rest. No window passing: the ratio closest to 1."""
    dim = space.dim
    best = usable.nsmallest(min(n_train_cap, len(usable)), "chi2")
    g0 = fit_gp(space.to_unit(best), best["chi2"].values, dim, seed, kernel)
    unit = max(1.0, float(np.sqrt(g0.kernel_.k2.noise_level) * np.std(best["chi2"].values)))
    d = usable["chi2"].values - usable["chi2"].min()
    fits, tried = [], set()
    for k in range(n_pass):
        for f in WINDOW_FRACTIONS:
            w = f * train_window * unit
            pool = usable[d <= w]
            if len(pool) < 10 * dim or len(pool) in tried:
                continue
            tried.add(len(pool))
            if len(pool) > n_train_cap:
                lo = pool.nsmallest(n_low, "chi2")
                pool = pd.concat([lo, pool.drop(lo.index).sample(n_train_cap - n_low, random_state=seed)])
            gp = PessimisticGP(space.to_unit(pool), pool["chi2"].values, level, dim, seed, kernel)
            gp.window = dict(window_dchi2=round(float(w), 1), window_sigma=round(f * train_window, 1),
                             unit=round(unit, 2), pass_=k + 1, **window_check(gp))
            fits.append(gp)
            if 1 / tol <= gp.window["ratio"] <= tol:
                gp.window_checks = [g.window for g in fits]
                return gp
        if not fits:
            return None
        new_unit = min(g.window["loo_rms_floor"] for g in fits)
        if new_unit >= 0.95 * unit:
            break
        unit = max(1.0, new_unit)
    chosen = min(fits, key=lambda g: abs(np.log(g.window["ratio"])))
    chosen.window_checks = [g.window for g in fits]
    return chosen


# ---------------------------------------------------------------- profiles of the surrogate
def profile_point(gp, j: int, uj: float, dim: int, rng, n_rand: int = 4000, n_top: int = 5, n_local: int = 160,
                  step: float = 0.03):
    """Minimum of the surrogate mean over the other coordinates with coordinate ``j`` fixed at ``uj`` (unit
    cube): random search, then Gaussian perturbations around the 5 best. Returns (u*, mean, total sd)."""
    U = rng.random((n_rand, dim))
    U[:, j] = uj
    m = gp.predict(U)
    top = U[np.argsort(m)[:n_top]]
    L = np.clip(np.repeat(top, n_local, axis=0) + step * rng.standard_normal((n_top * n_local, dim)), 0.0, 1.0)
    L[:, j] = uj
    U = np.vstack([top, L])
    m, s = gp.predict(U, return_std=True)
    k = int(np.argmin(m))
    return U[k], float(m[k]), float(s[k])


def gp_profiles(gp, space, seed: int = 0, n_grid: int = 25, levels: tuple = (4.0, 10.0)) -> dict:
    """Profile of the surrogate along each free parameter on ``n_grid`` values of its unit coordinate, with
    the latent sd (predictive sd with the scatter removed). For each Delta chi2 level, the range within that
    level of the minimum (linear interpolation between grid values), the latent sd at its ends, and whether
    an end is a box bound.

    The zero point is the lowest profile value where models constrain the surrogate (latent sd <= 2 x the
    scatter); a lower value elsewhere is extrapolation into empty space and is listed in
    ``"unsupported_dips"`` as a place to sample, not used.
    """
    space = as_space(space)
    dim = space.dim
    rng = np.random.default_rng([seed, 2718])
    nv = noise_var(gp)
    grid = np.linspace(0.0, 1.0, n_grid)
    prof = {}
    for j, p in enumerate(space.free_params):
        pts = [profile_point(gp, j, g, dim, rng) for g in grid]
        mu = np.array([x[1] for x in pts])
        lat = np.sqrt(np.maximum(np.array([x[2] for x in pts]) ** 2 - nv, 0.0))
        prof[p] = dict(u=grid, mu=mu, sd_latent=lat, path=np.array([x[0] for x in pts]))
    sup_max = 2.0 * np.sqrt(nv)
    supported = [v["mu"][v["sd_latent"] <= sup_max] for v in prof.values()]
    mu_min = min((float(x.min()) for x in supported if len(x)), default=min(float(v["mu"].min()) for v in prof.values()))
    dips = [(p, float(v["mu"][k]), float(v["sd_latent"][k])) for p, v in prof.items()
            for k in range(len(grid)) if v["sd_latent"][k] > sup_max and v["mu"][k] < mu_min - 2.0 * np.sqrt(nv)]
    out = {"mu_min": mu_min, "noise": float(np.sqrt(nv)), "levels": list(levels), "params": {},
           "unsupported_dips": [dict(param=p, dchi2=round(m - mu_min, 1), sd_latent=round(sd, 1))
                                for p, m, sd in dips[:8]]}
    values = space.from_unit(np.outer(grid, np.ones(dim)))
    for j, p in enumerate(space.free_params):
        v = prof[p]
        d = v["mu"] - mu_min
        rec = {"value": [float(x) for x in values[p].values], "dchi2": [float(x) for x in d],
               "sd_latent": [float(x) for x in v["sd_latent"]], "path": v["path"].tolist(), "edges": {}}
        for D in levels:
            inside = np.where(d <= D)[0]
            if not len(inside):
                continue
            ends = []
            for k, side in ((inside[0], -1), (inside[-1], +1)):
                n = k + side
                if n < 0 or n >= len(grid):
                    ends.append((float(grid[k]), float(v["sd_latent"][k]), True))
                    continue
                f = (D - d[k]) / max(d[n] - d[k], 1e-12)       # where the profile crosses D between k and n
                u = grid[k] + f * (grid[n] - grid[k])
                ends.append((float(u), float(v["sd_latent"][k] + f * (v["sd_latent"][n] - v["sd_latent"][k])), False))
            phys = space.from_unit(np.outer([ends[0][0], ends[1][0]], np.ones(dim)))[p].values
            rec["edges"][f"{D:g}"] = dict(u=[ends[0][0], ends[1][0]], value=[float(phys[0]), float(phys[1])],
                                          sd_latent=[ends[0][1], ends[1][1]], at_bound=[ends[0][2], ends[1][2]])
        out["params"][p] = rec
    return out


def profile_compact(prof: dict) -> dict:
    """The part of :func:`gp_profiles` worth logging: per parameter and level, the edges and their latent sd."""
    return {"mu_min": prof["mu_min"], "noise": prof["noise"], "levels": prof["levels"],
            "unsupported_dips": prof.get("unsupported_dips", []),
            "params": {p: {"edges": v["edges"]} for p, v in prof["params"].items()}}


def profile_verdict(before: dict | None, after: dict, params, sd_tol: float = 3.0, frac: float = 0.2) -> dict:
    """The error-bar stopping rule: done when, for every parameter in ``params`` and every level, the latent
    sd at each end of the range is <= ``sd_tol`` (ends at a box bound excepted) and neither end moved by
    ``frac`` x the width (unit cube) since ``before``. Returns ``{"done", "worst_sd", "worst_move"}``."""
    worst_sd, worst_move, ok = 0.0, 0.0, True
    for p in params:
        for D, e in after["params"][p]["edges"].items():
            for sd, at_b in zip(e["sd_latent"], e["at_bound"]):
                if not at_b:
                    worst_sd = max(worst_sd, sd)
            if before is None or D not in before["params"].get(p, {}).get("edges", {}):
                ok = False
                continue
            b = before["params"][p]["edges"][D]["u"]
            width = max(e["u"][1] - e["u"][0], 1e-6)
            worst_move = max(worst_move, max(abs(e["u"][0] - b[0]), abs(e["u"][1] - b[1])) / width)
    ok = ok and worst_sd <= sd_tol and worst_move < frac
    return {"done": bool(ok), "worst_sd": round(worst_sd, 2), "worst_move": round(worst_move, 3),
            "sd_tol": sd_tol, "frac": frac}


# ---------------------------------------------------------------- one error-bar round
def errorbar_round(table: pd.DataFrame, space, n_new: int, seed: int = DEFAULT_SEED, *, kernel: str = "matern52",
                   train_window: float | None = 30.0, dchi2: float | None = None, n_candidates: int = 2 ** 15,
                   n_train_cap: int = 1000, n_profile: int = 7, profile_dchi2: float | None = None,
                   profile_levels: tuple = (4.0, 10.0), auto_window: bool = False, round_index: int = 0) -> pd.DataFrame:
    """One error-bar round of ``n_new`` models: ``n_profile`` x dim on the profile paths, the rest filling the
    region within ``dchi2`` (default 10 + 2 sigma_chi2) of the surrogate minimum.

    A profile path is, for one parameter, the location of the surrogate minimum over the others at each of
    its values; its ``n_profile`` points span the range within ``profile_dchi2`` (default: the fill level)
    plus 15% (and 0.02 of the box) beyond each end. The fill uses scrambled-Sobol candidates over the box in
    Sobol order, which is space-filling within the region; if too few qualify, the lowest-predicted are added.

    Every model carries the surrogate's prediction (``pred_mu``; ``pred_sd``, the predictive sd including the
    scatter; ``pred_latent``, without it) for :func:`out_of_sample`. ``.attrs["profile_before"]`` holds the
    profiles before the round, for :func:`profile_verdict`. Pass the number of earlier error-bar rounds as
    ``round_index`` (used only for bookkeeping).
    """
    space = as_space(space)
    dim = space.dim
    fit = inference_gp(table, space, seed, kernel, train_window, n_train_cap, auto_window=auto_window)
    if fit is None:
        raise ValueError("too few finite models to fit the surrogate")
    gp = fit[0]
    scatter = max(1.0, float(np.sqrt(gp.kernel_.k2.noise_level) * gp.scale))
    dchi2 = 10.0 + 2.0 * scatter if dchi2 is None else dchi2

    U = qmc.Sobol(dim, scramble=True, seed=np.random.default_rng([seed, 31337, len(table)])).random(n_candidates)  # 2^15
    mu = gp.predict(U)

    prof_units, kinds, prof = [], [], None
    if n_profile > 0:
        prof = gp_profiles(gp, space, seed, levels=tuple(profile_levels))
        level = dchi2 if profile_dchi2 is None else profile_dchi2
        rng = np.random.default_rng([seed, 4711, len(table)])
        for j, p in enumerate(space.free_params):
            d = np.array(prof["params"][p]["dchi2"])
            grid = np.linspace(0.0, 1.0, len(d))
            inside = grid[d <= level] if (d <= level).any() else grid[[int(np.argmin(d))]]
            lo, hi = inside.min(), inside.max()
            pad = 0.15 * (hi - lo) + 0.02
            for uj in np.linspace(max(0.0, lo - pad), min(1.0, hi + pad), n_profile):
                prof_units.append(profile_point(gp, j, float(uj), dim, rng)[0])
                kinds.append(f"profile_{p}")
    n_fill = max(0, n_new - len(prof_units))

    inside = np.where(mu - mu.min() <= dchi2)[0]          # Sobol order preserved: space-filling
    pick = list(inside[:n_fill])
    if len(pick) < n_fill:
        rest = [i for i in np.argsort(mu) if i not in set(pick)]
        pick += rest[:n_fill - len(pick)]
    units = np.vstack([U[pick]] + ([np.array(prof_units)] if prof_units else []))
    kinds = ["fill"] * len(pick) + kinds
    out = space.from_unit(units, with_fixed=True)
    out["stage"] = "errorbar"
    out["kind"] = kinds
    m, sd = gp.predict(units, return_std=True)
    out["pred_mu"] = m
    out["pred_sd"] = sd
    out["pred_latent"] = np.sqrt(np.maximum(sd ** 2 - noise_var(gp), 0.0))
    out.attrs.update(fill_dchi2=float(dchi2), fill_scatter=scatter, fill_n_inside=int(len(inside)),
                     fill_round=round_index + 1, n_fill=len(pick), n_profile_points=len(prof_units),
                     fill_kernel=kernel, fill_train_window=train_window,
                     fill_window=getattr(gp, "window", None), fill_window_checks=getattr(gp, "window_checks", None))
    if prof is not None:
        out.attrs["profile_before"] = profile_compact(prof)
    return out


def out_of_sample(predicted: pd.DataFrame, chi2) -> dict:
    """Standardized residuals z = (chi2 - pred_mu) / pred_sd of models the surrogate had not seen, for a round
    from :func:`errorbar_round` (rows in the same order as ``chi2``; failed models, NaN, are skipped). For a
    calibrated surrogate z scatters as a unit normal."""
    z = (np.asarray(chi2, float) - predicted["pred_mu"].values) / predicted["pred_sd"].values
    ok = np.isfinite(z)
    z = z[ok]
    if not len(z):
        return dict(n=0)
    return dict(n=int(len(z)), rms=float(np.sqrt(np.mean(z ** 2))), mean=float(np.mean(z)),
                frac_above_2=float(np.mean(z > 2)), frac_below_2=float(np.mean(z < -2)), z=z.tolist())


__all__ = ["inference_gp", "window_check", "profile_point", "gp_profiles", "profile_compact", "profile_verdict",
           "errorbar_round", "out_of_sample"]
