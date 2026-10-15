"""Constraints from the final surrogate (Section 6.4 of the paper).

The posterior is sampled on the inference surrogate's mean, p(theta) ~ exp(-mu/2), with priors uniform over
the unit cube (log-uniform on logarithmic axes) and the posterior set to zero wherever the surrogate's latent
sd exceeds twice the scatter: there no models constrain it, and a sampler would settle in any low value the
surrogate extrapolates into an empty region. The surrogate's own uncertainty is propagated by reweighting the
samples with joint draws of its latent error (the surrogate-marginalized posterior), and the Delta chi2
profile intervals can be widened by parametric-bootstrap refits of the surrogate.

All functions take a surrogate with ``predict(U, return_std=False)`` in the unit cube, ``.gp`` (the
scikit-learn regressor) and ``.scale`` / ``.level`` -- a :class:`sagari.gp.PessimisticGP`, as returned
by :func:`sagari.errorbars.inference_gp`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .gp import noise_var
from .space import as_space


# ---------------------------------------------------------------- the sampler of Waters et al. (2026)
def run_emcee(f, lo, hi, nsteps: int = 25000, burn: int = 5000, thin: int = 5, seed: int = 42, nwalkers: int = 32):
    """Ensemble MCMC of ln L = -chi2/2 with chi2 = f(theta) and flat priors on [lo, hi].

    50/50 stretch (a = 3) and differential-evolution moves; walkers whose median log-probability lies more
    than 1.6 robust sd below the others are dropped, and so are samples more than 10 below the median
    log-probability. Returns ``(samples, mean acceptance, walkers kept)``.
    """
    import emcee
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    nd = len(lo)

    def logp(th):
        th = np.atleast_2d(th)
        out = np.full(len(th), -np.inf)
        ok = np.all((th >= lo) & (th <= hi), axis=1)
        if ok.any():
            c = f(th[ok])
            out[ok] = np.where(np.isfinite(c) & (c >= 0), -0.5 * c, -np.inf)
        return out
    rng = np.random.default_rng(seed)
    pos = lo + (hi - lo) * rng.random((nwalkers, nd))
    s = emcee.EnsembleSampler(nwalkers, nd, logp, vectorize=True,
                              moves=[(emcee.moves.StretchMove(a=3.0), 0.5), (emcee.moves.DEMove(), 0.5)])
    s.random_state = np.random.RandomState(seed).get_state()   # emcee's own generator is unseeded otherwise
    with np.errstate(invalid="ignore"):        # -inf - -inf for a walker in the zero-posterior region: rejected
        s.run_mcmc(pos, nsteps, progress=False)
    ch, lp = s.get_chain(discard=burn, thin=thin), s.get_log_prob(discard=burn, thin=thin)
    wm = np.median(lp, 0)
    med = np.median(wm)
    mad = np.median(np.abs(wm - med))
    kw = (wm - med) / (1.4826 * mad + 1e-12) >= -1.6
    if kw.sum() < max(8, nwalkers // 2):
        kw[:] = True
    flat, flp = ch[:, kw].reshape(-1, nd), lp[:, kw].reshape(-1)
    g = flp > np.median(flp) - 10
    if g.sum() >= 100:
        flat = flat[g]
    return flat, float(s.acceptance_fraction.mean()), int(kw.sum())


def quantiles(samples, names, w=None, q=(0.1585, 0.5, 0.8415)) -> dict:
    """``{name: (median, lo, hi)}`` from samples (columns in ``names`` order), optionally weighted."""
    out = {}
    for j, p in enumerate(names):
        x = samples[:, j]
        if w is None:
            v = np.percentile(x, [100 * qq for qq in q])
        else:
            o = np.argsort(x)
            cw = np.cumsum(w[o]) / w.sum()
            v = np.interp(q, cw, x[o])
        out[p] = (float(v[1]), float(v[0]), float(v[2]))
    return out


# ---------------------------------------------------------------- where the surface is trusted
def support_cap(gp, k: float = 2.0) -> float:
    """The surface is trusted where its latent sd is at most ``k`` x the scatter: there models constrain it."""
    return k * np.sqrt(noise_var(gp))


def _mean_latent(gp, U):
    m, s = gp.predict(U, return_std=True)
    return m, np.sqrt(np.maximum(s ** 2 - noise_var(gp), 0.0))


def supported_chi2(gp, support: float):
    """chi2 function for the sampler: the surrogate mean, infinite (zero likelihood) where the surface is
    unsupported (latent sd above ``support``)."""
    def f(U):
        m, lat = _mean_latent(gp, np.atleast_2d(U))
        return np.where(lat <= support, m, np.inf)
    return f


# ---------------------------------------------------------------- profiles
def polished_point(gp, j: int, u: float, rng, dim: int, n_rand: int = 6000, n_top: int = 5, n_local: int = 160,
                   support: float | None = None):
    """Minimum of the surrogate mean over the other coordinates with coordinate ``j`` fixed at ``u``: random
    search, perturbations around the best, then L-BFGS-B on the smooth mean. With ``support`` only points
    where models constrain the surface count (if none does, the plain minimum is returned and its latent sd
    shows it is unsupported). Returns (u*, mean, total sd)."""
    def score(V):
        if support is None:
            return gp.predict(V)
        m, lat = _mean_latent(gp, V)
        return m + 1e3 * np.maximum(lat - support, 0.0) ** 2 / support ** 2 * np.sqrt(noise_var(gp))
    U = rng.random((n_rand, dim))
    U[:, j] = u
    m = score(U)
    top = U[np.argsort(m)[:n_top]]
    L = np.clip(np.repeat(top, n_local, 0) + 0.03 * rng.standard_normal((n_top * n_local, dim)), 0, 1)
    L[:, j] = u
    U = np.vstack([top, L])
    m = score(U)
    k = int(np.argmin(m))
    x = U[k].copy()
    oth = [i for i in range(dim) if i != j]
    r = minimize(lambda z: float(score(np.insert(z, j, u)[None])[0]), x[oth], method="L-BFGS-B",
                 bounds=[(0.0, 1.0)] * len(oth), options={"maxiter": 60})
    if r.fun < m[k]:
        x[oth] = r.x
    m2, s2 = gp.predict(x[None], return_std=True)
    return x, float(m2[0]), float(s2[0])


def profile(gp, j: int, dim: int, rng, n_coarse: int = 25, n_fine: int = 41, window: float = 30.0,
            n_rand: int = 6000, support: float | None = None):
    """Profile along unit coordinate ``j``: a coarse grid over [0, 1], then a fine grid where the coarse
    profile is within ``window`` of its minimum. Returns (u, mean, latent sd, path), sorted in u."""
    nv = noise_var(gp)

    def at(us):
        pts = [polished_point(gp, j, float(u), rng, dim, n_rand=n_rand, support=support) for u in us]
        return (np.array(us), np.array([p[1] for p in pts]),
                np.sqrt(np.maximum(np.array([p[2] for p in pts]) ** 2 - nv, 0)), np.array([p[0] for p in pts]))
    u, mu, lat, path = at(np.linspace(0, 1, n_coarse))
    ins = u[mu - mu.min() <= window]
    step = 1.0 / (n_coarse - 1)
    uf, muf, latf, pathf = at(np.linspace(max(0, ins.min() - step), min(1, ins.max() + step), n_fine))
    allu = np.concatenate([u, uf])
    o = np.argsort(allu)
    return allu[o], np.concatenate([mu, muf])[o], np.concatenate([lat, latf])[o], np.vstack([path, pathf])[o]


def segments(u, d, D: float) -> list[tuple[float, float]]:
    """Connected pieces of {u : d(u) <= D}, ends interpolated linearly."""
    inside = d <= D
    segs = []
    k = 0
    while k < len(u):
        if not inside[k]:
            k += 1
            continue
        a = k
        while k + 1 < len(u) and inside[k + 1]:
            k += 1
        b = k
        lo = u[a] if a == 0 else u[a - 1] + (D - d[a - 1]) / (d[a] - d[a - 1]) * (u[a] - u[a - 1])
        hi = u[b] if b == len(u) - 1 else u[b] + (D - d[b]) / (d[b + 1] - d[b]) * (u[b + 1] - u[b])
        segs.append((float(lo), float(hi)))
        k += 1
    return segs


def profiles(gp, space, seed: int, support: float | None = None, params=None) -> dict:
    """Supported profiles of every free parameter (or ``params``), as in the paper: ``{param: (u, mean,
    latent sd, path)}`` plus ``"mu_min"``, the lowest supported profile value. One generator, seeded
    ``[seed, 11]``, is shared by the profiles in ``params`` order."""
    space = as_space(space)
    support = support_cap(gp) if support is None else support
    rng = np.random.default_rng([seed, 11])
    params = list(space.free_params) if params is None else list(params)
    out = {p: profile(gp, space.free_params.index(p), space.dim, rng, support=support) for p in params}
    noise = np.sqrt(noise_var(gp))
    supported = [m[lat <= 2 * noise] for (_, m, lat, _) in out.values()]
    out["mu_min"] = min(x.min() for x in supported if len(x))
    return out


def profile_intervals(prof: dict, space, levels=(1.0, 4.0, 10.0)) -> dict:
    """``{param: {level: [(lo, hi), ...]}}`` in physical units from :func:`profiles` (several pieces where the
    region within the level is disconnected)."""
    space = as_space(space)
    out = {}
    for p, v in prof.items():
        if p == "mu_min":
            continue
        u, mu, _, _ = v
        d = mu - prof["mu_min"]
        out[p] = {}
        for D in levels:
            segs = segments(u, d, D)
            out[p][D] = [(_u2v(space, p, a), _u2v(space, p, b)) for a, b in segs]
    return out


def _u2v(space, p, u):
    lo, hi = space.bounds[p]
    if p in space.log_params:
        return float(10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))
    return float(lo + u * (hi - lo))


# ---------------------------------------------------------------- the posterior
def sample_posterior(gp, space, seed: int = 42, support: float | None = None, **emcee_kwargs) -> pd.DataFrame:
    """Plug-in posterior exp(-mu/2) on the supported surface, priors uniform in the unit cube. Returns the
    samples in physical units; ``.attrs`` holds ``unit`` (the unit-cube samples), ``acceptance`` and
    ``walkers``."""
    space = as_space(space)
    support = support_cap(gp) if support is None else support
    flat, acc, nk = run_emcee(supported_chi2(gp, support), np.zeros(space.dim), np.ones(space.dim), seed=seed,
                              **emcee_kwargs)
    df = space.from_unit(flat)
    df.attrs.update(unit=flat, acceptance=acc, walkers=nk)
    return df


def surface_draws(gp, samples_unit, n_sub: int = 3000, n_latent: int = 2000, n_refit: int = 300,
                  seed: int = 20261009, scatter: float | None = None):
    """Joint draws of the surrogate's error at a subsample of posterior samples.

    Returns ``(idx, D_latent, D_refit)``: the subsample (indices into ``samples_unit``), ``n_latent`` draws of
    the latent error (from the surrogate's joint latent covariance, the scatter removed), and ``n_refit``
    differences between the surrogate and refits to its training models with chi2 redrawn as mean +
    N(0, scatter^2) (hyperparameters fixed; 0 to skip). Each row of a D array is one realization of the
    error at the ``n_sub`` points.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    rng = np.random.default_rng(seed)
    n_sub = min(n_sub, len(samples_unit))
    idx = rng.choice(len(samples_unit), n_sub, replace=False)
    Xs = samples_unit[idx]
    _, cov = gp.gp.predict(Xs, return_cov=True)
    cov = cov * gp.scale ** 2 - noise_var(gp) * np.eye(n_sub)
    D_lat = (_cholesky(cov) @ rng.standard_normal((n_sub, n_latent))).T
    D_ref = None
    if n_refit:
        scatter = np.sqrt(noise_var(gp)) if scatter is None else scatter
        g0 = gp.gp
        Xt = g0.X_train_
        mu_t, mu_s = gp.predict(Xt), gp.predict(Xs)
        D_ref = []
        for _ in range(n_refit):
            yb = mu_t + scatter * rng.standard_normal(len(mu_t))
            g2 = GaussianProcessRegressor(kernel=g0.kernel_, optimizer=None).fit(Xt, (yb - gp.level) / gp.scale)
            D_ref.append(g2.predict(Xs) * gp.scale + gp.level - mu_s)
        D_ref = np.array(D_ref)
    return idx, D_lat, D_ref


def _cholesky(cov):
    """Cholesky factor of a latent covariance with a jitter of 1e-6 x its mean variance (the paper's value),
    raised tenfold until the factorization succeeds: posterior samples crowded far closer together than
    the kernel's length scales make the matrix numerically singular."""
    scale = np.mean(np.diag(cov))
    for jit in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        try:
            L = np.linalg.cholesky(cov + jit * scale * np.eye(len(cov)))
        except np.linalg.LinAlgError:
            continue
        if jit > 1e-6:
            import warnings
            warnings.warn(f"surface_draws: latent covariance needed a jitter of {jit:g} x its mean variance", stacklevel=3)
        return L
    raise np.linalg.LinAlgError("latent covariance is not positive definite even with a jitter of 1e-2; "
                                "use a smaller n_sub")


def reweight(D, prior_w=None) -> np.ndarray:
    """Normalized weights (one row per draw) that turn the plug-in posterior samples into the posterior of
    each realization of the surface: exp(-(D - mean D)/2) x the prior ratio ``prior_w``."""
    W = np.exp(-(D - D.mean(1, keepdims=True)) / 2)
    if prior_w is not None:
        W = W * prior_w
    return W / W.sum(1, keepdims=True)


def pooled_quantiles(values, D, prior_w=None, q=(15.85, 50.0, 84.15)) -> np.ndarray:
    """Quantiles of ``values`` (one per subsampled posterior sample) under the surrogate-marginalized
    posterior: the samples reweighted by every draw of ``D`` and pooled with equal weight per draw."""
    W = reweight(D, prior_w)
    x = np.tile(np.asarray(values), len(D))
    w = W.ravel()
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return np.interp(np.array(q) / 100, cw, x[o])


def mode_shares(D, select, prior_w=None) -> np.ndarray:
    """Share (0-1) of the posterior in the samples where ``select`` is True, for every draw of ``D`` -- the
    uncertainty of the split between two solutions."""
    W = reweight(D, prior_w)
    return W[:, np.asarray(select)].sum(1)


# ---------------------------------------------------------------- profile intervals under the scatter
class _Refit:
    """The surrogate refitted to resampled chi2, with the interface of the inference surrogate."""

    def __init__(self, g2, ref):
        self.gp, self.scale, self.level, self.kernel_ = g2, ref.scale, ref.level, g2.kernel_

    def predict(self, U, return_std=False):
        if return_std:
            m, s_ = self.gp.predict(U, return_std=True)
            return m * self.scale + self.level, s_ * self.scale
        return self.gp.predict(U) * self.scale + self.level


def profile_interval_bootstrap(gp, space, prof: dict, params, level: float = 1.0, n_boot: int = 100, seed: int = 0,
                               support: float | None = None, zoom_dchi2: float = 6.0) -> pd.DataFrame:
    """The Delta chi2 = ``level`` profile interval as the scatter moves it: ``n_boot`` parametric-bootstrap
    refits of the surrogate (each training model's chi2 replaced by the surrogate mean there plus a draw from
    N(0, sigma_chi2^2), hyperparameters fixed), each with its own minimum and its own profile of each
    parameter in ``params`` (15 values over the region where the fiducial profile ``prof`` is within
    ``zoom_dchi2``, padded by 0.03). Returns one row per refit with ``<param>_lo`` and ``<param>_hi``; the
    paper quotes the 16th percentile of the lower ends and the 84th of the upper ends.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    space = as_space(space)
    dim = space.dim
    support = support_cap(gp) if support is None else support
    noise = float(np.sqrt(noise_var(gp)))
    Xt = gp.gp.X_train_
    mu_t = gp.predict(Xt)
    zoom = {}
    for p in params:
        u, mu = prof[p][0], prof[p][1]
        sel = u[mu - prof["mu_min"] <= zoom_dchi2]
        zoom[p] = (max(0, sel.min() - 0.03), min(1, sel.max() + 0.03))
    rngb = np.random.default_rng([seed, 99])
    rows = []
    for _ in range(n_boot):
        ys = mu_t + noise * rngb.standard_normal(len(mu_t))
        g2 = GaussianProcessRegressor(kernel=gp.gp.kernel_, optimizer=None, normalize_y=False).fit(
            Xt, (ys - gp.level) / gp.scale)
        G = _Refit(g2, gp)
        U = rngb.random((40000, dim))
        m = G.predict(U)
        top = U[np.argsort(m)[:20]]
        L = np.clip(np.repeat(top, 300, 0) + 0.02 * rngb.standard_normal((6000, dim)), 0, 1)
        ml = G.predict(L)
        k = int(np.argmin(ml))
        row = dict(space.from_unit(L[[k]]).iloc[0])
        m0 = float(ml[k])
        for p in params:
            j = space.free_params.index(p)
            us = np.linspace(*zoom[p], 15)
            mus = np.array([polished_point(G, j, float(u), rngb, dim, n_rand=3000, support=support)[1] for u in us])
            s = segments(us, mus - m0, level)
            row[p + "_lo"], row[p + "_hi"] = ((_u2v(space, p, min(a for a, _ in s)), _u2v(space, p, max(b for _, b in s)))
                                              if s else (np.nan, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def loo(gp):
    """Closed-form leave-one-out residuals of the training models (chi2 - prediction without the model), the
    predictive variance without it, and that prediction, in chi2 units."""
    from scipy.linalg import cho_solve
    g = gp.gp
    Kinv = cho_solve((g.L_, True), np.eye(len(g.X_train_)))
    r = g.alpha_ / np.diag(Kinv) * gp.scale
    y = g.y_train_ * gp.scale + gp.level
    return r, gp.scale ** 2 / np.diag(Kinv), y - r


__all__ = ["run_emcee", "quantiles", "support_cap", "supported_chi2", "polished_point", "profile", "segments",
           "profiles", "profile_intervals", "sample_posterior", "surface_draws", "reweight", "pooled_quantiles",
           "mode_shares", "profile_interval_bootstrap", "loo"]
