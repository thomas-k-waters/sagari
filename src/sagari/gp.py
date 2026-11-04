"""Gaussian-process surrogates of the chi2 surface (Section 3.4 of the paper).

All surrogates are fitted in the unit cube with an anisotropic kernel plus a white-noise (nugget) term. The
nugget is the estimate of the model-to-model scatter of chi2, sigma_chi2, which the rest of the method
uses as its unit: the training window, the temperature of the monitoring weights, and the band within
which a low model counts as indistinguishable from the best.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from .space import ParameterSpace

KERNELS = ("matern52", "rbf")


def make_kernel(dim: int, kernel: str = "matern52"):
    """Constant x (Matern 5/2 or squared-exponential, one length scale per axis) + white noise."""
    if kernel == "rbf":
        shape = RBF(length_scale=[0.3] * dim, length_scale_bounds=(1e-3, 10.0))
    elif kernel == "matern52":
        shape = Matern(length_scale=[0.3] * dim, length_scale_bounds=(1e-3, 10.0), nu=2.5)
    else:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    return (ConstantKernel(1.0, (1e-3, 1e6)) * shape
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e3)))


def fit_gp(X, y, dim: int, seed: int, kernel: str = "matern52") -> GaussianProcessRegressor:
    """A GP with constant prior mean (the training mean: ``normalize_y``) fitted by maximum marginal
    likelihood, with one optimizer restart."""
    gp = GaussianProcessRegressor(kernel=make_kernel(dim, kernel), normalize_y=True, n_restarts_optimizer=1,
                                  random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def fitted_scatter(gp: GaussianProcessRegressor, y) -> float:
    """sigma_chi2 of a GP from :func:`fit_gp`: its nugget in chi2 units (``normalize_y`` divides y by its sd)."""
    return float(np.sqrt(gp.kernel_.k2.noise_level) * np.std(y))


class PessimisticGP:
    """A GP whose prior mean is a fixed, pessimistic ``level`` instead of the training mean, so that far from
    every model it predicts ``level`` (the feasibility threshold) rather than an average of the models.

    The monitoring and inference surrogate of the paper. ``predict`` returns chi2 units; ``gp`` is the
    underlying scikit-learn regressor (fitted to ``(y - level) / scale``).

    The level is capped at min(y) + ``level_cap`` x the range of y (``level_cap=None``: the uncapped level of
    the paper and of versions before 1.1.0). A level far above every training model must be bridged by a
    broad bump: the fitted length scales then exceed the box, the mean never returns to the level inside it,
    and the minimum is a small ripple on the bump, so errors in the bump's shape appear as dips below the
    minimum far from every model. See ``docs/constants.md``.
    """

    def __init__(self, X, y, level: float, dim: int, seed: int, kernel: str = "matern52",
                 level_cap: float | None = 10.0):
        y = np.asarray(y, float)
        if level_cap is not None:
            level = min(float(level), float(y.min()) + level_cap * max(float(np.ptp(y)), 1.0))
        self.level, self.scale = float(level), max(float(np.std(y)), 1e-9)
        self.gp = GaussianProcessRegressor(kernel=make_kernel(dim, kernel), normalize_y=False, n_restarts_optimizer=1,
                                           random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(X, (np.asarray(y) - self.level) / self.scale)
        self.kernel_ = self.gp.kernel_

    def predict(self, X, return_std: bool = False):
        if return_std:
            m, sd = self.gp.predict(X, return_std=True)
            return m * self.scale + self.level, sd * self.scale
        return self.gp.predict(X) * self.scale + self.level

    @property
    def noise_var(self) -> float:
        """sigma_chi2^2, the nugget in chi2 units."""
        return noise_var(self)

    @property
    def scatter(self) -> float:
        """sigma_chi2, the model-to-model scatter the surrogate separates from the smooth surface."""
        return float(np.sqrt(noise_var(self)))

    def predict_latent(self, X):
        """(mean, latent sd): the predictive sd with the model-to-model scatter removed (s_f of the paper)."""
        m, s = self.predict(X, return_std=True)
        return m, np.sqrt(np.maximum(s ** 2 - noise_var(self), 0.0))


def noise_var(gp: PessimisticGP) -> float:
    """Nugget of a :class:`PessimisticGP` in chi2^2 units."""
    return float(gp.kernel_.k2.noise_level) * gp.scale ** 2


def feasibility_gp(table: pd.DataFrame, space: ParameterSpace, seed: int, feasible_factor: float = 10.0,
                   max_train: int = 2000, n_box: int = 20000):
    """The feasibility GP of Section 3.5 of the paper: a squared-exponential GP on log10(chi2 - min + 20) of all models
    (failed ones at the worst value seen; above ``max_train`` models, the 300 lowest plus a random sample of
    the rest), and the hopelessness threshold: ``feasible_factor`` x the median Delta chi2 that this GP
    predicts over the whole box. A box median, not the median of the models run, keeps the threshold from
    drifting inward as a campaign concentrates near the minimum.

    Returns ``(gp, chi2_min, threshold_dchi2)``.
    """
    finite = table[np.isfinite(table["chi2"])]
    chi2_min = float(finite["chi2"].min())
    allf = table
    if len(allf) > max_train:
        keep = finite.nsmallest(min(300, len(finite)), "chi2")
        allf = pd.concat([keep, allf.drop(keep.index).sample(max_train - len(keep), random_state=seed)])
    dim = space.dim
    Xf = space.to_unit(allf)
    yf = np.log10(allf["chi2"].values - chi2_min + 20.0)
    yf = np.where(np.isfinite(yf), yf, np.nanmax(yf))
    gpf = fit_gp(Xf, yf, dim, seed, kernel="rbf")
    box = np.random.default_rng([seed, 77]).random((n_box, dim))
    box_median = float(np.median(10 ** gpf.predict(box) - 20.0))
    return gpf, chi2_min, feasible_factor * max(box_median, 1e-9)


def fit_clipped(X, train: pd.DataFrame, usable: pd.DataFrame, space: ParameterSpace, level: float, seed: int,
                kernel: str = "matern52", clip: bool = True, level_cap: float | None = 10.0,
                n_iter: int = 3) -> PessimisticGP:
    """A :class:`PessimisticGP` on the training models (``X``, the rows of ``train``). With ``clip=True``, the
    other ``usable`` models (above the training window, or beyond the training cap) that the GP predicts below
    the training set's top are added at that top, the ceiling, and the GP refitted (up to ``n_iter`` times).

    Dropped, the models outside the window cannot stop the GP from extending the valley into a gap they
    surround: an NGC 4751 campaign put chi2_min + 3 between three models measured at chi2_min + 1068-1205.
    Only the models the fit contradicts are added, since all of them, clipped, make the GP fit a flat plateau
    around the valley, and the kink at its edge inflates the fitted scatter. ``clip=False``: the behavior of
    the paper and of versions before 1.2.0. The GP carries ``.ceiling`` (None without clipping) and
    ``.n_clipped``. See ``docs/constants.md``.
    """
    y = train["chi2"].values
    gp = PessimisticGP(X, y, level, space.dim, seed, kernel, level_cap=level_cap)
    gp.ceiling, gp.n_clipped = None, 0
    rest = usable.loc[~usable.index.isin(train.index)]
    if not clip or not len(rest):
        return gp
    ceiling = gp.ceiling = float(y.max())
    Xr = space.to_unit(rest)
    added = np.zeros(len(rest), bool)
    for _ in range(n_iter):
        new = ~added & (gp.predict(Xr) < ceiling)
        if not new.any():
            break
        added |= new
        gp = PessimisticGP(np.vstack([X, Xr[added]]), np.concatenate([y, np.full(int(added.sum()), ceiling)]),
                           level, space.dim, seed, kernel, level_cap=level_cap)
        gp.ceiling, gp.n_clipped = ceiling, int(added.sum())
    return gp


def training_window(X, chi2, dim: int, seed: int, train_window: float | None, kernel: str = "matern52",
                    n_min: int | None = None):
    """The training window of Section 3.4 of the paper: keep the models within ``train_window`` x sigma_chi2 of the
    minimum, sigma_chi2 from a first fit to all of ``X``; at least ``n_min`` (default 10 x dim) are kept.

    A stationary kernel fitted over a range many times the scatter has to bend steeply and overshoots
    (M 86: a GP fitted to chi2 = 992-3671 with scatter ~15 predicted 730 where models later measured
    1180-1290). Returns ``(mask, scatter)``; ``scatter`` is None when no window is applied.
    """
    n_min = n_min or max(10 * dim, 2 * dim + 2)
    if train_window is None or len(chi2) <= n_min:
        return np.ones(len(chi2), bool), None
    gp0 = fit_gp(X, chi2, dim, seed, kernel)
    scatter = fitted_scatter(gp0, chi2)
    d = chi2 - chi2.min()
    keep = d <= train_window * max(scatter, 1.0)
    if keep.sum() < n_min:
        keep = d <= np.sort(d)[n_min - 1]
    return keep, scatter
