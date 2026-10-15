"""The synthetic two-basin objective of Appendix B of the paper.

An analytic chi2 in the four parameters of the M 86 pilot campaign (Upsilon and v_c on linear axes, M_BH and
r_c on logarithmic ones): a broad, shallow basin and a narrow, tilted one deeper by 50, with Gaussian scatter
of sd 47 per model, the scatter fitted to the pilot. Optionally a region of failed and very-high-chi2 models
(``hopeless=True``): the 15% of the box with u_Upsilon > 0.7 and u_v > 0.5, where 15% of models fail.

    >>> obj = TwoBasin(hopeless=True)
    >>> chi2 = obj(design_dataframe, rng)            # NaN for failed models
    >>> obj.found(estimate)                          # is a smoothed best point in the deeper basin?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .space import ParameterSpace

PILOT_SPACE = ParameterSpace(
    free_params=["upsilon", "mbh", "v_c", "r_c"],
    bounds={"upsilon": (1.0, 15.0), "mbh": (3.0e7, 5.0e9), "v_c": (0.0, 500.0), "r_c": (0.05, 20.0)},
    fixed_values={"xi": 1.0, "inclination": 89.0},
    log_params=frozenset({"mbh", "r_c"}),
    name="two_basin",
)

NOISE = 47.0                                     # chi2 scatter per model
SCALE = 35.0                                     # chi2 rise at one basin width
CENTRE_BROAD = dict(mbh=1.5e8, r_c=2.0, upsilon=9.3, v_c=60.0)
CENTRE_NARROW = dict(mbh=8.0e8, r_c=0.3, upsilon=3.7, v_c=350.0)
WIDTH_BROAD = {"upsilon": 0.06, "mbh": 0.25, "v_c": 0.40, "r_c": 1.00}
WIDTH_NARROW = {"upsilon": 0.035, "mbh": 0.15, "v_c": 0.05, "r_c": 0.08}
FAIL_ABOVE, FAIL_FRAC = 5e4, 0.15                # models with true Delta chi2 > 5e4 fail 15% of the time


class TwoBasin:
    """Callable objective: ``obj(design, rng) -> chi2`` (NaN for failures)."""

    def __init__(self, hopeless: bool = False, noise: float = NOISE, space: ParameterSpace = PILOT_SPACE):
        self.space, self.hopeless, self.noise = space, hopeless, noise
        self.P = {p: i for i, p in enumerate(space.free_params)}
        self.c_b = space.to_unit(pd.DataFrame([CENTRE_BROAD]))[0]
        self.c_n = space.to_unit(pd.DataFrame([CENTRE_NARROW]))[0]

    def true(self, U):
        """Noise-free chi2 at unit-cube points, and whether the narrow basin is the lower one there."""
        U = np.atleast_2d(U)
        P, cb, cn = self.P, self.c_b, self.c_n
        qB = sum(((U[:, P[p]] - cb[P[p]]) / WIDTH_BROAD[p]) ** 2 for p in P)
        du = (U[:, P["upsilon"]] - cn[P["upsilon"]]) / WIDTH_NARROW["upsilon"]
        dv = (U[:, P["v_c"]] - cn[P["v_c"]]) / WIDTH_NARROW["v_c"]
        # tilted valley: raising v_c compensates lowering Upsilon (along du ~ -dv the rise is 5x slower)
        qN = ((du + dv) / np.sqrt(2)) ** 2 + (((du - dv) / np.sqrt(2)) / 5.0) ** 2 \
            + ((U[:, P["mbh"]] - cn[P["mbh"]]) / WIDTH_NARROW["mbh"]) ** 2 \
            + ((U[:, P["r_c"]] - cn[P["r_c"]]) / WIDTH_NARROW["r_c"]) ** 2
        fB, fN = 1050 + SCALE * qB, 1000 + SCALE * qN
        f = np.minimum(fB, fN)
        if self.hopeless:
            f = f + 6e4 * self.in_hopeless(U) * (1 + 2 * (U[:, P["v_c"]] - 0.5) / 0.5)
        return f, fN < fB

    def in_hopeless(self, U):
        U = np.atleast_2d(U)
        return ((U[:, self.P["upsilon"]] > 0.7) & (U[:, self.P["v_c"]] > 0.5)).astype(float)

    def __call__(self, design: pd.DataFrame, rng) -> np.ndarray:
        f, _ = self.true(self.space.to_unit(design))
        chi2 = f + rng.normal(0, self.noise, len(f))
        fail = (f - 1000 > FAIL_ABOVE) & (rng.random(len(f)) < FAIL_FRAC)
        chi2[fail] = np.nan
        return chi2

    def found(self, estimate: dict) -> bool:
        """True if the best point of a monitoring estimate lies where the narrow basin is the lower one."""
        u = self.space.to_unit(np.array([[estimate[p][0] for p in self.space.free_params]]))
        return bool(self.true(u)[1][0])

    def mbh_error(self, estimate: dict) -> float:
        """abs(log10(M_BH / 8e8)) of a monitoring estimate's best point (dex)."""
        return float(abs(np.log10(estimate["mbh"][0] / CENTRE_NARROW["mbh"])))
