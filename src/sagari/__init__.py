"""SAGARI: Schwarzschild Adaptive Gaussian-process Acquisition, Refinement, and Inference.

Adaptive Gaussian-process sampling for orbit-superposition modeling and other expensive, noisy chi2 surfaces.

The method of Waters & Gultekin (2026), "Schwarzschild Orbit-library Modeling with an Order of Magnitude
Fewer Models": a scrambled-Sobol initial design, refinement rounds by feasibility-weighted Expected
Improvement under a Gaussian-process surrogate with an explicit term for the model-to-model scatter of chi2,
a stopping rule on smoothed parameter ranges, error-bar rounds that test the surrogate out of sample, and a
posterior sampled on the final surrogate. It needs only the parameters and chi2 of each model, so it can
drive any modeling code.

Quick start::

    import sagari as sg
    space = sg.ParameterSpace.from_dict({"parameters": {
        "mbh": {"bounds": [3e7, 5e9], "scale": "log"}, "upsilon": {"bounds": [4, 12]}}})
    c = sg.Campaign.create("my_campaign", space, converge_on=["mbh", "upsilon"])
    results = c.run(my_chi2_function)              # or, on a cluster: c.next_round(results_csv)
    gp, _, _ = c.surrogate(results)
    post = sg.posterior.sample_posterior(gp, c.space)
"""
from . import errorbars, gp, monitor, posterior, replay
from ._version import __version__
from .acquisition import initial_design, propose_round
from .campaign import DEFAULTS, Campaign, read_results
from .errorbars import errorbar_round, gp_profiles, inference_gp, out_of_sample, profile_verdict
from .gp import PessimisticGP, feasibility_gp, fit_gp, training_window
from .monitor import estimate_converged, estimate_moved, smoothed_estimate, trusted_best_point
from .sobol import DEFAULT_SEED, SobolGlobalDesign
from .space import ParameterSpace

__all__ = [
    "__version__", "ParameterSpace", "SobolGlobalDesign", "DEFAULT_SEED", "Campaign", "DEFAULTS", "read_results",
    "initial_design", "propose_round", "smoothed_estimate", "estimate_moved", "estimate_converged",
    "trusted_best_point", "inference_gp", "errorbar_round", "gp_profiles", "profile_verdict", "out_of_sample",
    "PessimisticGP", "feasibility_gp", "fit_gp", "training_window",
    "gp", "monitor", "errorbars", "posterior", "replay",
]
