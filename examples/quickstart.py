"""A complete campaign on an analytic, noisy chi2 surface, in a few seconds.

Replace ``my_model`` by anything that returns chi2 for a set of parameters -- a wrapper around an orbit
code, a call to a cluster queue, a lookup in an archive. For a code that runs on a cluster one model per job,
use the file-based workflow instead (README.md, "On a cluster").

    python quickstart.py
"""
import numpy as np

import sagari as sg
from sagari import posterior as po

# 1. The search space: free parameters with bounds (and log axes where the range spans decades), fixed ones.
space = sg.ParameterSpace.from_dict({"name": "toy", "parameters": {
    "mbh": {"bounds": [1e6, 1e10], "scale": "log"},
    "upsilon": {"bounds": [1.0, 10.0]},
    "v_c": {"bounds": [0.0, 500.0]},
    "inclination": {"value": 60.0},
}})


# 2. The model: chi2 for each row of a DataFrame of parameters (NaN for a failed model). Here a tilted valley
#    in (log M_BH, Upsilon), insensitive to v_c, with model-to-model scatter of sd 5.
def my_model(design):
    x = np.log10(design["mbh"].values) - 8.3
    y = design["upsilon"].values - 6.0
    z = (design["v_c"].values - 200.0) / 300.0
    chi2 = 1000 + 60 * (x + 0.3 * y) ** 2 + 8 * y ** 2 + 5 * z ** 2
    return chi2 + np.random.default_rng(len(design)).normal(0, 5.0, len(chi2))


# 3. The campaign: 64 Sobol points, rounds of 32, stop on the ranges of M_BH and Upsilon, then error-bar rounds.
#    (The paper used 256 and 128 for four parameters; scale the round size with the dimension.)
c = sg.Campaign.create("toy_campaign", space, overwrite=True, n_initial=64, n_round=32,
                        converge_on=["mbh", "upsilon"], errorbars={"n_round": 32, "n_profile": 4})
results = c.run(my_model)

# 4. The posterior on the final surrogate, and the same averaged over the surrogate's own uncertainty.
gp, _, _ = c.surrogate(results)
post = po.sample_posterior(gp, space, nsteps=5000, burn=1000)
idx, D, _ = po.surface_draws(gp, post.attrs["unit"], n_sub=1500, n_latent=500, n_refit=0)
for p in ("mbh", "upsilon"):
    m, lo, hi = np.percentile(post[p], [50, 15.85, 84.15])
    lo2, m2, hi2 = po.pooled_quantiles(post[p].values[idx], D)
    print(f"{p:8s} plug-in {m:.4g} [{lo:.4g}, {hi:.4g}]   surrogate-marginalized {m2:.4g} [{lo2:.4g}, {hi2:.4g}]")
print(f"{len(results)} models in total")
