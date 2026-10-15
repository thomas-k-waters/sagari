# From the final surrogate to a measurement

The constraints come from the **inference surrogate**: a GP fitted to every model inside the training window,
with a nugget for the model-to-model scatter. Do not use the lowest model. With scatter, the lowest of many
models lies well below the smooth minimum (in M86, 17 below).

```python
import sagari as sg
from sagari import posterior as po

c = sg.Campaign("my_campaign")
gp, chi2_min, threshold = c.surrogate("results.csv")
print(gp.scatter)                    # sigma_chi2, the fitted model-to-model scatter
```

Outside a campaign directory, use `sg.inference_gp(table, space)` on any table of models.

## Profiles: Δχ² intervals

```python
prof = po.profiles(gp, space, seed=c.seed)                         # every free parameter
iv = po.profile_intervals(prof, space, levels=(1.0, 4.0, 10.0))     # {param: {level: [(lo, hi), ...]}}
```

A profile minimizes the surrogate mean over all other parameters at each value of one. Points are trusted only
where models constrain the surrogate (latent sd ≤ 2σ). Intervals can have several pieces when the region is
disconnected. The ends move when the scatter is redrawn; `profile_interval_bootstrap` refits the surrogate to
parametric-bootstrap copies and gives their spread.

## The posterior

```python
post = po.sample_posterior(gp, space)          # emcee: 32 walkers, 25000 steps, 5000 burn-in
post["mbh"].quantile([0.1585, 0.5, 0.8415])
```

This samples $p\propto\exp(-\mu/2)$ with priors uniform in the unit cube (log-uniform on log axes). The posterior
is zero wherever the surrogate's latent sd exceeds $2\sigma_{\chi^2}$: there no models constrain it, and a sampler
would settle in any low value the surrogate extrapolates. The samples in the unit cube are in
`post.attrs["unit"]`.

## Averaging over the surrogate's uncertainty

The surrogate is fitted to noisy models, so its mean is uncertain too. This matters most when two solutions
differ by less than the scatter:

```python
idx, D, D_refit = po.surface_draws(gp, post.attrs["unit"])        # 2000 latent draws at 3000 samples (+300 refits)
lo, med, hi = po.pooled_quantiles(post["mbh"].values[idx], D)     # the surrogate-marginalized posterior
shares = po.mode_shares(D, post["upsilon"].values[idx] > 7.0)     # how uncertain the split between two solutions is
```

`D` holds joint draws of the surrogate's latent error (the scatter removed, so it is not counted twice).
`D_refit` comes from refits to the models with their χ² redrawn from the scatter. Reweighting the samples by each
draw gives the posterior of that realization of the surface. Pooling the reweighted posteriors gives the
surrogate-marginalized posterior that the paper adopts.

## Other priors

Every prior is a reweighting of the samples. For a prior uniform in $r_c$ instead of $\log r_c$:

```python
w = post["r_c"].values[idx]
po.pooled_quantiles(post["mbh"].values[idx], D, prior_w=w)
po.mode_shares(D, select, prior_w=w)
```

## Checking the surrogate

- **Out of sample**: every error-bar round stores the prediction made before each model ran;
  `sagari status` reports the rms of $z=(\chi^2-\mu)/s$ (≈1 for a calibrated surrogate) and the fractions
  beyond ±2. `sg.out_of_sample(round_table, chi2)` computes it directly.
- **Leave-one-out**: `po.loo(gp)` returns the closed-form leave-one-out residuals of the training models.
- **The scatter**: `gp.scatter` should be stable once the region of the minimum is sampled.
  `sg.errorbars.window_check(gp)` compares it with the leave-one-out rms near the floor.

The [analysis notebook](notebooks/03_analysis.ipynb) runs all of this on a two-solution problem.
