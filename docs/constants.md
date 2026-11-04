# Fixed constants and their rationale

The [settings](configuration.md) are the choices a user is expected to make. Below them, the method uses a number
of fixed constants. Each one is a keyword argument of the function named in the table, so it can be changed for
an experiment, but the values here are the ones used in the paper (Waters & Gültekin 2026). All distances are in
the unit cube, and σ_χ² is the model-to-model scatter fitted by the surrogate's nugget.

## Surrogates

| constant | value | where | why |
| --- | --- | --- | --- |
| kernel length-scale bounds | 10⁻³–10 | `gp.make_kernel` | wide enough for any structure the models can resolve. A length scale at the upper bound means the surface is flat along that axis (see the training cap below) |
| minimum training set | 10 × d models | `gp.training_window` | the surrogate needs several models per parameter to fit d length scales |
| feasibility offset c | 20 | `gp.feasibility_gp` | keeps log₁₀(χ² − χ²_min + c) finite at the minimum and flat within the scatter there, so the feasibility GP describes the walls and not the noise at the floor |
| feasibility threshold t | 10 × the median Δχ² predicted over the box | `gp.feasibility_gp` (`feasible_factor`) | marks as unpromising only the regions far worse than a typical point of the box. The median is taken over the box, not the models, so the threshold does not shrink as a campaign concentrates near the minimum |
| feasibility training set | all models up to 2000, then the 300 lowest plus a random sample | `gp.feasibility_gp` (`max_train`) | keeps the fit cheap while keeping the region near the minimum and an even sample of the rest |
| prior level of the monitoring and inference surrogates | χ²_min + t, capped at the lowest training χ² + 10 × the training range | `gp.PessimisticGP` (`level_cap`; settings `estimate.level_cap`, `errorbars.level_cap`) | far from every model the surface is assumed bad, not average. The cap keeps the level within reach of the training models. Since version 1.1.0; see below |
| models outside the training window | added back at the window top where the surrogate predicts them below it, up to 3 refits | `gp.fit_clipped` (`clip`, `n_iter`; settings `estimate.clip_window`, `errorbars.clip_window`) | a model far above the window still says the surface is bad where it sits. Since version 1.2.0; see below |

## Choosing the models of a search round

| constant | value | where | why |
| --- | --- | --- | --- |
| uniform candidates | 5000 | `propose_round` (`n_candidates`) | spacing of ~0.1 in four dimensions: enough to find the broad structure of EI everywhere |
| local candidates | 4000, Gaussian perturbations of widths 0.01, 0.03, 0.08 around the 20 models with the lowest predicted χ² | `propose_round` (`n_local_candidates`, `local_scales`) | resolve minima narrower than the uniform spacing, at three scales so that both a narrow valley and its surroundings are covered |
| minimum separation within a round | 0.04 | `propose_round` (`min_separation`) | spreads a round over the high-EI region instead of stacking it at one optimum. Models closer than ~0.05 have partly correlated scatter (paper, Appendix B), so packing them more tightly adds little independent information |
| isolated low models | up to 2 models within 2σ_χ² of the best with no neighbor within 0.1 | `propose_round` (`verify_max`, `support_ok`) | a real narrow minimum stays low in its neighbors; a model low only by scatter does not |
| confirmation step | ±0.03 along each axis (2d models per isolated model) | `propose_round` (`verify_step`) | close enough to sample the same minimum. At this distance the scatter of neighbors is still partly correlated, which makes the check somewhat weaker than it would be for independent values (paper, Appendix B) |

## Stopping rule and monitoring

| constant | value | where | why |
| --- | --- | --- | --- |
| temperature T | max(σ_χ², 1) | `monitor.smoothed_estimate` | models closer than the scatter cannot be ranked, so the monitoring weights must not treat such differences as real |
| random points for the weights | 6 × 10⁴ | `monitor.smoothed_estimate` (`n_uniform`) | a dense sample of the whole cube, so that the weighted percentiles are not limited by the number of points |
| local refinement of the minimum | 200 perturbations of width 0.02 around the 20 lowest-predicted models | `monitor.smoothed_estimate` | locates a narrow minimum that the random points miss |

## Error-bar rounds

| constant | value | where | why |
| --- | --- | --- | --- |
| region filled | Δχ² ≤ 10 + 2σ_χ² of the surrogate's minimum | `errorbar_round` (`dchi2`) | about the joint 95% region for four parameters, widened by the scatter so that the ends of the intervals are inside it |
| profile extension | 15% of the width plus 0.02 beyond each end | `errorbar_round` | places some models just outside the region, so that the surrogate is constrained where the intervals end |
| fill candidates | 2¹⁵ Sobol points, used in Sobol order | `errorbar_round` (`n_candidates`) | Sobol order keeps the chosen models evenly spread within the region |
| supported profile points | latent sd ≤ 2σ_χ² | `errorbars.gp_profiles` | below this, models constrain the surface; above it, a low value is an extrapolation, reported as a place to sample |

## Posterior

| constant | value | where | why |
| --- | --- | --- | --- |
| support cut | posterior set to zero where the latent sd exceeds 2σ_χ² | `posterior.support_cap` | the surrogate can extrapolate low values into empty regions, and a sampler would settle there |
| sampler | 32 walkers, 25000 steps, first 5000 discarded, every 5th kept | `posterior.run_emcee` | the settings of the paper. Check the chains of a new problem as for any MCMC |
| surrogate-marginalized posterior | 2000 joint draws of the latent error at 3000 posterior samples | `posterior.surface_draws` | enough draws for the quantiles to be stable; the subsample keeps the covariance matrix tractable. Check the effective sample size of the reweighted draws (`posterior.reweight`) |

## The training cap of the monitoring surrogate

The monitoring surrogate of the search, which decides when the search stops, is fitted to at most
`estimate.n_train_cap` models. The paper (and version 1.0.0) used 300. Once a search has placed several hundred
models in one basin, the 300 lowest span only a narrow range of χ², the surrogate loses the walls of the basin,
and its minimum can drift (paper, Section 4.1). On the paper's two-basin benchmark, the estimate fitted with a cap
of 1000 was better at every budget from 384 models on, and for M86 the two caps gave the same stopping decisions.
The cost is negligible next to that of the models (a few seconds more per round), so since version 1.0.1 the
default is 1000. To reproduce the paper's campaigns exactly, set `estimate: {n_train_cap: 300}`.

The GP that chooses the refinement models by Expected Improvement keeps its cap of 300
(`propose_round(n_train_cap=300)`), the value used and validated in the paper.

## The prior level of the monitoring and inference surrogates

The surrogates of `smoothed_estimate` and `inference_gp` (`gp.PessimisticGP`) assume that the surface far from every
model is bad. Their prior mean is χ²_min + t, the feasibility threshold, instead of the mean of the training
models. Those models lie within the training window of the minimum, but t can be far larger: in an NGC 4751
campaign the training χ² spanned 911–1210 and the level was 40540. The GP then fits a broad bump to bridge the
gap, with length scales longer than the box. The mean never returns to the level inside the box, the valley
becomes a 1% ripple on the bump, and errors in the bump's shape appear as dips 130–330 below the minimum, 0.35 from
any model. They took most of the monitoring weight and drove the profiles to the box edges.

Since version 1.1.0 the level is capped at the lowest training χ² + 10 × the training range. For the NGC 4751 models
the length scales are then 0.5–0.7, there are no dips, and the Δχ² ≤ 4 range of M_BH is 1.92–2.25 × 10⁹ instead of
running to the box edge. The value 10 is a compromise:
- With caps of about 20 and up, the NGC 4751 dips return.
- With caps below about 10, surrogates fitted to small training sets (20 models of a two-parameter test problem)
  fit too small a scatter more often, and their error-bar predictions become overconfident.
- For M86 v5 the profile ranges change by less than 2%.

To reproduce the paper, set `estimate: {level_cap: null}` and `errorbars: {level_cap: null}` in a campaign, or pass
`level_cap=None` to `smoothed_estimate`, `inference_gp`, `errorbar_round` or `PessimisticGP`.

## The models outside the training window

The training window keeps the surrogates' fit to the models within 30 σ_χ² of the minimum, so that a stationary
kernel does not have to bend over the whole range of χ² (see the training window above). Before version 1.2.0 the
models outside it were dropped. A model far above the window still says the surface is bad where it sits, and
without it the GP can extend the valley into a gap that such models surround. In an NGC 4751 campaign the inference
surrogate put χ²_min + 3 at M_BH 4.7 × 10⁸, v_c 200 km/s, 0.05–0.12 (unit cube) from three models measured at
χ²_min + 1068–1205. The false basin took monitoring weight (the M_BH range of one round reached 4.7 × 10⁸), made
the stopping rule's M_BH move 4.5 in the next, and put second minima in the profiles. 37 of the 547 models outside
the window were predicted below it, one measured at +1105 at +87.

Since version 1.2.0 (`gp.fit_clipped`), the surrogate is fitted as before, and the models outside the training
set that it predicts below the training set's top are added at that top and the GP refitted, up to three times.
For the same NGC 4751 models 42 models are added, the fitted scatter is 14.9 (13.8 before), the length scales are
unchanged, no model outside the window is predicted more than 5 σ_χ² below its top, and the profiles have a
single valley. Only the contradicted models are added: adding all of them, clipped, makes the GP fit a flat
plateau around the valley, and the kink at its edge inflates the fitted scatter (16.2 for the same models).

The models are clipped, not entered at their measured χ², because the window exists to keep the fit away from the
steep walls. A clipped model says only "at least the window top", which is all the surrogate needs to know.

To reproduce the paper or earlier versions, set `estimate: {clip_window: false}` and `errorbars: {clip_window:
false}`, or pass `clip_window=False` to `smoothed_estimate`, `inference_gp` or `errorbar_round`.
