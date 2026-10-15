# How it works

A campaign is a sequence of **rounds**. Each round is a batch of parameter sets evaluated together, typically
as one job array, so wall-clock time is set by the number of rounds, not the number of models. The full
description, with the failures that motivated each choice, is Section 3 of Waters & Gültekin (2026).

## The problem: an expensive, noisy χ²

An orbit-superposition model returns a χ² that is **expensive** (hours per model), **not differentiable**, and
**noisy**. The orbit library is finite, its launch grid is discrete, the weight solve stops on a tolerance, and
each orbit is projected with pseudo-random azimuths. Two nearly identical potentials therefore return χ² values
that differ by $\sigma_{\chi^2}\sim10$ for typical data: comparable to the differences between competing
solutions. The method treats each evaluation as

$$ y_i = f(\boldsymbol\theta_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal N(0, \sigma_{\chi^2}^2), $$

and never trusts a single low model. With scatter, the lowest of many models is systematically too low.

## The stages

**1. The unit cube.** Every free parameter is mapped to [0, 1], linearly or in $\log_{10}$. Designs,
surrogates and distances all live in this cube, so no parameter's units enter the algorithm. The choice of axis
is a choice of prior: a design uniform in the cube is log-uniform on a log axis.

**2. The initial design.** A scrambled Sobol sequence (256 points for four parameters) leaves no large gap in
which a second solution could hide. The sequence is persisted as (seed, number drawn). Every later round
continues it with its global share, so the campaign's global coverage is one low-discrepancy sequence.

**3. The surrogate.** A Gaussian process with an anisotropic Matérn 5/2 kernel and a white-noise (nugget) term.
The nugget is the estimate of $\sigma_{\chi^2}$, and it becomes the unit of everything else. The main surrogate
is fitted only to models within $30\sigma_{\chi^2}$ of the minimum. A stationary kernel fitted across four orders
of magnitude of χ² overshoots badly.

**4. The acquisition.** Expected Improvement is computed against the surrogate's lowest prediction at an
evaluated model, not the lowest raw χ²: one downward-scattered model would otherwise set a target nothing can
beat. EI is multiplied by the probability that a candidate lies inside the training window. That probability
comes from a second GP fitted to $\log_{10}(\chi^2-\chi^2_{\min}+20)$ of *all* models, failures included, which
keeps the search out of regions known to be hopeless or to fail. Candidates come from a uniform pool plus
perturbations at three scales around the best models. The batch is chosen greedily with a minimum separation,
isolated low models are verified by their neighbours, and 35% of every round continues the Sobol sequence.

**5. Stopping.** The surrogate mean is turned into tempered weights $w\propto\exp[-(\mu-\mu_{\min})/2T]$ with
$T=\sigma_{\chi^2}$ (models closer than the noise cannot be ranked). Each parameter gets a 16–84% range. The
search stops when, for the parameters you name, both ends of the range move by less than 0.2 of its width
between rounds, after at least two refinement rounds. Ranges, not best-fit values, are compared: when two
solutions lie within the scatter of each other, the best value jumps between them while the range covering both
is stable.

**6. Error-bar rounds.** EI concentrates models at the minimum, while the intervals are read at its edges.
Error-bar rounds fill the region within $\Delta\chi^2 = 10+2\sigma_{\chi^2}$ of the minimum, space-filling, plus
points along each parameter's profile. The surrogate's prediction for every model is saved before it runs, so
each round tests the surrogate out of sample. The rounds stop when the surrogate's own (latent) uncertainty at
the ends of the $\Delta\chi^2=4$ and 10 ranges is at most 3, or after three rounds.

**7. The constraints.** The posterior is sampled on the surrogate mean, $p\propto\exp(-\mu/2)$, only where
models constrain it. It is then averaged over the surrogate's own uncertainty. Δχ² profile intervals are also
available. See [Analysis](analysis.md).

## Why each piece is there

| component | the failure it prevents |
| --- | --- |
| nugget and denoised incumbent | chasing a single downward-scattered model |
| training window | a surrogate that overshoots by hundreds in χ² between models |
| feasibility GP | rounds spent in regions where every model failed |
| two-scale candidates, minimum separation | missing a minimum narrower than the candidate spacing; a batch collapsing onto one point |
| global share | never finding a second basin |
| ranges in the stopping rule | stopping (or never stopping) while the best fit flips between two solutions |
| minimum number of refinement rounds | premature stability in the wrong basin |
| error-bar rounds with out-of-sample checks | intervals read where the surface is unknown |

On a benchmark with a narrow, deeper minimum beside a broad one and the scatter of a real campaign, the full
method finds the deeper minimum in 91 of 100 trials. The same search without these components finds it in 50,
and Sobol sampling alone in none ([notebook 4](notebooks/04_benchmark_and_replay.ipynb)).
