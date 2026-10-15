# FAQ and troubleshooting

**How many models will a campaign take?**
For four parameters the paper's replays took 512–896 models for the search and about 1000–1300 in total,
including three error-bar rounds and a separate campaign without a black hole. The search stops when the answer
settles, so easy problems stop at the earliest round the rule allows (two refinement rounds).

**`only N finite models -- too few to fit a d-D Gaussian process`**
Most of the initial design failed or is outside the box. Check the bounds (and the failure handling of your
code), or start with a larger `n_initial`.

**`next` says a round has models without results.**
Every model of a round needs a row in the results table before the next round. Failed models count: give them
`chi2 = nan`. Check `sagari status` for which rounds are pending.

**Can I use it with something that is not χ²?**
Yes: any objective that behaves like χ² (lower is better, differences of a few are meaningful), for example
$-2\ln L$. Δχ² thresholds such as the training window and the error-bar levels are on that scale.

**How many parameters can it handle?**
It was validated in four dimensions (and three, for the campaign without a black hole). Nothing in the code is
specific to four, and the GP cost grows with the number of training models rather than the dimension, but higher
dimensions have not been tested. Scale the initial design with the dimension ([Configuration](configuration.md))
and check the out-of-sample test of the error-bar rounds.

**My model gives the same χ² every time for the same parameters. Is there still scatter?**
Yes, if neighbouring parameter sets give χ² values that are not smooth: the scatter is the roughness of the
surface between nearby models, not run-to-run randomness. The fitted nugget (`gp.scatter`) measures it.
If your model is genuinely smooth, the nugget fits small and nothing is lost.

**The stopping rule never fires.**
Usually a watched parameter is unconstrained (its range keeps moving across the box). Leave nuisance parameters
out of `converge_on`. `stop.max_rounds` ends the search in any case.

**The error-bar rounds never declare themselves done.**
This happens on surfaces whose scatter grows away from the minimum: the paper's NGC 5102 replays never fired
the rule, and the paper followed a fixed three rounds. `errorbars.max_rounds` (default 3) ends them. The
out-of-sample rms in `status` tells you whether the surrogate can be trusted regardless.

**Can I reuse models from an earlier campaign?**
Two campaigns with the same seed and box propose the same global points. Match proposals against the earlier
results by parameter values and give them the stored χ², but only if every fixed input (data, PSF, deprojection)
is identical.
