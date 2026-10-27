# Configuration

## The space file

```yaml
name: M86
parameters:
  upsilon:     {bounds: [3.65, 10.94]}                # free, linear axis
  mbh:         {bounds: [3.0e+7, 5.0e+9], scale: log}  # free, log10 axis
  v_c:         {bounds: [0.0, 800.0]}
  r_c:         {bounds: [0.1, 30.0], scale: log}
  inclination: {value: 89.0}                           # fixed: copied into every round
  xi:          {value: 1.0}
```

- A parameter with `bounds` is free; with `value` it is fixed. The explicit form `fixed: true/false` is accepted
  too.
- `scale: log` samples uniformly in $\log_{10}$; the lower bound must be positive. Use it for anything that
  spans more than a decade or two. On a linear axis from $3\times10^7$ to $5\times10^9$, 95% of a design would
  lie above $2.5\times10^8$.
- The free parameters are ordered alphabetically (this fixes the Sobol dimensions), whatever the file order.
- Write exponents with a sign (`5.0e+9`): YAML reads `5.0e9` as a string. The package converts it, but other
  tools may not.

The same structure can be passed as a Python dict to `ParameterSpace.from_dict`.

## Settings

Pass settings to `Campaign.create(..., **settings)`, or to `sagari init --settings settings.yaml` (with
`--seed`, `--n-initial`, `--n-round`, `--converge-on` as shortcuts). They are stored in `campaign.yaml` and can be
edited there between rounds. **The defaults are the adopted settings of the paper**, validated on two real
campaigns (Section 4 of the paper). Change them only for a reason.

| setting | default | meaning |
| --- | --- | --- |
| `seed` | 20260817 | Sobol scramble and every random choice. Same seed and box → same global points |
| `n_initial` | 256 | size of round 0. A power of two keeps the Sobol net exact |
| `n_round` | 128 | models per search round |
| `converge_on` | all free | parameters the stopping rule watches. Leave nuisance parameters out |
| `stop.frac` | 0.2 | stop when both ends of every watched range move by less than this × its width |
| `stop.patience` | 1 | … for this many consecutive rounds |
| `stop.min_rounds` | 2 | … and not before this many refinement rounds |
| `stop.max_rounds` | 20 | give up the search after this many refinement rounds (then error bars) |
| `sampler.kernel` | matern52 | GP kernel (`rbf` is available, but fails on rough surfaces) |
| `sampler.train_window` | 30 | fit the main GP to models within this × σ_χ² of the minimum |
| `sampler.global_fraction` | 0.35 | share of each search round that continues the Sobol sequence |
| `estimate.n_train_cap` | 300 | most models the monitoring surrogate is fitted to |
| `errorbars.n_round` | 128 | models per error-bar round |
| `errorbars.n_profile` | 7 | points on each parameter's profile per round (the rest fill the region) |
| `errorbars.n_train_cap` | 1000 | most models the inference surrogate is fitted to: raise it until the window decides |
| `errorbars.sd_tol` | 3 | error bars are done when the latent sd at every interval end is at most this |
| `errorbars.frac` | 0.2 | … and the ends moved by less than this × the width during the round |
| `errorbars.levels` | [4, 10] | the Δχ² levels whose ends are checked |
| `errorbars.max_rounds` | 3 | most error-bar rounds |
| `errorbars.auto_window` | false | choose the training window from the scatter near the floor (unstable on some surfaces) |

The fixed constants below these settings, and the reason for each, are listed in [Fixed constants](constants.md).

Any other keyword of {py:func}`sagari.propose_round` can be added under `sampler` (for example
`noise_diagnostic: false`, as in the paper's replays).

## Round sizes and dimension

The paper used an initial design of $4^d$ points (256 for its four-parameter campaigns, 64 for its
three-parameter campaign without a black hole), the same density per parameter, and search rounds of half that.
Keep at least 64 initial points in any dimension: the surrogates need a few tens of finite models before their
first fit. Smaller rounds mean more rounds (more wall-clock time) but not fewer models; in the paper's replays,
rounds of 64 for four parameters lost recovery in some seeds.

## Which parameters to watch

`converge_on` should list the parameters you will report: for a black-hole campaign, the mass and the stellar
mass-to-light ratio. Halo parameters are often weakly constrained, and their ranges may never settle. Leaving them
out means the campaign stops when the quantities you care about have.
