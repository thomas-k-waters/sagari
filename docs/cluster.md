# Campaigns on a cluster

When every model is a batch job, the campaign cannot call your code. Instead it lives in a directory, and you
move it forward one round at a time:

1. `sagari next` writes the next round as a CSV of parameter sets;
2. you run those models with your own code and job scripts;
3. you add their χ² to a results table and call `sagari next` again.

The decisions take seconds, so the campaign can run on a login node or in a small batch job that submits the
model jobs.

## Step by step

```bash
sagari init my_campaign --space space.yaml --converge-on mbh,upsilon
sagari next my_campaign
```

`init` writes `my_campaign/campaign.yaml` (the space and every setting) and `next` writes round 0:

```text
my_campaign/
  campaign.yaml           space and settings; edit the bounds here to narrow the box
  sobol_state.json        position in the global Sobol sequence
  rounds/round_000.csv    the models of round 0
  rounds/round_000.json   what the round was, and diagnostics
  log.jsonl               one line per decision
```

A round file has one row per model:

```text
model_id,mbh,r_c,upsilon,v_c,inclination,stage
r000_0000,327526612.5,2.942406,2.610645,616.708115,89,sobol_global
r000_0001,1833280455,0.108644,8.878052,181.699415,89,sobol_global
...
```

`stage` says how the point was chosen: `sobol_global` (initial design), `gp_ei_local` (Expected Improvement),
`gp_ei_verify` (checking an isolated low model), `sobol_global_continuation` (the global share), or `errorbar`
(with `kind` = `fill` or `profile_<param>`, and the predictions `pred_mu`, `pred_sd`, `pred_latent`).

## The results table

A CSV (or whitespace-separated table with a header) with a row per evaluated model:

```text
model_id,chi2
r000_0000,1043.7
r000_0001,nan
```

- **Failed models**: write `nan`, `inf`, or leave `chi2` empty, or add a column `failed` with `1`/`true`. Keep
  them: the search uses them to stay out of regions where models fail.
- **Parameters actually run**: if your code rounds the values it was given (say to six significant digits), add
  the parameter columns with the values used; they replace the proposed ones.
- **No `model_id`?** Rows are matched to proposals by their parameter values (relative tolerance $10^{-6}$).
- **Order** does not matter; append as jobs finish.

`next` refuses while a round still has models without results (a failed model counts as evaluated).

```bash
sagari next   my_campaign --results results.csv    # the next round, or "done"
sagari status my_campaign --results results.csv    # where the campaign stands
```

`status` prints, for every search round, the monitored ranges and how far they moved, whether the stopping rule
has fired, and for every error-bar round the out-of-sample test and the error-bar verdict. Add
`--json status.json` to keep it.

## When it is done

```bash
sagari posterior my_campaign --results results.csv --out posterior --profiles
```

writes `posterior/posterior_samples.csv` and `posterior/summary.json`: the plug-in and surrogate-marginalized
16/50/84% quantiles of every free parameter and, with `--profiles`, the Δχ² = 1, 4 and 10 profile intervals. For
anything more, use the Python API ([Analysis](analysis.md)).

## Connecting your pipeline

All the glue a pipeline needs is two functions: round file → jobs, and finished jobs → results table.
`examples/sco_runalls.py` does this for the Fortran orbit-superposition pipeline of the paper:

```bash
python examples/sco_runalls.py commands my_campaign 3 --apparent-q 0.625 --out round_003.run
#   -> one "runallt <upsilon> <xi> <mbh> <v_c> <r_c> <apparent_q> <inclination>" line per model
python examples/sco_runalls.py collect my_campaign path/to/modeling/res --out results.csv
#   -> reads the cres.<libn> files of every model run so far
```

From Python, the same loop is

```python
c = sg.Campaign("my_campaign")
d = c.next_round("results.csv")        # a DataFrame, also written to rounds/round_NNN.csv; None when done
```

## Changing course

- **Narrow the box**: edit the bounds in `campaign.yaml`. Models outside the new box stay in the count but leave
  every fit, and the monitoring history is recomputed for the new box. (In the paper the floor on the stellar
  M/L was raised mid-campaign this way.)
- **Force a round type**: `sagari next ... --phase errorbar` (or `search`).
- **More error-bar rounds**: raise `errorbars.max_rounds` in `campaign.yaml`.
- **Restart cleanly**: a new directory. Two campaigns with the same seed and box propose the same global points,
  so models already computed can be reused by matching parameters.
