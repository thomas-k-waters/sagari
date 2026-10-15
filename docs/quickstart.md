# Quick start

## In Python

If your model can be called from Python, a whole campaign is a few lines:

```python
import numpy as np
import sagari as sg

space = sg.ParameterSpace.from_dict({"parameters": {
    "mbh":         {"bounds": [1e6, 1e10], "scale": "log"},
    "upsilon":     {"bounds": [1.0, 10.0]},
    "v_c":         {"bounds": [0.0, 500.0]},
    "inclination": {"value": 60.0},               # fixed
}})

def my_model(design):
    """design: a DataFrame of parameter sets. Return one chi2 per row (NaN for a model that failed)."""
    ...

c = sg.Campaign.create("my_campaign", space, converge_on=["mbh", "upsilon"])
results = c.run(my_model)        # Sobol design, search rounds, stopping rule, error-bar rounds

gp, _, _ = c.surrogate(results)
post = sg.posterior.sample_posterior(gp, space)
```

`c.run` prints a line per round and finishes with the status of the campaign. The results are also written to
`my_campaign/results.csv`, so an interrupted run can be resumed by calling `c.run` again.

## On a cluster

When each model is a batch job, move between rounds with the command line:

```bash
sagari init my_campaign --space space.yaml --converge-on mbh,upsilon
sagari next my_campaign                         # rounds/round_000.csv: 256 parameter sets
# ... run them; write model_id,chi2 to results.csv ...
sagari next my_campaign --results results.csv   # the next round, until it reports "done"
sagari posterior my_campaign --results results.csv --out posterior --profiles
```

See [Campaigns on a cluster](cluster.md) and the notebook
[A campaign on a cluster](notebooks/02_cluster_workflow.ipynb).

## What to read next

- [How it works](concepts.md): the ideas behind each stage, and why they matter for noisy χ².
- [Configuration](configuration.md): the space file and every setting.
- [Analysis](analysis.md): from the final surrogate to a measurement.
