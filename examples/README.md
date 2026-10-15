# Examples

## Notebooks (`notebooks/`, stored executed)

| notebook | what it shows | runtime |
| --- | --- | --- |
| `01_quickstart.ipynb` | a complete campaign on an analytic, noisy χ² in Python: space, model, run, status, posterior | ~15 s |
| `02_cluster_workflow.ipynb` | the file-based workflow for cluster jobs through the command line: round files, results tables, failed and pending models, narrowing the box, the posterior command | ~1 min |
| `03_analysis.ipynb` | the final analysis on a two-solution problem: the surrogate, Δχ² profiles and their bootstrap, the plug-in and surrogate-marginalized posteriors, the uncertainty of the split between solutions, priors as reweighting | ~2–3 min |
| `04_benchmark_and_replay.ipynb` | the two-basin benchmark of the paper (full method vs. ablated vs. Sobol) and a replay on an archive of models | ~15 min |

Run them with `pip install "sagari[plot]" jupyter`. They write their campaigns next to themselves.

## Scripts and files

- `quickstart.py`: the first notebook as a script.
- `sco_runalls.py`: the adapter between a campaign directory and the Fortran orbit-superposition pipeline of the
  paper. It writes a `runallt` commands file for a round and collects the `cres.*` files into a results table.
- `space_m86.yaml`, `settings_paper.yaml`: the search space and settings of the paper's M 86 campaign.
- `../benchmarks/two_basin.py`: the paper's two-basin benchmark (100 trials per sampler).
