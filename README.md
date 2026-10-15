# SAGARI

**Schwarzschild Adaptive Gaussian-process Acquisition, Refinement, and Inference**

[![Tests](https://github.com/thomas-k-waters/sagari/actions/workflows/tests.yml/badge.svg)](https://github.com/thomas-k-waters/sagari/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sagari.svg)](https://pypi.org/project/sagari/)
[![Documentation](https://readthedocs.org/projects/sagari/badge/?version=latest)](https://sagari.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23386326.svg)](https://doi.org/10.5281/zenodo.23386326)

Adaptive Gaussian-process sampling for Schwarzschild orbit-superposition modeling, and for any other model
whose goodness of fit is an expensive, noisy χ².

An orbit-superposition model costs hours of CPU time, and a nested parameter grid needs ~10⁴ of them.
SAGARI replaces the grid with a sequence of rounds, each placed where a surrogate of the χ² surface predicts
the largest gain, and stops the campaign when the answer stops moving. On the archived models of two published
black-hole mass measurements it recovered the published mass in every replayed trial with **9 to 30 times fewer
models** than the original campaigns.

It is the method of

> Waters & Gültekin (2026), *Schwarzschild Orbit-library Modeling with an Order of Magnitude Fewer Models:
> Adaptive Gaussian-process Sampling and a Case Study of M 86*

and reproduces the code used there exactly ([details](#fidelity-to-the-paper)).

The package never runs a model. It proposes parameter sets and reads back their χ², so it can drive any code: an
orbit-superposition pipeline on a cluster, a Jeans model, or a lookup in an archive.

📖 **Documentation:** https://sagari.readthedocs.io

## Installation

```bash
pip install sagari            # add [plot] for the example notebooks
```

Python ≥ 3.10. Depends on NumPy, SciPy, pandas, scikit-learn, emcee and PyYAML.

## Quick start

```python
import sagari as sg

space = sg.ParameterSpace.from_dict({"parameters": {
    "mbh":         {"bounds": [1e6, 1e10], "scale": "log"},
    "upsilon":     {"bounds": [1.0, 10.0]},
    "v_c":         {"bounds": [0.0, 500.0]},
    "inclination": {"value": 60.0},            # fixed
}})

def my_model(design):                          # a DataFrame of parameter sets -> one chi2 per row
    ...                                        # (NaN for a model that failed)

c = sg.Campaign.create("my_campaign", space, converge_on=["mbh", "upsilon"])
results = c.run(my_model)                      # Sobol design, search rounds, stopping rule, error-bar rounds

gp, _, _ = c.surrogate(results)                # the final surrogate of the chi2 surface
post = sg.posterior.sample_posterior(gp, space)
```

## On a cluster

When each model is a batch job, the campaign lives in a directory and moves one round at a time:

```bash
sagari init my_campaign --space space.yaml --converge-on mbh,upsilon
sagari next my_campaign                          # writes my_campaign/rounds/round_000.csv
#   ... run the models of the round with your own code; add model_id,chi2 rows to results.csv ...
sagari next my_campaign --results results.csv    # the next round (search or error bars), or "done"
sagari status my_campaign --results results.csv  # ranges per round, stopping rule, out-of-sample checks
sagari posterior my_campaign --results results.csv --out posterior --profiles
```

Failed models are written as `chi2 = nan` and kept: the search learns to avoid where models fail.
`examples/sco_runalls.py` connects a campaign to the Fortran pipeline of the paper (`runallt` commands file in,
`cres` files out).

## How it works

1. **Initial design**: a scrambled Sobol sequence over the box, continued by every later round.
2. **Search rounds**: a Gaussian process with a Matérn 5/2 kernel and an explicit nugget for the model-to-model
   scatter of χ², fitted near the minimum. A second GP fitted to every model keeps the search out of regions that
   are hopeless or where models fail. Each round places 65% of its models by feasibility-weighted Expected
   Improvement, with a denoised incumbent, two-scale candidates and a minimum separation, and verifies isolated
   low models by their neighbours. The other 35% continues the Sobol sequence.
3. **Stopping rule**: smoothed 16–84% ranges of the parameters you care about. The search stops when they move
   by less than 0.2 of their width, after at least two refinement rounds.
4. **Error-bar rounds**: models fill the acceptable region and the parameter profiles. Each model's χ² is
   predicted before it runs, so the surrogate is tested out of sample.
5. **Posterior**: MCMC on the final surrogate where models constrain it, averaged over the surrogate's own
   uncertainty, plus Δχ² profile intervals.

The defaults are the settings adopted and validated in the paper.

## Example notebooks

| notebook | |
| --- | --- |
| [01 Your first campaign](examples/notebooks/01_quickstart.ipynb) | a complete campaign in Python on a noisy analytic χ² |
| [02 A campaign on a cluster](examples/notebooks/02_cluster_workflow.ipynb) | round files, results tables, failures, narrowing the box, the CLI |
| [03 From the surrogate to a measurement](examples/notebooks/03_analysis.ipynb) | profiles, posteriors, two solutions, surrogate uncertainty |
| [04 Benchmark and replay](examples/notebooks/04_benchmark_and_replay.ipynb) | the paper's two-basin benchmark and a replay on an archive |

## Fidelity to the paper

The package is a port of the campaign code used in the paper, not a reimplementation:

- `tests/test_parity.py` runs the original code and the package side by side and requires identical output at
  every stage (needs the original code; see the docs).
- `tests/test_golden.py` checks the same outputs against a fixture written by the original code, and runs anywhere.
- Replays of the paper's NGC 5102 and NGC 3258 campaigns, and its two-basin benchmark
  (`benchmarks/two_basin.py`), are reproduced model for model.

Exact repetition from a seed also needs the same library versions and the same number of linear-algebra threads
(`OMP_NUM_THREADS=1`).

## Citing

Please cite Waters & Gültekin (2026) and the software, archived on Zenodo: [10.5281/zenodo.23386326](https://doi.org/10.5281/zenodo.23386326) (all versions; each release also has its own DOI on that page). See `CITATION.cff`.

## License

MIT. See [LICENSE](LICENSE).
