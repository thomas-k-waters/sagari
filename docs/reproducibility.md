# Reproducibility and fidelity to the paper

## Same seed, same campaign

A campaign is fully determined by its space, its settings (including `seed`) and the χ² of its models. Each
round's decisions use generators seeded from the seed and the number of models, and the global Sobol state is
persisted, so continuing a campaign in another process gives the same rounds.

Exact repetition also needs the same versions of NumPy, SciPy and scikit-learn and **the same number of
linear-algebra threads**. A GP fit is reproducible only to round-off, and round-off can break a near-tie between
candidates. Fix the threads before Python starts:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Results are statistically the same either way; this only matters when you want bit-identical rounds.

## The code of the paper

`sagari` is a port of the campaign code of Waters & Gültekin (2026), not a reimplementation. The numerical
functions are that code with the pipeline-specific parts removed, the defaults are the paper's adopted settings,
and the options that reproduce its benchmark variants are kept. This is checked three ways:

- `tests/test_parity.py` runs the original code and the package side by side and requires **identical** output
  for the Sobol design, refinement rounds, the monitoring estimate and stopping rule, error-bar rounds with their
  predictions and profiles, the automatic window, the posterior sampler, the profiles, and the surrogate draws.
  It needs the original code (`SAGARI_REFERENCE=/path/to/campaign`, and
  `SAGARI_REFERENCE_ANALYSIS` for the analysis functions).
- `tests/test_golden.py` checks the same outputs against a fixture written by the original code, so it runs
  anywhere. Refinement rounds are compared point for point with a small tolerance for platform round-off.
- `Campaign.run` with `replay.archive_evaluator` and `estimate={"n_train_cap": 300}` (the paper's value; the
  default is 1000 since version 1.0.1) reproduces the paper's NGC 5102 and NGC 3258 replays model for model, and `benchmarks/two_basin.py` reproduces its 100-trial benchmark exactly (with one thread).

## Deliberate differences from the paper's driver

- A campaign orders its models by proposal; the paper's driver read them in file-name order. The two differ only
  at round-off.
- A campaign drops exact repeats before every fit. A real campaign never repeats a model, but a replay on an
  archive can, and a GP reads exact repeats as noiseless.
- `posterior.surface_draws` raises the covariance jitter if the factorization fails. With the paper's data it
  never does, and the first attempt uses the paper's value.
- Since version 1.0.1, a campaign's monitoring surrogate is fitted to at most 1000 models instead of the paper's
  300 (`estimate.n_train_cap`; see [Fixed constants](constants.md)). Set it to 300 to reproduce the paper.
