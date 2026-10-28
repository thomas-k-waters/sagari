# Changelog

## Unreleased

- The default `estimate.n_train_cap` of a `Campaign` (the most models the monitoring surrogate of the search is
  fitted to) is now 1000 instead of the paper's 300. With 300, a search that packs several hundred models into one
  basin fits the surrogate to a narrow range of chi2, loses the walls of the basin, and its minimum drifts; on the
  paper's two-basin benchmark the cap of 1000 was better at every budget from 384 models on, and for M86 it gave
  the same stopping decisions. Set `estimate: {n_train_cap: 300}` to reproduce the paper. The function
  `smoothed_estimate` and the EI sampler `propose_round` keep their defaults of 300.
- Documentation: a page of the fixed constants and the reason for each (`docs/constants.md`).

## 1.0.0 (2026-10-15)

First public release of SAGARI (Schwarzschild Adaptive Gaussian-process Acquisition, Refinement, and
Inference), accompanying Waters & Gültekin (2026).

- `ParameterSpace`: bounds, linear and logarithmic axes, fixed parameters; YAML and dict input.
- `SobolGlobalDesign` / `initial_design`: persistent scrambled-Sobol global design.
- `propose_round`: refinement rounds by feasibility-weighted Expected Improvement with a nugget for the
  model-to-model scatter, a training window in units of that scatter, a denoised incumbent, a two-scale
  candidate pool, a minimum separation, verification of isolated low models, and a global Sobol share.
- `smoothed_estimate`, `estimate_converged`: monitoring estimate and stopping rule; `trusted_best_point`.
- `errorbar_round`, `inference_gp`, `gp_profiles`, `profile_verdict`, `out_of_sample`: error-bar rounds,
  the inference surrogate (optionally with an automatically chosen training window) and its checks.
- `posterior`: the ensemble-MCMC posterior on the supported surrogate, surrogate-marginalized quantiles and
  mode shares, Δχ² profiles and their parametric-bootstrap widening.
- `Campaign` and the `sagari` command line: file-based campaigns for clusters, and an in-process loop.
- `replay`, `benchmark`: archive replays and the two-basin objective of the paper; `benchmarks/two_basin.py`.
- Parity tests against the original campaign code of the paper, and a golden fixture written by it.
- Four executed example notebooks (a first campaign; the cluster workflow; the final analysis on a two-solution
  problem; the two-basin benchmark and an archive replay), a documentation site (concepts, cluster workflow,
  configuration reference, analysis, reproducibility, FAQ), and example space and settings files.
- `posterior.surface_draws` raises the covariance jitter when the factorization fails (posterior samples much
  closer together than the kernel's length scales); the first attempt uses the paper's value.
- Tested on Python 3.10–3.13, down to NumPy 1.26, SciPy 1.11, scikit-learn 1.3 and pandas 2.0, and up to
  pandas 3.0.
