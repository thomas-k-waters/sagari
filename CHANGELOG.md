# Changelog

## 1.2.0 (2026-11-04)

- **Fix: the monitoring and inference surrogates could put a false basin between models far above the minimum.**
  The training window drops the models more than 30 σ_χ² above the minimum, so the GP never saw the models that
  rule a region out and could extend the valley into a gap they surround. In an NGC 4751 campaign the surrogate put
  χ²_min + 3 at M_BH 4.7e8, v_c 200 km/s, 0.05–0.12 (unit cube) from three models measured at χ²_min + 1068–1205.
  The false basin took monitoring weight (the M_BH range of one round reached 4.7e8), made the stopping rule's M_BH
  move 4.5 in the next round, and put second minima in the profiles. The surrogate is now fitted as before, and the
  models outside the training set that it predicts below the training set's top are added at that top and the GP
  refitted, up to three times (`gp.fit_clipped`). For the same models, 42 of 547 are added, the fitted scatter is
  14.9 (13.8 before), the length scales are unchanged, no model outside the window is predicted more than 5 σ_χ²
  below its top (one at +1105 was predicted at +87), and the profiles have a single valley. Adding all the models
  outside the window, clipped, was rejected: the GP then has to fit a flat plateau around the valley, and the kink
  at its edge inflates the fitted scatter.
- `clip_window` is a keyword argument of `smoothed_estimate`, `inference_gp` and `errorbar_round`, and a campaign
  setting (`estimate.clip_window`, `errorbars.clip_window`, both true). `false` / `False` restores the behavior of
  the paper and earlier versions. Existing campaigns pick it up when they are next loaded, and their estimate
  history is recomputed with it. The automatic window (`auto_window=True`) does not clip.
- New: `sagari.fit_clipped`.
- Effect on the paper's problems:
  - M86 v5: 44 models are added. The profile curves move by at most 3 in Δχ² where Δχ² < 20 (median 0.1–1.1;
    σ_χ² is 10). The Δχ² ≤ 10 ranges change by less than 8% of their width. At Δχ² ≤ 4 the upper end of Υ drops
    from 8.3 to 7.0, because M86's second solution (Υ ≈ 8.3) sits at Δχ² 4.0 before and 4.1 after.
  - The two-basin benchmark: BENCHMARK_RESULT
  - Parity tests pass (the reference, the original campaign code, has the same change). The golden tests now run
    the GP stages with `clip_window=False`, since the fixture was written by the code of the paper. With clipping
    the golden table adds 20 of 128 models.
- New test: `test_clipped_window`.
- Documentation: the models outside the training window in `docs/constants.md`, and the new settings in
  `docs/configuration.md`.

## 1.1.0 (2026-11-03)

- **Fix: the monitoring and inference surrogates could dip far below the minimum away from every model.**
  `PessimisticGP`, the surrogate of `smoothed_estimate` (the stopping rule) and of `inference_gp` (profiles,
  error-bar rounds, posterior), uses the feasibility threshold as its prior mean, and that threshold can lie far
  above every training model. The GP then fits a broad bump to bridge the gap, with length scales longer than the
  box. The mean never returns to the prior level inside the box, the valley becomes a small ripple on the bump,
  and small errors in the bump's shape show up as deep dips far from any model. In an NGC 4751 campaign (training
  chi2 911–1210, prior level 40540) the dips lay 130–330 below the minimum at 0.35 from the nearest model. They
  took most of the monitoring weight, so the stopping rule saw ranges of M_BH 1.3–2.3e9 and v_c 27–452 km/s, and
  the Δχ² ≤ 4 profiles ran to the box edges. The prior level is now capped at the lowest training chi2 plus
  10 × the training range (`PessimisticGP(level_cap=10.0)`). The same models then give length scales of 0.5–0.7, no
  dips, and M_BH 1.92–2.25e9 at Δχ² ≤ 4. The dips return from a cap of about 20 up. Below about 10, small training
  sets (20 models) fit too small a scatter more often.
- `level_cap` is a keyword argument of `PessimisticGP`, `smoothed_estimate`, `inference_gp` and `errorbar_round`,
  and a campaign setting (`estimate.level_cap`, `errorbars.level_cap`, both 10). `null` / `None` restores the
  behavior of the paper and earlier versions. Existing campaigns pick up the cap when they are next loaded, and
  their estimate history is recomputed with it.
- Effect on the paper's problems:
  - M86 v5: the Δχ² ≤ 4 and ≤ 10 profile ranges change by less than 2%.
  - The two-basin benchmark: the cap never binds. The feasibility threshold stays below 10 × the training
    range in all 96 estimates of 16 trials of the full method (8 plain, 8 with the failure region; 6 rounds
    each), so Table 1 is unchanged.
  - Golden and parity tests pass. The parity reference (the original campaign code) has the same cap.
- `test_errorbar_round_predicts_its_models` now checks the out-of-sample calibration over four seeded campaigns
  instead of one. Each surrogate there is fitted to 20 models, whose scatter is only loosely identified, so a
  single campaign can come out at z rms 2–3 with or without the cap.
- Documentation: the prior level and its cap in `docs/constants.md`.

## 1.0.2 (2026-10-30)

- `benchmarks/two_basin.py` now runs all seven methods of the paper's benchmark (Table 1): `full`, `lhc` (the
  full method with Latin-hypercube space filling), `no_noise` (the full method without a noise term in the chi2
  surrogates), `ablated` (the basic search), `sobol`, `lhc_only`, and `nested_grid`. The library is unchanged.
- `benchmarks/two_basin.py` now scores each trial with the smoothed estimate at a training cap of 1000, as the
  paper does. It previously used the estimate's default of 300, which the paper reports only as a pitfall.
- The script's docstring and the documentation refer to the sections of the revised paper.

## 1.0.1 (2026-10-28)

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
