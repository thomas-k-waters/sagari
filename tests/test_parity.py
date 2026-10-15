"""Parity with the original campaign code of the paper.

Runs the original modules (``campaign/sampling`` of the paper's pipeline, and ``fit_methods.py`` of its
analysis notebooks) and this package side by side on the same synthetic campaign, and requires identical
output at every stage: the Sobol design, a refinement round with the production settings, the monitoring
estimate, the stopping rule, an error-bar round with its predictions and profiles, the automatic window, the
posterior sampler, the profiles, and the surrogate draws.

Skipped unless ``SAGARI_REFERENCE`` points to the original ``campaign`` directory (and, for the
analysis functions, ``SAGARI_REFERENCE_ANALYSIS`` to the directory holding ``fit_methods.py``).
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sagari as sg
from sagari.benchmark import TwoBasin

REF = os.environ.get("SAGARI_REFERENCE")
REF_ANALYSIS = os.environ.get("SAGARI_REFERENCE_ANALYSIS")
pytestmark = [pytest.mark.parity, pytest.mark.skipif(not REF, reason="SAGARI_REFERENCE not set")]

YAML = """galaxy: two_basin
apparent_q: 0.625
parameters:
  upsilon: {fixed: false, bounds: [1.0, 15.0]}
  xi: {fixed: true, value: 1.0}
  mbh: {fixed: false, bounds: [3.0e+7, 5.0e+9], scale: log}
  inclination: {fixed: true, value: 89.0}
  v_c: {fixed: false, bounds: [0.0, 500.0]}
  r_c: {fixed: false, bounds: [0.05, 20.0], scale: log}
"""
SEED = 1003
PRODUCTION = dict(kernel="matern52", train_window=30.0, global_fraction=0.35)


@pytest.fixture(scope="module")
def ref():
    sys.path[:0] = [str(Path(REF) / "sampling")]
    mods = {m: importlib.import_module(m) for m in ("design", "gp_ei", "estimate", "posterior_fill", "support_check")}
    yield mods


@pytest.fixture(scope="module")
def setup(ref, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    yaml_path = tmp / "two_basin.yaml"
    yaml_path.write_text(YAML)
    cfg = ref["design"].GalaxyConfig.from_yaml(yaml_path)
    space = sg.ParameterSpace.from_yaml(yaml_path)
    obj = TwoBasin(hopeless=True)
    return dict(yaml=yaml_path, cfg=cfg, space=space, obj=obj, tmp=tmp)


def _campaign(ref, s, n_rounds=2):
    """The same campaign through both codes: 256 Sobol points and ``n_rounds`` refinement rounds of 128."""
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        da = ref["design"].generate_global_design(s["yaml"], 256, ta, seed=SEED)
        sob = sg.SobolGlobalDesign(4, SEED, Path(tb) / "state.json")
        db = sg.initial_design(s["space"], 256, sob)
        free = s["cfg"].free_params
        assert free == s["space"].free_params
        assert np.array_equal(da[free].values, db[free].values)
        ta_tab = da[free + ["xi", "inclination"]].copy(); ta_tab["chi2"] = s["obj"](da, rng_a)
        tb_tab = db[free + ["xi", "inclination"]].copy(); tb_tab["chi2"] = s["obj"](db, rng_b)
        rounds = []
        for _ in range(n_rounds):
            ra = ref["gp_ei"].gp_ei_round(ta_tab, s["yaml"], 128, ta, seed=SEED, **PRODUCTION)
            rb = sg.propose_round(tb_tab, s["space"], 128, sob, seed=SEED, **PRODUCTION)
            rounds.append((ra, rb))
            na = ra[free + ["xi", "inclination"]].copy(); na["chi2"] = s["obj"](ra, rng_a)
            nb = rb[free + ["xi", "inclination"]].copy(); nb["chi2"] = s["obj"](rb, rng_b)
            ta_tab = pd.concat([ta_tab, na], ignore_index=True)
            tb_tab = pd.concat([tb_tab, nb], ignore_index=True)
    return ta_tab, tb_tab, rounds


@pytest.fixture(scope="module")
def campaign(ref, setup):
    return _campaign(ref, setup)


def test_design_and_rounds_identical(setup, campaign):
    ta, tb, rounds = campaign
    free = setup["cfg"].free_params
    for ra, rb in rounds:
        assert np.array_equal(ra[free].values, rb[free].values)
        assert list(ra["stage"]) == list(rb["stage"])
        for k in ("gp_noise_sd", "raw_best", "n_verify", "n_train", "window_dchi2", "feasible_frac", "thr_dchi2"):
            assert ra.attrs[k] == rb.attrs[k], k
    assert np.array_equal(ta["chi2"].values, tb["chi2"].values, equal_nan=True)


def test_estimate_and_stopping_rule(ref, setup, campaign):
    ta, tb, _ = campaign
    hist_a, hist_b = [], []
    for n in (256, 384, 512):
        ea = ref["estimate"].smoothed_estimate(ta.iloc[:n], setup["cfg"], seed=SEED, kernel="matern52", train_window=30.0)
        eb = sg.smoothed_estimate(tb.iloc[:n], setup["space"], seed=SEED, kernel="matern52", train_window=30.0)
        assert ea == eb
        hist_a.append(ea); hist_b.append(eb)
    sub_a = ref["design"].GalaxyConfig(galaxy="x", apparent_q=1.0, free_params=["mbh", "upsilon"],
                                       bounds={p: setup["cfg"].bounds[p] for p in ("mbh", "upsilon")},
                                       fixed_values={}, log_params=frozenset({"mbh"}))
    sub_b = setup["space"].subspace(["mbh", "upsilon"])
    for frac in (0.1, 0.2, 0.3):
        for pat in (1, 2):
            for mr in (1, 2):
                assert ref["estimate"].estimate_converged(hist_a, sub_a, frac=frac, patience=pat, min_rounds=mr) == \
                    sg.estimate_converged(hist_b, sub_b, frac=frac, patience=pat, min_rounds=mr)
    ba = ref["support_check"].trusted_best_point(ta, setup["cfg"].free_params, setup["cfg"].bounds,
                                                 log_params=setup["cfg"].log_params)
    bb = sg.trusted_best_point(tb, setup["space"])
    assert ba["chi2"] == bb["chi2"]


def test_errorbar_round_and_profiles(ref, setup, campaign):
    ta, tb, _ = campaign
    free = setup["cfg"].free_params
    fa = ref["posterior_fill"].posterior_fill_round(ta, setup["yaml"], 128, seed=SEED, kernel="matern52",
                                                    train_window=30.0, n_train_cap=1000, n_profile=7)
    fb = sg.errorbar_round(tb, setup["space"], 128, seed=SEED, kernel="matern52", train_window=30.0,
                            n_train_cap=1000, n_profile=7)
    assert np.array_equal(fa[free].values, fb[free].values)
    for c in ("pred_mu", "pred_sd", "pred_latent", "kind"):
        assert np.array_equal(fa[c].values, fb[c].values), c
    assert fa.attrs["profile_before"] == fb.attrs["profile_before"]
    ga = ref["estimate"].inference_gp(ta, setup["cfg"], SEED, "matern52", 30.0, 1000)[0]
    gb = sg.inference_gp(tb, setup["space"], SEED, "matern52", 30.0, 1000)[0]
    pa = ref["estimate"].gp_profiles(ga, setup["cfg"], SEED)
    pb = sg.gp_profiles(gb, setup["space"], SEED)
    assert ref["estimate"].profile_compact(pa) == sg.errorbars.profile_compact(pb)
    va = ref["estimate"].profile_verdict(fa.attrs["profile_before"], ref["estimate"].profile_compact(pa), ["mbh", "upsilon"])
    vb = sg.profile_verdict(fb.attrs["profile_before"], sg.errorbars.profile_compact(pb), ["mbh", "upsilon"])
    assert va == vb


def test_auto_window(ref, setup, campaign):
    ta, tb, _ = campaign
    ga = ref["estimate"].inference_gp(ta, setup["cfg"], SEED, "matern52", 30.0, 300, auto_window=True)[0]
    gb = sg.inference_gp(tb, setup["space"], SEED, "matern52", 30.0, 300, auto_window=True)[0]
    assert ga.window == gb.window and ga.window_checks == gb.window_checks
    U = np.random.default_rng(0).random((500, 4))
    assert np.array_equal(ga.predict(U), gb.predict(U))


@pytest.mark.skipif(not REF_ANALYSIS, reason="SAGARI_REFERENCE_ANALYSIS not set")
def test_posterior_analysis(ref, setup, campaign):
    sys.path[:0] = [REF_ANALYSIS]
    fm = importlib.import_module("fit_methods")
    ta, tb, _ = campaign
    ga = ref["estimate"].inference_gp(ta, setup["cfg"], SEED, "matern52", 30.0, 1000)[0]
    gb = sg.inference_gp(tb, setup["space"], SEED, "matern52", 30.0, 1000)[0]
    sup_a, sup_b = fm.support_cap(ga), sg.posterior.support_cap(gb)
    assert sup_a == sup_b
    fa, acc_a, nk_a = fm.run_emcee(fm.supported_chi2(ga, sup_a), np.zeros(4), np.ones(4), nsteps=1500, burn=300)
    fb, acc_b, nk_b = sg.posterior.run_emcee(sg.posterior.supported_chi2(gb, sup_b), np.zeros(4), np.ones(4),
                                              nsteps=1500, burn=300)
    assert np.array_equal(fa, fb) and acc_a == acc_b and nk_a == nk_b
    for j in range(4):
        a = fm.profile(ga, j, 4, np.random.default_rng([SEED, 11]), n_coarse=9, n_fine=9, n_rand=1500, support=sup_a)
        b = sg.posterior.profile(gb, j, 4, np.random.default_rng([SEED, 11]), n_coarse=9, n_fine=9, n_rand=1500,
                                  support=sup_b)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
        assert fm.segments(a[0], a[1] - a[1].min(), 4.0) == sg.posterior.segments(b[0], b[1] - b[1].min(), 4.0)
    # the surrogate draws of the paper's posterior notebook
    rng = np.random.default_rng(20261009)
    M = 300
    idx = rng.choice(len(fa), M, replace=False)
    Xs = fa[idx]
    _, cov = ga.gp.predict(Xs, return_cov=True)
    cov = cov * ga.scale ** 2 - fm.noise_var(ga) * np.eye(M)
    D_gp = (np.linalg.cholesky(cov + 1e-6 * np.mean(np.diag(cov)) * np.eye(M)) @ rng.standard_normal((M, 50))).T
    from sklearn.gaussian_process import GaussianProcessRegressor
    g0 = ga.gp; Xt = g0.X_train_; mu_t = ga.predict(Xt); mu_s = ga.predict(Xs); D_bs = []
    noise = float(np.sqrt(fm.noise_var(ga)))
    for _ in range(5):
        yb = mu_t + noise * rng.standard_normal(len(mu_t))
        g2 = GaussianProcessRegressor(kernel=g0.kernel_, optimizer=None).fit(Xt, (yb - ga.level) / ga.scale)
        D_bs.append(g2.predict(Xs) * ga.scale + ga.level - mu_s)
    idx_b, Dl, Dr = sg.posterior.surface_draws(gb, fb, n_sub=M, n_latent=50, n_refit=5)
    assert np.array_equal(idx, idx_b) and np.array_equal(D_gp, Dl) and np.array_equal(np.array(D_bs), Dr)
