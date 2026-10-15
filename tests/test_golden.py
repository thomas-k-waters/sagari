"""Outputs of the original campaign code of the paper, stored in fixtures/golden.json by
fixtures/make_golden.py, against this package, on any platform.

What does not involve a GP fit (the Sobol design, the global share of a round, the composition of a round) must
match exactly. What does is compared with tolerances set by perturbing the fixture's chi2 at the round-off level
(1e-12 to 1e-6 relative), which is what a different BLAS or CPU does: the placed points, the monitoring estimate
and the error-bar round stay put, while the GP's fitted scatter after one round of 128 models can move by a factor
of 2 (the marginal likelihood is nearly flat between more nugget and shorter length scales, and round-off decides
which side the optimizer settles on). The exact, same-machine comparison with the original code is
tests/test_parity.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sagari as sg
from sagari.acquisition import STAGE_GLOBAL
from sagari.benchmark import PILOT_SPACE

G = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text())
SPACE = PILOT_SPACE
FREE = SPACE.free_params


def _table():
    t = pd.DataFrame(G["table"], columns=FREE + ["chi2"])
    t["chi2"] = t["chi2"].astype(float)
    return t


def _nearest(a, b):
    """Distance from each point of a to the nearest point of b (unit cube)."""
    return np.array([np.min(np.linalg.norm(b - x, axis=1)) for x in a])


def test_sobol_design_exact():
    d = sg.initial_design(SPACE, G["n0"], sg.SobolGlobalDesign(4, G["seed"]))
    assert np.array_equal(d[FREE].values, np.array(G["design0"]))


def test_refinement_round():
    sob = sg.SobolGlobalDesign(4, G["seed"]); sob.extend(G["n0"])
    d = sg.propose_round(_table(), SPACE, G["n_round"], sob, seed=G["seed"], **G["production"])
    U, ref = SPACE.to_unit(d), SPACE.to_unit(np.array(G["round1"]))
    stage_ref = np.array(G["round1_stage"])
    # the global share continues the Sobol sequence: exact
    glob, glob_ref = (d["stage"] == STAGE_GLOBAL).values, stage_ref == STAGE_GLOBAL
    assert glob.sum() == glob_ref.sum()
    assert np.allclose(U[glob], ref[glob_ref], rtol=0, atol=1e-12)
    # the points placed by Expected Improvement: the same points, allowing a near-tie to break differently
    same = np.all(np.isclose(U, ref, rtol=0, atol=1e-6), axis=1)
    assert same.mean() >= 0.9
    assert np.median(_nearest(U[~glob], ref[~glob_ref])) < 0.02
    # the fitted scatter is poorly determined after one round (see the module docstring)
    noise = d.attrs["gp_noise_sd"]
    assert np.isfinite(noise) and G["round1_noise"] / 4 < noise < 4 * G["round1_noise"]


def test_estimate():
    e = sg.smoothed_estimate(_table(), SPACE, seed=G["seed"], kernel="matern52", train_window=30.0)
    for p in FREE:
        assert np.allclose(e[p], G["estimate"][p], rtol=1e-2)
    assert abs(e["chi2_min"] - G["estimate"]["chi2_min"]) < 0.5


def test_errorbar_round():
    d = sg.errorbar_round(_table(), SPACE, G["n_round"], seed=G["seed"], n_profile=7)
    U, ref = SPACE.to_unit(d), SPACE.to_unit(np.array(G["fill"]))
    assert (d["kind"] == "fill").sum() == G["n_round"] - 7 * SPACE.dim
    same = np.all(np.isclose(U, ref, rtol=0, atol=1e-6), axis=1)
    assert same.mean() >= 0.9
    assert np.median(_nearest(U, ref)) < 0.02
    assert np.allclose(d["pred_mu"].values[same], np.array(G["fill_pred_mu"])[same], rtol=1e-3)
