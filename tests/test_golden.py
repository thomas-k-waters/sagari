"""Outputs of the original campaign code of the paper, stored in fixtures/golden.json by
fixtures/make_golden.py, against this package. The Sobol design is compared exactly. The GP-dependent outputs
are compared to a tolerance, and a refinement round must match point for point in at least 95% of its points:
optimizer and BLAS differences between platforms can flip a near-tie in the candidate ranking, and the
exact, same-machine comparison is tests/test_parity.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sagari as sg
from sagari.benchmark import PILOT_SPACE

G = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text())
SPACE = PILOT_SPACE
FREE = SPACE.free_params


def _table():
    t = pd.DataFrame(G["table"], columns=FREE + ["chi2"])
    t["chi2"] = t["chi2"].astype(float)
    return t


def test_sobol_design_exact():
    d = sg.initial_design(SPACE, G["n0"], sg.SobolGlobalDesign(4, G["seed"]))
    assert np.array_equal(d[FREE].values, np.array(G["design0"]))


def test_refinement_round():
    sob = sg.SobolGlobalDesign(4, G["seed"]); sob.extend(G["n0"])
    d = sg.propose_round(_table(), SPACE, G["n_round"], sob, seed=G["seed"], **G["production"])
    ref = np.array(G["round1"])
    match = np.all(np.isclose(SPACE.to_unit(d), SPACE.to_unit(ref), rtol=0, atol=1e-6), axis=1)
    assert match.mean() >= 0.95
    assert list(d["stage"]) == G["round1_stage"]
    assert np.isclose(d.attrs["gp_noise_sd"], G["round1_noise"], rtol=1e-3)


def test_estimate():
    e = sg.smoothed_estimate(_table(), SPACE, seed=G["seed"], kernel="matern52", train_window=30.0)
    for p in FREE:
        assert np.allclose(e[p], G["estimate"][p], rtol=1e-3)
    assert np.isclose(e["chi2_min"], G["estimate"]["chi2_min"], rtol=1e-5)


def test_errorbar_round():
    d = sg.errorbar_round(_table(), SPACE, G["n_round"], seed=G["seed"], n_profile=7)
    ref = np.array(G["fill"])
    match = np.all(np.isclose(SPACE.to_unit(d), SPACE.to_unit(ref), rtol=0, atol=1e-6), axis=1)
    assert match.mean() >= 0.95
    assert np.allclose(d["pred_mu"].values[match], np.array(G["fill_pred_mu"])[match], rtol=1e-4)
