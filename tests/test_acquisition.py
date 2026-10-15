import numpy as np
import pytest
from helpers import SPACE, table

from sagari import SobolGlobalDesign, initial_design, propose_round
from sagari.acquisition import STAGE_GLOBAL, STAGE_LOCAL


def test_round_composition_and_bounds():
    t = table(96)
    sob = SobolGlobalDesign(2, 5)
    sob.extend(96)
    d = propose_round(t, SPACE, 40, sob, seed=5)
    assert len(d) == 40
    assert (d["stage"] == STAGE_GLOBAL).sum() == 14                # 35% of 40
    assert (d["c"] == 3.0).all()
    U = SPACE.to_unit(d)
    assert (U >= 0).all() and (U <= 1).all()
    assert d.attrs["gp_noise_sd"] > 0 and d.attrs["n_train"] >= 20


def test_local_points_go_to_the_minimum():
    t = table(128)
    sob = SobolGlobalDesign(2, 3)
    sob.extend(128)
    d = propose_round(t, SPACE, 64, sob, seed=3)
    U = SPACE.to_unit(d)
    loc = (d["stage"] == STAGE_LOCAL).values
    dist = np.hypot(U[:, 0] - 0.3, U[:, 1] - 0.6)
    assert np.median(dist[loc]) < 0.5 * np.median(dist[~loc])


def test_reproducible_and_global_share_continues_sequence():
    t = table(96)
    a = SobolGlobalDesign(2, 9); a.extend(96)
    b = SobolGlobalDesign(2, 9); b.extend(96)
    da = propose_round(t, SPACE, 32, a, seed=9)
    db = propose_round(t, SPACE, 32, b, seed=9)
    assert np.array_equal(da[SPACE.free_params].values, db[SPACE.free_params].values)
    g = da[da["stage"] == STAGE_GLOBAL]
    expected = SobolGlobalDesign(2, 9).extend(96 + len(g))[96:]
    assert np.allclose(SPACE.to_unit(g), expected)


def test_failed_models_and_too_few():
    t = table(96)
    t.loc[t.index[:10], "chi2"] = np.nan
    sob = SobolGlobalDesign(2, 1); sob.extend(96)
    assert len(propose_round(t, SPACE, 16, sob, seed=1)) == 16
    with pytest.raises(ValueError):
        propose_round(t.iloc[:4], SPACE, 16, sob, seed=1)
    with pytest.raises(ValueError):
        propose_round(t, SPACE, 16, None)


def test_initial_design():
    sob = SobolGlobalDesign(2, 11)
    d = initial_design(SPACE, 16, sob)
    assert len(d) == 16 and (d["stage"] == "sobol_global").all() and sob.num_generated == 16
