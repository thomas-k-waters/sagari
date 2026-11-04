import numpy as np
import pandas as pd
from helpers import SPACE, bowl, table

from sagari import PessimisticGP, errorbar_round, gp_profiles, inference_gp, out_of_sample, profile_verdict
from sagari.errorbars import profile_compact
from sagari.gp import fit_clipped


def test_errorbar_round_predicts_its_models():
    """The out-of-sample z of the predictions is about unit normal. Checked over four campaigns: each surrogate
    is fitted to the 20 models of its training window, whose scatter is only loosely identified, so a single
    campaign can come out at z rms 2-3 with or without the level cap."""
    rms = []
    for s in range(4):
        d = errorbar_round(table(160, seed=s), SPACE, 40, seed=1 + s, n_profile=4)
        assert len(d) == 40 and (d["kind"] == "fill").sum() == 32
        assert {"pred_mu", "pred_sd", "pred_latent"} <= set(d.columns)
        assert (d["pred_latent"] <= d["pred_sd"]).all()
        assert "profile_before" in d.attrs
        oos = out_of_sample(d, bowl(d, 7 + s))
        assert oos["n"] == 40
        rms.append(oos["rms"])
    assert 0.5 < np.median(rms) < 2.0 and max(rms) < 3.0, rms


def test_pessimistic_level_cap():
    """The prior level is capped at min + level_cap x the training range; a level below the cap, or
    level_cap=None (the paper), is kept."""
    rng = np.random.default_rng(5)
    X, y = rng.random((40, 2)), 900 + 300 * rng.random(40)
    assert PessimisticGP(X, y, 40000.0, 2, 1).level == y.min() + 10 * np.ptp(y)
    assert PessimisticGP(X, y, 1500.0, 2, 1).level == 1500.0
    assert PessimisticGP(X, y, 40000.0, 2, 1, level_cap=None).level == 40000.0
    t = table(200)
    g, chi2_min, thr = inference_gp(t, SPACE, 2, level_cap=None)
    assert g.level == chi2_min + thr
    y = g.gp.y_train_ * g.scale + g.level
    g1 = inference_gp(t, SPACE, 2, level_cap=1.0)[0]
    assert np.isclose(g1.level, min(chi2_min + thr, y.min() + np.ptp(y)))


def test_clipped_window():
    """Models outside the training window that the surrogate predicts below the window top are added back at the
    top (an NGC 4751 campaign: dropped, three models at chi2_min + 1068-1205 could not stop the GP from putting
    chi2_min + 3 between them); clip=False, or a fit that agrees with them, leaves the floor as it was."""
    t = table(200)
    train = t.nsmallest(40, "chi2")
    X = SPACE.to_unit(train)
    base = fit_clipped(X, train, t, SPACE, 40000.0, 1, clip=False)
    agree = fit_clipped(X, train, t, SPACE, 40000.0, 1)
    low = SPACE.to_unit(train.nsmallest(10, "chi2"))
    assert base.ceiling is None and base.n_clipped == 0
    assert np.abs(agree.predict(low) - base.predict(low)).max() < 3
    # three bad models next to the floor, which the unclipped fit cannot see
    bad = train.nsmallest(3, "chi2").copy()
    bad["a"] = bad["a"] + 0.01
    bad["chi2"] = 5000.0
    bad.index = bad.index + 10_000
    t2 = pd.concat([t, bad])
    clipped = fit_clipped(X, train, t2, SPACE, 40000.0, 1)
    assert clipped.n_clipped >= 3 and clipped.ceiling == train["chi2"].max()
    assert np.all(clipped.predict(SPACE.to_unit(bad)) > base.predict(SPACE.to_unit(bad)) + 50)
    assert inference_gp(t, SPACE, 2, clip_window=False)[0].ceiling is None


def test_profiles_and_verdict():
    t = table(200)
    gp = inference_gp(t, SPACE, 2)[0]
    prof = gp_profiles(gp, SPACE, 2, levels=(100.0,))
    e = prof["params"]["a"]["edges"]["100"]
    assert e["value"][0] < 0.3 < e["value"][1]
    c = profile_compact(prof)
    v = profile_verdict(c, c, ["a", "b"], sd_tol=100.0)
    assert v["done"] and v["worst_move"] == 0
    assert not profile_verdict(None, c, ["a"])["done"]


def test_auto_window_option():
    gp = inference_gp(table(300, noise=40.0), SPACE, 2, n_train_cap=300, auto_window=True)[0]
    assert hasattr(gp, "window") and gp.window["ratio"] > 0
