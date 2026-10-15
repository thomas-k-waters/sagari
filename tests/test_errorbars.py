from helpers import SPACE, bowl, table

from sagari import errorbar_round, gp_profiles, inference_gp, out_of_sample, profile_verdict
from sagari.errorbars import profile_compact


def test_errorbar_round_predicts_its_models():
    t = table(160)
    d = errorbar_round(t, SPACE, 40, seed=1, n_profile=4)
    assert len(d) == 40 and (d["kind"] == "fill").sum() == 32
    assert {"pred_mu", "pred_sd", "pred_latent"} <= set(d.columns)
    assert (d["pred_latent"] <= d["pred_sd"]).all()
    oos = out_of_sample(d, bowl(d, 7))
    assert oos["n"] == 40 and 0.5 < oos["rms"] < 2.0
    assert "profile_before" in d.attrs


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
