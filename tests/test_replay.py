import numpy as np
from helpers import SPACE, bowl

from sagari import Campaign
from sagari.replay import archive_evaluator, kdtree_snap


def _archive(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = SPACE.from_unit(rng.random((n, 2)))
    a["chi2"] = bowl(a, seed)
    a.loc[a.index[:5], "chi2"] = np.nan          # a few failed models
    return a


def test_snap_returns_archived_models():
    arch = _archive()
    q = arch.iloc[[3, 50, 200]][["a", "b"]].copy()
    q["a"] += 1e-6
    s = kdtree_snap(q, arch, ["a", "b"])
    assert np.allclose(s[["a", "b"]].values, arch.iloc[[3, 50, 200]][["a", "b"]].values)
    assert np.isnan(s["chi2"].iloc[0]) and (s["snap_distance"] < 1e-4).all()


def test_replay_campaign_stores_snapped_parameters(tmp_path):
    arch = _archive(600)
    c = Campaign.create(tmp_path / "c", SPACE, seed=2, n_initial=48, n_round=24,
                        errorbars={"n_round": 24, "n_profile": 3, "max_rounds": 1})
    res = c.run(archive_evaluator(arch, SPACE.free_params), verbose=False)
    merged = res.merge(arch, on=["a", "b"], how="left", suffixes=("", "_arch"))
    assert np.allclose(merged["chi2"], merged["chi2_arch"], equal_nan=True)   # every result is an archived model
    assert c.status(res, verbose=False)["next"] == "done"
