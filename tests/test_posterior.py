import numpy as np
import pytest
from helpers import SPACE, TRUE, table

from sagari import inference_gp
from sagari import posterior as po


@pytest.fixture(scope="module")
def gp():
    return inference_gp(table(200), SPACE, 4)[0]


def test_supported_chi2(gp):
    f = po.supported_chi2(gp, po.support_cap(gp))
    assert np.isfinite(f(np.array([[0.3, 0.6]]))).all()
    assert np.isinf(po.supported_chi2(gp, 1e-9)(np.array([[0.3, 0.6]]))).all()


def test_segments():
    u = np.linspace(0, 1, 11)
    d = (u - 0.5) ** 2 * 100
    (lo, hi), = po.segments(u, d, 4.0)
    assert np.isclose(lo, 0.3, atol=0.02) and np.isclose(hi, 0.7, atol=0.02)
    assert len(po.segments(u, np.r_[0, 9, 0, 9, 9, 9, 9, 9, 9, 9, 0], 1.0)) == 3


@pytest.mark.slow
def test_posterior_brackets_the_minimum(gp):
    post = po.sample_posterior(gp, SPACE, nsteps=3000, burn=1000)
    for p in ("a", "b"):
        q = np.percentile(post[p], [2.5, 97.5])
        assert q[0] < TRUE[p] < q[1]
    idx, D, R = po.surface_draws(gp, post.attrs["unit"], n_sub=400, n_latent=100, n_refit=3)
    assert D.shape == (100, 400) and R.shape == (3, 400)
    lo, med, hi = po.pooled_quantiles(post["a"].values[idx], D)
    assert lo < med < hi
    shares = po.mode_shares(D, post["a"].values[idx] > 0.3)
    assert ((shares > 0) & (shares < 1)).all()


@pytest.mark.slow
def test_profiles_and_bootstrap(gp):
    prof = po.profiles(gp, SPACE, 4, params=["a"])
    iv = po.profile_intervals(prof, SPACE, levels=(4.0,))
    (lo, hi), = iv["a"][4.0]
    assert lo < TRUE["a"] < hi
    boot = po.profile_interval_bootstrap(gp, SPACE, prof, ["a"], level=1.0, n_boot=3)
    assert len(boot) == 3 and {"a_lo", "a_hi"} <= set(boot.columns)
