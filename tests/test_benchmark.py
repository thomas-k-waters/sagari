import numpy as np
import pandas as pd

from sagari.benchmark import CENTRE_BROAD, CENTRE_NARROW, PILOT_SPACE, TwoBasin


def test_minima_and_failure_region():
    plain, hopeless = TwoBasin(), TwoBasin(hopeless=True)
    pts = pd.DataFrame([CENTRE_NARROW, CENTRE_BROAD])
    f, in_n = plain.true(PILOT_SPACE.to_unit(pts))
    assert np.allclose(f, [1000.0, 1050.0]) and list(in_n) == [True, False]
    U = np.random.default_rng(0).random((5000, 4))
    bad = hopeless.in_hopeless(U).astype(bool)
    assert 0.12 < bad.mean() < 0.18                                       # 15% of the box
    assert (hopeless.true(U)[0][bad] - plain.true(U)[0][bad] >= 6e4 - 1e-6).all()
    chi2 = hopeless(PILOT_SPACE.from_unit(U), np.random.default_rng(1))
    assert np.isnan(chi2[~bad]).sum() == 0 and 0.10 < np.isnan(chi2[bad]).mean() < 0.20


def test_found_and_error():
    obj = TwoBasin()
    est = {p: [v, v, v] for p, v in CENTRE_NARROW.items()}
    assert obj.found(est) and obj.mbh_error(est) == 0.0
    est = {p: [v, v, v] for p, v in CENTRE_BROAD.items()}
    assert not obj.found(est)
