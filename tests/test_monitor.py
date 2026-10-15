import pandas as pd
from helpers import SPACE, TRUE, table

from sagari import estimate_converged, estimate_moved, smoothed_estimate, trusted_best_point


def test_estimate_brackets_the_minimum():
    e = smoothed_estimate(table(160), SPACE, seed=0)
    for p in ("a", "b"):
        assert e[p][1] <= TRUE[p] <= e[p][2]
    assert e["temperature"] >= 1 and e["ess"] > 10
    assert smoothed_estimate(table(160).iloc[:4], SPACE) is None


def test_stopping_rule():
    sub = SPACE.subspace(["a"])
    h = [{"a": [0.5, 0.2, 0.8]}, {"a": [0.5, 0.3, 0.6]}, {"a": [0.45, 0.31, 0.61]}]
    assert estimate_moved(h[1], h[2], sub)["a"] < 0.2
    assert estimate_converged(h, sub, frac=0.2, patience=1, min_rounds=2)
    assert not estimate_converged(h[:2], sub, frac=0.2, patience=1, min_rounds=2)    # one refinement round only
    assert not estimate_converged(h, sub, frac=0.2, patience=2, min_rounds=2)        # 0.2 -> 0.3 moved too much
    assert not estimate_converged(h + [None], sub)


def test_trusted_best_ignores_isolated_low_model():
    t = pd.DataFrame({"a": [0.5, 0.505, 0.51, 0.95], "b": [1.0, 1.02, 1.04, 50.0], "chi2": [100, 98, 99, 10]})
    best = trusted_best_point(t, SPACE)
    assert best["chi2"] == 98
    assert trusted_best_point(t.iloc[[3]], SPACE) is None
