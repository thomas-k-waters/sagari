import numpy as np
import pandas as pd
import pytest

from sagari import ParameterSpace

SPEC = {"name": "t", "parameters": {"upsilon": {"bounds": [4, 12]}, "mbh": {"bounds": [3e7, 5e9], "scale": "log"},
                                    "inclination": {"value": 89.0}}}


def test_order_and_fixed():
    s = ParameterSpace.from_dict(SPEC)
    assert s.free_params == ["mbh", "upsilon"]          # alphabetical: the Sobol axis order
    assert s.fixed_values == {"inclination": 89.0}
    assert s.log_params == frozenset({"mbh"})


def test_roundtrip_and_log_axis():
    s = ParameterSpace.from_dict(SPEC)
    U = np.random.default_rng(0).random((50, 2))
    df = s.from_unit(U, with_fixed=True)
    assert np.allclose(s.to_unit(df), U)
    assert (df["inclination"] == 89.0).all()
    # half way along a log axis is the geometric mean
    assert np.isclose(s.from_unit([[0.5, 0.5]])["mbh"][0], np.sqrt(3e7 * 5e9))


def test_original_format_and_yaml(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("galaxy: M86\napparent_q: 0.6\nparameters:\n  upsilon: {fixed: false, bounds: [4, 12]}\n"
                 "  mbh: {fixed: false, bounds: [3.0e6, 5.0e9], scale: log}\n  xi: {fixed: true, value: 1.0}\n")
    s = ParameterSpace.from_yaml(p)
    assert s.name == "M86" and s.bounds["mbh"] == (3e6, 5e9) and s.fixed_values == {"xi": 1.0}
    assert ParameterSpace.from_dict(s.to_dict()).bounds == s.bounds


@pytest.mark.parametrize("bad", [
    {"a": {"bounds": [1, 1]}},
    {"a": {"bounds": [0, 1], "scale": "log"}},
    {"a": {"bounds": [0, 1], "scale": "cubic"}},
    {"a": {"fixed": True}},
    {"a": {"fixed": False}},
])
def test_invalid(bad):
    with pytest.raises(ValueError):
        ParameterSpace.from_dict({"parameters": bad})


def test_contains_and_subspace():
    s = ParameterSpace.from_dict(SPEC)
    t = pd.DataFrame({"mbh": [1e8, 1e8, 1e10], "upsilon": [4.0 - 1e-6, 3.0, 5.0]})
    assert list(s.contains(t)) == [True, False, False]
    sub = s.subspace(["upsilon"])
    assert sub.free_params == ["upsilon"] and not sub.fixed_values
    assert s.with_bounds(upsilon=(5, 12)).bounds["upsilon"] == (5.0, 12.0)
