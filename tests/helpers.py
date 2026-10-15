"""A small noisy test problem shared by the tests: a 2-D quadratic bowl with scatter."""
import numpy as np
import pandas as pd

from sagari import ParameterSpace

SPACE = ParameterSpace.from_dict({"name": "bowl", "parameters": {
    "a": {"bounds": [0.0, 1.0]}, "b": {"bounds": [0.01, 100.0], "scale": "log"}, "c": {"value": 3.0}}})
TRUE = {"a": 0.3, "b": 10 ** 0.4}          # minimum of the bowl
NOISE = 3.0


def bowl(design: pd.DataFrame, seed: int = 0, noise: float = NOISE) -> np.ndarray:
    U = SPACE.to_unit(design)
    f = 100 + 400 * ((U[:, 0] - 0.3) ** 2 / 0.05 + (U[:, 1] - 0.6) ** 2 / 0.1)
    return f + np.random.default_rng([seed, len(design)]).normal(0, noise, len(f))


def table(n: int = 96, seed: int = 0, noise: float = NOISE) -> pd.DataFrame:
    from sagari import SobolGlobalDesign
    U = SobolGlobalDesign(2, seed).extend(n)
    t = SPACE.from_unit(U)
    t["chi2"] = bowl(t, seed, noise)
    return t
