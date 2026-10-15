"""The search space: free parameters with bounds and an axis scale, fixed parameters, and the map to the
unit cube in which every design, surrogate and distance of the method is defined (Section 3.2 of the paper).

A logarithmic axis runs uniformly in log10 between its bounds, so a design that is uniform in the unit
cube is log-uniform in that parameter. The free parameters are ordered alphabetically by name: the
order fixes the Sobol dimensions, so it must not depend on the order in which a configuration file
happens to list them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ParameterSpace:
    """Free parameters (with bounds and scale) and fixed parameters of a campaign.

    Attributes
    ----------
    free_params : list of str
        Names of the free parameters, sorted alphabetically (the unit-cube axis order).
    bounds : dict
        ``{name: (lower, upper)}`` for every free parameter.
    fixed_values : dict
        ``{name: value}`` for every fixed parameter; carried into every design unchanged.
    log_params : frozenset
        Free parameters with a logarithmic axis.
    name : str
        Label of the campaign (used only in messages and file headers).
    """
    free_params: list[str]
    bounds: dict[str, tuple[float, float]]
    fixed_values: dict[str, float] = field(default_factory=dict)
    log_params: frozenset[str] = frozenset()
    name: str = "campaign"

    def __post_init__(self):
        self.free_params = sorted(self.free_params)
        self.bounds = {p: (float(lo), float(hi)) for p, (lo, hi) in self.bounds.items()}
        self.log_params = frozenset(self.log_params)
        missing = set(self.free_params) - set(self.bounds)
        if missing:
            raise ValueError(f"free parameters without bounds: {sorted(missing)}")
        for p in self.free_params:
            lo, hi = self.bounds[p]
            if not lo < hi:
                raise ValueError(f"'{p}': bounds must satisfy lower < upper, got {(lo, hi)}")
            if p in self.log_params and lo <= 0:
                raise ValueError(f"'{p}' has a logarithmic axis, so its lower bound must be > 0, got {lo}")
        if self.log_params - set(self.free_params):
            raise ValueError(f"log_params not among the free parameters: {sorted(self.log_params - set(self.free_params))}")
        overlap = set(self.free_params) & set(self.fixed_values)
        if overlap:
            raise ValueError(f"parameters both free and fixed: {sorted(overlap)}")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, spec: dict, name: str | None = None) -> "ParameterSpace":
        """Build from ``{"parameters": {name: entry}}`` (or the parameter mapping itself).

        An entry is free if it has ``bounds: [lower, upper]`` (optional ``scale: linear | log``) and
        fixed if it has ``value: x``. The explicit form ``fixed: true/false`` is also accepted.
        """
        params = spec.get("parameters", spec)
        free, bounds, fixed, logp = [], {}, {}, set()
        for p, e in params.items():
            if not isinstance(e, dict):
                fixed[p] = float(e)
                continue
            is_fixed = e.get("fixed", "value" in e and "bounds" not in e)
            if is_fixed is None:
                raise ValueError(f"'{p}': no fixed/free decision (set 'fixed', or give 'bounds' or 'value')")
            if is_fixed:
                if e.get("value") is None:
                    raise ValueError(f"'{p}' is fixed but has no value")
                fixed[p] = float(e["value"])
                continue
            if e.get("bounds") is None:
                raise ValueError(f"'{p}' is free but has no bounds")
            # float() also catches YAML's "3.0e6" (no exponent sign), which PyYAML reads as a string
            lo, hi = float(e["bounds"][0]), float(e["bounds"][1])
            scale = e.get("scale", "linear")
            if scale not in ("linear", "log"):
                raise ValueError(f"'{p}': scale must be 'linear' or 'log', got {scale!r}")
            if scale == "log":
                logp.add(p)
            free.append(p)
            bounds[p] = (lo, hi)
        return cls(free_params=free, bounds=bounds, fixed_values=fixed, log_params=frozenset(logp),
                   name=name or spec.get("name", spec.get("galaxy", "campaign")))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ParameterSpace":
        """Read a YAML file in the format of :meth:`from_dict`."""
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        params = {}
        for p in self.free_params:
            params[p] = {"bounds": list(self.bounds[p]), "scale": "log" if p in self.log_params else "linear"}
        for p, v in self.fixed_values.items():
            params[p] = {"value": v}
        return {"name": self.name, "parameters": params}

    def subspace(self, names) -> "ParameterSpace":
        """The space of a subset of the free parameters (no fixed ones), e.g. those the stopping rule uses."""
        names = list(names)
        unknown = set(names) - set(self.free_params)
        if unknown:
            raise ValueError(f"not free parameters: {sorted(unknown)}")
        return ParameterSpace(free_params=names, bounds={p: self.bounds[p] for p in names},
                              log_params=frozenset(set(names) & self.log_params), name=self.name)

    def with_bounds(self, **bounds) -> "ParameterSpace":
        """A copy with some bounds changed (e.g. a floor raised during a campaign)."""
        return ParameterSpace(free_params=list(self.free_params), bounds={**self.bounds, **bounds},
                              fixed_values=dict(self.fixed_values), log_params=self.log_params, name=self.name)

    # ------------------------------------------------------------------ geometry
    @property
    def dim(self) -> int:
        return len(self.free_params)

    def _axes(self, names=None):
        names = self.free_params if names is None else names
        return [(np.log10(self.bounds[p][0]), np.log10(self.bounds[p][1]), True) if p in self.log_params
                else (self.bounds[p][0], self.bounds[p][1], False) for p in names]

    def to_unit(self, X) -> np.ndarray:
        """Physical values -> unit-cube coordinates, shape (n, dim). ``X`` is a DataFrame with the free
        parameters as columns, or an (n, dim) array in :attr:`free_params` order."""
        if isinstance(X, pd.DataFrame):
            X = X[self.free_params].values
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty_like(X)
        for j, (lo, hi, is_log) in enumerate(self._axes()):
            x = np.log10(X[:, j]) if is_log else X[:, j]
            out[:, j] = (x - lo) / max(hi - lo, 1e-12)
        return out

    def from_unit(self, U, with_fixed: bool = False) -> pd.DataFrame:
        """Unit-cube coordinates (n, dim) -> DataFrame of physical values (free parameters, and the fixed
        ones too if ``with_fixed``)."""
        U = np.atleast_2d(np.asarray(U, dtype=float))
        out = {}
        for j, (p, (lo, hi, is_log)) in enumerate(zip(self.free_params, self._axes())):
            x = lo + U[:, j] * (hi - lo)
            out[p] = 10.0 ** x if is_log else x
        df = pd.DataFrame(out)
        if with_fixed:
            for p, v in self.fixed_values.items():
                df[p] = v
        return df

    def contains(self, table: pd.DataFrame, rtol: float = 1e-5) -> np.ndarray:
        """Boolean mask of the rows of ``table`` inside the box. The tolerance covers parameters written
        with six significant digits. Failed models are judged by their parameters like any other."""
        keep = np.ones(len(table), bool)
        for p, (lo, hi) in self.bounds.items():
            tol = rtol * max(abs(lo), abs(hi))
            v = table[p].values.astype(float)
            keep &= (v >= lo - tol) & (v <= hi + tol)
        return keep

    def __repr__(self) -> str:
        axes = ", ".join(f"{p}[{self.bounds[p][0]:g}, {self.bounds[p][1]:g}{', log' if p in self.log_params else ''}]"
                         for p in self.free_params)
        fixed = ", ".join(f"{p}={v:g}" for p, v in self.fixed_values.items())
        return f"ParameterSpace({self.name}: {axes}" + (f"; fixed {fixed}" if fixed else "") + ")"


def as_space(space) -> ParameterSpace:
    """Accept a ParameterSpace, a dict, or a YAML path."""
    if isinstance(space, ParameterSpace):
        return space
    if isinstance(space, dict):
        return ParameterSpace.from_dict(space)
    return ParameterSpace.from_yaml(space)
