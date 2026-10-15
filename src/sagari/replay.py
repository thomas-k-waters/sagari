"""Replaying a campaign on an archive of already-evaluated models (Section 4.1 of the paper).

Each proposed point is snapped to the nearest archived model in min-max-scaled space and returns that
model's real chi2 (failed models as failures), so no interpolated value enters the loop. Several proposals can
snap to the same archived model; they count as models, but every surrogate should see each model once
(:meth:`sagari.Campaign` drops exact repeats before fitting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def kdtree_snap(query: pd.DataFrame, archive: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """For each row of ``query``, the nearest row of ``archive`` in the parameters ``dims`` (each scaled to
    the archive's range), with ``snap_distance``. Works for any archive, not only a Cartesian grid."""
    mins = archive[dims].min().values
    ranges = np.maximum(archive[dims].max().values - mins, 1e-12)
    tree = cKDTree((archive[dims].values - mins) / ranges)
    dist, idx = tree.query((query[dims].values - mins) / ranges, k=1)
    out = archive.iloc[idx].reset_index(drop=True).copy()
    out["snap_distance"] = dist
    return out


def archive_evaluator(archive: pd.DataFrame, dims: list[str]):
    """An ``evaluate`` function for :meth:`sagari.Campaign.run` that answers from ``archive`` (columns
    ``dims`` and ``chi2``, NaN for failed models): returns the snapped chi2 and records the snapped
    parameters, so the campaign stores the model that was actually returned."""
    def evaluate(design: pd.DataFrame):
        s = kdtree_snap(design, archive, dims)
        return s["chi2"].values, s[dims]
    return evaluate
