"""Scrambled-Sobol global design with persistent, extendable state (Section 3.3 of the paper).

A fresh ``scipy.stats.qmc.Sobol(seed=seed)`` advanced with ``fast_forward(n)`` and then ``random(m)``
reproduces exactly the tail of a single ``random(n + m)`` draw from the same seed, so the only state that
has to persist between rounds is ``{dim, seed, num_generated}``. Every refinement round continues the same
sequence for its global share, so the campaign's global coverage is one low-discrepancy sequence.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import qmc

DEFAULT_SEED = 20260817


class SobolGlobalDesign:
    """A scrambled Sobol sequence over the unit cube whose position can be saved to a JSON file.

    Parameters
    ----------
    dim : int
        Number of free parameters.
    seed : int
        Scramble seed. Two campaigns with the same seed and dimension propose the same global points.
    state_path : str or Path, optional
        JSON file holding ``{dim, seed, num_generated}``. Without it the state lives in memory only.
    """

    def __init__(self, dim: int, seed: int = DEFAULT_SEED, state_path: str | Path | None = None):
        self.dim, self.seed = int(dim), int(seed)
        self.state_path = Path(state_path) if state_path is not None else None
        self.num_generated = 0
        if self.state_path is not None and self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state["dim"] != self.dim:
                raise ValueError(f"{self.state_path}: stored dim={state['dim']} but {self.dim} requested; a change of "
                                 "the free parameters invalidates the sequence -- start a new state file.")
            if state["seed"] != self.seed:
                raise ValueError(f"{self.state_path}: stored seed={state['seed']} but {self.seed} requested; use the "
                                 "stored seed to extend the sequence, or start a new state file deliberately.")
            self.num_generated = int(state["num_generated"])

    def _save(self):
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"dim": self.dim, "seed": self.seed,
                                               "num_generated": self.num_generated}, indent=2))

    def peek(self, n_new: int) -> np.ndarray:
        """The next ``n_new`` points without advancing the state."""
        gen = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        if self.num_generated > 0:
            gen.fast_forward(self.num_generated)
        with warnings.catch_warnings():       # rounds of 128 after 256 are not powers of 2, by design
            warnings.filterwarnings("ignore", message="The balance properties of Sobol")
            return gen.random(n_new)

    def extend(self, n_new: int) -> np.ndarray:
        """Draw the next ``n_new`` points in [0, 1)^dim, advancing (and saving) the state.

        The net property of the sequence is exact when the cumulative count is a power of two (e.g. an
        initial design of 256); other counts still improve the discrepancy monotonically.
        """
        pts = self.peek(n_new)
        self.num_generated += n_new
        self._save()
        return pts
