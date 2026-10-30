"""The two-basin benchmark of the paper (Section 4.1, Table 1 and Appendix A), run with sagari.

Each trial is a campaign of 256 points and five rounds of 128 (896 models) on the synthetic objective of
:mod:`sagari.benchmark`, for the seven methods of Table 1:

  full         the method of the paper (Matern 5/2 kernel, 30 sigma training window, 35% global share, plug-in
               incumbent, two-scale candidates, minimum separation, verification of isolated models)
  lhc          the full method with Latin-hypercube space filling: the first round and the global share of each
               later round are new, independent Latin hypercubes (a Latin hypercube of 256 points stays one
               only if it grows by a multiple of 256)
  no_noise     the full method without a noise term in the chi2 surrogates: their nugget is fixed at 1e-8 of the
               variance of the training values, in the search and in the smoothed estimate, so everything
               measured in units of the scatter uses sigma_chi2 = 1. The feasibility GP keeps its noise term.
  ablated      the same search without its robustness measures (the "basic search" of the paper):
               squared-exponential kernel, no training window, 15% global share, the lowest model as the
               incumbent, no local candidates, no minimum separation or verification, one candidate pool for
               the whole campaign (the feasibility weighting is kept)
  sobol        Sobol sampling alone
  lhc_only     Latin-hypercube sampling alone (a new Latin hypercube each round)
  nested_grid  a coarse grid of 4 values per parameter (cell centres), then finer 4^d grids centred on the
               lowest-chi2 model so far, each half the extent of the previous one, run in two interleaved
               (checkerboard) halves of 128 models

After every round it records whether the best point of the smoothed estimate lies in the deeper basin, its
M_BH error, the models within 2 sigma of the deeper minimum, and the share of local points in the failure
region. The smoothed estimate is fitted with the training cap of 1000 models of the final constraints, as in
the paper. Trials use seeds 1000, 1001, ... as in the paper (1000 trials each in Table 1).

    python two_basin.py [n_trials] [--hopeless] [--variants full,lhc,...] [--out results/two_basin.csv] [--procs N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import qmc  # noqa: E402
from sklearn.gaussian_process.kernels import WhiteKernel  # noqa: E402

import sagari as sg  # noqa: E402
import sagari.gp  # noqa: E402
from sagari.benchmark import NOISE, PILOT_SPACE, TwoBasin  # noqa: E402

N0, NR, N_ROUNDS = 256, 128, 5
WASTE_ABOVE = 2000.0           # a local point with true Delta chi2 above this is wasted
EST_CAP = 1000                 # training cap of the smoothed estimate that scores each trial
FULL = dict(kernel="matern52", train_window=30.0, global_fraction=0.35)
VARIANTS = {
    "full": FULL,
    "lhc": FULL,
    "no_noise": FULL,
    "ablated": dict(kernel="rbf", train_window=None, global_fraction=0.15, incumbent="raw", n_local_candidates=0,
                    min_separation=0.0, verify=False, redraw_candidates=False),
    "sobol": None,
    "lhc_only": None,
    "nested_grid": "grid",
}
JITTER = 1e-8                  # nugget of the no_noise variant, in units of the variance of the training values
_make_kernel = sagari.gp.make_kernel


class LatinHypercubeDesign(sg.SobolGlobalDesign):
    """The space-filling design of the Latin-hypercube variants. Each call draws a new, independent Latin
    hypercube, seeded by the campaign seed and the number of points drawn so far."""

    def peek(self, n_new: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, 31, self.num_generated])
        try:
            gen = qmc.LatinHypercube(d=self.dim, rng=rng)
        except TypeError:                        # SciPy < 1.15
            gen = qmc.LatinHypercube(d=self.dim, seed=rng)
        return gen.random(n_new)


def _exact_kernel(dim, kernel="matern52"):
    """The kernel of the no_noise variant: the chi2 surrogates (Matern) get a fixed, negligible nugget, and the
    feasibility GP (squared-exponential) keeps its fitted one."""
    k = _make_kernel(dim, kernel)
    if kernel == "matern52":
        k = k.k1 + WhiteKernel(noise_level=JITTER, noise_level_bounds="fixed")
    return k


def record(obj, space, table, variant, seed, rnd, waste):
    """The quantities recorded after each round, from the smoothed estimate with the training cap of 1000."""
    f, in_n = obj.true(space.to_unit(table))
    row = dict(variant=variant, seed=seed, round=rnd, n_models=len(table), n_failed=int(table["chi2"].isna().sum()),
               waste=float(np.mean(waste)) if waste else np.nan, n_in_N=int((((f - 1000) < 2 * NOISE) & in_n).sum()))
    est = sg.smoothed_estimate(table, space, seed=seed, kernel="matern52", train_window=30.0, n_train_cap=EST_CAP)
    if est is not None:
        row.update(est_found=obj.found(est), est_mbh_err=obj.mbh_error(est),
                   est_mbh_covers=bool(est["mbh"][1] <= 8.0e8 <= est["mbh"][2]))
    return row


def grid_trial(seed, hopeless):
    """The nested grid: each 4^d grid is evaluated in grid order (one draw of the scatter per model) and then
    run in two checkerboard halves of 128, so that each half spans the whole grid."""
    obj, space = TwoBasin(hopeless=hopeless), PILOT_SPACE
    rng = np.random.default_rng(seed)
    table, rows = None, []
    centre, half = np.full(space.dim, 0.5), 0.5
    while table is None or len(table) < N0 + N_ROUNDS * NR:
        axes = [np.clip(c + half * ((np.arange(4) + 0.5) / 4 * 2 - 1), 0, 1) for c in centre]
        U = np.array(list(itertools.product(*axes)))
        order = np.argsort(np.array(list(itertools.product(range(4), repeat=space.dim))).sum(1) % 2, kind="stable")
        d = space.from_unit(U, with_fixed=True)
        new = d[space.free_params].copy()
        new["chi2"] = obj(d, rng)
        new = new.iloc[order]
        for part in ([new] if table is None else [new.iloc[:NR], new.iloc[NR:]]):
            if table is not None and len(table) >= N0 + N_ROUNDS * NR:
                break
            table = part.reset_index(drop=True) if table is None else pd.concat([table, part], ignore_index=True)
            rows.append(record(obj, space, table, "nested_grid", seed, len(rows), []))
        ok = np.isfinite(table["chi2"].values)
        i_best = np.flatnonzero(ok)[np.argmin(table["chi2"].values[ok])]
        centre, half = space.to_unit(table.iloc[[i_best]])[0], half / 2     # refine around the lowest model
    return rows


def trial(args):
    variant, seed, hopeless = args
    if VARIANTS[variant] == "grid":
        return grid_trial(seed, hopeless)
    sagari.gp.make_kernel = _exact_kernel if variant == "no_noise" else _make_kernel
    obj = TwoBasin(hopeless=hopeless)
    space = PILOT_SPACE
    rng = np.random.default_rng(seed)
    design = LatinHypercubeDesign if variant in ("lhc", "lhc_only") else sg.SobolGlobalDesign
    sob = design(space.dim, seed)
    d = sg.initial_design(space, N0, sob)
    table = d[space.free_params].copy()
    table["chi2"] = obj(d, rng)
    rows, waste = [], []
    for rnd in range(N_ROUNDS + 1):
        if rnd > 0:
            if VARIANTS[variant] is None:
                d = sg.initial_design(space, NR, sob)
            else:
                # noise_diagnostic=False: the band for verifying isolated models is twice the sd of the 20 lowest
                # chi2, as in the benchmark runs of the paper
                d = sg.propose_round(table, space, NR, sob, seed=seed, noise_diagnostic=False, **VARIANTS[variant])
                loc = (d["stage"] != "sobol_global_continuation").values
                if loc.any():
                    waste.append(float(((obj.true(space.to_unit(d[loc]))[0] - 1000) > WASTE_ABOVE).mean()))
            new = d[space.free_params].copy()
            new["chi2"] = obj(d, rng)
            table = pd.concat([table, new], ignore_index=True)
        # The runs of the paper drew a search pool for a diagnostic not reported here from the generator that also
        # draws the chi2 scatter; draw the same numbers so that every later round sees identical scatter.
        rng.random((20000, space.dim))
        for _ in range(50):
            rng.standard_normal((20, space.dim))
        rows.append(record(obj, space, table, variant, seed, rnd, waste))
    return rows


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n_trials", type=int, nargs="?", default=20)
    ap.add_argument("--hopeless", action="store_true", help="add the region of failed and very-high-chi2 models")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated, from: " + ", ".join(VARIANTS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out or f"results/two_basin_{'hopeless' if a.hopeless else 'plain'}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".jsonl")
    done = set()
    if part.exists():          # finished trials are kept, so an interrupted run loses nothing
        done = {(r[0]["variant"], r[0]["seed"]) for r in map(json.loads, part.read_text().splitlines())}
    jobs = [(v, 1000 + s, a.hopeless) for v in a.variants.split(",") for s in range(a.n_trials)
            if (v, 1000 + s) not in done]
    print(f"{len(jobs)} trials to run ({len(done)} already in {part})", flush=True)
    with Pool(a.procs) as pool:
        for rs in pool.imap_unordered(trial, jobs):
            with open(part, "a") as f:
                f.write(json.dumps(rs, default=float) + "\n")
            print(f"{rs[0]['variant']} {rs[0]['seed']}: found={rs[-1].get('est_found')}", flush=True)
    rows = [r for rs in map(json.loads, part.read_text().splitlines()) for r in rs if r["seed"] < 1000 + a.n_trials]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    fin = df[df["n_models"] == N0 + N_ROUNDS * NR]
    print(f"\nat {N0 + N_ROUNDS * NR} models ({'with' if a.hopeless else 'without'} the failure region):")
    for v, g in fin.groupby("variant"):
        k, n = int(g["est_found"].sum()), len(g)
        lo, hi = wilson(k, n)
        print(f"  {v:8s} deeper basin found in {k}/{n} ({100 * lo:.0f}-{100 * hi:.0f}%), median M_BH error "
              f"{g['est_mbh_err'].median():.2f} dex, median models within 2 sigma of it {g['n_in_N'].median():.0f}, "
              f"local points wasted {100 * g['waste'].mean():.0f}%")


if __name__ == "__main__":
    main()
