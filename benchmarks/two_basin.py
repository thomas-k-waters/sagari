"""The two-basin benchmark of the paper (Section 4.3 and Appendix B), run with sagari.

Each trial is a campaign of 256 Sobol points and five rounds of 128 (896 models) on the synthetic objective
of :mod:`sagari.benchmark`, for three samplers:

  full      the method of the paper (Matern 5/2 kernel, 30 sigma training window, 35% global share, plug-in
            incumbent, two-scale candidates, minimum separation, verification of isolated models)
  ablated   the same search without its robustness measures: squared-exponential kernel, no training window,
            15% global share, the lowest model as the incumbent, no local candidates, no minimum separation
            or verification, one candidate pool for the whole campaign (the feasibility weighting is kept)
  sobol     Sobol sampling alone

After every round it records whether the best point of the smoothed estimate lies in the deeper basin, its
M_BH error, the models within 2 sigma of the deeper minimum, and the share of local points in the failure
region. Trials use seeds 1000, 1001, ... as in the paper (100 trials: Figure 1).

    python two_basin.py [n_trials] [--hopeless] [--out results/two_basin.csv] [--procs N]
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sagari as sg  # noqa: E402
from sagari.benchmark import NOISE, PILOT_SPACE, TwoBasin  # noqa: E402

N0, NR, N_ROUNDS = 256, 128, 5
WASTE_ABOVE = 2000.0           # a local point with true Delta chi2 above this is wasted
VARIANTS = {
    "full": dict(kernel="matern52", train_window=30.0, global_fraction=0.35),
    "ablated": dict(kernel="rbf", train_window=None, global_fraction=0.15, incumbent="raw", n_local_candidates=0,
                    min_separation=0.0, verify=False, redraw_candidates=False),
    "sobol": None,
}


def trial(args):
    variant, seed, hopeless = args
    obj = TwoBasin(hopeless=hopeless)
    space = PILOT_SPACE
    rng = np.random.default_rng(seed)
    sob = sg.SobolGlobalDesign(space.dim, seed)
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
        f, in_n = obj.true(space.to_unit(table))
        row = dict(variant=variant, seed=seed, round=rnd, n_models=len(table), n_failed=int(table["chi2"].isna().sum()),
                   waste=float(np.mean(waste)) if waste else np.nan, n_in_N=int((((f - 1000) < 2 * NOISE) & in_n).sum()))
        est = sg.smoothed_estimate(table, space, seed=seed, kernel="matern52", train_window=30.0)
        if est is not None:
            row.update(est_found=obj.found(est), est_mbh_err=obj.mbh_error(est),
                       est_mbh_covers=bool(est["mbh"][1] <= 8.0e8 <= est["mbh"][2]))
        rows.append(row)
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
    ap.add_argument("--variants", default="full,ablated,sobol")
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
