"""Adapter between an sagari campaign and the Fortran orbit-superposition pipeline of the paper
(runalls/runallt and the SLURM job scripts that run one model per line of a commands file).

    # commands file for a round: one "runallt <upsilon> <xi> <mbh> <v_c> <r_c> <apparent_q> <inclination>" line
    python sco_runalls.py commands my_campaign 3 --apparent-q 0.625 --out round_003.run [--runner runallt]

    # results table from the res/ directory of the modeling directory, for every round written so far
    python sco_runalls.py collect my_campaign path/to/modeling/res --out results.csv

The parameter values are written with six significant digits (``{:g}``), as the pipeline names its models
(libn = m<upsilon>b<mbh>v<v_c>r<r_c>i<inclination>x<xi>), and the results table carries those rounded values,
so the campaign fits exactly the models that were run. A model counts as failed when its cres file has no
result, chi2 is not finite and positive, or alpha < 1e-8 (the weight solve did not get past its first step);
a model without a cres file has not run yet and is left out.

The space file must name the parameters ``upsilon``, ``xi``, ``mbh``, ``v_c``, ``r_c`` and ``inclination``
(free or fixed); ``apparent_q`` is passed on the command line here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sagari as sg

ORDER = ["upsilon", "xi", "mbh", "v_c", "r_c", "apparent_q", "inclination"]      # runalls' positional arguments
CRES_COLS = ["upsilon", "mbh", "v_c", "r_c", "inclination", "xi", "chi2", "alpha"]


def fmt(v: float) -> str:
    return f"{v:g}"


def libn(row) -> str:
    return (f"m{fmt(row['upsilon'])}b{fmt(row['mbh'])}v{fmt(row['v_c'])}r{fmt(row['r_c'])}"
            f"i{fmt(row['inclination'])}x{fmt(row['xi'])}")


def commands(campaign: sg.Campaign, i: int, apparent_q: float, out: Path, runner: str = "runallt"):
    d = campaign.round_table(i)
    d["apparent_q"] = apparent_q
    missing = [p for p in ORDER if p not in d.columns]
    if missing:
        raise SystemExit(f"round {i} lacks {missing}: add them to the space as free or fixed parameters")
    out.write_text("".join(f"{runner} " + " ".join(fmt(r[p]) for p in ORDER) + "\n" for _, r in d.iterrows()))
    print(f"wrote {out}: {len(d)} models of round {i}")


def read_cres(path: Path):
    text = path.read_text().split()
    try:
        params = [float(x) for x in text[:6]]
        chi2, alpha = (float(text[6]), float(text[7])) if len(text) == 8 else (np.nan, np.nan)
    except (ValueError, IndexError):
        return None
    if len(params) != 6:
        return None
    rec = dict(zip(CRES_COLS, params + [chi2, alpha]))
    if not (np.isfinite(chi2) and chi2 > 0 and alpha >= 1e-8):
        rec["chi2"] = np.nan                         # failed model: kept, so the search avoids its region
    return rec


def collect(campaign: sg.Campaign, res_dir: Path, out: Path):
    prop = campaign.proposals()
    rows, n_missing = [], 0
    for _, r in prop.iterrows():
        f = res_dir / f"cres.{libn(r)}"
        rec = read_cres(f) if f.exists() else None
        if rec is None:
            n_missing += 1
            continue
        rows.append({"model_id": r["model_id"], **{p: rec[p] for p in campaign.space.free_params}, "chi2": rec["chi2"],
                     "alpha": rec["alpha"]})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, float_format="%.10g")
    print(f"wrote {out}: {len(df)} models ({int(df['chi2'].isna().sum()) if len(df) else 0} failed), "
          f"{n_missing} not run yet")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("commands")
    s.add_argument("campaign")
    s.add_argument("round", type=int)
    s.add_argument("--apparent-q", type=float, required=True)
    s.add_argument("--runner", default="runallt")
    s.add_argument("--out", required=True)
    s = sub.add_parser("collect")
    s.add_argument("campaign")
    s.add_argument("res_dir")
    s.add_argument("--out", default="results.csv")
    a = ap.parse_args()
    c = sg.Campaign(a.campaign)
    if a.cmd == "commands":
        commands(c, a.round, a.apparent_q, Path(a.out), a.runner)
    else:
        collect(c, Path(a.res_dir), Path(a.out))


if __name__ == "__main__":
    main()
