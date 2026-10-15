"""Command-line interface: ``sagari init | next | status | posterior``.

A campaign on a cluster::

    sagari init my_campaign --space space.yaml --converge-on mbh,upsilon
    sagari next my_campaign                       # writes rounds/round_000.csv (256 Sobol points)
    # ... run the models of the round with your own code; append model_id,chi2 to results.csv ...
    sagari next my_campaign --results results.csv     # the next round (search, error bars, or done)
    sagari status my_campaign --results results.csv
    sagari posterior my_campaign --results results.csv --out posterior
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ._version import __version__


def _cmd_init(a):
    from .campaign import Campaign
    settings = {}
    if a.settings:
        import yaml
        settings = yaml.safe_load(Path(a.settings).read_text()) or {}
    if a.seed is not None:
        settings["seed"] = a.seed
    if a.n_initial is not None:
        settings["n_initial"] = a.n_initial
    if a.n_round is not None:
        settings["n_round"] = a.n_round
    if a.converge_on:
        settings["converge_on"] = a.converge_on.split(",")
    c = Campaign.create(a.directory, a.space, overwrite=a.overwrite, **settings)
    print(f"created {c.dir / 'campaign.yaml'}: {c.space}")
    print(f"converge on: {', '.join(c.converge_on)}; next: sagari next {a.directory}")


def _cmd_next(a):
    from .campaign import Campaign
    c = Campaign(a.directory)
    d = c.next_round(a.results, phase=a.phase)
    if d is None:
        print("The campaign is done: sample the posterior with `sagari posterior`.")
        return
    path = c.dir / "rounds" / f"round_{d.attrs['round']:03d}.csv"
    stages = d["stage"].value_counts().to_dict()
    print(f"wrote {path}: {len(d)} models ({d.attrs['phase']}; " + ", ".join(f"{k} {v}" for k, v in stages.items()) + ")")
    print("Evaluate them and add a row per model (model_id, chi2) to your results table; chi2 = nan for a failure.")


def _cmd_status(a):
    from .campaign import Campaign
    st = Campaign(a.directory).status(a.results)
    if a.json:
        Path(a.json).write_text(json.dumps(st, indent=1, default=str))


def _cmd_posterior(a):
    from .campaign import Campaign
    from .posterior import pooled_quantiles, profile_intervals, profiles, sample_posterior, support_cap, surface_draws
    c = Campaign(a.directory)
    gp, _, _ = c.surrogate(a.results)
    print(f"surrogate: {len(gp.gp.y_train_)} training models, scatter sigma_chi2 = {gp.scatter:.2f}")
    post = sample_posterior(gp, c.space, seed=a.seed, nsteps=a.nsteps, burn=a.burn)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    post.to_csv(out / "posterior_samples.csv", index=False, float_format="%.8g")
    print(f"plug-in posterior: {len(post)} samples (acceptance {post.attrs['acceptance']:.2f}) -> "
          f"{out / 'posterior_samples.csv'}")
    summary = {"plug_in": {}, "marginalized": {}, "profile": {}}
    for p in c.space.free_params:
        q = np.percentile(post[p].values, [15.85, 50, 84.15])
        summary["plug_in"][p] = [float(q[1]), float(q[0]), float(q[2])]
    if a.marginalize:
        idx, D, _ = surface_draws(gp, post.attrs["unit"], n_refit=0)
        for p in c.space.free_params:
            lo, med, hi = pooled_quantiles(post[p].values[idx], D)
            summary["marginalized"][p] = [float(med), float(lo), float(hi)]
    if a.profiles:
        prof = profiles(gp, c.space, c.seed, support=support_cap(gp))
        summary["profile"] = {p: {f"{D:g}": v for D, v in iv.items()}
                              for p, iv in profile_intervals(prof, c.space, levels=(1.0, 4.0, 10.0)).items()}
        summary["profile_minimum"] = float(prof["mu_min"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for kind in ("plug_in", "marginalized"):
        for p, (m, lo, hi) in summary[kind].items():
            print(f"  {kind:12s} {p:>12s}: {m:.4g} (+{hi - m:.3g} / -{m - lo:.3g})")
    for p, levels in summary["profile"].items():
        print(f"  profile      {p:>12s}: " + "; ".join(
            f"dchi2<={D}: " + " + ".join(f"{a:.4g}-{b:.4g}" for a, b in segs) for D, segs in levels.items()))
    print(f"summary -> {out / 'summary.json'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sagari", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"sagari {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a campaign directory")
    s.add_argument("directory")
    s.add_argument("--space", required=True, help="YAML file with the parameters (bounds, scale, fixed values)")
    s.add_argument("--settings", help="YAML file of settings (see sagari.campaign.DEFAULTS)")
    s.add_argument("--seed", type=int)
    s.add_argument("--n-initial", type=int)
    s.add_argument("--n-round", type=int)
    s.add_argument("--converge-on", help="comma-separated parameters the stopping rule watches (default: all free)")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=_cmd_init)

    s = sub.add_parser("next", help="write the next round (initial design, search, error bars) or report done")
    s.add_argument("directory")
    s.add_argument("--results", help="results table (model_id or parameters, and chi2)")
    s.add_argument("--phase", choices=["search", "errorbar"], help="force the kind of round")
    s.set_defaults(func=_cmd_next)

    s = sub.add_parser("status", help="monitoring ranges, stopping rule, error-bar verdicts")
    s.add_argument("directory")
    s.add_argument("--results")
    s.add_argument("--json", help="also write the status as JSON to this file")
    s.set_defaults(func=_cmd_status)

    s = sub.add_parser("posterior", help="sample the posterior on the final surrogate")
    s.add_argument("directory")
    s.add_argument("--results", required=True)
    s.add_argument("--out", default="posterior")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--nsteps", type=int, default=25000)
    s.add_argument("--burn", type=int, default=5000)
    s.add_argument("--no-marginalize", dest="marginalize", action="store_false",
                   help="skip the average over the surrogate's uncertainty")
    s.add_argument("--profiles", action="store_true", help="also compute the Delta chi2 profile intervals (slower)")
    s.set_defaults(func=_cmd_posterior)

    a = p.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
