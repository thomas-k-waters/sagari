"""A campaign on disk: the rounds proposed so far, the Sobol state, and the rules that decide the next round.

The campaign never runs a model itself. Each round is written as a CSV of parameter sets with a ``model_id``;
you evaluate them with your own code, however and wherever you like, and hand back a results table with
``model_id`` (or the parameter columns) and ``chi2`` (empty, NaN or ``inf`` for a model that failed). The
campaign then decides what comes next, as in the paper (the outline of Section 3):

1. round 0: ``n_initial`` scrambled-Sobol points;
2. search rounds of ``n_round`` by feasibility-weighted Expected Improvement, until the smoothed ranges of the
   ``converge_on`` parameters stop moving (threshold 0.2 of their width, patience 1, at least 2 refinement
   rounds);
3. error-bar rounds that fill the region near the minimum and the parameter profiles, each model's chi2
   predicted before it runs, until the latent sd at the ends of the Delta chi2 = 4 and 10 ranges is <= 3
   and the ends stop moving, or ``errorbars.max_rounds`` rounds;
4. done: sample the posterior on the final surrogate (:meth:`Campaign.surrogate`, :mod:`sagari.posterior`).

Directory layout::

    campaign.yaml          parameter space and settings (edit the bounds here to narrow the box)
    sobol_state.json       position in the global Sobol sequence
    rounds/round_NNN.csv   the models of each round (model_id, parameters, stage, predictions)
    rounds/round_NNN.json  what the round was and the diagnostics of its proposal
    log.jsonl              one line per decision

For in-process use (an evaluator written in Python, or a replay on an archive) see :meth:`Campaign.run`.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ._version import __version__
from .acquisition import initial_design, propose_round
from .errorbars import errorbar_round, gp_profiles, inference_gp, out_of_sample, profile_compact, profile_verdict
from .monitor import estimate_converged, estimate_moved, smoothed_estimate, trusted_best_point
from .sobol import DEFAULT_SEED, SobolGlobalDesign
from .space import ParameterSpace, as_space

DEFAULTS = {
    "seed": DEFAULT_SEED,
    "n_initial": 256,
    "n_round": 128,
    "converge_on": None,            # parameters the stopping rule watches (default: all free)
    "stop": {"frac": 0.2, "patience": 1, "min_rounds": 2, "max_rounds": 20},
    "sampler": {"kernel": "matern52", "train_window": 30.0, "global_fraction": 0.35},
    "estimate": {"n_train_cap": 1000},   # 300 in the paper (v1.0.0); see docs/constants.md
    "errorbars": {"n_round": 128, "n_profile": 7, "n_train_cap": 1000, "sd_tol": 3.0, "frac": 0.2,
                  "levels": [4.0, 10.0], "max_rounds": 3, "auto_window": False},
}

SEARCH, ERRORBAR = "search", "errorbar"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_results(path) -> pd.DataFrame:
    """A results table (CSV, or whitespace-separated with a header line). ``chi2`` that is empty, NaN, inf or
    non-numeric marks a failed model; a column ``failed`` (true/1) does too."""
    path = Path(path)
    with open(path) as f:
        head = f.readline()
    df = pd.read_csv(path, sep="," if "," in head else r"\s+", comment=None)
    return _clean_results(df)


def _clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "chi2" not in df.columns:
        raise ValueError("the results table needs a 'chi2' column")
    df["chi2"] = pd.to_numeric(df["chi2"], errors="coerce").astype(float)
    df.loc[~np.isfinite(df["chi2"]), "chi2"] = np.nan
    if "failed" in df.columns:
        failed = df["failed"].astype(str).str.lower().isin(("1", "true", "yes"))
        df.loc[failed, "chi2"] = np.nan
    return df


class Campaign:
    """An adaptive campaign stored in ``directory``. Create one with :meth:`Campaign.create`."""

    def __init__(self, directory):
        self.dir = Path(directory)
        cfg_path = self.dir / "campaign.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} not found; create the campaign with Campaign.create")
        raw = yaml.safe_load(cfg_path.read_text())
        self.space = ParameterSpace.from_dict(raw["space"], name=raw.get("name"))
        self.settings = _merge(DEFAULTS, raw.get("settings", {}))
        self.converge_on = list(self.settings["converge_on"] or self.space.free_params)
        unknown = set(self.converge_on) - set(self.space.free_params)
        if unknown:
            raise ValueError(f"converge_on has parameters that are not free: {sorted(unknown)}")
        self.seed = int(self.settings["seed"])
        (self.dir / "rounds").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, directory, space, overwrite: bool = False, **settings) -> "Campaign":
        """Start a campaign in ``directory`` for ``space`` (a ParameterSpace, a dict, or a YAML path).

        Settings (defaults are those of the paper): ``seed``, ``n_initial`` (256), ``n_round`` (128),
        ``converge_on`` (list of parameters; default all free), ``stop`` (``frac`` 0.2, ``patience`` 1,
        ``min_rounds`` 2, ``max_rounds`` 20), ``sampler`` (keyword arguments of
        :func:`sagari.propose_round`), ``estimate`` (``n_train_cap`` 1000; the paper used 300), ``errorbars`` (``n_round``
        128, ``n_profile`` 7, ``n_train_cap`` 1000, ``sd_tol`` 3, ``frac`` 0.2, ``levels`` [4, 10],
        ``max_rounds`` 3, ``auto_window`` False).
        """
        d = Path(directory)
        if (d / "campaign.yaml").exists() and not overwrite:
            raise FileExistsError(f"{d / 'campaign.yaml'} exists; open it with Campaign({str(d)!r}) instead")
        space = as_space(space)
        d.mkdir(parents=True, exist_ok=True)
        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)} (known: {sorted(DEFAULTS)})")
        doc = {"name": space.name, "sagari_version": __version__, "space": space.to_dict(),
               "settings": _merge(DEFAULTS, settings)}
        (d / "campaign.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        return cls(d)

    # ------------------------------------------------------------------ rounds on disk
    @property
    def sobol(self) -> SobolGlobalDesign:
        return SobolGlobalDesign(self.space.dim, self.seed, state_path=self.dir / "sobol_state.json")

    def rounds(self) -> list[dict]:
        """Metadata of every round written so far, in order."""
        return [json.loads(p.read_text()) for p in sorted((self.dir / "rounds").glob("round_[0-9][0-9][0-9].json"))]

    def round_table(self, i: int) -> pd.DataFrame:
        return pd.read_csv(self.dir / "rounds" / f"round_{i:03d}.csv")

    def proposals(self) -> pd.DataFrame:
        """Every model proposed so far, with its round."""
        tabs = []
        for meta in self.rounds():
            t = self.round_table(meta["round"])
            t["round"] = meta["round"]
            t["phase"] = meta["phase"]
            tabs.append(t)
        return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()

    def _write_round(self, design: pd.DataFrame, phase: str, extra: dict) -> pd.DataFrame:
        i = len(self.rounds())
        design = design.copy()
        design.insert(0, "model_id", [f"r{i:03d}_{k:04d}" for k in range(len(design))])
        cols = ["model_id"] + self.space.free_params + list(self.space.fixed_values) + \
               [c for c in design.columns if c not in ["model_id", *self.space.free_params, *self.space.fixed_values]]
        design = design[cols]
        design.to_csv(self.dir / "rounds" / f"round_{i:03d}.csv", index=False, float_format="%.10g")
        attrs = {k: v for k, v in design.attrs.items() if k != "profile_before"}
        meta = {"round": i, "phase": phase, "n": len(design), "time": time.time(),
                "bounds": {p: list(self.space.bounds[p]) for p in self.space.free_params},
                "attrs": json.loads(json.dumps(attrs, default=_jsonable)), **extra}
        if "profile_before" in design.attrs:
            meta["profile_before"] = json.loads(json.dumps(design.attrs["profile_before"], default=_jsonable))
        (self.dir / "rounds" / f"round_{i:03d}.json").write_text(json.dumps(meta, indent=1, default=_jsonable))
        self._log({"action": "write_round", "round": i, "phase": phase, "n": len(design)})
        # hand back exactly what is on disk, so a campaign continued in another process sees the same values
        on_disk = self.round_table(i)
        on_disk.attrs.update(design.attrs, round=i, phase=phase)
        return on_disk

    def _log(self, rec: dict):
        with open(self.dir / "log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.time(), **rec}, default=_jsonable) + "\n")

    # ------------------------------------------------------------------ results
    def results_table(self, results) -> pd.DataFrame:
        """Join a results table (DataFrame or path) to the proposals: one row per proposed model with its
        ``chi2`` (NaN: failed) and ``evaluated`` (False: no result yet). Results are matched by ``model_id``
        if they have it, otherwise by their parameter values (relative tolerance 1e-6). A result may carry
        the parameters actually run (e.g. rounded by the modeling code); those replace the proposed ones."""
        res = read_results(results) if not isinstance(results, pd.DataFrame) else _clean_results(results)
        prop = self.proposals()
        if prop.empty:
            raise ValueError("no rounds have been written yet")
        if "model_id" in res.columns:
            res = res.drop_duplicates("model_id", keep="last").set_index("model_id")
            out = prop.copy()
            out["evaluated"] = out["model_id"].isin(res.index)
            out["chi2"] = out["model_id"].map(res["chi2"])
            for p in self.space.free_params:
                if p in res.columns:
                    v = out["model_id"].map(res[p])
                    out[p] = np.where(v.notna(), v, out[p])
        else:
            missing = [p for p in self.space.free_params if p not in res.columns]
            if missing:
                raise ValueError(f"the results table has no model_id and lacks the parameters {missing}")
            Xr = res[self.space.free_params].values.astype(float)
            Xp = prop[self.space.free_params].values.astype(float)
            out = prop.copy()
            out["evaluated"] = False
            out["chi2"] = np.nan
            scale = np.maximum(np.abs(Xp), 1e-300)
            for i in range(len(prop)):
                hit = np.where(np.all(np.abs(Xr - Xp[i]) <= 1e-6 * scale[i], axis=1))[0]
                if len(hit):
                    out.loc[out.index[i], "evaluated"] = True
                    out.loc[out.index[i], "chi2"] = res["chi2"].values[hit[-1]]
        return out

    def _fit_table(self, joined: pd.DataFrame, upto_round: int | None = None) -> pd.DataFrame:
        """The evaluated models inside the current box (failed ones included as NaN), up to a round, each
        set of parameters once: a surrogate reads exact repeats as noiseless."""
        t = joined[joined["evaluated"]]
        if upto_round is not None:
            t = t[t["round"] <= upto_round]
        t = t[self.space.contains(t)]
        return t.drop_duplicates(subset=self.space.free_params + list(self.space.fixed_values), keep="first")

    # ------------------------------------------------------------------ status
    def _estimate(self, table):
        s = self.settings
        return smoothed_estimate(table, self.space, seed=self.seed, train_window=s["sampler"].get("train_window", 30.0),
                                 kernel=s["sampler"].get("kernel", "matern52"),
                                 n_train_cap=s["estimate"].get("n_train_cap", 1000))

    def _inference(self, table):
        e = self.settings["errorbars"]
        return inference_gp(table, self.space, self.seed, self.settings["sampler"].get("kernel", "matern52"),
                            self.settings["sampler"].get("train_window", 30.0), e["n_train_cap"],
                            auto_window=e.get("auto_window", False))

    def status(self, results=None, verbose: bool = True) -> dict:
        """Where the campaign stands: the monitoring estimate after every search round (recomputed for the
        current box from the models of that round and the ones before it), whether the stopping rule has
        fired, the out-of-sample test and verdict of every error-bar round, and the next step."""
        rounds = self.rounds()
        st = {"n_rounds": len(rounds), "history": [], "errorbar_rounds": [], "next": None, "complete": True}
        if not rounds:
            st["next"] = "initial"
            return self._report(st, verbose)
        joined = self.results_table(results) if results is not None else None
        if joined is None:
            st["complete"] = False
            st["next"] = "results"
            return self._report(st, verbose)
        st["n_proposed"] = int(len(joined))
        st["n_evaluated"] = int(joined["evaluated"].sum())
        st["n_failed"] = int((joined["evaluated"] & joined["chi2"].isna()).sum())
        st["n_outside_box"] = int((~self.space.contains(joined)).sum())
        pending = [m["round"] for m in rounds if not joined.loc[joined["round"] == m["round"], "evaluated"].all()]
        st["pending_rounds"] = pending
        st["complete"] = not pending
        sub = self.space.subspace(self.converge_on)
        stop = self.settings["stop"]
        hist, prev = [], None
        converged_at = None
        for m in rounds:
            if m["phase"] != SEARCH or m["round"] in pending:
                continue
            est = self._estimate(self._fit_table(joined, m["round"]))
            rec = {"round": m["round"], "n_models": int((joined["round"] <= m["round"]).sum()), "estimate": est}
            if est is not None and prev is not None:
                rec["moves"] = {k: round(float(v), 3) for k, v in estimate_moved(prev, est, sub).items()}
            hist.append(est)
            rec["converged"] = bool(estimate_converged(
                [None if h is None else {p: h[p] for p in self.converge_on} for h in hist], sub,
                frac=stop["frac"], patience=stop["patience"], min_rounds=stop["min_rounds"]))
            if rec["converged"] and converged_at is None:
                converged_at = m["round"]
            prev = est
            st["history"].append(rec)
        st["search_converged"] = converged_at is not None
        st["converged_at_round"] = converged_at
        last = self._fit_table(joined)
        tb = trusted_best_point(last, self.space)
        st["trusted_best"] = None if tb is None else {k: float(tb[k]) for k in self.space.free_params + ["chi2"]}

        e = self.settings["errorbars"]
        eb_done = False
        for m in rounds:
            if m["phase"] != ERRORBAR or m["round"] in pending:
                continue
            t = joined[joined["round"] == m["round"]]
            oos = out_of_sample(t, t["chi2"].values)
            oos.pop("z", None)
            after = self._profile_after(joined, m["round"])
            verdict = profile_verdict(m.get("profile_before"), after, self.converge_on, sd_tol=e["sd_tol"],
                                      frac=e["frac"])
            st["errorbar_rounds"].append({"round": m["round"], "out_of_sample": oos, "verdict": verdict,
                                          "profile": after})
            eb_done = eb_done or verdict["done"]
        n_eb = sum(m["phase"] == ERRORBAR for m in rounds)
        n_search_ref = sum(m["phase"] == SEARCH for m in rounds) - 1
        if pending:
            st["next"] = "results"
        elif not st["search_converged"] and n_eb == 0:
            st["next"] = SEARCH if n_search_ref < stop["max_rounds"] else ERRORBAR
        elif not eb_done and n_eb < e["max_rounds"]:
            st["next"] = ERRORBAR
        else:
            st["next"] = "done"
        st["errorbars_done"] = bool(eb_done)
        return self._report(st, verbose)

    def _profile_after(self, joined, i: int) -> dict:
        """Profiles of the surrogate after error-bar round ``i`` (cached in rounds/round_NNN_after.json, keyed
        by the models it was fitted to, the box and the settings)."""
        table = self._fit_table(joined, i)
        e = self.settings["errorbars"]
        key = json.dumps({"chi2": np.round(table["chi2"].fillna(-1.0).values, 6).tolist(),
                          "x": np.round(self.space.to_unit(table), 9).tolist(),
                          "bounds": self.space.bounds, "sampler": self.settings["sampler"], "errorbars": e},
                         default=_jsonable, sort_keys=True)
        import hashlib
        h = hashlib.sha1(key.encode()).hexdigest()
        cache = self.dir / "rounds" / f"round_{i:03d}_after.json"
        if cache.exists():
            c = json.loads(cache.read_text())
            if c.get("key") == h:
                return c["profile"]
        fit = self._inference(table)
        prof = json.loads(json.dumps(profile_compact(gp_profiles(fit[0], self.space, self.seed,
                                                                  levels=tuple(e["levels"]))), default=_jsonable))
        cache.write_text(json.dumps({"key": h, "profile": prof}))
        return prof

    def _report(self, st, verbose):
        if not verbose:
            return st
        print(f"{self.space}")
        if st.get("n_proposed") is not None:
            print(f"{st['n_rounds']} rounds, {st['n_proposed']} models proposed, {st['n_evaluated']} evaluated "
                  f"({st['n_failed']} failed, {st['n_outside_box']} outside the current box)")
        for h in st["history"]:
            e = h["estimate"]
            if e is None:
                print(f"  round {h['round']}: too few finite models for an estimate")
                continue
            rng = "; ".join(f"{p} {e[p][1]:.4g}-{e[p][2]:.4g}" for p in self.converge_on)
            mv = f"  moves {h['moves']}" if "moves" in h else ""
            print(f"  round {h['round']} ({h['n_models']} models): {rng}; chi2_min {e['chi2_min']:.1f}, T {e['temperature']:.1f}"
                  f"{mv}{'  -> stopping rule met' if h['converged'] else ''}")
        for r in st["errorbar_rounds"]:
            o, v = r["out_of_sample"], r["verdict"]
            if o.get("n"):
                print(f"  error-bar round {r['round']}: out-of-sample rms z {o['rms']:.2f} over {o['n']} models "
                      f"({100 * o['frac_above_2']:.1f}% above +2, {100 * o['frac_below_2']:.1f}% below -2); worst latent sd "
                      f"{v['worst_sd']} (tol {v['sd_tol']}), worst end move {v['worst_move']}{'  -> done' if v['done'] else ''}")
        if st.get("pending_rounds"):
            print(f"  waiting for results of round(s) {st['pending_rounds']}")
        print(f"next: {st['next']}")
        return st

    # ------------------------------------------------------------------ proposing
    def next_round(self, results=None, phase: str | None = None) -> pd.DataFrame | None:
        """Write and return the next round (or None when the campaign is done). ``phase`` forces ``"search"``
        or ``"errorbar"``; by default the status decides."""
        rounds = self.rounds()
        if not rounds:
            return self.initial(self.settings["n_initial"])
        st = self.status(results, verbose=False)
        if st["next"] == "results":
            raise RuntimeError(f"round(s) {st.get('pending_rounds')} have models without results; evaluate them first "
                               "(a failed model counts as evaluated: give it chi2 = nan)")
        what = phase or st["next"]
        if what == "done":
            self._log({"action": "done"})
            return None
        joined = self.results_table(results)
        table = self._fit_table(joined)
        if what == SEARCH:
            d = propose_round(table, self.space, self.settings["n_round"], self.sobol, seed=self.seed,
                              **self.settings["sampler"])
            return self._write_round(d, SEARCH, {"estimate_before": st["history"][-1]["estimate"] if st["history"] else None})
        if what == ERRORBAR:
            e = self.settings["errorbars"]
            n_eb = sum(m["phase"] == ERRORBAR for m in rounds)
            d = errorbar_round(table, self.space, e["n_round"], seed=self.seed,
                               kernel=self.settings["sampler"].get("kernel", "matern52"),
                               train_window=self.settings["sampler"].get("train_window", 30.0),
                               n_train_cap=e["n_train_cap"], n_profile=e["n_profile"],
                               profile_levels=tuple(e["levels"]), auto_window=e.get("auto_window", False),
                               round_index=n_eb)
            return self._write_round(d, ERRORBAR, {})
        raise ValueError(f"unknown phase {what!r}")

    def initial(self, n: int | None = None) -> pd.DataFrame:
        """Write round 0, the first ``n`` (default ``n_initial``) points of the Sobol sequence."""
        if self.rounds():
            raise RuntimeError("round 0 already exists")
        d = initial_design(self.space, n or self.settings["n_initial"], self.sobol)
        return self._write_round(d, SEARCH, {})

    # ------------------------------------------------------------------ results of the campaign
    def surrogate(self, results):
        """The inference surrogate on every evaluated model inside the box: ``(gp, chi2_min, threshold)``."""
        return self._inference(self._fit_table(self.results_table(results)))

    # ------------------------------------------------------------------ in-process loop
    def run(self, evaluate, results_path=None, max_models: int | None = None, verbose: bool = True) -> pd.DataFrame:
        """Run the whole campaign in this process. ``evaluate(design) -> chi2`` takes a DataFrame of parameter
        sets and returns one chi2 per row (NaN for a failure); it may instead return ``(chi2, params)`` with
        the parameters actually evaluated (e.g. an archive replay, :func:`sagari.replay.archive_evaluator`).
        Results accumulate in ``results_path`` (default ``<dir>/results.csv``). Returns the results table."""
        results_path = Path(results_path) if results_path else self.dir / "results.csv"
        res = read_results(results_path) if results_path.exists() else pd.DataFrame(columns=["model_id", "chi2"])
        while True:
            d = self.next_round(res if len(res) else None)
            if d is None:
                break
            out = evaluate(d)
            chi2, params = (out if isinstance(out, tuple) else (out, None))
            new = pd.DataFrame({"model_id": d["model_id"].values})
            for p in self.space.free_params:
                new[p] = (params[p].values if params is not None else d[p].values)
            new["chi2"] = np.asarray(chi2, float)
            res = pd.concat([res, new], ignore_index=True) if len(res) else new
            res.to_csv(results_path, index=False, float_format="%.10g")
            if verbose:
                print(f"round {d.attrs['round']} ({d.attrs['phase']}): {len(d)} models, "
                      f"{int(np.isfinite(new['chi2']).sum())} finite", flush=True)
            if max_models is not None and len(res) >= max_models:
                if verbose:
                    print(f"stopped at max_models={max_models}")
                break
        if verbose:
            self.status(res)
        return res


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
