import json

import numpy as np
import pandas as pd
import pytest
import yaml
from helpers import SPACE, TRUE, bowl

from sagari import Campaign, read_results
from sagari.cli import main

SMALL = dict(n_initial=48, n_round=24, errorbars={"n_round": 24, "n_profile": 3, "max_rounds": 1})


def _evaluate_round(c, results, i, fail=()):
    d = c.round_table(i)
    new = pd.DataFrame({"model_id": d["model_id"], "chi2": bowl(d, i)})
    new.loc[new.index[list(fail)], "chi2"] = np.nan
    return pd.concat([results, new], ignore_index=True) if results is not None else new


def test_file_workflow(tmp_path):
    c = Campaign.create(tmp_path / "c", SPACE, seed=3, **SMALL)
    with pytest.raises(FileExistsError):
        Campaign.create(tmp_path / "c", SPACE)
    d0 = c.next_round()
    assert len(d0) == 48 and d0["model_id"].iloc[0] == "r000_0000"
    with pytest.raises(RuntimeError):
        c.next_round(pd.DataFrame({"model_id": ["r000_0000"], "chi2": [1.0]}))   # round incomplete
    res = _evaluate_round(c, None, 0, fail=[5])
    d1 = Campaign(tmp_path / "c").next_round(res)          # a fresh process continues the campaign
    assert d1.attrs["phase"] == "search" and len(d1) == 24
    res = _evaluate_round(c, res, 1)
    st = c.status(res, verbose=False)
    assert st["n_failed"] == 1 and st["next"] == "search" and len(st["history"]) == 2
    res.to_csv(tmp_path / "results.csv", index=False)
    assert len(read_results(tmp_path / "results.csv")) == 72


def test_results_matched_by_parameters(tmp_path):
    c = Campaign.create(tmp_path / "c", SPACE, seed=3, **SMALL)
    d0 = c.next_round()
    res = d0[["a", "b"]].copy()
    res["chi2"] = bowl(d0)
    res = res.sample(frac=1.0, random_state=1)            # any order
    j = c.results_table(res)
    assert j["evaluated"].all()
    assert np.allclose(j["chi2"].values, bowl(d0))


def test_box_narrowed_mid_campaign(tmp_path):
    c = Campaign.create(tmp_path / "c", SPACE, seed=3, **SMALL)
    c.next_round()
    res = _evaluate_round(c, None, 0)
    doc = yaml.safe_load((tmp_path / "c" / "campaign.yaml").read_text())
    doc["space"]["parameters"]["a"]["bounds"] = [0.1, 1.0]
    (tmp_path / "c" / "campaign.yaml").write_text(yaml.safe_dump(doc))
    c2 = Campaign(tmp_path / "c")
    st = c2.status(res, verbose=False)
    assert st["n_outside_box"] > 0
    d1 = c2.next_round(res)
    assert (d1["a"] >= 0.1 - 1e-9).all()


@pytest.mark.slow
def test_in_process_run_to_completion(tmp_path):
    c = Campaign.create(tmp_path / "c", SPACE, seed=3, converge_on=["a", "b"], **SMALL)
    res = c.run(lambda d: bowl(d, len(d)), verbose=False)
    st = c.status(res, verbose=False)
    assert st["search_converged"] and st["next"] == "done" and len(st["errorbar_rounds"]) == 1
    e = st["history"][-1]["estimate"]
    for p in ("a", "b"):
        assert e[p][1] <= TRUE[p] <= e[p][2]
    gp, _, _ = c.surrogate(res)
    assert gp.scatter > 0


def test_cli(tmp_path, capsys):
    (tmp_path / "space.yaml").write_text(yaml.safe_dump(SPACE.to_dict()))
    assert main(["init", str(tmp_path / "c"), "--space", str(tmp_path / "space.yaml"), "--n-initial", "32",
                 "--n-round", "16", "--seed", "4", "--converge-on", "a"]) == 0
    assert main(["next", str(tmp_path / "c")]) == 0
    c = Campaign(tmp_path / "c")
    res = _evaluate_round(c, None, 0)
    res.to_csv(tmp_path / "results.csv", index=False)
    assert main(["next", str(tmp_path / "c"), "--results", str(tmp_path / "results.csv")]) == 0
    assert main(["status", str(tmp_path / "c"), "--results", str(tmp_path / "results.csv"),
                 "--json", str(tmp_path / "st.json")]) == 0
    assert json.loads((tmp_path / "st.json").read_text())["next"] == "results"
    out = capsys.readouterr().out
    assert "round_001.csv" in out and "waiting for results" in out


def test_read_results_formats(tmp_path):
    (tmp_path / "r.dat").write_text("model_id chi2\nr000_0000 12.5\nr000_0001 nan\nr000_0002 inf\n")
    r = read_results(tmp_path / "r.dat")
    assert r["chi2"].iloc[0] == 12.5 and r["chi2"].iloc[1:].isna().all()
    (tmp_path / "r.csv").write_text("model_id,chi2,failed\nr000_0000,12.5,0\nr000_0001,13.0,true\nr000_0002,,0\n")
    r = read_results(tmp_path / "r.csv")
    assert r["chi2"].iloc[0] == 12.5 and r["chi2"].iloc[1:].isna().all()


def test_errorbar_round_status_and_verdict_cache(tmp_path):
    c = Campaign.create(tmp_path / "c", SPACE, seed=3, **SMALL)
    c.next_round()
    res = _evaluate_round(c, None, 0)
    d = c.next_round(res, phase="errorbar")                  # forced: an error-bar round right away
    assert d.attrs["phase"] == "errorbar" and {"pred_mu", "pred_sd", "kind"} <= set(d.columns)
    meta = c.rounds()[1]
    assert "profile_before" in meta
    res = _evaluate_round(c, res, 1)
    st = c.status(res, verbose=False)
    r = st["errorbar_rounds"][0]
    assert r["out_of_sample"]["n"] == 24 and "worst_sd" in r["verdict"]
    assert (tmp_path / "c" / "rounds" / "round_001_after.json").exists()
    st2 = c.status(res, verbose=False)                         # second call reads the cache
    assert st2["errorbar_rounds"][0]["verdict"] == r["verdict"]


@pytest.mark.slow
def test_cli_posterior(tmp_path, capsys):
    c = Campaign.create(tmp_path / "c", SPACE, seed=3, **SMALL)
    res = c.run(lambda d: bowl(d, len(d)), verbose=False)
    res.to_csv(tmp_path / "results.csv", index=False)
    assert main(["posterior", str(tmp_path / "c"), "--results", str(tmp_path / "results.csv"), "--out",
                 str(tmp_path / "post"), "--nsteps", "1500", "--burn", "300", "--profiles"]) == 0
    s = json.loads((tmp_path / "post" / "summary.json").read_text())
    assert set(s["plug_in"]) == {"a", "b"} and set(s["marginalized"]) == {"a", "b"} and "profile_minimum" in s
    assert (tmp_path / "post" / "posterior_samples.csv").exists()
