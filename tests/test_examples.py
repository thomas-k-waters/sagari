"""The adapter of examples/sco_runalls.py, on fake cres files."""
import importlib.util
from pathlib import Path

import numpy as np

from sagari import Campaign, ParameterSpace

EX = Path(__file__).resolve().parents[1] / "examples" / "sco_runalls.py"
spec = importlib.util.spec_from_file_location("sco_runalls", EX)
sco = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sco)

SPACE = ParameterSpace.from_dict({"parameters": {
    "upsilon": {"bounds": [4, 12]}, "mbh": {"bounds": [3e7, 5e9], "scale": "log"},
    "v_c": {"bounds": [0, 800]}, "r_c": {"bounds": [0.1, 30], "scale": "log"},
    "xi": {"value": 1.0}, "inclination": {"value": 89.0}}})


def test_commands_and_collect(tmp_path):
    c = Campaign.create(tmp_path / "c", SPACE, n_initial=8)
    d = c.next_round()
    sco.commands(c, 0, 0.625, tmp_path / "round_000.run")
    lines = (tmp_path / "round_000.run").read_text().splitlines()
    assert len(lines) == 8 and lines[0].startswith("runallt ") and len(lines[0].split()) == 8
    assert lines[0].split()[6] == "0.625"                       # apparent_q, 6th positional argument
    res = tmp_path / "res"
    res.mkdir()
    for i, (_, r) in enumerate(d.iloc[:6].iterrows()):           # 6 of 8 have run; one of them failed
        vals = [float(sco.fmt(r[p])) for p in ("upsilon", "mbh", "v_c", "r_c", "inclination", "xi")]
        chi2, alpha = (1000.0 + i, 2.5) if i != 2 else (float("nan"), 0.0)
        (res / f"cres.{sco.libn(r)}").write_text(" ".join(map(str, vals + [chi2, alpha])) + "\n")
    sco.collect(c, res, tmp_path / "results.csv")
    j = c.results_table(tmp_path / "results.csv")
    assert j["evaluated"].sum() == 6 and np.isnan(j["chi2"].iloc[2]) and j["chi2"].iloc[0] == 1000.0
    assert np.isclose(j["upsilon"].iloc[0], float(sco.fmt(d["upsilon"].iloc[0])))   # values as run (6 digits)
