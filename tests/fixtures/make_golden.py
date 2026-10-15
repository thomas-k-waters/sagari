"""Write golden.json from the ORIGINAL campaign code of the paper (not from this package):

    SAGARI_REFERENCE=/path/to/campaign python tests/fixtures/make_golden.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(os.environ["SAGARI_REFERENCE"]) / "sampling")]
from design import GalaxyConfig, generate_global_design  # noqa: E402
from estimate import smoothed_estimate  # noqa: E402
from gp_ei import gp_ei_round  # noqa: E402
from posterior_fill import posterior_fill_round  # noqa: E402

from sagari.benchmark import TwoBasin  # noqa: E402  (the objective only)

YAML = """galaxy: two_basin
apparent_q: 0.625
parameters:
  upsilon: {fixed: false, bounds: [1.0, 15.0]}
  xi: {fixed: true, value: 1.0}
  mbh: {fixed: false, bounds: [3.0e+7, 5.0e+9], scale: log}
  inclination: {fixed: true, value: 89.0}
  v_c: {fixed: false, bounds: [0.0, 500.0]}
  r_c: {fixed: false, bounds: [0.05, 20.0], scale: log}
"""
SEED, N0, NR = 2024, 128, 64
PRODUCTION = dict(kernel="matern52", train_window=30.0, global_fraction=0.35)

with tempfile.TemporaryDirectory() as tmp:
    y = Path(tmp) / "g.yaml"; y.write_text(YAML)
    cfg = GalaxyConfig.from_yaml(y)
    free = cfg.free_params
    d0 = generate_global_design(y, N0, tmp, seed=SEED)
    t = d0[free].copy(); t["chi2"] = TwoBasin(hopeless=True)(d0, np.random.default_rng(1))
    r1 = gp_ei_round(t, y, NR, tmp, seed=SEED, **PRODUCTION)
    est = smoothed_estimate(t, cfg, seed=SEED, kernel="matern52", train_window=30.0)
    fill = posterior_fill_round(t, y, NR, seed=SEED, kernel="matern52", train_window=30.0, n_train_cap=1000, n_profile=7)
out = dict(seed=SEED, n0=N0, n_round=NR, production=PRODUCTION, design0=d0[free].values.tolist(),
           table=[[*map(float, r[free]), (None if not np.isfinite(r["chi2"]) else float(r["chi2"]))] for _, r in t.iterrows()],
           round1=r1[free].values.tolist(), round1_stage=list(r1["stage"]), round1_noise=r1.attrs["gp_noise_sd"],
           estimate=est, fill=fill[free].values.tolist(), fill_pred_mu=fill["pred_mu"].tolist())
(Path(__file__).parent / "golden.json").write_text(json.dumps(out))
print("wrote golden.json")
