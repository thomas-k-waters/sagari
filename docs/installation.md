# Installation

```bash
pip install sagari
```

`sagari` needs Python 3.10 or later. NumPy, SciPy, pandas, scikit-learn, emcee and PyYAML are installed with
it. The example notebooks also use matplotlib (`pip install "sagari[plot]"`).

## From source

```bash
git clone https://github.com/thomas-k-waters/sagari.git
cd sagari
pip install -e ".[dev,plot]"
pytest
```

## On a shared cluster

Install into your own environment; nothing needs root access.

```bash
python3 -m venv ~/venvs/sagari
source ~/venvs/sagari/bin/activate
pip install sagari
sagari --version
```

The campaign itself is light (a round's decisions take seconds), so it can run on a login node or inside a small
batch job that submits the model jobs. The models are your code's business.
