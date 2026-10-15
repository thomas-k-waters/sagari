# Contributing

Bug reports, questions and pull requests are welcome on GitHub.

## Development setup

```bash
git clone https://github.com/thomas-k-waters/sagari.git
cd sagari
pip install -e ".[dev,plot,docs]"
pytest                      # the full suite, ~30 s
ruff check src tests
```

The numerical code reproduces the paper exactly. A change to `acquisition.py`, `gp.py`, `monitor.py`,
`errorbars.py` or `posterior.py` that alters results must say so in `CHANGELOG.md`. It must also keep
`tests/test_golden.py` passing, or regenerate the fixture deliberately (`tests/fixtures/make_golden.py`, from the
original code) and explain why.

## Documentation

```bash
cd docs && sphinx-build -W -b html . _build/html
```

The example notebooks are stored executed. After changing one, re-execute it with one linear-algebra thread
(`OMP_NUM_THREADS=1 jupyter nbconvert --to notebook --execute --inplace examples/notebooks/NAME.ipynb`).

## Releases

1. Update `src/sagari/_version.py`, `CHANGELOG.md` and `CITATION.cff`.
2. Tag `vX.Y.Z` and publish a GitHub release. The `publish.yml` workflow tests, builds and uploads to PyPI, and
   Zenodo archives the release.
