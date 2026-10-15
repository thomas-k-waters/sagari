# SAGARI

**Schwarzschild Adaptive Gaussian-process Acquisition, Refinement, and Inference**

Adaptive Gaussian-process sampling for Schwarzschild orbit-superposition modeling, and for any other model
whose goodness of fit is an expensive, noisy χ².

An orbit-superposition model costs hours of CPU time, and a nested parameter grid needs $\sim10^4$ of them.
SAGARI replaces the grid with a sequence of rounds, each placed where a surrogate of the χ² surface
predicts the largest gain, and stops when the answer stops moving. On the archived models of two published
black-hole mass measurements it recovered the published mass in every replayed trial with **9 to 30 times fewer
models**. It is the method of Waters & Gültekin (2026) and reproduces the code used there exactly.

The package never runs a model. It proposes parameter sets and reads back their χ², so it can drive any code:
a Fortran pipeline on a cluster, a Jeans model, or a lookup in an archive.

```{tip}
New here? Start with the [quick start](quickstart.md), then the
[first notebook](notebooks/01_quickstart.ipynb). On a cluster, read [Campaigns on a cluster](cluster.md).
```

```{toctree}
:maxdepth: 2
:caption: Using sagari

installation
quickstart
concepts
cluster
configuration
analysis
```

```{toctree}
:maxdepth: 1
:caption: Example notebooks

notebooks/01_quickstart
notebooks/02_cluster_workflow
notebooks/03_analysis
notebooks/04_benchmark_and_replay
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
reproducibility
faq
citing
changelog
```
