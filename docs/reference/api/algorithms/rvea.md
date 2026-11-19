# RVEA configuration

Configure `rvea` through `vamos.algorithms.RVEAConfig`. This page is the parameter reference, not a tutorial about the algorithm.

**Stable public API.** The [1.x compatibility policy](../../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos.algorithms import RVEAConfig
```

[Configuration overview](index.md) · [Algorithm contracts](../../algorithms.md) · [Optimization](../optimization.md)

**Development after 1.4.0:** `reference_vectors` gives RVEA the reference
vectors of a file, a directory or a two-layer lattice, with the keys of MOEA/D's
`weight_vectors` and NSGA-III's `reference_directions` (see
[reference vectors](../../algorithms.md#reference-and-weight-vectors)), so that
the three algorithms can start from the same vectors, as jMetal and Evolver
configure them:

```python
from vamos import optimize
from vamos.algorithms import RVEAConfig

config = (
    RVEAConfig.builder()
    .pop_size(100)
    .reference_vectors(path="weights/W3D_100.dat")  # a copy of jMetal's file, which VAMOS also ships
    .alpha(2.0)
    .adapt_freq(0.1)
    .crossover("sbx", prob=0.9, eta=30.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .build()
)
result = optimize("dtlz2", algorithm="rvea", algorithm_config=config, max_evaluations=1000, seed=1)
```

RVEA keeps one individual per vector, so the file must hold `pop_size`
vectors; a directory is searched for `W{n_obj}D_{pop_size}.dat`, and a missing
or mismatching file raises before any evaluation. `n_partitions` is not used
with `reference_vectors`; the builder does not need it and records the
`divisions` (12 for a file). `reference_vectors(divisions=12)` runs the same
lattice as `n_partitions(12)`. `to_dict()` leaves the field out while it is
`None`, so a configuration without it keeps the resolved specification, and the
task id, of VAMOS 1.4.0. The resolved spec of a run with a file records it in
`algorithm.resolution.reference_vectors`, for example
`{"source": "file", "file": "W3D_100.dat", "vectors": 100}`. Study planning
rejects a path, as it does for NSGA-III, and checks that `pop_size` equals the
vectors of `divisions` and `inner_divisions`.

::: vamos.algorithms.RVEAConfig
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Builder methods

Use `RVEAConfig.builder()` to obtain the fluent builder; do not import the private builder class directly. The methods below are generated from that returned object, so their signatures are not maintained as a second manual parameter table.

::: vamos.engine.algorithm.config.rvea._RVEAConfigBuilder
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false
