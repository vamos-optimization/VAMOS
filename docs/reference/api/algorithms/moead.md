# MOEA/D configuration

Configure `moead` through `vamos.algorithms.MOEADConfig`. This page is the parameter reference, not a tutorial about the algorithm.

**Stable public API.** The [1.x compatibility policy](../../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos.algorithms import MOEADConfig
```

[Configuration overview](index.md) · [Algorithm contracts](../../algorithms.md) · [Optimization](../optimization.md)

**Development after 1.2.1:** the resolved spec of a MOEA/D run records where its
weight vectors come from, in `algorithm.resolution.weight_vectors`, beside the
population in `population.initial_size`: `{"source": "das_dennis", "divisions": 12, "vectors": 91}`,
`{"source": "packaged_file", "file": "W3D_100.dat", "vectors": 100}`, `{"source": "file", "file": ...}`
for a file of `weight_vectors(path=...)`, or `{"source": "uniform", ...}` with two objectives.
A run whose `weight_vectors(path=...)` directory lacks the file generates the
lattice and saves it there; a file that holds exactly that lattice is recorded as
`das_dennis`, so every seed of a study records the same source whether or not an
earlier run wrote the file.
A complete lattice comes before a packaged file of the same size, so 91 and 120
individuals with three objectives use the lattice although VAMOS ships
`W3D_91.dat` and `W3D_120.dat`. Once normalized, `W3D_120.dat` differs from the
lattice by about `1e-16`, so a tool that reads the file can drift from VAMOS with
the same seed. `vamos tune` records the population a trial ran
when it differs from the sampled one (see [Tuning](../../../topics/tuning.md#populations-that-depend-on-the-number-of-objectives)).

::: vamos.algorithms.MOEADConfig
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Builder methods

Use `MOEADConfig.builder()` to obtain the fluent builder; do not import the private builder class directly. The methods below are generated from that returned object, so their signatures are not maintained as a second manual parameter table.

::: vamos.engine.algorithm.config.moead._MOEADConfigBuilder
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false
