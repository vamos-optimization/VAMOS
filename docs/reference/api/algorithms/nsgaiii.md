# NSGA-III configuration

Configure `nsgaiii` through `vamos.algorithms.NSGAIIIConfig`. This page is the parameter reference, not a tutorial about the algorithm.

**Stable public API.** The [1.x compatibility policy](../../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos.algorithms import NSGAIIIConfig
```

[Configuration overview](index.md) · [Algorithm contracts](../../algorithms.md) · [Optimization](../optimization.md)

**VAMOS 1.3.0:** `pop_size_auto` and `enforce_ref_dirs` have no
effect: NSGA-III runs the largest reference-direction lattice that fits in
`pop_size` and keeps at least one individual per direction. Setting them with
the builder, or `enforce_ref_dirs=False` in the constructor or `from_dict()`,
emits a `FutureWarning`; both fields are planned for removal in 2.0 (see
[stable configuration](../../../project/stability-and-versioning.md#stable-configuration)).
`NSGAIIIConfig.default()` does not warn. `mutation_prob_factor` now sets the
mutation probability to `factor / n_var`, as in NSGA-II.

**Development after 1.4.0:** the file of `reference_directions(path=...)` is
read as MOEA/D's weight files: one direction per line, the components separated
by whitespace or by commas (VAMOS 1.4.0 read commas only, so jMetal's
`W3D_100.dat` failed), every row non-negative and summing to one within the
rounding of jMetal's files (`1e-5`). A path that is not an existing file raises
`ConfigurationError` when the run starts; VAMOS 1.4.0 generated the lattice
instead and saved it at that path.

**Development after 1.4.0:** `reference_directions` takes the keys of MOEA/D's
`weight_vectors` and RVEA's `reference_vectors` (see
[reference and weight vectors](../../algorithms.md#reference-and-weight-vectors)):
`path` may also name a directory, from which NSGA-III reads jMetal's
`W{n_obj}D_{pop_size}.dat` (a directory without it raises), and
`reference_directions(divisions=3, inner_divisions=2)` runs Deb and Jain's
two-layer lattice, 156 directions with eight objectives. The resolved spec of a
run with a path records the file in `algorithm.resolution.reference_directions`,
`{"source": "file", "file": "W3D_100.dat"}`; a run without one keeps the
resolved spec of VAMOS 1.4.0. Study planning still rejects a path, and checks
that `pop_size` covers a two-layer lattice.

::: vamos.algorithms.NSGAIIIConfig
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Builder methods

Use `NSGAIIIConfig.builder()` to obtain the fluent builder; do not import the private builder class directly. The methods below are generated from that returned object, so their signatures are not maintained as a second manual parameter table.

::: vamos.engine.algorithm.config.nsgaiii._NSGAIIIConfigBuilder
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false
