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

**Development after 1.4.0:** `selection` is `("random", {})`, the default, or
`("tournament", {"size": k})`, and any other method or option raises
`ConfigurationError`; NSGA-III used to run a binary tournament on rank and
crowding distance whatever its selection. See
[variants and their effect](../../../algorithms/nsgaiii.md#variants-and-their-effect).

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
