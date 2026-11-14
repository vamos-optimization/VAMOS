# CMOEA-MS configuration

Configure `cmoeams` through `vamos.algorithms.CMOEAMSConfig`. This page is the parameter reference, not a tutorial about the algorithm; see [CMOEA-MS](../../../algorithms/cmoeams.md) for the algorithm, its two stages, its variations and its budget rule.

**Development after 1.1.0.** CMOEA-MS and `CMOEAMSConfig` are new in the development series after VAMOS 1.1.0; the [1.0 compatibility baseline](../../../project/stability-and-versioning.md) does not cover them yet.

```python
from vamos.algorithms import CMOEAMSConfig
```

[Configuration overview](index.md) · [Algorithm contracts](../../algorithms.md) · [Optimization](../optimization.md)

::: vamos.algorithms.CMOEAMSConfig
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Builder methods

Use `CMOEAMSConfig.builder()` to obtain the fluent builder; do not import the private builder class directly. The methods below are generated from that returned object, so their signatures are not maintained as a second manual parameter table.

::: vamos.engine.algorithm.config.cmoeams._CMOEAMSConfigBuilder
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false
