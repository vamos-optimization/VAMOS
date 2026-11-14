# CCMO configuration

Configure `ccmo` through `vamos.algorithms.CCMOConfig`. This page is the parameter reference, not a tutorial about the algorithm; see [CCMO](../../../algorithms/ccmo.md) for the algorithm, its variations and its budget rule.

**Development after 1.1.0.** CCMO and `CCMOConfig` are new in the development series after VAMOS 1.1.0; the [1.0 compatibility baseline](../../../project/stability-and-versioning.md) does not cover them yet.

```python
from vamos.algorithms import CCMOConfig
```

[Configuration overview](index.md) · [Algorithm contracts](../../algorithms.md) · [Optimization](../optimization.md)

::: vamos.algorithms.CCMOConfig
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Builder methods

Use `CCMOConfig.builder()` to obtain the fluent builder; do not import the private builder class directly. The methods below are generated from that returned object, so their signatures are not maintained as a second manual parameter table.

::: vamos.engine.algorithm.config.ccmo._CCMOConfigBuilder
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false
