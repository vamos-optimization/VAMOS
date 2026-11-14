# C-TAEA configuration

Configure `ctaea` through `vamos.algorithms.CTAEAConfig`. This page is the parameter reference, not a tutorial about the algorithm; see [C-TAEA](../../../algorithms/ctaea.md) for the algorithm, its weight vectors and its budget rule.

**Development after 1.1.0.** C-TAEA and `CTAEAConfig` are new in the development series after VAMOS 1.1.0; the [1.0 compatibility baseline](../../../project/stability-and-versioning.md) does not cover them yet.

```python
from vamos.algorithms import CTAEAConfig
```

[Configuration overview](index.md) · [Algorithm contracts](../../algorithms.md) · [Optimization](../optimization.md)

::: vamos.algorithms.CTAEAConfig
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Builder methods

Use `CTAEAConfig.builder()` to obtain the fluent builder; do not import the private builder class directly. The methods below are generated from that returned object, so their signatures are not maintained as a second manual parameter table.

::: vamos.engine.algorithm.config.ctaea._CTAEAConfigBuilder
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false
