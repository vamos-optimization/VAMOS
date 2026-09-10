# Results

Inspect the result of an optimization. `X` contains decision variables and `F` contains their objective values; persistence has its own reference.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import OptimizationResult, StudyResult
```

[Quickstart](../../guide/zero_to_hero.md) · [Save and replay a run](runs.md) · [Durable study reports](study-models.md)

Do not infer the number of returned solutions from population size alone. Consult the [algorithm/result-mode contract](../algorithms.md) when choosing between the population, non-dominated results, and an external archive.

::: vamos.OptimizationResult
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.StudyResult
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
