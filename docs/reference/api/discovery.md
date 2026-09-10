# Problem and operator discovery

Find supported identifiers through the public facades instead of importing internal registries. The catalogues explain the corresponding capabilities and dependencies.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import available_problem_names, make_problem_selection
from vamos.algorithms import (
    available_algorithms,
    available_crossover_methods,
    available_mutation_methods,
)
```

[Problems](../problems.md) · [Algorithms and backends](../algorithms.md) · [Configuration classes](algorithms/index.md)

::: vamos.available_problem_names
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.make_problem_selection
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.algorithms.available_algorithms
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.algorithms.available_crossover_methods
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.algorithms.available_mutation_methods
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
