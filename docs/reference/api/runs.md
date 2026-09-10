# Save, load, and replay runs

Persist an individual optimization, inspect its stored results, check its integrity, or explicitly reproduce it in a compatible environment.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import save_result, load_run, load_result, verify_run, reproduce
```

[Run artifacts guide](../../guide/run-artifacts.md) · [Stored run and report models](run-models.md) · [Results](results.md)

Loading and verification are data-only operations. `reproduce` is a separate executable operation: it creates a new run and does not overwrite the source. Exact replay is limited to reconstructable built-ins in a materially matching environment and backend.

::: vamos.run_artifacts.save_result
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.run_artifacts.load_run
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.run_artifacts.load_result
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.run_artifacts.verify_run
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.run_artifacts.reproduce
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
