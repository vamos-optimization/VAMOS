# Study lifecycle

Create or load a durable study, then run, inspect, summarize, resume, or retry it through the `Study` object.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import create_study, load_study, Study
```

[Durable studies guide](../../guide/studies.md) · [Specification and planning](study-specification.md) · [Limits and reports](study-models.md)

Creation performs no optimization; loading is data-only. Study mutation is single-owner and sequential in VAMOS 1.0.0: do not run concurrent `run`, `resume`, or `retry` operations against the same study. There is no cross-process cancellation command.

::: vamos.study_artifacts.create_study
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.study_artifacts.load_study
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.study_artifacts.Study
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
