# Adding a quality indicator or metric

Use a quality indicator for a numerical property of an objective front. Execution counters and timing belong to experiment outcomes; analysis-only derived columns belong to UX/study reporting rather than the kernel interface.

## Workflow

1. Put front indicators under `src/vamos/foundation/quality_indicators/`. Implement the `QualityIndicator` protocol when the metric fits the common `compute(front, reference_front, maximise, **kwargs)` contract and return `IndicatorResult`.
2. Validate that fronts are finite two-dimensional arrays and that reference fronts/points match objective dimension. Define minimization/maximization semantics explicitly.
3. Add the canonical name and any essential spelling to `get_indicator(...)`; export intentional classes from the package `__init__.py`. Do not create a parallel registry.
4. Keep optional MooCore use guarded. If a NumPy implementation exists, test it as the reference; otherwise raise the established actionable optional-dependency error.
5. Add a `KernelBackend` hook only for a high-call-count primitive that algorithms need directly. Advertise it through `quality_indicators()` and test capability before dispatch.
6. Update study/UX choices only when the indicator is supported there, and document its reference data, direction, units, and failure behavior.

Do not persist a second metric file beside a canonical run. Bounded scalar metrics that belong to a run go in the manifest outcome; study tables are derived exports.

## Required tests

- Known analytical examples and invalid shapes/reference dimensions.
- Optional-dependency behavior with and without the provider.
- Alias/name resolution through `get_indicator`.
- Backend parity when adding a native hook.
- Study integration only through canonical RunManifest metrics projected by `StudySummary`.

Run:

```bash
python -m pytest -q tests/foundation/test_moocore_indicators.py tests/foundation/test_hypervolume_fallback.py
python -m pytest -q tests/engine/test_tuning_hypervolume_nd.py
```

```agent-docs
path: src/vamos/foundation/quality_indicators
path: src/vamos/foundation/quality_indicators/moocore_indicators.py
path: src/vamos/foundation/kernel/backend.py
path: tests/foundation/test_moocore_indicators.py
path: tests/foundation/test_hypervolume_fallback.py
path: tests/engine/test_tuning_hypervolume_nd.py
symbol: vamos.foundation.quality_indicators.moocore_indicators:QualityIndicator
symbol: vamos.foundation.quality_indicators.moocore_indicators:IndicatorResult
symbol: vamos.foundation.quality_indicators.moocore_indicators:get_indicator
command: python -m pytest -q tests/foundation/test_moocore_indicators.py tests/foundation/test_hypervolume_fallback.py
command: python -m pytest -q tests/engine/test_tuning_hypervolume_nd.py
```
