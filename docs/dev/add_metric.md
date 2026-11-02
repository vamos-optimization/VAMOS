# Adding a quality indicator or metric

Use a quality indicator for a numerical property of an objective front. Execution counters and timing belong to experiment outcomes; analysis-only derived columns belong to UX/study reporting rather than the kernel interface.

## Workflow

1. Put front indicators under `src/vamos/foundation/quality_indicators/`. Implement the `QualityIndicator` protocol when the metric fits the common `compute(front, reference_front, maximise, **kwargs)` contract and return `IndicatorResult`.
2. Validate that fronts are finite two-dimensional arrays and that reference fronts/points match objective dimension. Define minimization/maximization semantics explicitly.
3. Add the canonical name and any essential spelling to the alias table behind `get_indicator(...)` and `canonical_indicator_name(...)`; export intentional classes from the package `__init__.py`. Do not create a parallel registry.
4. Keep optional MooCore use guarded. If a NumPy implementation exists, test it as the reference; otherwise raise the established actionable optional-dependency error.
5. Add a `KernelBackend` hook only for a high-call-count primitive that algorithms need directly. Advertise it through `quality_indicators()` and test capability before dispatch.
6. Update study/UX choices only when the indicator is supported there, and document its reference data, direction, units, and failure behavior.

## Indicator contract

**Development after 1.1.0:** every indicator path validates its inputs the same way.

- **Finite inputs:** a NaN or infinite value in a front, a reference front or a reference point raises `ValueError`. MooCore used to drop such rows silently.
- **Reference point:** `hypervolume()`, `hypervolume_contributions()`, `HVIndicator`, `HVContributionsIndicator`, `HVApproxIndicator` and `WHVRectIndicator` require an explicit reference point. VAMOS 1.1.0 used `max(front) + 1` of the measured front, so adding a dominated point changed the hypervolume.
- **Points outside the reference box:** every path ignores them, as MooCore does. `hypervolume()` no longer raises for them, and its `allow_ref_expand` argument is ignored with a `FutureWarning`.
- **Contributions:** they follow MooCore's `ignore_dominated` semantics in every environment. A dominated, duplicated or outside point contributes 0. Without MooCore, the fallback for three or more objectives used to keep dominated points in, and could return small negative values.
- **`pareto_filter`:** it rejects rows with NaN and inputs that are not two-dimensional.
- **Study and benchmark indicators:** they accept every alias of `get_indicator`. The hypervolume family receives the problem's reference point, and the other indicators receive its reference front.

Definitions to keep in mind when comparing with other tools:

- **IGD:** `igd` is the mean distance from each reference point to the front (p = 1, Coello and Cortés, as in pymoo). jMetal's default IGD is `sqrt(sum d^2) / n`, which gives smaller values.
- **Multiplicative epsilon:** `epsilon_mult` needs strictly positive values in the front and in the reference front. Shift both explicitly when objectives can be 0, as on ZDT and DTLZ fronts.
- **Exact fallback cost:** without MooCore, the exact hypervolume for four or more objectives grows exponentially with the number of objectives. SMS-EMOA and IBEA with the hypervolume indicator need the `[compute]` extra in practice from M = 4.

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
