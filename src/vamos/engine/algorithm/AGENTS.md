# Scope

Applies only to `src/vamos/engine/algorithm/**`.

Inherits all repository-wide rules from `/AGENTS.md`. This file contains local deltas only.

## Responsibility and invariants

- Each built-in algorithm owns its search lifecycle under its package; shared construction helpers stay in `_builder_*.py` and `builders.py`.
- Typed algorithm configuration lives under `config/` and must serialize into the resolved experiment specification without hidden state.
- `registry.py` owns built-in names and builder resolution. A builder accepts `(config_mapping, kernel)` and returns an object satisfying `AlgorithmLike`.
- Shared crossover/mutation orchestration lives in `src/vamos/engine/variation/`; algorithm packages import that canonical pipeline directly.
- External archive configuration is `ExternalArchiveConfig` in `src/vamos/engine/archive/config.py`. Experiment specs use only `archive.external`, parsed by `build_archive_cfg` in `src/vamos/engine/hooks/config_parse.py`.
- Run loops must honor exact evaluation budgets, explicit RNG ownership, encoding support, observer/hooks, and warm-start contracts.
- Reference-direction algorithms must validate population/reference-direction cardinality. Algorithms use `KernelBackend` methods rather than backend-specific imports. MOEA/D's neighbourhood update is the documented exception: it runs compiled whenever Numba is installed, whatever the engine ([Automatic engine selection](/docs/reference/algorithms.md#automatic-engine-selection)).

## Extension touchpoints

Follow [Adding an algorithm](/docs/dev/add_algorithm.md). Update configuration, builder/registry, public facade, CLI/spec plumbing, tests, and docs only where the new algorithm actually requires them.

## Targeted validation

Run `python -m pytest -q tests/engine/test_algorithm_registry.py tests/engine/test_algorithms_smoke.py tests/engine/test_algorithm_encodings.py tests/engine/test_algorithm_problem_matrix.py` plus focused algorithm tests.

```agent-docs
path: src/vamos/engine/algorithm/config
path: src/vamos/engine/algorithm/registry.py
path: src/vamos/engine/algorithm/builders.py
path: src/vamos/engine/variation
path: src/vamos/engine/archive/config.py
path: src/vamos/engine/hooks/config_parse.py
path: tests/engine/test_algorithm_registry.py
path: tests/engine/test_algorithms_smoke.py
path: tests/engine/test_algorithm_encodings.py
path: tests/engine/test_algorithm_problem_matrix.py
path: docs/dev/add_algorithm.md
symbol: vamos.engine.algorithm.registry:AlgorithmLike
symbol: vamos.engine.algorithm.registry:resolve_algorithm
symbol: vamos.engine.variation:VariationPipeline
symbol: vamos.engine.archive.config:ExternalArchiveConfig
symbol: vamos.engine.hooks.config_parse:build_archive_cfg
command: python -m pytest -q tests/engine/test_algorithm_registry.py tests/engine/test_algorithms_smoke.py tests/engine/test_algorithm_encodings.py tests/engine/test_algorithm_problem_matrix.py
```
