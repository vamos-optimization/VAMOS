# Adding an algorithm

Add an algorithm when VAMOS needs a new search/update lifecycle. A new crossover, mutation, repair, or numerical primitive belongs at its narrower extension boundary.

## Built-in workflow

1. Study `AlgorithmLike` in `src/vamos/engine/algorithm/registry.py` and the closest built-in package. The run method accepts a problem, termination, seed, evaluation strategy, and optional live visualization, and returns the current result mapping.
2. Implement the lifecycle under `src/vamos/engine/algorithm/<name>/`. Reuse shared builders/components without importing orchestration or CLI modules into `engine`.
3. Add a typed config under `src/vamos/engine/algorithm/config/` when the algorithm has a supported declarative configuration. Its serialized form must contain every choice needed by execution and exact replay.
4. Add a builder with the registry signature `(config_mapping, kernel) -> AlgorithmLike`, then register the canonical lower-case name in `_register_algorithms(...)` and `get_builtin_algorithm_names()`.
5. Wire current construction in `src/vamos/engine/algorithm/builders.py` and related focused builder modules. Validate encoding, reference directions, population size, operator choices, and backend capability before entering the run loop.
6. Export a stable config through `vamos.algorithms` only if public callers need it. Update CLI/spec parsing, artifact reconstruction, and public API snapshots whenever the public/resolved configuration surface changes.

The implementation must honor exact evaluation budgets and explicit seed/RNG ownership. It must not switch backends silently. Reference-direction algorithms must define and validate how direction count constrains population size.

Use `VariationPipeline` from `src/vamos/engine/variation/` for shared operator orchestration. If the algorithm exposes an external result archive, use `ExternalArchiveConfig` from `src/vamos/engine/archive/config.py`; experiment specifications contain only `archive.external`, parsed by `build_archive_cfg` in `src/vamos/engine/hooks/config_parse.py`.

## Package plugin workflow

External packages may expose a builder using the `vamos.algorithms` entry-point group. Discovery is lazy through `vamos.algorithms.available_algorithms()`, `resolve_algorithm(...)`, or explicit plugin discovery. A plugin is not automatically eligible for exact replay.

```toml
[project.entry-points."vamos.algorithms"]
my_algorithm = "my_package.vamos_plugin:build_my_algorithm"
```

**Development after 1.1.0:** a plugin whose name is that of a built-in algorithm
(`nsgaii`, `moead`, and the others, in any case) is ignored with a
`RuntimeWarning`; VAMOS 1.1.0 ignored it silently. Setting the environment
variable `VAMOS_DISABLE_PLUGINS=1` stops the first use of the algorithm registry
(`optimize`, `available_algorithms()`, `resolve_algorithm(...)`) from loading
installed plugins, for example to rule a plugin out of a run;
`discover_algorithm_plugins()` still loads them when called. The VAMOS 1.0
compatibility snapshot lists only the built-in algorithms, whatever plugins are
installed.

## Required tests

- Registry name and builder resolution.
- Tiny exact-budget smoke with deterministic seed.
- Supported and rejected encodings.
- Config serialization/defaults and CLI/spec parsing when exposed.
- Backend parity/capability, warm start, hooks/observers, and reference directions where applicable.
- Replay reconstruction matrix when adding a supported built-in replay component.

Run:

```bash
python -m pytest -q tests/engine/test_algorithm_registry.py tests/engine/test_algorithms_smoke.py
python -m pytest -q tests/engine/test_algorithm_encodings.py tests/engine/test_algorithm_problem_matrix.py tests/engine/test_warm_start_algorithms.py
```

Add the focused algorithm tests and apply the full tier from `/AGENTS.md` before completion.

```agent-docs
path: src/vamos/engine/algorithm/registry.py
path: src/vamos/engine/algorithm/builders.py
path: src/vamos/engine/algorithm/config
path: src/vamos/engine/variation
path: src/vamos/engine/archive/config.py
path: src/vamos/engine/hooks/config_parse.py
path: src/vamos/experiment/artifacts/reconstruction.py
path: src/vamos/algorithms.py
path: tests/engine/test_algorithm_registry.py
path: tests/engine/test_algorithms_smoke.py
path: tests/engine/test_algorithm_encodings.py
path: tests/engine/test_algorithm_problem_matrix.py
path: tests/engine/test_warm_start_algorithms.py
symbol: vamos.algorithms:available_algorithms
symbol: vamos.algorithms:resolve_algorithm
symbol: vamos.engine.algorithm.registry:AlgorithmLike
symbol: vamos.engine.algorithm.registry:AlgorithmBuilder
symbol: vamos.engine.variation:VariationPipeline
symbol: vamos.engine.archive.config:ExternalArchiveConfig
symbol: vamos.engine.hooks.config_parse:build_archive_cfg
command: python -m pytest -q tests/engine/test_algorithm_registry.py tests/engine/test_algorithms_smoke.py
command: python -m pytest -q tests/engine/test_algorithm_encodings.py tests/engine/test_algorithm_problem_matrix.py tests/engine/test_warm_start_algorithms.py
```
