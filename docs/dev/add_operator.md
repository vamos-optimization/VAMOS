# Adding an operator

Operators are reusable variation or repair components. If the change alters a search lifecycle or survival policy, it belongs in an algorithm instead.

## Workflow

1. Choose the encoding package under `src/vamos/engine/operators/impl/`: `real`, `integer`, `binary`, `permutation`, or `mixed`. Shared repair and registry modules live directly under `impl/`.
2. Copy the protocol and call shape of a neighboring operator of the same kind. Real mutation operators, for example, are constructed with their parameters/bounds and called with `(offspring, rng)`; other encodings have their own current protocols.
3. Vectorize over individuals, accept an explicit `numpy.random.Generator`, validate shapes/dtypes, and document whether mutation is in place. Never obtain randomness from global NumPy state or from an unseeded `np.random.default_rng()`; `vamos.foundation.rng.require_generator` raises `ConfigurationError` when a caller passes no generator.
4. Export the class from the encoding package and register its stable method name in `src/vamos/engine/operators/impl/registry.py`.
5. Update `src/vamos/engine/operators/policies/` only when config-driven construction or encoding/algorithm policy must select the operator. Keep algorithm-specific decisions out of the generic class.
6. If it becomes a supported public config value, update discovery helpers and CLI/config validation together; do not add a second registry.

Shared algorithm-side composition uses `VariationPipeline` from `src/vamos/engine/variation/`. Extend that canonical pipeline only when the operator requires a new cross-algorithm composition contract; keep the concrete operator in `engine/operators`.

Repair is an explicit pipeline policy. An operator that may leave bounds must not claim that its standalone output is feasible; test repair separately from raw mutation/crossover behavior.

## Required tests

- Fixed-seed regression or invariant checks for shape, dtype, input mutation, and parameter validation.
- Bounds/permutation/binary/integer invariants appropriate to the encoding.
- Registry resolution under the chosen stable name.
- Policy construction and the smallest algorithm smoke that exercises the new choice.

Run:

```bash
python -m pytest -q tests/engine/operators tests/engine/test_discrete_and_mixed_operators.py
python -m pytest -q tests/engine/test_operator_combo_validation.py tests/engine/test_algorithm_encodings.py
```

Add the focused operator test during development, then run the higher validation tier from `/AGENTS.md`.

```agent-docs
path: src/vamos/engine/operators/impl
path: src/vamos/engine/operators/impl/registry.py
path: src/vamos/engine/operators/policies
path: src/vamos/engine/variation
path: tests/engine/operators
path: tests/engine/test_discrete_and_mixed_operators.py
path: tests/engine/test_operator_combo_validation.py
path: tests/engine/test_algorithm_encodings.py
symbol: vamos.engine.operators.impl.registry:get_operator_registry
symbol: vamos.engine.variation:VariationPipeline
symbol: vamos.algorithms:available_crossover_methods
symbol: vamos.algorithms:available_mutation_methods
command: python -m pytest -q tests/engine/operators tests/engine/test_discrete_and_mixed_operators.py
command: python -m pytest -q tests/engine/test_operator_combo_validation.py tests/engine/test_algorithm_encodings.py
```
