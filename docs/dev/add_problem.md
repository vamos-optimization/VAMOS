# Adding a problem

Use `vamos.make_problem(...)` for user-local objectives. Add a built-in only when the problem belongs in VAMOS's named benchmark or real-world catalog.

## User-local problem

`make_problem` adapts a scalar callable by default. Set `vectorized=True` only when the callable accepts an `(n_points, n_var)` array and returns an `(n_points, n_obj)` array.

```python
import vamos


def objectives(x):
    return [x[0], (1.0 + x[1]) * (1.0 - x[0] ** 0.5)]


problem = vamos.make_problem(
    objectives,
    n_var=2,
    n_obj=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    encoding="real",
)
result = vamos.optimize(problem, algorithm="nsgaii", max_evaluations=200, seed=42)
```

The CLI scaffold is discoverable with `vamos create-problem --help`.

## Built-in problem workflow

1. Read `ProblemProtocol` in `src/vamos/foundation/problem/types.py` and a neighboring implementation with the same encoding.
2. Implement `n_var`, `n_obj`, `n_constraints`, `xl`, `xu`, `encoding`, and `evaluate(X, out)`. Evaluation is batched, writes `out["F"]`, and writes `out["G"]` for constraints with `g <= 0` feasible.
3. Add a `ProblemSpec` to the appropriate `src/vamos/foundation/problem/registry/families/*.py` module. The family exposes `get_specs()`; `registry/specs.py` assembles those maps and is not the per-problem registration file. The factory takes `n_var` and `n_obj` positionally; the options a user may set by name (`problem_kwargs`) are its keyword-only parameters, with the class's defaults, and their values must be JSON scalars so that replay rebuilds the problem.
4. Add packaged reference data under `src/vamos/resources/` only when the problem has an authoritative dataset/front, and verify package-data coverage.
5. Export the class from `vamos.problems` only when direct construction is part of the intentional public API. Named access through `vamos.optimize("key", ...)` does not require a class export.
6. Document dimensions, objective direction, constraints, encoding, and source. Never silently choose dimensions that the `ProblemSpec` marks fixed.
7. **VAMOS 1.5.1:** when an optional package evaluates the problem, as `modact` does the MODAct problems, import it inside the constructor and raise `ImportError` with the install command when it is missing; the registry-wide tests then skip the problem, and `_OPTIONAL_DEPENDENCIES` in `tests/foundation/test_problem_registry.py` names its package. Copy none of the package's code, and test the conversion to VAMOS's conventions with a stub of the package as well as against the package itself.

Use the canonical encoding names `real`, `integer`, `binary`, `permutation`, and `mixed`. For mixed problems, provide the current mixed specification expected by `MixedProblemProtocol` consumers.

## Required tests

- Direct evaluation: bounds, batch shape, finite values, and constraint shape/sign.
- Registry: key discovery, dimension resolution, factory instantiation, and duplicate-free assembly.
- Algorithm smoke: one supported algorithm/encoding at a tiny exact budget.
- Reference data/package test when adding resources.

Run:

```bash
python -m pytest -q tests/foundation/test_problem_registry.py tests/foundation/test_problem_zoo.py tests/foundation/test_problem_evaluation_edge_cases.py
python -m pytest -q tests/engine/test_algorithm_problem_matrix.py
```

Add the focused new-problem test to these commands during development, then run the repository validation tier required by `/AGENTS.md`.

```agent-docs
path: src/vamos/foundation/problem/types.py
path: src/vamos/foundation/problem/registry/common.py
path: src/vamos/foundation/problem/registry/families
path: src/vamos/foundation/problem/registry/specs.py
path: src/vamos/resources
path: src/vamos/problems.py
path: tests/foundation/test_problem_registry.py
path: tests/foundation/test_problem_zoo.py
path: tests/foundation/test_problem_evaluation_edge_cases.py
path: tests/engine/test_algorithm_problem_matrix.py
symbol: vamos:make_problem
symbol: vamos:optimize
symbol: vamos.foundation.problem.types:ProblemProtocol
symbol: vamos.foundation.problem.registry.common:ProblemSpec
cli: vamos create-problem --help
command: python -m pytest -q tests/foundation/test_problem_registry.py tests/foundation/test_problem_zoo.py tests/foundation/test_problem_evaluation_edge_cases.py
command: python -m pytest -q tests/engine/test_algorithm_problem_matrix.py
```
