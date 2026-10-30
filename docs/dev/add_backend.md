# Adding a kernel backend

A kernel backend accelerates stable numerical primitives without changing their semantics. Put control-heavy orchestration, algorithm policy, CLI, and study behavior elsewhere.

## Current interface

`KernelBackend` requires:

- `nsga2_ranking(F)`;
- `tournament_selection(ranks, crowding, pressure, rng, n_parents)`;
- `sbx_crossover(X_parents, params, rng, xl, xu)`;
- `polynomial_mutation(X, params, rng, xl, xu)`;
- `nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=False)`.

Optional archive and indicator hooks must advertise capability before use. `variation_kernels()` is the optional hook for compiled SBX and polynomial-mutation arithmetic: it returns `None` by default, and a backend that returns kernels must draw from the caller's generator in exactly the order and shape of the reference operators. `nsga2_steady_state_kernels()` (**development after 1.1.0**) is the optional hook for the compiled one-offspring NSGA-II step: it also returns `None` by default, and returned kernels never draw random numbers and must reproduce the reference steady-state ask/tell path exactly. `moead_steady_state_kernels()` (**development after 1.1.0**) is the matching optional hook for MOEA/D's one-offspring step (`batch_size=1`); its contract, aggregation ids and workspace layout live in `vamos.foundation.kernel.moead_kernels`. Read the abstract class rather than copying a method list from documentation when the interface changes.

## Workflow

1. Treat `NumPyKernel` as the reference behavior and identify a measured hot path.
2. Implement the abstract interface under `src/vamos/foundation/kernel/`. Preserve input/output shapes, tie handling, stable ordering expectations, RNG consumption contracts, and error semantics.
3. Add a lazy factory to `KERNELS` in `registry.py`. Import the optional package only inside that factory or backend invocation.
4. Put a non-core dependency in the appropriate optional extra and provide an actionable import error. Do not make `import vamos` load it.
5. Add parity tests against NumPy, failure tests, and a tiny end-to-end algorithm run. Benchmark only after correctness is established.
6. Update backend reference docs and exact-replay reconstruction only when the backend becomes an intentionally supported built-in replay target.

Never substitute NumPy or another engine after the caller selects an unavailable backend. An explicit `auto` choice may resolve through the orchestration policy; a named backend either runs or fails.

## Required validation

```bash
python -m pytest -q tests/foundation/test_backends_smoke.py tests/foundation/test_kernel_failures.py
python -m pytest -q tests/foundation/test_numba_backend_parity.py tests/engine/test_kernel_selection_dispatch.py
```

Add focused parity/performance coverage for the changed primitive and run the full tier from `/AGENTS.md`.

```agent-docs
path: src/vamos/foundation/kernel/backend.py
path: src/vamos/foundation/kernel/numpy_backend.py
path: src/vamos/foundation/kernel/registry.py
path: tests/foundation/test_backends_smoke.py
path: tests/foundation/test_kernel_failures.py
path: tests/foundation/test_numba_backend_parity.py
path: tests/engine/test_kernel_selection_dispatch.py
path: docs/reference/algorithms.md
symbol: vamos.foundation.kernel.backend:KernelBackend
symbol: vamos.foundation.kernel.numpy_backend:NumPyKernel
symbol: vamos.foundation.kernel.registry:resolve_kernel
command: python -m pytest -q tests/foundation/test_backends_smoke.py tests/foundation/test_kernel_failures.py
command: python -m pytest -q tests/foundation/test_numba_backend_parity.py tests/engine/test_kernel_selection_dispatch.py
```
