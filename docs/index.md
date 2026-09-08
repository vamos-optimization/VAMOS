# VAMOS

VAMOS (Vectorized Architecture for Multiobjective Optimization Studies) is a Python framework for multi-objective evolutionary optimization and reproducible optimization studies.

VAMOS 1.0 distinguishes stable optimization, run-artifact, and single-owner study surfaces from experimental features such as Studio, provider integrations, and tuning. Check [Stability and versioning](project/stability-and-versioning.md) before depending on an API as a 1.x compatibility commitment.

## Choose your path

### Try VAMOS

Install the package, run a small built-in problem, and learn what the returned objective and decision arrays contain.

Start with [Installation](guide/installation.md), then follow the [Quickstart](guide/zero_to_hero.md).

### Solve your own problem

Define objectives, bounds, encodings, and constraints with the public `make_problem(...)` facade, then optimize the result with the same `optimize(...)` entry point used for built-in problems.

Follow [Solve your own problem](guide/custom-problem.md).

### Run a reproducible study

Freeze a problem-by-algorithm-by-seed matrix, persist canonical runs, inspect results, resume interrupted work, and use exact replay only when the stored run is reconstructable in a materially matching environment.

Follow [Durable studies](guide/studies.md) and [Run artifacts & replay](guide/run-artifacts.md).

## What VAMOS includes

- **Algorithms:** NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, and RVEA.
- **Encodings:** real, integer, binary, permutation, and mixed decision variables where supported by the selected problem and algorithm.
- **Backends:** NumPy as the deterministic reference path, optional Numba acceleration for core kernels, and MooCore-backed indicator acceleration.
- **Research tooling:** durable studies, benchmarking, tuning, result inspection, verification, replay, analysis, and optional interactive tools.

For precise signatures and configuration contracts, use the [API reference](reference/api_reference.md), [algorithm reference](reference/algorithms.md), and [problem reference](reference/problems.md).

## Citation and project information

Citation metadata is maintained in [`CITATION.cff`](https://github.com/vamos-optimization/VAMOS/blob/main/CITATION.cff). See the [security policy](https://github.com/vamos-optimization/VAMOS/blob/main/SECURITY.md) and use [private vulnerability reporting](https://github.com/vamos-optimization/VAMOS/security/advisories/new) for security issues. See [Known limitations](project/known-limitations.md), the [roadmap](roadmap.md), and [repository governance](project/repository-governance.md) for project-level information.
