# Reference Verification Baselines

These tests verify **scientific correctness** by running algorithms on standard benchmarks with deterministic seeds.

## ZDT1 (Convex, 2D)
*   **Problem**: `ZDT1Problem(n_var=30)`
*   **Goal**: Minimize f1, f2
*   **Budget**: 2,500 evaluations
*   **Metric**: Hypervolume (Ref Point: [1.1, 1.1])
*   **Baselines**:
    *   **NSGA-II**: Expected HV > 0.60
    *   **MOEA/D**: Expected HV > 0.60

## ZDT2 (Non-Convex, 2D)
*   **Problem**: `ZDT2Problem(n_var=30)`
*   **Budget**: 3,000 evaluations
*   **Baselines**:
    *   **NSGA-II**: Expected HV > 0.30

## DTLZ2 (Concave, 3D)
*   **Problem**: `DTLZ2Problem(n_var=12, n_obj=3)`
*   **Budget**: 5,000 evaluations
*   **Metric**: Hypervolume (Ref Point: [1.1, 1.1, 1.1])
*   **Baselines**:
    *   **NSGA-III**: Expected HV > 0.50

> **Note**: These thresholds are lower bounds to catch significant regressions (e.g., algorithm collapses or fails to progress). They do not represent state-of-the-art performance, but rather "acceptable functioning" for regression testing.

## Conformance with a reference implementation
*   **AGE-MOEA on ZDT1**: `ZDT1Problem(n_var=30)`, population 100, 10,000 evaluations, seeds 0 to 9, Hypervolume (Ref Point: [1.1, 1.1])
*   **Baseline**: jMetal 7.9 `AGEMOEABuilder` over 30 seeds, HV 0.8545 +- 0.0040. The mean must lie within two standard errors of it.
