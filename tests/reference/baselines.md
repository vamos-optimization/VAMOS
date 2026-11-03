# Reference Verification Baselines

These tests verify **scientific correctness** by running algorithms on standard benchmarks with deterministic seeds.

## ZDT1 (Convex, 2D)
*   **Problem**: `ZDT1Problem(n_var=30)`
*   **Goal**: Minimize f1, f2
*   **Budget**: 10,000 evaluations, seed 42
*   **Metric**: Hypervolume (Ref Point: [1.1, 1.1]; the front's maximum is 0.8716)
*   **Baselines**:
    *   **NSGA-II**: Expected HV > 0.80 (seed 42 reaches 0.8466; seeds 0 to 9, 0.8431 to 0.8512)
    *   **MOEA/D** (SBX, Tchebycheff): Expected HV > 0.79 (seed 42 reaches 0.8394; seeds 0 to 9, 0.8341 to 0.8590)

## ZDT2 (Non-Convex, 2D)
*   **Problem**: `ZDT2Problem(n_var=30)`
*   **Budget**: 10,000 evaluations, seed 42
*   **Baselines**:
    *   **NSGA-II**: Expected HV > 0.47 (seed 42 reaches 0.5027). Other seeds can lose the upper part of the front at this budget (seeds 0 to 9, 0.12 to 0.50), so the threshold holds for seed 42 only.

## DTLZ2 (Concave, 3D)
*   **Problem**: `DTLZ2Problem(n_var=12, n_obj=3)`
*   **Budget**: 5,000 evaluations, seed 42
*   **Metric**: Hypervolume (Ref Point: [1.1, 1.1, 1.1]; the front's maximum is 0.8074)
*   **Baselines**:
    *   **NSGA-III** (12 divisions, 91 individuals): Expected HV > 0.67 (seed 42 reaches 0.7192; seeds 0 to 9, 0.7055 to 0.7205)

> **Note**: Each threshold is about 93 to 95 % of the hypervolume that its configuration reaches, so a drop of more than about 5 % fails (the thresholds of VAMOS 1.1.0 let drops of 29 to 40 % pass). They catch regressions; they are not a measure of comparative performance.

## Conformance with a reference implementation
*   **AGE-MOEA on ZDT1**: `ZDT1Problem(n_var=30)`, population 100, 10,000 evaluations, seeds 0 to 9, Hypervolume (Ref Point: [1.1, 1.1])
*   **Baseline**: jMetal 7.9 `AGEMOEABuilder` over 30 seeds, HV 0.8545 +- 0.0040. The mean must lie within two standard errors of it.
*   **NSGA-II on ZDT1** and **NSGA-III on DTLZ2** (`test_external_conformance.py`): pymoo 0.6.2 with the same configuration over 30 seeds, stored with its configuration in `data/external_conformance.json` by `tools/reference_conformance_baselines.py`. The VAMOS mean over seeds 0 to 9 must lie within 3 % of pymoo's mean. Measured: NSGA-II 0.8467 against pymoo's 0.8438 (0.3 %); NSGA-III 0.7385 against 0.7384 (0.01 %). pymoo's MOEA/D (no replacement limit), SMS-EMOA (tournament mating) and SPEA2 differ from VAMOS's, so they are not compared with it.
