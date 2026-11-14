# VAMOS 1.2.1 release notes

Released 2026-11-14.

VAMOS 1.2.1 is a patch release of [VAMOS 1.2.0](release-notes-1.2.0.md). It
corrects C2-DTLZ2's default radius with two objectives, which changes the
results of affected experiments, and the constraint count of eight ZXH-CF
problems. It keeps persistent Optuna studies of the tuning CLI from mixing with
those of earlier versions, and corrects two install instructions in the
documentation. The stable 1.x API, the omitted-engine NumPy default, and the
1.0.0 run and study schemas remain the compatibility baseline.

## Which results change

| Area | Change | Rerun |
| --- | --- | --- |
| C2-DTLZ2 with two objectives | The default radius is 0.5, as in Jain and Deb (2014) and jMetal, instead of 0.2. The whole quarter circle of the front is feasible, and the packaged two-objective front is regenerated. | Every two-objective C2-DTLZ2 run with the default radius, and its indicators against the packaged front. `r=0.2` gives the former problem. |
| ZXH-CF5–8 and ZXH-CF13–16 | One constraint per constrained angle, the k_theta constraints of Zhou, Xiang and He (2021), as ZXH-CF9–12 do: `k + 1` or `k + 2` constraints instead of `2k + 1` or `2k + 2`. Feasibility and the overall constraint violation do not change. | Runs whose constraint handling uses the individual constraints, such as `constraint_handling="violated_constraints"`, and anything that reads the number of constraints. |

C2-DTLZ2 with three objectives keeps its default radius, and no other problem,
algorithm, operator or indicator changes.

## Persistent Optuna studies

When `optuna` or `bohb_optuna` uses `--optuna-storage`, `vamos tune` now stores
its studies under `__vamos_cli_final_population_hv_v2`. VAMOS 1.2.0 changed the
built-in search spaces and the scores of the same configurations but kept the
`_v1` namespace of VAMOS 1.1.0. Resuming an earlier study therefore mixed trials
of two scoring contracts, or stopped with Optuna's `ValueError:
CategoricalDistribution does not support dynamic value space` when a parameter
offered other choices. With 1.2.1 the same command starts a new study, and the
earlier one stays in the storage under its `_v1` name. Persistent SMPSO studies
keep `__vamos_cli_leaders_hv_v1`, the namespace that VAMOS 1.2.0 introduced. See
[tuning](../topics/tuning.md#persistent-optuna-studies-are-scoring-contract-versioned).

## Documentation

- The troubleshooting guide no longer lists the `research` extra, which VAMOS
  1.2.0 removed, and gives the `[analysis,examples]` install that notebooks
  need.
- The algorithms reference describes the removed third-party baselines with the
  VAMOS 1.1.0 install command, the last version that had them.
