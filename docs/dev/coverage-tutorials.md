# Maintain coverage tutorials

The phase-4 lessons cover the six additional algorithm families, four discrete/mixed representations, three practical cases and three advanced workflows. Sources are `examples/tutorials/capability_examples.py` and `examples/tutorials/advanced_workflows.py`; `tools/build_coverage_tutorials.py` owns downloads, code/output blocks, figures, observations and evidence under `docs/assets/coverage/`.

## Regenerate from executed code

Use a clean non-editable environment with the exact [coverage requirements](../assets/coverage/requirements.txt), then run from the checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python tools/build_coverage_tutorials.py --write --work-dir ../coverage-recording-new
python tools/build_coverage_tutorials.py --check
```

The work directory must be new. It preserves complete ablation studies and the tuner’s original files. The checked-in observation JSON is a bounded derived dataset, not a canonical replay artifact. Source/asset hashes, package versions and base commit are recorded. Standard-library `--check` detects stale sources, downloads, generated blocks, dependency recipes and asset bytes; it does not rerun optimization or establish scientific validity.

Before creating any output, `--write` checks the resolved VAMOS module against
the installed distribution and rejects source-checkout imports and editable
distribution metadata. A matching version string alone does not establish
that the published wheel supplied the code. `--check` remains independent of
an installed VAMOS package.

The generated timing figure intentionally records real elapsed time, so a regeneration changes those values. Capture thread settings and platform metadata, inspect the output, and avoid interpreting incidental timing changes as regressions. Figures are generated with Matplotlib from real arrays or samples. Do not hand-edit outputs or asset hashes. Keep ordinary assets under 250 KB and the observation file under 750 KB.

## Validate behavior and interpretation

Follow the repository’s [validation tiers](testing.md) and [documentation delivery](documentation-delivery.md). Run `tests/docs/test_coverage_tutorials.py`, the docs smoke manifest and both strict documentation builds. Test downloads copied outside the repository in the clean installed-wheel environment. Check meaningful invariants: exact evaluation counts, encoding domains, row alignment, emitted/declared constraints, non-overwrite behavior, changed ablation component and independently recomputed HV. Timings must not be subject to a speed threshold in CI.

Visually inspect representative two/three-objective plots, practical cases and the advanced figures. Check navigation, every new download and existing routes/anchors in the PR preview. Public examples use only curated facades; the tuner uses its Experimental CLI. Never label that CLI Stable merely because the wrapper is tested.

## Coverage map

| Capability | Precise reference | Verified teaching example |
| --- | --- | --- |
| Nine built-in algorithms | [Catalogue and capability matrix](../reference/algorithms.md) | Each catalogue row links to its complete guide and executable script. |
| Real decisions | [Problem protocol](../reference/api/problem-definition.md) | [T01](../tutorials/t01-first-result.md) |
| Binary, integer, permutation, mixed decisions | [Problem catalogue](../reference/problems.md) | [Four representation lessons](../tutorials/encodings.md) |
| Mixed constrained design | [Constraint convention](../reference/constraints.md) | [Welded beam](../tutorials/engineering-design.md) |
| Feature selection and model hyperparameters | [Problem catalogue](../reference/problems.md) | [Masks](../tutorials/feature-selection.md), [mixed SVC settings](../tutorials/model-hyperparameters.md) |
| Algorithm configuration tuning | [Experimental tuning](../topics/tuning.md) | [Train/validation/test seeds](../tutorials/tuning-experiment.md) |
| Component ablation | [StudySpec](../reference/api/study-specification.md) | [Mutation removal](../tutorials/controlled-ablation.md) |
| Backend performance | [Backend roles](../reference/algorithms.md#backends) | [First and subsequent calls](../tutorials/measure-performance.md) |

This is teaching coverage of explicitly identified cases, not proof of the full Cartesian product of capabilities, convergence, model generalization or universal speedups.

The expanded menu uses Material's `navigation.prune` to avoid embedding inactive
branches in every page. Inactive groups link to their first page; their complete
contents appear after entering the group. This keeps the modular API hub within
its existing HTML-size gate. Keep the first page of each group a useful entry
point and verify navigation in both documentation engines.
