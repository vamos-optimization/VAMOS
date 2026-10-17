# Coverage tutorial setup

These lessons extend the first-result and study tutorials to all nine built-in algorithm families, five representations, three practical cases and advanced experiments. The new downloads are verified against the published **VAMOS 1.0.1** wheel, using Python 3.12.14. They import only the public VAMOS facades.

## Install and run

Start with a fresh virtual environment using the [installation guide](../guide/installation.md). Core runs need `vamos-optimization==1.0.1`; figures add Matplotlib, the two learning cases add scikit-learn, and the advanced comparison needs MooCore and Numba. For the complete recorded environment, download [requirements.txt](../assets/coverage/requirements.txt) and run:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Download the script linked by a lesson into your working directory. No repository checkout is required. Algorithm and representation examples default to 400 objective evaluations; feature selection and model hyperparameters use 80. Plotting is optional and reruns the optimization:

```bash
python smsemoa.py
python smsemoa.py --budget 800 --output smsemoa-800.svg
```

Existing figure files are refused. Advanced examples require `--output` pointing to a **new directory**, preserve their original artifacts there, and refuse an existing directory. They do not resume implicitly; use the [study lifecycle](../guide/studies.md) for recovery of a saved study.

## Recorded evidence

[Observations](../assets/coverage/observations.json) retain actual decision/objective arrays, constraint residuals when applicable, original row indices, explicit configurations, seeds and evaluation counts. The advanced sections retain tuner summaries, verified ablation fronts and measured timing samples. [Evidence](../assets/coverage/evidence.json) records source and asset hashes, dependency versions, the generation command and base commit.

These are derived teaching observations, **not canonical RunManifest or StudyManifest bundles**. They do not support `reproduce()` directly. Full ablation studies and original tuner files remain in the generation work directory; see [maintainer regeneration](../dev/coverage-tutorials.md).

All recorded optimization examples use minimization. Example seeds are explicit; that does not guarantee bitwise equality across platforms or dependency versions. Timing samples are specific to the recorded host and thread settings. No table is an algorithm ranking, and none of the teaching budgets constitutes a convergence guarantee.

## Choose your next lesson

- [Algorithm catalogue](../reference/algorithms.md#algorithm-catalogue): one teaching guide for every built-in identifier.
- [Representations](encodings.md): binary, integer, permutation and mixed decisions; [T01](t01-first-result.md) covers real variables.
- [Engineering design](engineering-design.md), [feature selection](feature-selection.md), or [model hyperparameters](model-hyperparameters.md).
- [Experimental algorithm tuning](tuning-experiment.md), [controlled ablation](controlled-ablation.md), or [backend measurements](measure-performance.md).
