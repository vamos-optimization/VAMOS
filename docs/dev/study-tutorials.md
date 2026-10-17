# Maintaining study tutorials

T06–T08 follow the [common lesson template](fundamental-tutorials.md#common-lesson-template).
Their canonical executable source is `examples/tutorials/study_workflow.py`.
`tools/build_study_tutorials.py` composes the three standalone downloads from
complete functions and generates the marked code/output blocks, real figures,
retained measurements, CSV tables and recorded report. Do not edit those
generated outputs by hand.

## Scientific boundaries

Keep two protocols distinct: eight small runs check that the workflow works;
the recorded experiment has two problems, two algorithms, ten seeds and three
evaluation budgets, totaling 120 runs and 120,000 objective evaluations.
These are descriptive observations for the declared settings. Two problems
do not justify a general ranking or an inferential test.

The budget axis compares **fresh runs**. The stable study API does not capture
per-generation history; do not relabel these observations as checkpoints or a
continuous convergence trajectory. Charge each run in full. Fixed references,
metric directions, aggregation and failure policy must remain explicit.

The measurement step retains every planned row, with unavailable metrics as
null. The report requires a complete successful design and checks retained
front hashes and recomputed metrics before producing figures. It reuses the
public Experimental rank and plotting helpers; it does not implement a second
statistics framework. Shared seed labels alone do not establish pairing.

## Generate the owned content

Use a clean environment containing the installed PyPI wheel, with no editable
checkout or `PYTHONPATH` pointing to `src`:

```bash
python -m pip install -r docs/assets/studies/study-requirements.txt
python tools/build_study_tutorials.py --write --work-dir /path/to/new-study-recording
```

The generator checks all six pinned package versions and rejects imports from
the checkout. The work directory must be new. It retains the full canonical
studies and intermediate reports for inspection; do not commit those run
directories. All tutorial commands likewise refuse existing destinations.

The repository retains a bounded **derived documentation snapshot**, including
non-dominated objective arrays and per-task metric/provenance rows. This lets
readers recompute each scalar and rebuild the report without optimization.
It is not a StudyManifest, RunManifest or exact-replay bundle: it omits decision
arrays and full canonical run files. Recreate those files by following the
recorded protocol in [Study setup](../tutorials/study-setup.md#the-recorded-experiment).
Keep complete canonical evidence separately when publishing scientific work.

`evidence.json` identifies the source hash, base commit, environment, command,
asset hashes and actual output. Its source hash identifies the exact source
before the change has a commit. Each retained JSON asset has a 750 kB budget;
each other owned asset has a 250 kB budget.

## Validate and review

```bash
python tools/build_study_tutorials.py --check
python -m pytest -q tests/docs/test_study_tutorials.py
```

The synchronization check is standard-library-only. Tests exercise standalone
downloads, plan-only behavior, persisted provenance, metric direction and
references, missing/duplicate/altered evidence, incomplete studies, and report
creation without optimization. The smoke manifest also registers all three
canonical lessons. Optional metric/plot tests skip explicitly without those
dependencies; the T06 study workflow remains core-only.

Run the published-wheel checks and the [documentation validation tier](testing.md#canonical-tiers),
then strict MkDocs/Zensical and portal checks from [Documentation delivery](documentation-delivery.md).
Review actual figures and rendered pages. Preserve existing routes and anchors
when linking the study lifecycle and analysis guides to this sequence.
