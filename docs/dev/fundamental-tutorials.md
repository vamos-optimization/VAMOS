# Maintaining fundamental tutorials

T01–T03 use one source, `examples/tutorials/fundamentals.py`, for the complete
code, downloadable scripts, printed output, and figure data. The process
model and feasibility workflow are shared by T02 and T03. Their standalone
downloads are generated compositions, not separately maintained examples.

## Common lesson template

Use this structure for a maintained lesson:

1. **Goal:** one observable outcome for the learner.
2. **Before you start:** prerequisites, tested package and environment,
   dependencies, interface stability, explicit seed, backend, and budget.
3. **Model or method:** explain the decisions that make the code meaningful.
4. **Complete code:** downloadable runnable source and a generated code block.
5. **Observed result:** real printed output and a figure from that execution.
6. **Interpretation:** what the result establishes, what it cannot establish,
   and which rows, units, feasibility conditions, or preferences matter.
7. **Try one change:** modify one assumption and state what to check.
8. **Next step:** a named lesson or canonical guide.

Source, version, and dependency provenance must remain visible. Separate an
algorithm's returned set from a non-dominated subset; verify feasibility before
selection; do not infer convergence or comparative evidence from a small run.

## Generate the owned content

Use a clean environment with the **installed PyPI wheel**, not an editable
checkout. From a source checkout, install the tutorial recipe and plotting
dependency into that environment:

```bash
python -m pip install -r docs/assets/tutorials/tutorial-requirements.txt "matplotlib==3.10.9"
python tools/build_fundamental_tutorials.py --write
```

Clear any `PYTHONPATH` that points at `src` first. The generator rejects a
package imported from this checkout and rejects numerical/plotting versions
that differ from the declared recipe. `--write` explicitly replaces only
the owned downloads, figures, evidence record, dependency recipe, and marked
code/output blocks in the three lesson pages.

The generator extracts complete functions from the canonical source and
executes the assembled standalone scripts. Matplotlib receives their actual
arrays. Do not edit generated code, point positions, output, or hashes by hand.
The source hash identifies the exact example even when its source has not yet
been committed; `base_commit` records the checkout used for generation.

## Validate and review

```bash
python tools/build_fundamental_tutorials.py --check
python -m pytest -q tests/docs/test_fundamental_tutorials.py
```

The synchronization check needs only the standard library. The focused tests
exercise the downloaded scripts, actual objective values, original row
alignment, filtering before Pareto analysis, declared preferences, guided
modifications, and rejection of invalid preferences. The documentation smoke
manifest also executes the three canonical lesson entry points.

Run the examples and focused numerical tests in the clean wheel environment,
then follow the [documentation validation tier](testing.md#canonical-tiers)
and the strict MkDocs/Zensical and portal checks in
[Documentation delivery](documentation-delivery.md). Review the generated
figures and rendered lessons. Keep existing routes and anchors when adding
entry points or extracting content from a longer guide.

The evidence record is documentation provenance, not a VAMOS RunManifest.
Each SVG has a 250 kB size budget; no large arrays, run directories, notebook
outputs, or benchmark claims are committed for these demonstrations.
