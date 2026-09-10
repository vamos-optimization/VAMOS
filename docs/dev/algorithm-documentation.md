# Maintaining algorithm tutorials

The canonical algorithm index remains at [Algorithms and backends](../reference/algorithms.md). Teaching pages live in `docs/algorithms/`; [NSGA-II](../algorithms/nsgaii.md) is the first complete template. Generated signatures and builder methods belong to the separate [configuration reference](../reference/api/algorithms/index.md). Do not maintain a second manual table of all API defaults or promote undocumented internals to a public interface.

## Source and figure ownership

The NSGA-II tutorial is grounded in the current implementation and the original paper, with bibliographic attribution retained from the legacy page. Its claims deliberately do not inherit fixed objective-count cut-offs or performance guarantees from legacy marketing text.

`examples/journeys/nsgaii_zdt1.py` owns the numerical illustration at `docs/assets/algorithms/nsgaii-zdt1.svg`. The plotted points are the actual returned objectives. The dashed line is analytical and must never be described as a second empirical run. No run directory, solution archive or benchmark leaderboard is committed for this illustration.

The initial illustration was generated with VAMOS 1.0.0, Python 3.13.5, NumPy 2.3.5 and Matplotlib 3.10.8, using the explicit NumPy backend, seed 42, population 100 and 10,000 evaluations. It returned 100 solutions. This records that illustration's environment, not a new minimum dependency policy or a guarantee of identical arrays in different environments.

To update it, run the example with `--output` pointing to a **new temporary SVG file**, review the output, then replace the maintained illustration deliberately. The generator refuses overwrites. Keep the visible settings and this provenance aligned with the regenerated file. Do not edit point positions by hand or generate a front independently of the example.

## Validation

Follow the [canonical validation tiers](testing.md#canonical-tiers). The focused regression file is `tests/docs/test_nsgaii_tutorial.py`; the no-plot command is also registered in `tests/docs/smoke_manifest.py`.

The focused tests execute the tutorial's first Python block, compare it with the example at the same settings, validate shapes, bounds, objective values and exact evaluation count, exercise the explicit builder, and check optional SVG generation and overwrite refusal. They do not impose a convergence threshold, compare wall-clock performance or assert bitwise identity across environments. Review the rendered page, figure, table and code at mobile/desktop widths and in both themes; strict builds are not a substitute for that review. Zensical compatibility and the versioned portal checks must remain green.

## Extending the catalogue

Add a teaching page only when it has a checked example and an interpretation of its output. Until then, the index should link to the available configuration reference rather than to an empty tutorial. MOEA/D and NSGA-III are the next candidates after the NSGA-II template is accepted. Do not change hosting, runtime algorithms, dependency policy or archive-preservation rules as incidental documentation cleanup.
