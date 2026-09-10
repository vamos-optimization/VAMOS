# Analysis and Visualization

VAMOS includes optional post-processing helpers for statistical comparison,
visualization, and multi-criteria decision making (MCDM). These tools are useful
after optimization; they are not required by the core optimization runtime.

!!! warning "Experimental surface in VAMOS 1.0.0"
    `vamos.ux.api`, statistical analysis, visualization, and MCDM helpers are
    **experimental**. They are supported for evaluation, but may change
    incompatibly in a minor release. See
    [Stability and versioning](../project/stability-and-versioning.md).

## Install the analysis extras

```bash
pip install "vamos-optimization[analysis]"
```

## Statistical comparison

The public `vamos.ux.api` facade exposes non-parametric helpers for comparing
algorithms across benchmark problems.

The statistical matrix has shape **`(n_problems, n_algorithms)`**:

- each row is one benchmark problem (the paired experimental block);
- each column is one algorithm;
- each cell is one pre-defined summary of the stochastic repeats for that
  problem/algorithm pair, such as the median hypervolume across seeds.

Do **not** put individual seeds in separate rows merely to increase the sample
size. Seeds are stochastic replicates of a problem/algorithm cell, not
additional benchmark problems. Decide the per-cell estimator before looking at
the statistical result.

The example below uses hypervolume, so larger values are better. For a metric
such as IGD+ where smaller values are better, pass `higher_is_better=False`.

```python
import numpy as np

from vamos.ux.api import friedman_test, pairwise_wilcoxon

algorithm_names = ["NSGA-II", "MOEA/D", "SPEA2"]
scores = np.array(
    [
        [0.85, 0.82, 0.84],
        [0.79, 0.75, 0.77],
        [0.91, 0.88, 0.89],
        [0.72, 0.70, 0.71],
        [0.81, 0.78, 0.80],
        [0.87, 0.83, 0.85],
    ],
    dtype=float,
)

friedman = friedman_test(scores, higher_is_better=True)
print(f"Friedman statistic: {friedman.statistic:.3f}")
print(f"Friedman p-value: {friedman.p_value:.4f}")
print("Average ranks:", dict(zip(algorithm_names, friedman.avg_ranks, strict=True)))

comparisons = pairwise_wilcoxon(
    scores,
    algorithm_names,
    higher_is_better=True,
)
for comparison in comparisons:
    print(comparison.algo_i, comparison.algo_j, comparison.p_value)
```

`friedman_test(...)` returns a `FriedmanResult` with `statistic`, `p_value`,
`ranks`, and `avg_ranks`. `pairwise_wilcoxon(...)` returns a list of
`WilcoxonResult` objects with `algo_i`, `algo_j`, `statistic`, and `p_value`.

The pairwise p-values are **raw Wilcoxon p-values**. The helper does not apply
Holm or another multiple-comparison correction. If your inferential procedure
requires multiplicity control, apply the correction explicitly outside this
helper and report which procedure you used.

### Critical-distance plot

`plot_critical_distance(...)` visualizes average ranks and, when
`n_problems` is supplied, a Nemenyi-style critical-distance bar. It does not
infer or draw significance cliques for you.

The plotting helpers return a Matplotlib `Axes`; save the figure through that
object instead of passing a filename argument.

```python
from vamos.ux.api import plot_critical_distance

ax = plot_critical_distance(
    friedman.avg_ranks,
    algorithm_names,
    n_problems=scores.shape[0],
    show=False,
)
ax.figure.savefig("cd_plot.png", dpi=200, bbox_inches="tight")
```

## Pareto-front visualization

Pass the objective matrix you want to display. `plot_pareto_front_2d(...)`
expects exactly two objectives and plots the supplied rows; it does not perform
an additional non-dominated filtering step.

```python
from vamos.ux.api import plot_pareto_front_2d

F = np.array(
    [
        [0.20, 0.90],
        [0.35, 0.65],
        [0.55, 0.45],
        [0.80, 0.25],
    ],
    dtype=float,
)

ax = plot_pareto_front_2d(
    F,
    labels=("f1", "f2"),
    title="Example objective front",
    show=False,
)
ax.figure.savefig("front.png", dpi=200, bbox_inches="tight")
```

The same facade also exposes `plot_pareto_front_3d`,
`plot_parallel_coordinates`, and `plot_hv_convergence`.

## MCDM: selecting one solution from a front

The public MCDM helpers in `vamos.ux.api` are:

- `weighted_sum_scores(F, weights)` — weighted scalarization;
- `tchebycheff_scores(F, weights, reference=None)` — weighted Chebyshev
  distance to a reference point (the component-wise minimum by default);
- `reference_point_scores(F, reference)` — Euclidean distance to a supplied
  reference point;
- `knee_point_scores(F)` — a geometric knee heuristic for **2D fronts only**.

They return an `MCDMResult` containing `scores`, `best_index`, and `best_point`.
The scoring convention is minimization: the selected index is the one preferred
by the helper under the supplied objectives/preferences.

Weighted methods are sensitive to objective scale. If objectives use different
units or magnitudes, normalize them deliberately before applying weights and
keep the returned index to recover the original objective vector (and the
corresponding decision vector, if you have an aligned `X` matrix).

```python
from vamos.ux.api import weighted_sum_scores

span = np.ptp(F, axis=0)
safe_span = np.where(span == 0.0, 1.0, span)
scaled_F = (F - F.min(axis=0)) / safe_span
weights = np.array([0.5, 0.5], dtype=float)

choice = weighted_sum_scores(scaled_F, weights)
best_index = choice.best_index
best_objectives = F[best_index]
print("Selected row:", best_index, best_objectives)
```

Normalization is a modeling choice, not a neutral preprocessing detail. For a
publication-quality analysis, state the scaling rule, weights/reference point,
metric direction, replicate aggregation rule, and statistical correction (if
any) in the experimental protocol.

## Landscape analysis

For landscape-analysis workflows such as random walks, autocorrelation, and
ruggedness, see `notebooks/2_advanced/25_landscape_analysis.ipynb`.
