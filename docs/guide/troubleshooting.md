# Troubleshooting

These checks target the published VAMOS 1.0.0 package. See
[Installation](installation.md) for supported environments and extras, and
[Stability and versioning](../project/stability-and-versioning.md) for the
**Stable**, **Experimental**, and **Internal** labels.

## Check the selected environment

Run these commands in the same terminal or notebook environment as your code:

```bash
python -c "import sys, vamos; print(sys.executable); print(vamos.__version__); print(vamos.__file__)"
python -m pip show vamos-optimization
```

The version should be `1.0.0` when following this release's examples. If the
package is missing, activate the intended virtual environment and follow
[Installation](installation.md). Use `python -m pip` so installation targets the
selected interpreter. Restart a notebook kernel after changing packages.

`vamos check` provides an additional **Experimental** diagnostic report.
Missing optional packages do not necessarily mean that the core NumPy backend
is unusable.

## Missing optional dependencies

Install the relevant extra into the active environment; these commands work
without a source checkout:

| Capability | Installation command | Status |
| --- | --- | --- |
| Numba/MooCore kernels and Dask dependency | `python -m pip install "vamos-optimization[compute]==1.0.0"` | Kernel selection is Stable; distributed evaluation is Experimental |
| Plots, pandas, and notebooks | `python -m pip install "vamos-optimization[analysis]==1.0.0"` | Analysis helpers are Experimental |
| Local Studio | `python -m pip install "vamos-optimization[studio]==1.0.0"` | Experimental |
| Selected domain examples | `python -m pip install "vamos-optimization[examples]==1.0.0"` | Check the individual example |
| External research baselines | `python -m pip install "vamos-optimization[research]==1.0.0"` | Experimental integrations |
| Optional tuning backends | `python -m pip install "vamos-optimization[tuning]==1.0.0"` | Experimental |

An explicitly requested unavailable backend fails; it is not silently replaced
by NumPy. Choose `engine="numpy"` explicitly if you want a core-only run.

## Unknown algorithm, engine, or incompatible encoding

- Consult the [algorithm capability matrix](../reference/algorithms.md) for
  supported algorithms, encodings, constraints, and backends.
- Check exact `--algorithm` and `--engine` spellings in the
  [CLI reference](cli.md).
- Match the problem's encoding with the configured crossover and mutation.
  The [problem catalog](../reference/problems.md) documents problem properties.
- Keep the budget at least as large as the initial population. For
  reference-direction algorithms, also respect their population/direction
  cardinality requirements.

## Invalid values from a custom problem

Objective values must be finite throughout the declared bounds. Check boundary
points explicitly: a denominator must not become zero, and square roots and
logarithms must stay in their domains. A batched objective must return an array
of shape `(batch_size, n_obj)`.

For constraints, `g(x) <= 0` means feasible. Set `n_constraints` consistently
and return the corresponding number of constraint values. See
[Solving your own problem](custom-problem.md).

## A run or replay destination already exists

Canonical run writers do not overwrite or merge an existing result. Use a new
`--output-root` for a new CLI experiment, a new destination for `save_result`,
or a new `--output` for replay. Keep existing runs intact when comparing
settings. See [Run artifacts](run-artifacts.md).

## Replay is refused

First inspect and verify the source run:

```bash
vamos results inspect results/first-run/ZDT1/nsgaii/numpy/seed_7
vamos results verify results/first-run/ZDT1/nsgaii/numpy/seed_7 --require-level exact
```

These paths refer to the [Minimal Python Track](minimal-python.md); replace
them with your actual run directory. Read the reported integrity, environment,
and reconstructability results. Exact replay requires supported built-ins
and matching material environment evidence. Loading arrays with `load_result`
is a separate data-only operation and does not require executing the problem.
Do not edit manifests to force compatibility.

## Benchmark suite names

The benchmark command is **Experimental**. Use `vamos bench --list` to discover
available suites and match their names exactly. See the
[CLI reference](cli.md#other-subcommands).
