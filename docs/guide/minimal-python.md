# Minimal Python Track

Run an experiment and inspect its results from a terminal. You do not need to
write Python code for this path.

> **Stable in VAMOS 1.0.0:** the single-run, inspection, verification, and replay
> commands below. The optional quickstart wizard is **Experimental**. See
> [Stability and versioning](../project/stability-and-versioning.md).

## 1. Install

Follow [Installation](installation.md) to create and activate a virtual
environment for your operating system. Install the published package from any
working folder:

```bash
python -m pip install "vamos-optimization==1.0.0"
python -c "import vamos; print(vamos.__version__)"
```

The second command should print `1.0.0`. No repository checkout is needed.
Optional plotting dependencies are described in
[Installation: optional extras](installation.md#optional-extras).

## 2. Run a first experiment

This small, seeded run uses NSGA-II on the two-objective ZDT1 benchmark:

```bash
vamos --problem zdt1 --algorithm nsgaii --engine numpy --population-size 40 --max-evaluations 400 --seed 7 --output-root results/first-run
```

The budget is deliberately small: it checks the complete workflow, not
convergence to the true Pareto front.

## 3. Inspect the result

```bash
vamos results inspect results/first-run/ZDT1/nsgaii/numpy/seed_7
vamos results verify results/first-run/ZDT1/nsgaii/numpy/seed_7
```

The run directory contains:

- `manifest.json`: requested and resolved settings, actual seed, outcome, and hashes.
- `result.npz`: numerical decision, objective, population, and archive arrays.
- `environment.json`: the recorded runtime environment.

Inspection reports the run and array shapes. Verification checks the stored
artifact and reports replay compatibility without rerunning optimization.
See [Understanding results](understanding-results.md) for the difference
between a returned population, its non-dominated subset, and the true front.

## 4. Change the budget or replay exactly

Give a larger run its own output root so the first result remains available:

```bash
vamos --problem zdt1 --algorithm nsgaii --engine numpy --population-size 40 --max-evaluations 4000 --seed 7 --output-root results/larger-run
```

To repeat the first run with its **stored** configuration and seed:

```bash
vamos reproduce results/first-run/ZDT1/nsgaii/numpy/seed_7 --output results/replays/first-run
```

Exact replay requires supported built-in components and the same materially
relevant environment. A replay creates a new run; its destination must not
already exist. See [Run artifacts](run-artifacts.md).

## Optional: use the guided wizard

**Experimental.** The wizard asks questions, writes a config file, and runs an
experiment:

```bash
vamos quickstart
```

For a bounded demonstration without prompts or plotting dependencies:

```bash
vamos quickstart --template demo --yes --no-plot --engine numpy --pop-size 40 --budget 400 --seed 7 --output-root results/wizard --config-path quickstart.json
```

The wizard prints the actual config and result paths. Validate its generated
config, then run it into a new output root:

```bash
vamos --config quickstart.json --validate-config
vamos --config quickstart.json --max-evaluations 800 --output-root results/wizard-repeat
```

Discover other templates with `vamos quickstart --template list`. The biology
and chemistry templates need the `examples` extra from
[Installation](installation.md#optional-extras).

## Glossary

- **Problem:** the model or function to optimize.
- **Objective:** one quantity to minimize; there can be several competing objectives.
- **Non-dominated set:** solutions for which no other retained solution is no worse in every objective and strictly better in at least one.
- **Pareto front:** the objective values of globally Pareto-optimal solutions; a short run usually returns only an approximation.
- **Budget:** the maximum number of evaluations.
- **Population size:** the number of candidates maintained by the algorithm.
- **Seed:** controls randomness for repeatable runs in the same environment.
- **Engine:** the computational backend; `numpy` is the reference default.

## Next steps

- [Troubleshooting](troubleshooting.md) for installation and run errors.
- [Quickstart](getting-started.md) to move from terminal commands to Python.
- [CLI and config files](cli.md) for the complete command reference.
- [Cookbook](cookbook.md) for copy-paste Python recipes.
