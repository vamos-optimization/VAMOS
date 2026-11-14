# VAMOS: Vectorized Architecture for Multiobjective Optimization Studies

> **A high-performance, unified framework for Multi-Objective Evolutionary Algorithms (MOEA) in Python.**

![VAMOS Banner](https://raw.githubusercontent.com/vamos-optimization/VAMOS/main/docs/assets/vamos1.png)

Official repository: [vamos-optimization/VAMOS](https://github.com/vamos-optimization/VAMOS).
Issues, pull requests, releases, and development are coordinated there. See
[repository governance](https://vamos-optimization.org/project/repository-governance/).
Documentation: [vamos-optimization.org](https://vamos-optimization.org/).

VAMOS bridges the gap between simple research scripts and large-scale optimization studies. It provides a unified API for running state-of-the-art algorithms across diverse problems, backed by vectorized kernels with NumPy as the exact reference path and optional Numba acceleration for core kernels.

VAMOS 1.0.0 is the first official public release and compatibility baseline.
VAMOS 1.2.0 brings many built-in algorithms, operators, problems and indicators
back to their published definitions, which changes results of affected
experiments, and adds 115 constrained benchmark problems, jMetal's
constraint-handling techniques and the CCMO and C-TAEA algorithms. It supports
Python 3.13 and 3.14 and is the last minor release for Python 3.10. See the
[1.2.0 release notes](https://vamos-optimization.org/project/release-notes-1.2.0/), including
which results to rerun, and the earlier
[1.1.0 release notes](https://vamos-optimization.org/project/release-notes-1.1.0/).
Earlier version strings and Git tags were internal pre-public development
markers, not prior public releases. See the
[stability policy](https://vamos-optimization.org/project/stability-and-versioning/) and
[known limitations](https://vamos-optimization.org/project/known-limitations/).

## Key Features

- **Unified API**: A clear, fluent interface `vamos.optimize()` for all workflows.
- **Nine Algorithms**: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA. [Known limitations](https://vamos-optimization.org/project/known-limitations/) lists those not yet validated independently against a reference implementation.
  **VAMOS 1.2.0:** CCMO (Tian et al., 2021), a coevolutionary algorithm for constrained problems whose fitness and selection are checked against jMetal's port and whose IGD on MW1, MW3 and C1-DTLZ1 matches the published values.
  **VAMOS 1.2.0:** C-TAEA (Li et al., 2019), a two-archive algorithm for constrained problems whose weight vectors, archive updates and mating selection are checked against jMetal's port and whose IGD+ on C1-DTLZ1, C3-DTLZ4 and MW3 and IGD on MW1, MW3 and LIR-CMOP7 match jMetal's runs and the published values.
- **Unified Archiving**: Consistent external archive configuration via `.external_archive(capacity=..., pruning=...)`, with bounded or unbounded archives and pruning policies `crowding`, `hv`, `mc_hv`, `knn`, `maxmin`, and `ref_dirs`. When an external archive is enabled, top-level results come from it by default unless `result_mode="population"` is requested.
- **Multi-Fidelity Tuning**: Hyperband-style racing with warm-start checkpoints for sample-efficient algorithm configuration.
- **Ready-to-use Tuning Backends**: `racing` and `random` work out of the box; install the optional `tuning` extra to enable `optuna`, `bohb_optuna`, `smac3`, and `bohb` via `vamos tune`.
- **Performance Driven**: Vectorized NumPy kernels with optional Numba JIT acceleration for core kernels.
- **Interactive Analysis**: Built-in dashboards with `explore_result_front(result)` and publication-ready LaTeX tables.
- **Visual Problem Builder**: Define custom problems in the experimental VAMOS Studio; local Python preview requires an explicit trusted-code opt-in.
- **Extensible**: Standardized protocols for adding custom problems, operators, and algorithms.

Canonical customization guide: [Extending VAMOS](https://vamos-optimization.org/topics/extending/).

## Quick Install

```bash
pip install vamos-optimization
```

For development and extras (Windows PowerShell):

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core + essential extras
pip install "vamos-optimization[compute,analysis]"
```

For development and extras (Linux / macOS):

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core + essential extras
pip install "vamos-optimization[compute,analysis]"
```

Optional model-based tuning backends (`optuna`, `smac3`, `bohb`):

```bash
pip install "vamos-optimization[tuning]"
```

`smac3` in VAMOS is provided by the PyPI package `smac` (SMAC3):

```bash
pip install "smac>=2.0"
```

## Backend Capability Matrix

| Backend | Status | Role |
|---------|--------|------|
| `numpy` | Stable | Exact reference backend and deterministic default. |
| `numba` | Stable optional | Accelerates core numeric kernels such as variation, tournament selection, and MOEA/D neighborhood updates. |
| `moocore` | Stable optional | Adds accelerated quality-indicator support, especially hypervolume-style metrics. |

In VAMOS 1.2.0, use `engine="auto"` (CLI: `--engine auto`)
to prefer loadable Numba for all nine built-in algorithms, with a NumPy fallback.
Numba remains optional. Omitting `engine` still selects NumPy, preserving the 1.x
default; explicit `engine="numba"` fails if unavailable. First use can spend more
time compiling than a short optimization takes. See the
[selection and replay policy](https://vamos-optimization.org/reference/algorithms/#automatic-engine-selection)
before comparing timings or moving runs between environments.

Operator shorthand accepts either numeric probabilities or the string literal `"1/n"` for per-variable mutation rates.

## API Contracts

- `max_evaluations` is a strict evaluation budget. It must be at least the resolved population size, and the final generation is truncated when needed so budget-terminated runs report exactly `result.data["evaluations"] == max_evaluations`.
- Public algorithm results expose `evaluations` as the evaluation-count metric. Internal checkpoints may still store `n_eval`.
- **VAMOS 1.2.0:** `hypervolume` leaves out the points that the reference point does not dominate, as the two-objective helper `compute_hypervolume` does. `allow_ref_expand` is accepted for compatibility only; it is ignored with a `FutureWarning`.
- **VAMOS 1.2.0:** NSGA-III needs at least one individual per reference direction (`pop_size` at least the number of directions). RVEA and MOEA/D need exactly one individual per reference or weight vector. Incompatible configurations fail early with an actionable error.

## Quickstart

Solve the ZDT1 benchmark problem with NSGA-II in just a few lines:

```python
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=10000,
    pop_size=100,
    engine="numpy",
    seed=42,
)

front = result.front()
print(f"Non-dominated solutions: {len(front) if front is not None else 0}")
```

Prefer a guided CLI? Run:

```bash
vamos quickstart
```

This wizard writes a reusable config and stores results under `results/quickstart/`.

Use `vamos quickstart --template list` to inspect domain templates.

New to Python? Start with the [Minimal Python Track](https://vamos-optimization.org/guide/minimal-python/).

After a run, summarize results with:

```bash
vamos summarize --results results/quickstart
```

Inspect, fully verify, or exactly replay a canonical built-in run:

```bash
vamos results inspect RUN_DIR
vamos results verify RUN_DIR --require-level exact
vamos reproduce RUN_DIR
```

For a small study in one call, compare the seeds with a quality indicator. Every
run spends exactly `max_evaluations`, so the evaluation count cannot tell them
apart. The hypervolume below uses MooCore, installed by the `compute` extra:

```python
import numpy as np
from moocore import hypervolume
from vamos import optimize

study = optimize("zdt1", algorithm="nsgaii", max_evaluations=4000, seed=[0, 1, 2])
reference_point = np.array([1.1, 1.1])  # fixed: ZDT1's Pareto-front nadir plus 10%
hv = np.array([hypervolume(run.front(), ref=reference_point) for run in study])
print(f"Hypervolume: mean {hv.mean():.4f}, std {hv.std(ddof=1):.4f}")
print("Best seed:", study[int(hv.argmax())].meta["seed"])
```

Compare values computed against the same reference point only. For durable,
resumable experiment matrices, use [studies](https://vamos-optimization.org/guide/studies/).

All functionality lives under one command. Run `vamos help` to list everything:

| Command | What it does |
|---------|-------------|
| `vamos quickstart` | Guided wizard that writes a config |
| `vamos create-problem` | Scaffold a custom problem file |
| `vamos summarize` | Table/JSON summary of recent runs |
| `vamos results` | Inspect or verify one canonical run |
| `vamos reproduce` | Execute an exact same-environment built-in replay |
| `vamos check` | Verify installation and backends |
| `vamos bench` | Benchmark suite across algorithms |
| `vamos studio` | Launch interactive dashboard |
| `vamos tune` | Hyperparameter tuning |
| `vamos profile` | Performance profiling |
| `vamos zoo` | Problem zoo presets |

## Tuning Quick Start

You can use the implemented tuning backends directly from `vamos tune`:
`racing`, `random`, `optuna`, `bohb_optuna`, `smac3`, `bohb`.

Check backend availability in your current environment:

```bash
vamos tune --list-backends
```

Quick verification with the built-in backend and tiny budgets:

```bash
vamos tune --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 --algorithm nsgaii --backend random --smoke --output-dir report/tuning_smoke
```

Note: `racing` and `random` require no extra dependencies. The model-based backends (`optuna`, `bohb_optuna`, `smac3`, `bohb`) require the optional `tuning` extra: `pip install "vamos-optimization[tuning]"`. The `smac3` backend uses the `smac` package.

For paper-grade comparisons against other frameworks, install them yourself (VAMOS depends on none) and record a pinned environment or lockfile alongside reported results.

Recommended robust command (fallback + suite-stratified split):

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend optuna \
  --backend-fallback random \
  --split-strategy suite_stratified \
  --budget 5000 \
  --tune-budget 200 \
  --n-jobs -1
```

Canonical tuning reference: [Hyperparameter tuning](https://vamos-optimization.org/topics/tuning/).

New to hands-on learning? Open the **interactive tutorial notebook**:

```bash
jupyter notebook notebooks/0_basic/05_interactive_tutorial.ipynb
```

## Define Your Own Problem

Use `make_problem()` to turn any Python function into a VAMOS-compatible problem
-- no classes, no protocols, no NumPy vectorization required:

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
    encoding="real",
)

result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
```

Your function receives a single solution `x` (array of length `n_var`) and returns
a list of `n_obj` objective values. When `vectorized=False`, VAMOS adapts that
scalar callable by evaluating one row at a time.

For actual batch performance, pass `vectorized=True` and write a function that
handles `(N, n_var)` batches directly.

Prefer a file template? The CLI wizard scaffolds a ready-to-run `.py` file:

```bash
vamos create-problem
# Prompts for: name, variables, objectives, bounds, style
# Generates a .py file with TODO markers -- fill in your math and run it
```

Or use the experimental **visual builder** in VAMOS Studio -- review your objectives
in the browser, pick an algorithm, and see the Pareto front update on each run:

```bash
vamos studio
# Open the "Problem Builder" tab
```

See [Adding a problem](https://vamos-optimization.org/dev/add_problem/) for all approaches (function, class, or registry).

## VAMOS Assist (no-code workflow)

VAMOS Assist provides an end-to-end no-code flow for creating validated experiment plans, materializing runnable projects, and optionally running smoke checks. You can start with deterministic templates (no API keys), and optionally use provider-backed auto planning.

```bash
vamos assist go "template-first example" --template demo --smoke
```

```bash
pip install vamos-optimization[openai]
setx OPENAI_API_KEY "..."
vamos assist go "..." --mode auto --provider openai --smoke
```

See [VAMOS Assist](https://vamos-optimization.org/assist/) for the full guide (billing, privacy, artifacts, troubleshooting).

Preferred path: start with `optimize(...)`. Use config objects only when you need fully specified, reproducible runs or plugin algorithms.
See [Getting started](https://vamos-optimization.org/guide/getting-started/) for a short decision guide.

Advanced path (explicit config objects):

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
algo = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)

result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=algo,
    max_evaluations=10000,
    seed=42,
    engine="numpy",
)
```

Reminder: plain dict configs are intentionally not accepted (use `GenericAlgorithmConfig` for plugin algorithms).

For comparative evidence against pymoo on fixed seeded cases, install it (`pip install -c constraints/ci.txt pymoo` in a source checkout) and run:

```bash
python tools/benchmark_compare_pymoo.py --output artifacts/performance/pymoo_comparison.json --markdown artifacts/performance/pymoo_comparison.md
```

## Notes

- For reproducible results, set `seed`; NumPy/Numba/MooCore backends share the same RNG-driven stochastic operators.
- Troubleshooting guide: [Troubleshooting](https://vamos-optimization.org/guide/troubleshooting/).
- Algorithm-specific notes (reference directions, operator defaults): [Algorithms & Backends](https://vamos-optimization.org/reference/algorithms/).
- Release packaging smoke checklist: [Release verification](https://vamos-optimization.org/release_smoke/).

## Examples & Notebooks

The repository's [`notebooks/`](https://github.com/vamos-optimization/VAMOS/tree/main/notebooks) folder holds a comprehensive suite of Jupyter notebooks organized by tier:

- **0. Basic**: Essential concepts and API basics.
  - `notebooks/INDEX.ipynb` -- Maintained catalog of the full learning surface
  - `notebooks/0_basic/01_quickstart.ipynb` -- First optimization run
  - `notebooks/0_basic/05_interactive_tutorial.ipynb` -- Guided hands-on walkthrough
  - `notebooks/0_basic/06_optuna_tuning_basics.ipynb` -- Introductory Optuna tuning
- **1. Intermediate**: Real-world problems, constraints, and deeper analysis.
  - `notebooks/1_intermediate/10_discrete_problems.ipynb` -- Binary, integer, and permutation encodings
  - `notebooks/1_intermediate/11_constrained_optimization.ipynb` -- Constraint handling
  - `notebooks/1_intermediate/15_mcdm.ipynb` -- Multi-criteria decision making
  - `notebooks/1_intermediate/16_interactive_explorer.ipynb` -- Interactive Pareto front explorer
  - `notebooks/1_intermediate/19_algorithm_families_beyond_nsgaii.ipynb` -- Current recipes for the other VAMOS MOEAs
- **2. Advanced**: Custom extensions, tuning, and research benchmarks.
  - `notebooks/2_advanced/21_programmatic_tuning.ipynb` -- Canonical Optuna tuning workflow
  - `notebooks/2_advanced/23_backends_and_performance.ipynb` -- Backend tradeoffs and benchmarking
  - `notebooks/2_advanced/30_paper_benchmarking.ipynb` -- Inspect retained benchmark data
  - `notebooks/2_advanced/27_operator_efficacy.ipynb` -- Operator efficacy analysis
  - `notebooks/2_advanced/32_ablation_planning.ipynb` -- Ablation studies
  - `notebooks/2_advanced/33_optuna_tuning_advanced.ipynb` -- Multi-fidelity and persistent Optuna workflows
  - `notebooks/2_advanced/34_extension_workflows.ipynb` -- Custom problem and custom algorithm extension patterns

## Tooling Ecosystem

All tools are available as `vamos <subcommand>`. Run `vamos help` for the full list.

- **`vamos profile`**: Analyze the performance overhead of your experiments.
  ```bash
  vamos profile --problem zdt1 --engines numpy,numba --budget 2000 --output report/profile.csv
  ```
- **`vamos bench`**: Generate full reports comparing multiple algorithms, plus jMetalPy-compatible lab outputs (`summary/lab/QualityIndicatorSummary.csv`, Wilcoxon tables, boxplots). Boxplots require `matplotlib`.
  ```bash
  vamos bench ZDT_small --algorithms nsgaii moead --output report/
  ```
  ```bash
  vamos bench ZDT_small --algorithms nsgaii --output report/ --smoke
  ```
- **`vamos tune`**: You can use the implemented tuners directly from CLI (`racing`, `random`, `optuna`, `bohb_optuna`, `smac3`, `bohb`). `--tune-budget` counts configuration evaluations; `--budget` is per-run evaluations.
  ```bash
  vamos tune --problem zdt1 --algorithm nsgaii --budget 5000 --tune-budget 200 --n-seeds 5
  ```
  - Quick verification path:
    ```bash
    vamos tune --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 --algorithm nsgaii --backend random --smoke --output-dir report/tuning_smoke
    ```
  - Recommended robust invocation (backend fallback + suite-stratified split):

    ```bash
    vamos tune --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 --algorithm nsgaii --backend optuna --backend-fallback random --split-strategy suite_stratified --budget 5000 --tune-budget 200 --n-jobs -1
    ```
- Full tuning reference (canonical docs): [Hyperparameter tuning](https://vamos-optimization.org/topics/tuning/).
- **`vamos check`**: Verify your installation and backend availability.

## Citation

The [retained benchmark data guide](https://github.com/vamos-optimization/VAMOS/blob/main/experiments/REFERENCE_RESULTS.md) documents
the available scientific inputs and their provenance. Manuscript sources and
author revision materials are maintained separately from this public repository.

If you use VAMOS in published work, cite it directly:

```bibtex
@software{vamos_2026,
  title = {VAMOS: Vectorized Architecture for Multiobjective Optimization Studies},
  author = {Rodriguez Uribe, Nicolas and Herr{\'a}n, Alberto and Nebro, Antonio J. and Del Ser, Javier and Colmenar, J. Manuel},
  year = {2026},
  version = {1.2.0},
  url = {https://github.com/vamos-optimization/VAMOS}
}
```

The maintained citation metadata lives in [CITATION.cff](https://github.com/vamos-optimization/VAMOS/blob/main/CITATION.cff).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/vamos-optimization/VAMOS/blob/main/CONTRIBUTING.md) for guidelines.

- **Found a bug?** [Open an issue](https://github.com/vamos-optimization/VAMOS/issues).
- **Want to add an algorithm?** Check [Adding an algorithm](https://vamos-optimization.org/dev/add_algorithm/) in the docs.
- **Using AI tools?** Read [AGENTS.md](https://github.com/vamos-optimization/VAMOS/blob/main/AGENTS.md) for our AI coding standards.
- **Troubleshooting**: [Troubleshooting](https://vamos-optimization.org/guide/troubleshooting/).
- **Security issues**: See [SECURITY.md](https://github.com/vamos-optimization/VAMOS/blob/main/SECURITY.md) for private reporting.
- **Contributors**: See [AUTHORS.md](https://github.com/vamos-optimization/VAMOS/blob/main/AUTHORS.md).

---

**VAMOS** is a research-oriented multi-objective optimization framework.
