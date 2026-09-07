# VAMOS: Vectorized Architecture for Multiobjective Optimization Studies

> **A high-performance, unified framework for Multi-Objective Evolutionary Algorithms (MOEA) in Python.**

![VAMOS Banner](docs/assets/vamos1.png)

Official repository: [vamos-optimization/VAMOS](https://github.com/vamos-optimization/VAMOS).
Issues, pull requests, releases, and development are coordinated there. See
[repository governance](docs/project/repository-governance.md).

VAMOS bridges the gap between simple research scripts and large-scale optimization studies. It provides a unified API for running state-of-the-art algorithms across diverse problems, backed by vectorized kernels with NumPy as the exact reference path and optional Numba acceleration for core kernels.

VAMOS 1.0.0 is the first official public release and compatibility baseline.
Earlier version strings and Git tags were internal pre-public development
markers, not prior public releases. See the
[stability policy](docs/project/stability-and-versioning.md) and
[known limitations](docs/project/known-limitations.md).

## Key Features

- **Unified API**: A clear, fluent interface `vamos.optimize()` for all workflows.
- **Battle-Tested Algorithms**: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA.
- **Unified Archiving**: Consistent external archive configuration via `.external_archive(capacity=..., pruning=...)`, with bounded or unbounded archives and pruning policies `crowding`, `hv`, `mc_hv`, `knn`, `maxmin`, and `ref_dirs`. When an external archive is enabled, top-level results come from it by default unless `result_mode="population"` is requested.
- **Multi-Fidelity Tuning**: Hyperband-style racing with warm-start checkpoints for sample-efficient algorithm configuration.
- **Ready-to-use Tuning Backends**: `racing` and `random` work out of the box; install the optional `tuning` extra to enable `optuna`, `bohb_optuna`, `smac3`, and `bohb` via `vamos tune`.
- **Performance Driven**: Vectorized NumPy kernels with optional Numba JIT acceleration for core kernels.
- **Interactive Analysis**: Built-in dashboards with `explore_result_front(result)` and publication-ready LaTeX tables.
- **Visual Problem Builder**: Define custom problems in the experimental VAMOS Studio; local Python preview requires an explicit trusted-code opt-in.
- **Extensible**: Standardized protocols for adding custom problems, operators, and algorithms.

Canonical customization guide: `docs/topics/extending.md`.

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
pip install "vamos-optimization[compute,research,analysis]"
```

For development and extras (Linux / macOS):

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core + essential extras
pip install "vamos-optimization[compute,research,analysis]"
```

Optional model-based tuning backends (`optuna`, `smac3`, `bohb`):

```bash
pip install "vamos-optimization[tuning]"
```

`smac3` in VAMOS is provided by the PyPI package `smac` (SMAC3):

```bash
pip install "smac>=2.0"
```

For publication benchmarking, prefer the pinned paper environment:

```bash
pip install -e .
pip install -r paper/requirements-publication.txt
```

## Backend Capability Matrix

| Backend | Status | Role |
|---------|--------|------|
| `numpy` | Stable | Exact reference backend and deterministic default. |
| `numba` | Stable optional | Accelerates core numeric kernels such as variation, tournament selection, and MOEA/D neighborhood updates. |
| `moocore` | Stable optional | Adds accelerated quality-indicator support, especially hypervolume-style metrics. |

Operator shorthand accepts either numeric probabilities or the string literal `"1/n"` for per-variable mutation rates.

## API Contracts

- `max_evaluations` is a strict evaluation budget. It must be at least the resolved population size, and the final generation is truncated when needed so budget-terminated runs report exactly `result.data["evaluations"] == max_evaluations`.
- Public algorithm results expose `evaluations` as the evaluation-count metric. Internal checkpoints may still store `n_eval`.
- Hypervolume utilities require the reference point to dominate all points by default. Pass `allow_ref_expand=True` only when automatic reference-point expansion is intended.
- NSGA-III, RVEA, and MOEA/D require `pop_size` to match their configured reference directions or weight lattice; incompatible configurations fail early with an actionable error.

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

New to Python? Start with the Minimal Python Track: `docs/guide/minimal-python.md`.

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

For a small study in one call:

```python
study = optimize("zdt1", algorithm="nsgaii", max_evaluations=4000, seed=[0, 1, 2])
print(study.mean("evaluations"))
print(study.best_run("evaluations").meta["seed"])
```

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

For paper-grade comparisons against external frameworks, do not rely on the open-ended `research` extra alone. Record a pinned environment or lockfile alongside reported results.

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

Canonical tuning reference: `docs/topics/tuning.md`.

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

See `docs/dev/add_problem.md` for all approaches (function, class, or registry).

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

See `docs/assist.md` for the full guide (billing, privacy, artifacts, troubleshooting).

Preferred path: start with `optimize(...)`. Use config objects only when you need fully specified, reproducible runs or plugin algorithms.
See `docs/guide/getting-started.md` for a short decision guide.

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

For comparative evidence against pymoo on fixed seeded cases, run:

```bash
python tools/benchmark_compare_pymoo.py --output artifacts/performance/pymoo_comparison.json --markdown artifacts/performance/pymoo_comparison.md
```

## Notes

- For reproducible results, set `seed`; NumPy/Numba/MooCore backends share the same RNG-driven stochastic operators.
- Troubleshooting guide: `docs/guide/troubleshooting.md`.
- Algorithm-specific notes (reference directions, operator defaults): `docs/reference/algorithms.md`.
- Release packaging smoke checklist: `docs/release_smoke.md`.

## Examples & Notebooks

VAMOS comes with a comprehensive suite of Jupyter notebooks organized by tier:

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
  - `notebooks/2_advanced/30_paper_benchmarking.ipynb` -- Publication-ready benchmarks
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
- Full tuning reference (canonical docs): `docs/topics/tuning.md`.
- **`vamos check`**: Verify your installation and backend availability.

## Citation

Reviewing the Neurocomputing manuscript? The [paper reviewer guide](paper/README.md)
links the retained CSV data, a small executable check, isolated analysis/PDF
rebuilds and the relevant documentation.

If you use VAMOS in published work, cite it directly:

```bibtex
@software{vamos_2026,
  title = {VAMOS: Vectorized Architecture for Multiobjective Optimization Studies},
  author = {Rodriguez Uribe, Nicolas and Herr{\'a}n, Alberto and Nebro, Antonio J. and Del Ser, Javier and Colmenar, J. Manuel},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/vamos-optimization/VAMOS}
}
```

The maintained citation metadata lives in `CITATION.cff`.

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

- **Found a bug?** Open an issue.
- **Want to add an algorithm?** Check `dev/add_algorithm.md` in the docs.
- **Using AI tools?** Read `.agent/docs/AGENTS.md` for our AI coding standards.
- **Troubleshooting**: `docs/guide/troubleshooting.md`.
- **Security issues**: See `SECURITY.md` for private reporting.
- **Contributors**: See `AUTHORS.md`.

---

**VAMOS** is a research-oriented multi-objective optimization framework.
