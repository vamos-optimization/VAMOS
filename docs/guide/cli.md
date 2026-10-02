# CLI and config files

Install the [published package](installation.md) before using these commands.
The examples target VAMOS 1.0.1; use `vamos --help` or `vamos <command> --help`
for the parser in your installed version.

| Command family | Status in 1.0.1 | Use it for |
| --- | --- | --- |
| Main optimization runner | **Stable** core invocation | One run or a configured problem set |
| `results inspect`, `results verify`, `reproduce` | **Stable** | Canonical run inspection, integrity, and exact replay |
| `study plan/create/run/inspect/resume/retry/summarize` | **Stable**, single-owner | Durable experiment matrices |
| `quickstart`, `summarize`, `open-results` | **Experimental** | Interactive onboarding and convenience helpers |
| `bench`, `tune`, `ablation`, `profile`, `zoo`, `studio`, `assist`, `check` | **Experimental** | Optional workflows and diagnostics |

Stable commands have compatibility guarantees under the
[stability policy](../project/stability-and-versioning.md). Experimental
workflows may change in a minor release. Dask, third-party integrations, and
visualization remain Experimental even when selected through the main runner.

Quickstart wizard
-----------------

The **Experimental** wizard writes a config file and executes a single run:

```bash
vamos quickstart
```

If you are new to Python, start with [Minimal Python](minimal-python.md).

List available templates:

```bash
vamos quickstart --template list
```

Run a template non-interactively:

```bash
vamos quickstart --template physics_design --yes --no-plot
```

Skip optional dependency warnings:

```bash
vamos quickstart --no-preflight
```

Template keys (short list):

- `demo`: quick benchmark demo (ZDT1)
- `physics_design`: mixed-variable structural design (welded beam)
- `bio_feature_selection`: real-data feature selection (requires `examples` extra)
- `chem_hyperparam_tuning`: SVM hyperparameter tuning (requires `examples` extra)

The config is saved under `results/quickstart/` and can be re-run with `vamos --config <path>`.

Results helpers
---------------

The **Experimental** convenience helpers summarize or open run folders.
For stable data access, use the canonical commands in the next section.

Summarize recent runs:

```bash
vamos summarize --results results
```

Show only the latest run:

```bash
vamos summarize --latest
```

Open the latest run folder:

```bash
vamos open-results --open
```

## Canonical run commands (Stable)

Inspect one canonical run without materializing arrays:

```bash
vamos results inspect results/ZDT1/nsgaii/numpy/seed_7
vamos results inspect results/ZDT1/nsgaii/numpy/seed_7 --json
```

Fully verify integrity and exact replay compatibility without optimization:

```bash
vamos results verify results/ZDT1/nsgaii/numpy/seed_7
vamos results verify results/ZDT1/nsgaii/numpy/seed_7 --require-level exact
```

Execute a verified same-environment built-in replay as a new canonical run:

```bash
vamos reproduce results/ZDT1/nsgaii/numpy/seed_7
vamos reproduce results/ZDT1/nsgaii/numpy/seed_7 --output results/replays/zdt1-seed-7
```

Add `--json` to any of these commands for one machine-readable stdout
document. Replay never overwrites or modifies its source.

Durable studies
---------------

Resolve a durable study before creating a directory or executing an objective:

```json
{
  "problems": ["zdt1", "zdt2"],
  "algorithms": ["nsgaii"],
  "seeds": [0, 1],
  "max_evaluations": 10000,
  "pop_size": 80
}
```

```bash
vamos study plan study.json
vamos study plan study.json --output studies/comparison-01
vamos study plan study.json --json
vamos study create study.json --output studies/comparison-01
vamos study run studies/comparison-01
vamos study inspect studies/comparison-01 --json
vamos study summarize studies/comparison-01
vamos study summarize studies/comparison-01 --format csv --output artifacts/studies/tasks.csv
```

Planning is read-only: it creates no study, runs no task, and does not reserve
the proposed output. Its `plan_id` and task IDs match later Python
`vamos.create_study(...)` creation from the same `StudySpec`. Every study
command emits exactly one `vamos.study-command-result` version `1.0.0`
document in JSON mode. Creation and execution are separate. `inspect` and an
in-memory `summarize` are read-only; JSON/CSV summary files are written only
when `--output` is explicit and never overwrite an existing path.

Use `vamos study resume STUDY_DIR` for eligible pending/interrupted work and
`vamos study retry STUDY_DIR --failed` for explicit bounded failed-task retry.
Concurrent mutation is unsupported: one process must remain the only mutation
owner for a study. There is no cross-process cancel command; foreground Ctrl+C
uses graceful durable cancellation.

Main runner
-----------

Use `vamos` for single runs and problem sets.

Quick walkthroughs
------------------

Single run (default output under `results/`):

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --population-size 80 --seed 7
```

Python equivalent (preferred for scripting):

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", max_evaluations=5000, pop_size=80, seed=7)
```

Run a predefined continuous problem set with one compatible algorithm:

```bash
vamos --problem-set zdt --algorithm nsgaii --max-evaluations 3000 --seed 7
```

The `families` preset mixes real and permutation problems, so it is not
compatible with every algorithm selected by `--algorithm both`. Check the
[capability matrix](../reference/algorithms.md#capability-matrix) before
combining problem sets and algorithms.

Compare backends on one problem:

```bash
vamos --problem zdt1 --experiment backends --max-evaluations 2000
```

Install optional kernels with
`python -m pip install "vamos-optimization[compute]==1.0.1"`.
The `--experiment backends` comparison skips unavailable optional kernels with
a warning. An explicit single-run `--engine numba` or `--engine moocore`
instead fails if that dependency is missing; it never switches silently.

Multiprocessing evaluation for expensive problems:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 8000 --eval-strategy multiprocessing --n-workers 4
```

Enable Experimental live visualization and save plots (requires the
[`analysis` extra](installation.md#optional-extras)):

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 2000 --live-viz --plot
```

Early stop when hypervolume reaches a target fraction:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 15000 --hv-threshold 0.9
```

Include Experimental third-party baselines (ZDT1 example; requires the
[`research` extra](installation.md#optional-extras)):

```bash
vamos --problem zdt1 --algorithm both --include-external --external-problem-source native
```

Walkthrough: run and inspect outputs
------------------------------------

1) Run a single optimization:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --population-size 80 --seed 7
```

2) Inspect the canonical artifact under `results/` (default):

- `manifest.json`: requested/resolved configuration, actual seed, outcome, provenance, and hashes
- `result.npz`: objective, decision, constraint, population, and archive arrays
- `environment.json`: bounded runtime environment details

3) Save plots as presentation output outside the canonical run leaf:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --population-size 80 --seed 7 --plot
```

Key flags
---------

- `--algorithm`: agemoea, ibea, moead, nsgaii, nsgaiii, rvea, smpso, smsemoa, spea2, both, or Experimental external baselines (pymoo_nsga2, jmetalpy_nsga2, pygmo_nsga2)
- `--engine`: numpy | numba | moocore | auto. The deterministic default is `numpy`; use `auto` when you want heuristic backend selection.
- `--problem`: any registry key (see [Problems](../reference/problems.md))
- `--problem-set`: predefined sets (e.g., `families`)
- `--validate-config`: validate `--config` and exit
- `--output-root`: directory for run artifacts (default: `results/`)
- `--no-preflight`: skip optional dependency warnings
- `--population-size`, `--offspring-population-size`
- `--max-evaluations`
- `--hv-threshold` and `--hv-reference-front`
- `--selection-pressure`, `--external-archive-size`
- `--eval-strategy`: serial | multiprocessing (with `--n-workers`) | dask (Experimental; with `--dask-address`, see [Dask](../scaling/dask.md))
- `--live-viz` with `--live-viz-interval`, `--live-viz-max-points`
- `--plot`: save Pareto front plots after runs
- Variation overrides per algorithm (examples):
  - `--nsgaii-crossover sbx --nsgaii-crossover-prob 1.0 --nsgaii-mutation polynomial --nsgaii-mutation-prob 1/n`
  - `--moead-crossover sbx --moead-mutation polynomial --moead-aggregation pbi`
  - `--smsemoa-mutation polynomial --nsga3-crossover sbx`

Config files (YAML/JSON)
------------------------

Use `--config path/to/spec.yaml`; CLI flags override file values.

```yaml
version: "1"
defaults:
  title: My run
  algorithm: moead
  engine: numpy
  population_size: 120
  max_evaluations: 20000
  hv_threshold: 0.8
  moead:
    crossover: {method: sbx, prob: 1.0, eta: 20}
    mutation: {method: pm, prob: "1/n", eta: 20}
problems:
  bin_knapsack:
    algorithm: nsgaii
    n_var: 30
    population_size: 150
    nsgaii:
      crossover: {method: uniform}
      mutation: {method: bitflip, prob: "1/n"}
```

Validate a config without running:

```bash
vamos --config configs/experiment.yaml --validate-config
```

Run a config with a CLI override:

```bash
vamos --config configs/experiment.yaml --algorithm smsemoa --max-evaluations 10000
```

Other subcommands
-----------------

The following commands are **Experimental**. Run `vamos help` for the full list.
Install any required [optional extras](installation.md#optional-extras) first.

- Self-check: `vamos check`
- Benchmarking: `vamos bench --list` and `vamos bench ZDT_small --algorithms nsgaii moead --output report/`
- Fast benchmark verification: `vamos bench ZDT_small --algorithms nsgaii --output report/ --smoke`
- Tuning: `vamos tune --instances zdt1,zdt2,zdt3 --algorithm nsgaii --backend optuna --backend-fallback random --split-strategy suite_stratified --budget 5000 --tune-budget 200 --n-jobs -1`
- Ablation plans: `vamos ablation --config configs/ablation.yaml`
- Profiling: `vamos profile --problem zdt1 --engines numpy,numba --budget 2000 --output report/profile.csv`
- Problem zoo: `vamos zoo list`, `vamos zoo info zdt1`, `vamos zoo run zdt1 --algorithm nsgaii --budget 3000`
- Studio (interactive, needs `studio` extra): read the [Studio walkthrough and launcher limitation](studio.md) before starting.

Tuning quick notes (`vamos tune`)
---------------------------------

Use this guide for quick usage. For the complete, maintained `tune` reference
(all backends, split/fallback behavior, finisher/validation/test, and artifact
contracts), see:

- [Hyperparameter tuning](../topics/tuning.md)

Recommended robust invocation:

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

Quick verification path (built-in backend, tiny budgets):

```bash
vamos tune --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 --algorithm nsgaii --backend random --smoke --output-dir results/tuning_smoke
```

Ablation config example
-----------------------

```yaml
algorithm: nsgaii
engine: numpy
output_root: results/ablation_demo
default_max_evals: 2000
problems: [zdt1]
seeds: [1, 2, 3]
base_config:
  population_size: 60
  offspring_population_size: 60
variants:
  - name: baseline
summary_dir: results/ablation_demo/summary
```

The CLI writes a summary CSV by default to `<output_root>/summary/ablation_metrics.csv` (override with `summary_path` or `summary_dir`).

## Experimental graphical interface

`vamos gui [RESULTS_ROOT]` starts a local NiceGUI interface to explore canonical
runs and to launch built-in runs with live progress. It requires the optional
`gui` extra (`pip install "vamos-optimization[gui]"`) and binds to loopback by
default. See [VAMOS GUI](gui.md).
