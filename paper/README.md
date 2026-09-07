# VAMOS paper: reviewer guide

The manuscript was submitted to **Neurocomputing**. This directory contains the
article and supplementary sources, analysis scripts and links to the retained
numerical inputs. The official repository is
<https://github.com/vamos-optimization/VAMOS>.

## Start here

All commands below run from the repository root. Download or clone the repository
to inspect data; install VAMOS only for the small optimization example. No Overleaf
account is needed for any reviewer command.

| Task | Entry point | What to expect |
|---|---|---|
| Read the article | [Main manuscript](manuscript/main.tex) and [separate supplement](manuscript/supplementary.tex) | Primary results in the article; detailed analyses in the supplement |
| Browse the measurements | [Data inventory and provenance](../experiments/REFERENCE_RESULTS.md#reading-the-data) | CSV links, column meanings, coverage and experimental field conventions |
| Inspect data without installing packages | `python paper/review.py data` | Row counts, coverage and data notes; normally a few seconds |
| Run a small example | `python paper/review.py smoke` | A 200-evaluation run, saved artifact, verification and exact replay |
| Recompute supplementary analyses | `python paper/review.py rebuild` | Tables, confidence intervals and plots from existing CSVs; no new benchmark campaign |
| Build article and supplement PDFs | `python paper/review.py rebuild --pdf` | Four separate documents; requires the analysis environment and LaTeX |
| Learn the library | [Getting started](../docs/guide/getting-started.md), [examples](../examples/), [CLI guide](../docs/guide/cli.md) | Supported public APIs and commands |

The CSVs record the experimental development build described in the manuscript.
Running current VAMOS checks the current implementation. It does **not** establish
that historical timings were measured with the public 1.0.0 release. Recomputing
statistics from retained observations and repeating optimization experiments are
separate operations.

## 1. Inspect and export the data

Python 3.12 is the common environment for all paths below. Data inspection itself
uses only the standard library and also supports Python 3.10 or newer.
Check `python --version` before creating the publication environment below.

```bash
python paper/review.py data
python paper/review.py data --output-dir paper/generated/reviewer-data
```

The second command creates:

- `data-report.json`: file hashes, columns, row counts, frameworks, seeds,
  evaluation budgets and metadata notes for the twelve retained experiment CSVs.
- `summary.csv`: per-condition medians and observation counts, readable in a
  spreadsheet. Evaluation budgets, problem dimensions and algorithms remain
  separate. This is a descriptive summary, not a replacement for inferential tests.

The command requires all twelve CSVs and rejects malformed numerical data or
duplicate observation keys. It reports evaluation budgets and empty fields,
and includes a recorded-value count for each metric. In the scaling experiment,
hypervolume calculation is disabled for the 40 objective-count observations;
the JIT timing-policy label is inapplicable to the 95 NumPy observations. Runtime
is recorded for all 220 scaling observations. The steady-state CSV also includes
one 200-evaluation observation, which is summarized separately from the
50,000-evaluation campaign. See the
[data notes](../experiments/REFERENCE_RESULTS.md#experimental-field-conventions-and-analysis-scope)
for the exact conditions and aggregation rules.
Hashes identify the local inputs; they are not an independent authenticity check.

## 2. Run a small, reproducible example

Use a Git clone for this path: exact replay records the source revision. A ZIP
download suffices for data inspection and analysis rebuilds, but does not supply
that Git evidence. With Git installed:

```bash
git clone https://github.com/vamos-optimization/VAMOS.git
cd VAMOS
```

Create and activate an environment, then install the checkout:

```bash
python -m venv .venv-review
```

PowerShell:

```powershell
.\.venv-review\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path src).Path
```

Linux/macOS:

```bash
source .venv-review/bin/activate
export PYTHONPATH="$(pwd)/src"
```

```bash
python -m pip install -e .
python -c "import pathlib, vamos; print(pathlib.Path(vamos.__file__).resolve())"
python paper/review.py smoke
```

The printed module path must be inside this checkout's `src/`. The example runs
NSGA-II on ZDT1 using NumPy, population size 40, seed 42 and 200 evaluations. It
saves and reloads the result, verifies the artifact, and executes an exact replay
in the same environment. Success ends with `PASS`; this is a functionality check,
not a timing or statistical-quality benchmark.

Outputs are in `paper/generated/reviewer-smoke/`: `run/`, `replay/`,
`verification.json`, `smoke-report.json` and the data reports. The run directories use the current
[run artifact contract](../docs/dev/run_artifacts_and_replay.md). For example:

```bash
vamos results inspect paper/generated/reviewer-smoke/run --json
vamos results verify paper/generated/reviewer-smoke/run --require-level exact --json
```

Every reviewer command that writes files requires a **new directory**. To repeat
it, choose another destination, for example `--output-dir
paper/generated/reviewer-smoke-2`. Existing outputs are never replaced.

For the automated reviewer and documentation tests, install the development
environment specified in [Testing and validation](../docs/dev/testing.md), then:

```bash
python -m pytest -q tests/docs/test_paper_review.py tests/docs/test_publication_script_outputs.py tests/docs/test_docs_smoke.py
```

## 3. Rebuild analyses and PDFs from retained measurements

Use a separate Python 3.12 environment for the pinned publication dependencies;
follow the activation and import-path steps above, using `.venv-paper` as its name.

```bash
python -m venv .venv-paper
```

After activating `.venv-paper`:

```bash
python -m pip install -e .
python -m pip install -r paper/requirements-publication.txt
python -m pip check
python paper/review.py rebuild
```

This typically takes minutes, depending on the machine. It runs the maintained
statistical analysis for NSGA-II, SMS-EMOA and MOEA/D, then the convergence, memory,
runtime heatmap, forest-interval and accessibility generators. It uses a fresh
source-and-data copy under `paper/generated/reviewer-rebuild/workspace/`.
Existing local generated figures are not used as inputs. The checkout's CSVs,
TeX sources and red editorial markings are preserved.

The output directory contains input hashes, per-command logs, descriptive data
reports and `rebuild-report.json` with Python/package versions and output paths.
Generated tables and images are in `workspace/paper/generated/`; statistical CSVs
are in its `data/` subdirectory. These are **recomputed analyses**. The primary
article's embedded tables remain the submitted values, so compare the regenerated
results with the submitted manuscript before adopting an editorial update.

To also compile the documents, install a TeX distribution providing `latexmk`,
`pdflatex`, BibTeX, `elsarticle` and the packages imported by the TeX sources. Then:

```bash
python paper/review.py rebuild --pdf --output-dir paper/build/reviewer-pdfs
```

Open `paper/build/reviewer-pdfs/pdf/main/main.pdf` and
`paper/build/reviewer-pdfs/pdf/supplementary/supplementary.pdf`. The same directory
contains `manuscript_anonymous/manuscript_anonymous.pdf` and
`title_page/title_page.pdf` as editorial alternatives. This path invokes LaTeX
locally and performs no Overleaf synchronization.

## Manuscript organization

The [main article](manuscript/main.tex) contains the framework, experimental
protocol, primary runtime comparisons, summary quality results and conclusions.
The [supplement](manuscript/supplementary.tex) contains convergence plots, detailed
quality and statistical results, additional runtime summaries, memory measurements,
the accessibility protocol and NSGA-II parameter spaces for the comparison
frameworks. Its sections, tables, figures and equations have independent S-prefixed
numbering. Neither document imports the other.

Editorial changes are marked red with `\revision{...}`, defined in
[revision_marks.tex](manuscript/revision_marks.tex). Include that file when moving
sources to another compilation environment. It also marks the changed
supplementary tables and the VecMetaPy and PlatEMO references.

[Highlights](highlights.txt), the [graphical abstract](graphical_abstract.md),
[anonymous manuscript](manuscript/manuscript_anonymous.tex) and
[title page](manuscript/title_page.tex) are separate submission materials, not
additional sections of the main article. The frozen [Pareto-front example data](manuscript/scripts/pareto_front_plots/README.md)
remain available for inspection even though their figure block is currently
commented out of the main manuscript.

The [response letter](manuscript/response_letter.tex) addresses the reviewers of
Neurocomputing submission NEUCOM-D-26-13650. It is a separate editorial document,
with reviewer comments in black and author responses in red. Compile it from
`paper/manuscript/` with `latexmk -pdf response_letter.tex`; it is independent of
the four-document reviewer rebuild.

## Author maintenance and full experiment campaigns

The numbered scripts remain available for author work. Some write into the
manuscript or retained CSVs; use a disposable checkout for a new campaign and
record its command, environment and source revision. Long cross-framework runs
are not needed for the reviewer paths above.

| Script | Purpose and effects |
|---|---|
| [01](01_run_paper_benchmark.py) | Cross-framework optimization campaign; writes benchmark CSVs; controlled by the environment variables documented in the script |
| [04](04_update_paper_tables_from_csv.py) | Regenerates runtime/quality tables and edits `main.tex`; `--main-tex` selects a working copy |
| [05](05_run_statistical_tests.py) | Statistical analysis; set `VAMOS_PAPER_UPDATE_MAIN_TEX=0` to export tables without editing the article; the reviewer helper does this automatically |
| [14](14_update_frameworks_perf_variant_tables_from_csv.py) | Regenerates variant tables and also edits the primary article |
| [03](03_run_scaling_experiment.py), [18](18_run_convergence_experiment.py), [23](23_run_memory_benchmark.py), [31](31_run_zcat_all_tables.py) | New scaling, convergence, memory and ZCAT campaigns; write retained-data paths |
| [30](30_plot_scaling.py), [32](32_run_pareto_front_variants.py) | Scaling plot from CSV; new Pareto-front variant runs and plots, respectively |
| [08](08_compile_manuscript_pdf.py) | Author build of the current main article; use `--no-sync` for local compilation because its default also synchronizes sources to Overleaf |
| [11](11_sync_overleaf_sources.py), [15](15_sync_overleaf_to_local_sources.py) | Author-only source upload/download; require Overleaf credentials; only the download script (15) supports `--dry-run` |

Table-update scripts require the experiment CSVs; they do not provide an
`--empty` option. Generated files belong under
ignored `paper/generated/`, `paper/build/` or `paper/dist/`; final submission
packages belong in the intended external archive or release attachment. See the
[artifact contract](../experiments/ARTIFACT_CONTRACT.md) and
[repository hygiene policy](../docs/dev/repository_hygiene.md).

New durable optimization campaigns use the [study lifecycle](../docs/dev/studies.md).
The retained benchmark CSVs are scientific observations, not RunManifest or
StudyManifest directories and not resume authority for current studies.
