# Set up the study-to-report tutorials

T06–T08 form one executable workflow: plan and persist runs, measure their
quality, then create a descriptive report. Each lesson has a standalone script.
The scripts are supplied with these docs and are **not in the older v1.0.1 tag
ZIP**. You do not need a repository clone.

## Environment and downloads

Create an isolated environment using [Installation](../guide/installation.md).
The recorded experiment uses Python 3.12 and the published **VAMOS 1.0.1 wheel**.
Download [study-requirements.txt](../assets/studies/study-requirements.txt), then:

```bash
python -m pip install -r study-requirements.txt
```

The recipe pins VAMOS 1.0.1, NumPy 2.1.3, SciPy 1.15.3, joblib 1.4.2,
MooCore 0.3.2, and Matplotlib 3.10.9. T06 needs only core VAMOS; T07 adds
MooCore for indicators; T08 adds Matplotlib for plots. These optional packages
do not change the optimization backend, which remains NumPy. No notebook,
GPU, service account, or API key is needed.

| Lesson | Standalone download | Input → output |
| --- | --- | --- |
| [T06. Fair study](t06-fair-study.md) | [t06-fair-study.py](../assets/studies/t06-fair-study.py) | Explicit plan → canonical study and runs |
| [T07. Quality and budget](t07-quality-and-budget.md) | [t07-quality-and-budget.py](../assets/studies/t07-quality-and-budget.py) | Saved studies → derived observations |
| [T08. Study report](t08-study-report.md) | [t08-study-report.py](../assets/studies/t08-study-report.py) | Observations → CSV, Markdown, and SVG |

Use a **new output directory** for each command that writes. Scripts refuse to
overwrite existing output. To resume a study, use the canonical
[resume/retry lifecycle](../guide/studies.md#resume-and-retry), then measure it
into a new directory. A report or measurement file never controls study recovery.

## The small workflow

Run these commands in the directory containing the downloads:

```bash
python t06-fair-study.py --output results/tutorial-study
python t06-fair-study.py --output results/tutorial-study --run
python t07-quality-and-budget.py --studies results/tutorial-study --output results/tutorial-metrics
python t08-study-report.py --measurements results/tutorial-metrics/measurements.json --output results/tutorial-report
```

The first command only plans. The second creates and executes eight tasks:
ZDT1/ZDT2 × NSGA-II/MOEA/D × seeds 0/1, each with 200 evaluations, population
20, and 10 variables. The initial total is **1,600 evaluations**. Retries are
separate, explicit attempts and consume additional evaluations.

This validates the workflow. Two repeats and a short budget do not establish
scientific adequacy. T07/T08 execute no optimization.

## The recorded experiment

The [recorded report](recorded-study-report.md) uses a larger, still bounded,
descriptive experiment. Its design was fixed before generating the displayed
observations:

| Factor | Frozen choice |
| --- | --- |
| Problems | ZDT1 and ZDT2; 10 variables; both objectives minimized; no constraints |
| Algorithms | NSGA-II and MOEA/D; VAMOS 1.0.1 resolved defaults, no tuning campaign |
| Population/backend | 40; NumPy; serial evaluation; one study owner |
| Seed schedule | 0 through 9 for every problem/algorithm/budget cell |
| Budgets | 200, 800, 2,000; a separate durable study at each budget |
| Initial cost | 120 tasks; **120,000 evaluations**, counting every fresh run in full |
| Failures | `on_error="continue"`, at most two attempts per task; no automatic retry; incomplete evidence blocks reporting |
| Indicators | HV with fixed point `[1.1, 11.0]`; IGD+ against 201 points sampled uniformly in f1 from each analytical front |
| Summary | Per-cell median, Q1 and Q3 using linear interpolation; no cross-problem raw-HV average |
| Claims | Descriptive observations on these two problems; no p-values or general algorithm ranking |

To recreate the experiment, use three new study directories:

```bash
python t06-fair-study.py --profile recorded --budget 200 --output results/recorded-200 --run
python t06-fair-study.py --profile recorded --budget 800 --output results/recorded-800 --run
python t06-fair-study.py --profile recorded --budget 2000 --output results/recorded-2000 --run
python t07-quality-and-budget.py --studies results/recorded-200 results/recorded-800 results/recorded-2000 --output results/recorded-metrics
python t08-study-report.py --measurements results/recorded-metrics/measurements.json --output results/recorded-report
```

A seed is an RNG initialization label, not proof of a paired design across
algorithms. These budgets are separate fresh runs, not resumed checkpoints or
samples from a saved trajectory. Total work is the **sum** of all budgets.

## Retained evidence and reproducibility

Download [recorded-measurements.json](../assets/studies/recorded-measurements.json)
to rebuild the report without rerunning the experiment. It preserves the small
non-dominated objective sets, scalar observations, every planned task row,
selected attempt/run IDs, source manifest hashes, study designs, metric policy,
package versions, and source revision hash.

This is a **derived documentation snapshot**, not a loadable StudyManifest or
RunManifest. It supports recomputing the reported indicators and summaries;
it does not include decision arrays, full attempt journals, or everything
needed for `reproduce()`. Keep the original study directories for canonical
verification and exact replay. A fresh execution creates new execution evidence
and can differ in runtime or numerical last digits across environments.

The [generation record](../assets/studies/evidence.json) owns the source/output
hashes and size limits. [Maintaining study tutorials](../dev/study-tutorials.md)
explains regeneration. The retained data are pedagogical observations, not a
replacement for a broad benchmark campaign.
