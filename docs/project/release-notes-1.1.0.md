# VAMOS 1.1.0 release notes

Released 2026-10-22.

VAMOS 1.1.0 fixes constraint handling and IBEA's hypervolume indicator, both of
which could produce scientifically invalid results, and expands the
experimental Studio workflow. The stable 1.x API,
omitted-engine NumPy default, Python 3.10 support, and 1.0.0 run and study
schemas remain the compatibility baseline.

## Constraint corrections

- `cec2009_cf1` and `welded_beam` now declare the constraints they emit, so
  registry-created problems enforce them. CF1 uses `[0, 1]^n` and negates the
  original CEC2009 constraint to follow the VAMOS `G <= 0` convention.
- IBEA removes infeasible solutions before feasible ones, in decreasing order
  of violation. Its previous penalty had the wrong sign.
- NSGA-III uses feasibility-first survival and parent selection. Its ask/tell
  path rejects missing or misaligned offspring constraints without consuming
  the pending offspring, so the caller can correct and retry them.
- Non-dominated result selection respects feasibility across the built-in
  algorithms. When no feasible solution exists, the least violated candidates
  remain available for inspection.
- SPEA2 accepts `constraint_mode="none"` on constrained problems without the
  previous archive shape failure. All algorithms reject unsupported constraint
  modes; only `"feasibility"` and `"none"` are implemented.

**Scientific impact:** affected constrained runs can differ from 1.0.2 even
with the same seed. Reassess conclusions based on those runs and rerun affected
experiments. Unconstrained regression and replay checks are retained. Exact
replay requires a matching material environment; keep the original environment
when reproducing historical artifacts.

## IBEA indicator corrections

- With `indicator="hypervolume"`, IBEA used the hypervolume-difference
  indicator with the opposite orientation and kept dominated solutions; on ZDT1
  and DTLZ2 its final populations never reached the reference box. It now uses
  Zitzler and Künzli's `I_HD` with their adaptive scaling, as jMetal does, and
  its final fronts match jMetal 7.9's IBEA on ZDT1, DTLZ2 and DTLZ1.
- An exponential indicator contribution beyond the float64 range counted as
  zero and made a far-dominated solution look best. Selection then works in the
  log domain and recomputes each fitness after every removal, so overflow no
  longer changes which solutions survive.

**Scientific impact:** discard hypervolume-based IBEA results from earlier
versions and rerun them. Epsilon-based IBEA runs without overflow are unchanged;
runs whose objectives differed by more than about `709 * kappa` can change.

## Tuning conditional parameters

The experimental model-based tuning backends `optuna`, `bohb_optuna`, `smac3`
and `bohb` now honor the conditions of the tuning space, and `random` and
`racing` also leave out descendants of inactive parameters. Trials contain only
active parameters, Optuna suggests parents first, and SMAC3 and BOHB receive
native ConfigSpace conditions. Unsupported conditions, unknown parents and
dependency cycles fail before sampling. `mutation_eta` is active only for polynomial mutations in real-valued
operator spaces. Native BOHB no longer fails on an unsupported seed argument,
and a single-worker native BOHB search is reproducible for a seed.

**Reproducibility:** inactive parameters never changed the candidate
algorithm, so scores of persisted Optuna studies stay comparable and keep their
namespace. Seeded `random` and `racing` searches propose the same candidates as
1.0.2, except for MOEA/D, whose `mutation_eta` moved into the
polynomial-mutation block. See [conditional parameters](../topics/tuning.md#conditional-parameters).

## Studio

- The launcher serves the Panel application and honors `--study-dir`.
- Run Experiment and Problem Builder apply the selected algorithm settings,
  run in the background, report live progress, and support stopping.
- Batch Comparison creates persistent StudyManifest studies with planning,
  progress, stop/resume, retry, and access to stored results.
- Statistical Comparison adds normalized HV and available IGD+, corrected
  pairwise tests, effect sizes, ranks, plots, and exports.
- The problem and algorithm catalogs, environment panel, feasible/objective
  filters, and run-again workflow support inspecting and repeating experiments.
- Stop requests target a unique worker token. The matching worker exits itself;
  Studio does not signal a possibly reused process ID or confirm a stop without
  the worker's acknowledgment after file cleanup. Interrupted atomic writes
  do not strand temporary attempt files that prevent resume. Wildcard remote binding accepts the computer's
  host names and addresses; proxies and aliases have an explicit origin option. Final generations are
  recorded once in the live history.

Studio remains experimental and intended for trusted local use. Customized
stored configurations may require `vamos reproduce` instead of the run-again
form. Stopping a batch loses the in-progress run; resume reconciles it.

## Automatic backend selection and dependencies

Explicit `engine="auto"` prefers loadable, JIT-enabled Numba for all nine
built-in algorithms, with NumPy fallback before execution. Omitting `engine`
still selects NumPy. Explicit engine requests and persisted resolved engines
never silently substitute. Numba is optional and compilation can dominate
short runs.

The reviewed dependency set uses mypy 2.4.0 and preserves the compatible
PyMdown/mkdocstrings pair on Python 3.10 while using PyMdown 12.3 on Python 3.11+.

## Installation and remaining scope

```bash
python -m pip install --upgrade "vamos-optimization==1.1.0"
python -m pip install --upgrade "vamos-optimization[studio,compute]==1.1.0"
```

The welded-beam variant, UF3 domain, independent algorithm validation,
full-source typing debt and multi-worker study coordination remain documented
limitations. This release does not claim those research goals are complete.
See [known limitations](known-limitations.md),
[stability and versioning](stability-and-versioning.md),
[the 1.0.2 release notes](release-notes-1.0.2.md), and
[release verification](../release_smoke.md).
