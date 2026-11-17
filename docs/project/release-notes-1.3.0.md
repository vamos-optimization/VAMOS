# VAMOS 1.3.0 release notes

Released 2026-11-16.

VAMOS 1.3.0 is a minor release of [VAMOS 1.2.1](release-notes-1.2.1.md). It
requires Python 3.11 or newer, as VAMOS 1.2.0 announced. It corrects operators
and settings that departed from their published definitions, which changes the
results of affected experiments: read [which results change](#which-results-change)
before you compare new runs with earlier ones. It rebuilds the built-in tuning
spaces and racing's statistics, gives every algorithm one live-callback
contract, makes durable studies robust to interruptions, and adds a wall-clock
time limit, problem options by name, jMetal's reference fronts and MOEA/D
configurations as options, a final subset of the external archive, and an
access token for VAMOS Studio. The stable 1.x API, the omitted-engine NumPy
default, and the 1.0.0 run and study schemas remain the compatibility baseline.

## Which results change

A built-in component that departs from its published definition is a defect,
and its fix ships even when it changes results; see
[scientific corrections](stability-and-versioning.md#scientific-corrections).
The [changelog](https://github.com/vamos-optimization/VAMOS/blob/main/CHANGELOG.md)
lists each change with its pull request. Rerun the experiments that use any of
the following before comparing them with results of earlier versions.

| Area | Change | Rerun |
| --- | --- | --- |
| Non-uniform mutation | `non_uniform` is Michalewicz's mutation, as in jMetal: a variable moves towards a random bound by a fraction of its distance to it and never leaves the bounds. | Runs with `mutation="non_uniform"`. |
| Gaussian and Cauchy mutations | `sigma` and `gamma` are fractions of each variable's range, as in pymoo, and `gaussian` defaults to `sigma = 0.1`. On a range of 1 the draws are the same. | Runs with `gaussian` or `cauchy` on variables whose range is not 1. |
| Integer and mixed crossovers | The integer SBX rounds instead of truncating towards zero, the integer arithmetic crossover and the real part of the mixed crossover give two complementary children instead of the parents' mean, and `blx_alpha_beta` defaults to `alpha = beta = 0.5`. | Mixed problems with the default crossover, integer runs with SBX or `arithmetic`, and `blx_alpha_beta` without explicit parameters. |
| `mutation_prob_factor` | MOEA/D, NSGA-III, SMS-EMOA, SPEA2, AGE-MOEA and RVEA apply it, as NSGA-II and IBEA did. | Runs of those six algorithms that set it. |
| Command-line NSGA-III | `vamos --algorithm nsgaiii` and configuration files use SBX with distribution index 30, as `optimize` and Deb and Jain (2014) do. | Command-line NSGA-III runs. |
| Tuning spaces and racing | The spaces drop dimensions that never reached the algorithm, take Evolver's names and ranges, and add choices; racing eliminates with F-race. | Seeded tuning runs draw other configurations; stored assignments build the same configurations. |
| Live callbacks | Every algorithm sends the same payload, the whole population and `stats["evals"]`, and MOEA/D reports one generation per `pop_size` evaluations instead of one per evaluation. | Code that reads live-callback payloads. |
| Archive and monitor streams | An external archive's `ref_dirs` pruning and the HV convergence monitor's bootstrap derive their streams from the run seed, unless `rng_seed` is set. | Runs that use them without an explicit seed. |

Runs with explicit settings that none of these rows names, and replays of stored
runs, give the same results as before.

## New options

- `optimize(termination=("max_time", seconds))` stops a run on wall-clock time.
  Such a run is recorded as not reproducible.
- A registered problem takes the options its factory declares by name, through
  `optimize(name, problem_kwargs=...)`, and replay rebuilds the same problem.
- `MOEADConfig.default(variant="de")` gives jMetal's MOEA/D-DE and
  `variant="sbx"` its MOEA/D with SBX; without `variant` the default does not
  change. VAMOS 2.0 will make `"de"` the default.
- `SMSEMOAConfig.builder().reference_point(rule="jmetal")` measures the
  hypervolume contributions from jMetal's reference point.
- jMetal's reference fronts of ZDT and DTLZ are packaged as an optional source,
  `vamos --hv-reference-front jmetal`; VAMOS's own fronts stay the default.
  DTLZ1–DTLZ7 also gain two-objective reference fronts.
- `ExternalArchiveConfig.final_selection` and `final_size` report a subset of
  the external archive when the run ends, as jMetal's and Evolver's results do,
  while the saved run keeps the whole archive. A saved SPEA2 run whose result
  does not hold its external archive, as with a final subset or
  `result_mode="population"`, now stores that archive too; VAMOS 1.2.1 cannot
  read such a run.
- Every algorithm with an external archive takes `live_callback_mode`, and
  `population_archive` sends the archive to a live view, which ends with what
  the run returns.
- `select_top_k_farthest(..., normalize=True)` makes the farthest-point subset
  independent of the objectives' units.
- MOEA/D's resolved specification records where its weight vectors come from,
  and `vamos tune` records the population a MOEA/D trial ran. Its results do
  not change, but a MOEA/D run stored by an earlier version, which lacks that
  record, is no longer verified as exact.

## Studies

- A Ctrl+C or a Studio Stop no longer leaves a study that cannot be loaded.
- `load_study(path, verify="metadata")` inspects a study whose runs are damaged,
  and `run_limits=` and larger `StudyLoadLimits` apply to every later reload.
- After Ctrl+C, the CLI no longer suggests resuming a cancelled study.

## VAMOS Studio

`vamos studio` requires the access token of its launch: it prints a link with
the token, and the browser keeps a cookie. With a non-loopback address, Studio
no longer offers the pages that run code, unless
`--allow-remote-code-execution` is given.

## Deprecations

Stable configuration fields that have no effect emit a `FutureWarning` when they
are set and are planned for removal in VAMOS 2.0; the
[changelog](https://github.com/vamos-optimization/VAMOS/blob/main/CHANGELOG.md)
lists them.

VAMOS 1.4.0 removes three Experimental features that depend on LLM providers:
VAMOS Assist with the `vamos assist` command, Studio's Problem
Builder, and the `openai`, `anthropic` and `gemini` extras. Define a problem
with `make_problem` or `vamos create-problem`, and run it from Python, the
command line or Studio.

## Python support

VAMOS 1.3.0 requires Python 3.11 or newer and is tested on Python 3.11 to 3.14.
Python 3.10 reached its end of life in October 2026; VAMOS 1.2.1 is the last
release that installs on it. The declared minimum of NumPy rises to 1.23.2, the
first NumPy release for Python 3.11, and VAMOS no longer depends on
typing-extensions, which only Python 3.10 needed.

## Known limitations

In the mixed-encoding tuning spaces of MOEA/D, NSGA-III, SMS-EMOA and SPEA2, the
per-type segment settings do not reach the run, so tuning them has no effect.
The next minor release fixes them with the grammar-based spaces. See
[known limitations](known-limitations.md) for the rest.
