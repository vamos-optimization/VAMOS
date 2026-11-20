# VAMOS 1.5.1 release notes

Released 2026-11-20.

VAMOS 1.5.1 is a patch release of [VAMOS 1.5.0](release-notes-1.5.0.md). It
adds the 20 MODAct problems, which the separately installed `modact` package
evaluates, with the authors' best-known fronts and a command that downloads the
fronts too large to package, and the Foundations learning path in the
documentation. Both are additions: no existing problem, algorithm, default,
configuration or result changes, and runs saved with VAMOS 1.5.0 verify and
replay as before. The stable 1.x API, the omitted-engine NumPy default, and the
1.0.0 run and study schemas remain the compatibility baseline.

## MODAct problems

The 20 problems of Picard and Schiffmann's MODAct suite (IEEE Transactions on
Evolutionary Computation, 2021) design an electro-mechanical actuator, a
stepper motor and three spur-gear stages: 20 variables, 2 to 5 objectives
(cost, torque excess, safety factor, efficiency and transmission ratio) and 7
to 10 constraints. They are registered as `modact_cs1` to `modact_ctsei4`; see
the [MODAct section](../reference/problems.md#modact-actuator-design-evaluated-by-the-modact-package) of the problems catalog.

- **The authors' package evaluates them.** VAMOS includes no MODAct code. It
  imports `modact` (GPL-3.0) when a problem is created; without it, creating
  one raises `ImportError` with the install command,
  `python -m pip install "vamos-optimization[modact]"`. The new `modact` extra
  is left out of `all`, so no blanket install pulls GPL code.
- **VAMOS's conventions.** Objectives that MODAct maximizes change sign, so
  every column of `F` is minimized, and `G = c_weights * g`, so `G <= 0` is
  feasible. Both only multiply by 1 or -1, as pymoo's conversion does.
- **Fronts.** The best-known fronts of CS1–CS4 and CT1–CT4 are packaged as
  `modact_cs1.csv` to `modact_ct4.csv`, under CC BY 4.0, with their attribution
  in `LICENSE-modact-fronts.txt`. `vamos zoo fetch-fronts` (Experimental)
  downloads the other twelve from Zenodo, checks their MD5 checksums, converts
  them in the same way and writes them, with their notice, to a per-user
  directory that the reference-front lookup searches after the packaged fronts.
- **Exact replay.** `verify_run` and `reproduce` compare the recorded versions
  of `modact`, `trimesh` and `python-fcl` for a MODAct run, as they compare
  NumPy's and SciPy's for every run: with another version, or without the
  package, the run is not exactly replayable.

Evaluation is slow: each design builds and checks a geometric model, so a run
of 25 000 evaluations takes 5 to 15 minutes.

## Foundations learning path

A new **Foundations** tab teaches the ideas behind VAMOS in 25 lessons at three
levels: Basic (problems, Pareto dominance, quality aspects and indicators),
Intermediate (NSGA-II step by step, SPEA2, MOEA/D, SMS-EMOA, IBEA and SMPSO)
and Advanced (many objectives, NSGA-III, RVEA, AGE-MOEA, constraints, CCMO,
C-TAEA, ToP, CMOEA-MS and rigorous comparison). Each lesson has interactive
figures that work with the keyboard and keep a text fallback, runnable VAMOS
examples with their output, verbatim excerpts of the source that implements each
step, and references. See the [Foundations overview](../foundations/index.md).

The measurement section of the tutorial
[T07](../tutorials/t07-quality-and-budget.md) is now
headed "Fix the measurement rule: hypervolume and IGD+", which changes its
anchor.

## Which results change

None. The new problems and the per-user front directory add to what VAMOS
resolves: every problem key and reference front that resolved with VAMOS 1.5.0
resolves to the same problem or file, and the replay comparison adds findings
only for runs of a MODAct problem.

## Dependencies

The core dependencies do not change. The new optional `modact` extra installs
`modact>=1.0.1`, which brings trimesh and python-fcl; `all` does not include it.

## Known limitations

See [known limitations](known-limitations.md).
