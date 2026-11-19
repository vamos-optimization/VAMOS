# VAMOS 1.5.0 release notes

Released 2026-11-19.

VAMOS 1.5.0 is a minor release of [VAMOS 1.4.0](release-notes-1.4.0.md). It
adds AGE-MOEA-II and the authors' reading of RVEA as variants, one source of
reference vectors for MOEA/D, NSGA-III and RVEA, the CTP and IDTLZ test
problems, DTLZ reference fronts with up to eight objectives, and the
approximate hypervolume in the experimental analysis facade. It corrects
NSGA-III's parent selection, which changes the results of every NSGA-III run,
and the packaged MOEA/D weight files, which changes MOEA/D's results with four
objectives and 100 to 119 individuals. Read [which results change](#which-results-change)
before comparing results with those of earlier versions. The stable 1.x API, the
omitted-engine NumPy default, and the 1.0.0 run and study schemas remain the
compatibility baseline.

## NSGA-III selects its parents as configured

NSGA-III ran a binary tournament on rank and crowding distance, as NSGA-II,
whatever its configured selection, `"random"` included. Crowding distance is
not part of NSGA-III, so VAMOS's NSGA-III was not the published algorithm. Its
parents now follow the configuration:

- `selection("random")` is the default of `NSGAIIIConfig.default()`, the
  command line and the tuning spaces: parents are drawn at random, as in Deb
  and Jain (2014). With constraints and feasibility first, each parent wins a
  binary tournament on the constraint violation, decided at random between
  feasible solutions, as in Jain and Deb's constrained NSGA-III and in pymoo.
- `selection("tournament", size=k)` is jMetal's NSGA-III tournament on rank,
  with ties drawn at random.
- Any other selection method or option raises `ConfigurationError`, and
  `--selection-pressure` no longer applies to NSGA-III.
- The NSGA-III tuning spaces choose `selection`, `random` or `tournament` with
  `tournament_size`, as Evolver's space does; assignments of earlier spaces
  still build their tournament.

On ZDT4 with 100 individuals and 25,000 evaluations (seeds 0 to 9), the mean
hypervolume of the default configuration goes from 0.863 to 0.836. On DTLZ2 with
three objectives, VAMOS's mean hypervolume is within 0.1 % of pymoo's NSGA-III.
See [variants and their effect](../algorithms/nsgaiii.md#variants-and-their-effect).

## Reference and weight vectors

MOEA/D, NSGA-III and RVEA take their vectors from one kind of option, as jMetal
and Evolver configure them; see
[reference and weight vectors](../reference/algorithms.md#reference-and-weight-vectors).

- MOEA/D's `weight_vectors`, NSGA-III's `reference_directions` and RVEA's new
  `reference_vectors` accept a file, with whitespace or commas between the
  components, a directory with jMetal's `W<m>D_<n>.dat` files, the lattice of
  `divisions`, or, with the new `inner_divisions`, Deb and Jain's two-layer
  lattice (156 vectors with eight objectives, 3 and 2 divisions).
- The vectors are the population, so RVEA can run 100 individuals with three
  objectives from `W3D_100.dat`, and the three algorithms can start from the
  same vectors.
- `vamos.algorithms.moead_populations(n_obj)` lists the populations that MOEA/D
  runs without weight options.
- Every packaged weight file loads, also from the command line, where 14 of the
  31 failed. `W4D_100.dat` held its 100 vectors twice and was skipped, so MOEA/D
  with four objectives and 100 to 119 individuals ran 84; it now runs the 100
  vectors of the file.
- A run that names a file records the file it read and the `sha256` of its
  vectors, and replay refuses the file if its vectors have changed, where VAMOS
  1.4.0 replayed whatever the file then held. The run still verifies without the
  file.
- Validation is stricter: a path that does not exist, an unknown key in
  `weight_vectors` or `reference_directions`, and a two-layer lattice whose
  layers repeat a direction raise `ConfigurationError`. NSGA-III generated a
  lattice and saved it at a missing path, and MOEA/D created a missing directory
  for its lattice.

## Algorithm variants

- **AGE-MOEA-II** (Panichella, 2022) is a variant of AGE-MOEA:
  `AGEMOEAConfig.builder().variant("agemoea_ii")`, and `replacement:
  agemoea_ii` in every AGE-MOEA tuning space (Evolver's `agemoeaII`). Its
  survival estimates the front's exponent by Newton-Raphson from the point
  nearest to the ideal point and measures diversity with geodesic distances. Its
  exponents, distances and survival scores are checked against jMetal's
  `AGEMOEA2EnvironmentalSelection`. The default, `"agemoea"`, does not change;
  see [AGE-MOEA-II](../reference/algorithms.md#age-moea-ii).
- **RVEA's `variant("platemo")`** runs the authors' reading of the survival,
  that of PlatEMO and jMetal: the number of objectives in the angle penalty also
  with two objectives, translation by each generation's minimum, and adaptation
  of the vectors to the population's range. The default, `"pymoo"`, gives the
  results of VAMOS 1.4.0, bit for bit; see [RVEA](../algorithms/rvea.md).

## Problems and reference fronts

- **CTP1–CTP7** (Deb, Pratap and Meyarivan, 2001), registered as `ctp1`–`ctp7`
  with the paper's definition and parameters and two-objective reference fronts
  generated from it; `vamos.problems.CTP` builds other instances. Their
  constraint is the paper's difference, not the ratio of the NSGA-II code that
  pymoo uses.
- **IDTLZ1 and IDTLZ2**, the inverted DTLZ1 and DTLZ2 of Jain and Deb (2014) as
  PlatEMO defines them, registered as `idtlz1` and `idtlz2` with any number of
  objectives, and reference fronts for two and three objectives.
- **DTLZ reference fronts with 4, 6 and 8 objectives** for DTLZ1–DTLZ4 and
  DTLZ7, which the resolver finds by `n_obj`, so IGD, IGD+, epsilon and the
  Hausdorff distance have an automatic reference on these runs. DTLZ1–DTLZ4 use
  the Das-Dennis lattice of PlatEMO's reference fronts, whose points are those
  of jMetal's DTLZ2–DTLZ4 files.

The [problems catalog](../reference/problems.md) lists 260 registered problems.

## Approximate hypervolume

The experimental `vamos.ux.api` exports `HVIndicator`, `HVApproxIndicator`,
`HV_APPROX_METHODS` and `IndicatorResult`. `HVApproxIndicator.compute` passes
`seed` and `method` to MooCore's `hv_approx` and records the samples, the seed
and the method in its details; without them it computes the value it computed
before. The exact hypervolume can take minutes from about six objectives; see
[hypervolume, exact and approximate](../topics/analysis.md#hypervolume-exact-and-approximate).

## Faster unbounded archive

An unbounded external archive keeps the rows that each generation offers and
merges them when the archive is read, instead of refiltering itself every
generation. The rows, their order and every result are the same. A run that
reads its archive only at the end filters it once: 0.28 s instead of 85 s for a
112,000-row archive over 200,000 evaluations.

## Which results change

The [changelog](https://github.com/vamos-optimization/VAMOS/blob/main/CHANGELOG.md)
lists each change with its pull request. Rerun or redo the following before
comparing them with results of earlier versions.

| Area | Change | Rerun |
| --- | --- | --- |
| NSGA-III | Parents are drawn at random by default, or by a tournament on rank when configured; NSGA-III ran a tournament on rank and crowding distance whatever its selection. | Every NSGA-III run, by default and with a tournament: Python API, command line, studies and tuning. |
| MOEA/D with four objectives | A population of 100 to 119 runs the 100 vectors of `W4D_100.dat` instead of the 84 of the six-division lattice, and `vamos tune` gives 100 as the effective population. | MOEA/D runs and tuning with four objectives and 100 to 119 individuals. |
| Tuning spaces | The NSGA-III spaces choose `selection` (`random`, the default, or `tournament` with `tournament_size`), and the AGE-MOEA spaces offer `replacement` (`agemoea`, the default, or `agemoea_ii`). | Seeded tuning runs of the NSGA-III and AGE-MOEA spaces draw different trials. Stored assignments build the configurations they built. |
| Vector options | A path that does not exist, an unknown key and a two-layer lattice whose layers repeat a direction raise `ConfigurationError`; NSGA-III generated a lattice at a missing path, and MOEA/D created a missing directory. | Configurations that relied on these fallbacks fail; name an existing file or directory. |

The other algorithms, AGE-MOEA and RVEA with their default variants, MOEA/D with
other populations, runs with an unbounded external archive, and the
approximate hypervolume without the new options give the same results as with
VAMOS 1.4.0.

## Dependencies

The declared dependencies do not change. CI builds the documentation with
mkdocs-material 9.8.1.

## Known limitations

See [known limitations](known-limitations.md).
