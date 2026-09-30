# Problems catalog

Choose a problem by its **formulation, encoding and dimensions**, then check the
[algorithm compatibility matrix](algorithms.md). This catalog describes the
136 registered problems in VAMOS **1.0.0**. The named selection helpers are
[stable public APIs](../project/stability-and-versioning.md). A synthetic or
engineering benchmark is a test model, not a validated model of your application.

## Choose a starting point

| What you want to exercise | Start with | Why |
| --- | --- | --- |
| A first continuous run and a visible two-objective front | `zdt1` | Inexpensive evaluation, 30 variables, packaged 2D reference sample |
| Three or more objectives | `dtlz2` | Scalable objectives and a standard dimension rule |
| More varied geometry and difficulty | `wfg1`–`wfg9`, `zcat1`–`zcat20` | Configurable objective count; record the family parameters |
| Hundreds of variables | `lsmop1`–`lsmop9` | 300 variables by default; group structure matters |
| Explicit constraint handling | `c1dtlz1`, `mw1` | Emit inequality constraints separately from objectives |
| Binary, integer or mixed operators | `zdt5`, `int_alloc`, `mixed_design` | Small, deterministic examples with explicit encodings |
| Permutation operators | `tsp6`, then `kroa100` | Closed tours with two specified route objectives |
| Application-inspired test functions | RE and RWA tables below | Fixed engineering formulations with different encodings and objective counts |
| Expensive model evaluations | `fs_real`, `ml_tuning` | Train a classifier for each candidate; require scikit-learn |

Use `optimize("zdt1", ...)` in Python or `--problem zdt1` in the
[CLI](../guide/cli.md). Inspect the installed version through the public facade:

```python
from vamos import available_problem_names, make_problem_selection

print(sorted(available_problem_names()))
selection = make_problem_selection("dtlz2", n_obj=3)
problem = selection.instantiate()
print(problem.n_var, problem.n_obj, problem.encoding)  # 12 3 real
print(problem.xl, problem.xu)
```

The tables report the **instantiated** default dimensions. `n` is the number of
decision variables, `M` the number of objectives, and `G` the number of explicit
inequality outputs (`G <= 0` is feasible). Zero G means no separate inequality
outputs; bounds and encoding rules still apply. All objectives returned in `F`
are minimized, including sign-flipped maximization objectives.

**Reference sample** means a CSV shipped with the package, with its objective
dimension shown. It does not certify an exact Pareto front or guarantee suitability
after changing the formulation. See [Reference fronts](#reference-fronts) before
computing indicators.

## Continuous test suites

These families use real-valued variables (`real` or `continuous` in the metadata)
and need only the base installation, **including WFG**. Source links point to
the corresponding implementation or family registration in the 1.0.0 release.

| Registered keys | Default n | Default M | G | Default parameters / dimension rule | Source | Reference sample |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `zdt1`, `zdt2`, `zdt3` | 30 | 2 | 0 | Variables in `[0, 1]`; fixed M | [ZDT][zdt] | 2D, each |
| `zdt4` | 10 | 2 | 0 | First variable `[0, 1]`, remainder `[-5, 5]`; fixed M | [ZDT][zdt] | 2D |
| `zdt6` | 10 | 2 | 0 | Variables in `[0, 1]`; fixed M | [ZDT][zdt] | 2D |
| `dtlz1` | 7 | 3 | 0 | `n = M + 4` (`k = 5`) when n is omitted | [DTLZ][dtlz] | 3D |
| `dtlz2`–`dtlz6` | 12 | 3 | 0 | `n = M + 9` (`k = 10`); DTLZ4 uses `alpha = 100` | [DTLZ][dtlz] | 3D, each |
| `dtlz7` | 22 | 3 | 0 | `n = M + 19` (`k = 20`) | [DTLZ][dtlz] | 3D |
| `wfg1`–`wfg9` | 24 | 3 | 0 | `k = 4`, `l = 20`; bounds `[0, 2i]` for variable i | [WFG][wfg] | **2D only**, each |
| `lz09_f1`, `lz09_f7`, `lz09_f8` | 10 | 2 | 0 | Fixed M; family-specific linkage | [Li–Zhang LZ09][lz] | None packaged |
| `lz09_f2`–`lz09_f5`, `lz09_f9` | 30 | 2 | 0 | Fixed M; family-specific linkage | [Li–Zhang LZ09][lz] | None packaged |
| `lz09_f6` | 10 | 3 | 0 | Fixed M | [Li–Zhang LZ09][lz] | None packaged |
| `cec2009_uf1`–`cec2009_uf7` | 30 | 2 | 0 | Fixed M; UF5: `N=10`, UF6: `N=2`, both `epsilon=0.1` | [CEC2009][cec] | 2D, each |
| `cec2009_uf8`–`cec2009_uf10` | 30 | 3 | 0 | Fixed M; UF9: `epsilon=0.1` | [CEC2009][cec] | 3D, each |
| `lsmop1`–`lsmop9` | 300 | 3 | 0 | `nk = 5` subcomponents per group | [LSMOP][lsmop] | **2D only**, each |
| `zcat1`–`zcat20` | 30 | 2 | 0 | `level = 1`; complicated Pareto set, bias and imbalance all `False` | [ZCAT][zcat] | 2D, 3D, 4D and 6D, each |

DTLZ, WFG, LSMOP and ZCAT allow changing `n_obj` in named selection. For DTLZ,
omitting `n_var` recalculates the standard dimension from M; setting a different
n is allowed but changes the experimental setting. For WFG, `k = 4` at M = 2,
otherwise `k = 2(M - 1)`, and `l = n - k`; WFG2/3 require an even distance-variable
count. LSMOP needs enough variables for all groups to remain nonempty.
Family constructor checks still apply to dimension overrides.

## Explicitly constrained suites

All entries in this table are real-valued. They emit G alongside F; choose an
algorithm with the required constraint support in the [matrix](algorithms.md).

| Registered keys | Default n | Default M | G | Default parameters / objective count | Source | Reference sample |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `cec2009_cf1` | 30 | 2 | 1 | Fixed M and constraint function | [CEC2009][cec] | None packaged |
| `c1dtlz1` | 12 | 3 | 1 | Configurable M; n remains 12 unless supplied | [C-DTLZ][constrained] | **2D only** |
| `c1dtlz3` | 12 | 3 | 1 | Default radius `r = 9` at M = 3; varies with M | [C-DTLZ][constrained] | **2D only** |
| `c2dtlz2` | 12 | 3 | 1 | Default radius `r = 0.4` at M = 3; varies with M | [C-DTLZ][constrained] | **2D only** |
| `c3dtlz1` | 12 | 3 | M | Configurable M | [C-DTLZ][constrained] | None packaged |
| `dc1dtlz1`, `dc1dtlz3` | 12 | 3 | 1 | Configurable M; `a = 5`, `b = 0.95` | [DC-DTLZ][constrained] | **2D only**, each |
| `dc2dtlz1`, `dc2dtlz3` | 12 | 3 | 2 | Configurable M; `a = 3`, `b = 0.9` | [DC-DTLZ][constrained] | **2D only**, each |
| `dc3dtlz1`, `dc3dtlz3` | 12 | 3 | M | Configurable M; `a = 5`, `b = 0.5` | [DC-DTLZ][constrained] | None packaged |
| `mw1`, `mw2`, `mw6`, `mw9` | 15 | 2 | 1 | Actual M fixed at 2 | [MW][constrained] | 2D for MW1/2/6; none for MW9 |
| `mw3`, `mw7`, `mw12`, `mw13` | 15 | 2 | 2 | Actual M fixed at 2 | [MW][constrained] | 2D for MW3/7; none for MW12/13 |
| `mw5`, `mw10` | 15 | 2 | 3 | Actual M fixed at 2 | [MW][constrained] | 2D for MW5; none for MW10 |
| `mw11` | 15 | 2 | 4 | Actual M fixed at 2 | [MW][constrained] | None packaged |
| `mw4`, `mw8`, `mw14` | 15 | 3 | 1 | Configurable M | [MW][constrained] | None packaged |

In 1.0.0, the registry accepts an objective override for every MW key, but only
MW4, MW8 and MW14 actually change their objective count. Use the instantiated
problem's `n_obj` and keep the other MW cases at two objectives.

## Binary, integer, mixed and permutation examples

These examples are useful for checking representations and operators. The
synthetic instances have fixed generation seeds, independent of the optimizer's
seed. Changing n constructs a different instance; it does not merely increase
the search budget.

| Key | Encoding | Default n / M | G | Default parameters | Source | Reference sample |
| --- | --- | --- | ---: | --- | --- | --- |
| `zdt5` | Binary | 80 / 2 | 0 | 30-bit first block, ten 5-bit tail blocks | [ZDT5][zdt5] | 2D, 31 points for default n |
| `bin_feat` | Binary | 50 / 2 | 0 | Instance seed 12345 | [VAMOS synthetic][binary] | None packaged |
| `bin_knapsack` | Binary | 50 / 2 | 0 | Capacity ratio 0.4; instance seed 2023 | [VAMOS synthetic][binary] | None packaged |
| `bin_qubo` | Binary | 30 / 2 | 0 | Symmetric Q; instance seed 7 | [VAMOS synthetic][binary] | None packaged |
| `int_alloc` | Integer | 20 / 2 | 0 | Each allocation in `0..10`; instance seed 321 | [VAMOS synthetic][integer] | None packaged |
| `int_jobs` | Integer | 30 / 2 | 0 | Five job labels `0..4`; instance seed 99 | [VAMOS synthetic][integer] | None packaged |
| `mixed_design` | Mixed | 9 / 2 | 0 | Three real, three integer, three categorical; instance seed 123 | [VAMOS synthetic][mixed] | None packaged |
| `tsp6` | Permutation | 6 / 2 | 0 | Six equally spaced cities on the unit circle | [VAMOS TSP][tsp] | None packaged |
| `kroa100`, `krob100`, `kroc100`, `krod100`, `kroe100` | Permutation | 100 / 2 | 0 | Fixed packaged coordinates from the respective TSPLIB instance | [TSP formulation][tsp], [coordinates][tsplib] | None packaged |

### What the objectives actually mean

For binary x, each bit selects one item. Every expression below is minimized.

| Key | First objective | Second objective | Interpretation |
| --- | --- | --- | --- |
| `bin_feat` | Negative sum of selected utilities | Sum of selected feature costs | Synthetic utility/cost trade-off; no classifier is trained |
| `bin_knapsack` | Absolute difference between selected weight and target capacity | Negative selected value | Capacity deviation is an objective; overweight selections are allowed |
| `bin_qubo` | `xᵀQx + bᵀx` | Negative number of selected bits | A two-objective synthetic QUBO extension |
| `int_alloc` | Total allocation cost | Negative sum of weighted square-root returns | Diminishing returns; no global budget constraint |
| `int_jobs` | Weighted mismatches against preferred job labels | Largest job-type count divided by n | Encourages a balanced use of labels; does not model machines or makespan |
| `mixed_design` | Squared real/integer target errors plus category mismatches | Integer and category costs | Synthetic mixed-variable design trade-off |
| All TSP keys | Sum of edge lengths in the closed tour | Maximum edge length in that tour | Total distance versus longest leg |

TSP decisions must contain every city exactly once. Both objectives use the
**same coordinates and unrounded Euclidean distances**, including the closing
edge. A `kroa100` run therefore uses TSPLIB coordinates in VAMOS's two-objective
formulation; published single-objective TSPLIB tour costs with integer distance
rounding are not directly comparable. The named `tsp6` uses circle coordinates;
constructing `TSP()` without arguments uses a separate asymmetric six-city toy.

ZDT5 requires `n >= 35` and `(n - 30)` divisible by 5. Changing its number of
tail blocks changes its front, so the packaged 80-bit reference is not universal.
The TSPLIB factories have fixed dimensions; supplying `n_var` does not resize
their coordinate files.

## Application-inspired benchmarks

### RE: Tanabe–Ishibuchi engineering suite

These are fixed formulations: keep their dimensions as listed. **G = 0** for all
RE entries. Several include the **sum of constraint violations as an additional
objective**; that trade-off differs from enforcing feasibility through G.
The default bounds and discrete choices are fixed by the linked implementation.

| Key / formulation | Encoding | n / M | Violation objective | Default parameters | Source | Reference sample |
| --- | --- | --- | --- | --- | --- | --- |
| `re21` — four-bar truss | Real | 4 / 2 | No | Fixed design bounds | [RE2][re2] | 2D |
| `re22` — reinforced concrete beam | Mixed | 3 / 2 | f₂ | Discrete reinforcement-area choices | [RE2][re2] | None packaged |
| `re23` — pressure vessel | Mixed | 4 / 2 | f₂ | Thickness step 0.0625; two integer counts | [RE2][re2] | None packaged |
| `re24` — hatch cover | Real | **2 / 2** | f₂ | Bounds `[0.5, 4]` to `[4, 50]` | [RE2][re2] | 2D |
| `re25` — coil compression spring | Mixed | 3 / 2 | f₂ | Integer coil count, discrete wire diameter | [RE2][re2] | None packaged |
| `re31` — two-bar truss | Real | 3 / 3 | f₃ | Fixed design bounds | [RE3a][re3a] | 3D |
| `re32` — welded beam | Real | 4 / 3 | f₃ | Fixed design bounds | [RE3a][re3a] | 3D |
| `re33` — disc brake | Real | 4 / 3 | f₃ | Fixed design bounds | [RE3a][re3a] | 3D |
| `re34` — vehicle crashworthiness | Real | 5 / 3 | No | Fixed response surfaces | [RE3a][re3a] | 3D |
| `re35` — speed reducer | Mixed | 7 / 3 | f₃ | One integer gear-teeth variable | [RE3b][re3b] | None packaged |
| `re36` — gear train | Integer | 4 / 3 | f₃ | Integer tooth counts `12..60` | [RE3b][re3b] | None packaged |
| `re37` — rocket injector | Real | 4 / 3 | No | Fixed response surfaces | [RE3b][re3b] | 3D |
| `re41` — car side impact | Real | 7 / 4 | f₄ | Fixed response surfaces | [RE4][re4] | 4D |
| `re42` — conceptual marine design | Real | 6 / 4 | f₄ | Fixed design bounds | [RE4][re4] | 4D |
| `re61` — water resource planning | Real | 3 / 6 | f₆ | Fixed design bounds | [RE6/9][re69] | 6D |
| `re91` — car cab | Real | 7 / 9 | No separate violation objective | **Deterministic** mean-value evaluation (`stochastic=False`) | [RE6/9][re69] | 9D |

Version 1.0.0 has stale descriptive labels for several RE registry entries.
In particular, the `re24` selection metadata reports four variables while its
factory creates the two-variable hatch-cover formulation. The table follows
the implementation; inspect `problem.n_var` after instantiation and use
`make_problem_selection("re24", n_var=2)` to keep this selection's dimension
metadata aligned. Keep RE91's deterministic setting explicit when comparing it
with stochastic formulations.

### RWA: Zapotecas-Martínez and collaborators

All RWA entries use real variables, fixed dimensions and bounds, and **zero
separate G outputs**. They evaluate application response functions. The
implementation converts maximization goals to negative objective values.

| Key / application | n / M | Default parameters | Source | Reference sample |
| --- | --- | --- | --- | --- |
| `rwa1` — honeycomb heat sink | 5 / 2 | Fixed response surface; maximize Nu, minimize friction factor | [RWA1–4][rwa14] | 2D |
| `rwa2` — vehicle crashworthiness | 5 / 3 | Fixed response surface | [RWA1–4][rwa14] | 3D |
| `rwa3` — synthesis gas production | 3 / 3 | Maximize CH₄ and CO; minimize H₂/CO | [RWA1–4][rwa14] | 3D |
| `rwa4` — wire electrical discharge machining | 5 / 3 | Maximize cutting rate; minimize roughness and dimensional deviation | [RWA1–4][rwa14] | 3D |
| `rwa5` — thermal storage | 9 / 3 | Fixed packed-bed response surface | [RWA5][rwa5] | 3D |
| `rwa6` — steel milling | 4 / 3 | `z = 1`, `d = 1`; maximize removal rate | [RWA6][rwa6] | 3D |
| `rwa7` — rocket injector | 4 / 3 | Three response objectives | [RWA7–8][rwa78] | 3D |
| `rwa8` — rocket injector | 4 / 4 | Four response objectives | [RWA7–8][rwa78] | 4D |
| `rwa9` — ultra-wideband antenna | 10 / 5 | Fixed passband/stopband response functions | [RWA9–10][rwa910] | 5D |
| `rwa10` — repellent fabric | 3 / 7 | Fixed fabric performance response functions | [RWA9–10][rwa910] | 7D |

### Learning and mixed engineering examples

Install `python -m pip install "vamos-optimization[examples]==1.0.0"` for the two
classifier examples. The base package is sufficient for `welded_beam`.
See [Installation](../guide/installation.md) for environments and extras.

| Key | Encoding | Default n / M | G | Defaults and objective meaning | Source | Reference sample |
| --- | --- | --- | ---: | --- | --- | --- |
| `fs_real` | Binary | 30 / 2 | 0 | Breast-cancer data; stratified 70/30 split, `random_state=0`; logistic-regression validation error vs selected-feature count | [Feature selection][fs] | None packaged |
| `ml_tuning` | Mixed | 4 / 2 | 0 | Same dataset/split; SVC validation error vs support-vector count plus `0.1 × degree` | [SVM tuning][ml] | None packaged |
| `welded_beam` | Mixed | 6 / 2 | 4 | Four real dimensions, one material category, one stiffener count; fabrication cost vs deflection | [Engineering example][beam] | None packaged |

`fs_real` penalizes the empty mask with error 1 and feature-count objective n.
`ml_tuning` searches C in `[10⁻², 10²]`, gamma in `[10⁻³, 10¹]`, kernels
`rbf/linear/poly` and degrees 2–5. These are validation objectives, not unbiased
held-out test estimates. Both public classes also accept the wine dataset,
which changes `fs_real` to 13 decision variables; named registry selection uses
the defaults above.

`welded_beam` is VAMOS's six-variable mixed design example; it is a different
formulation from the four-variable, three-objective `re32`. Its four emitted G
columns are shear stress, normal stress, deflection and the `h <= b` condition.

## Reference fronts

The [packaged CSV directory][fronts] contains finite **reference samples**. Check
the CSV's number of columns, objective definitions, signs, scaling and parameter
settings against the run. A file's existence alone is insufficient evidence of
compatibility or a certified optimum.

- ZDT fronts have two columns; ZDT5's sample is tied to its default 80 bits.
- DTLZ fronts have three columns. Changing M needs a corresponding reference.
- WFG, LSMOP and the listed C-/DC-DTLZ samples have **two columns**, even though
  their registered default is three objectives. Supply a matching reference for
  default runs; do not use those two-column files for three-objective indicators.
- ZCAT supplies dimension-specific samples at M = 2, 3, 4 and 6. Other objective
  counts need a supplied reference; 1.0.0's automatic lookup can otherwise fall
  back to a 2D file.
- RE/RWA samples have the dimensions shown above. Their presence does not imply
  that every sample point is a certified global Pareto optimum.
- Families marked “None packaged” require a suitable external reference or a
  documented empirical reference construction when using reference-front metrics.

The CLI accepts `--hv-reference-front path/to/front.csv` with `--hv-threshold`;
use one column per objective and validate the selected file before launching a
study. A reference **front** is a set of objective vectors; a hypervolume
reference **point** is one vector bounding the evaluated region. See
[Analysis](../topics/analysis.md) for indicators and comparison guidance.

## CLI presets

| Preset | Members |
| --- | --- |
| `families` | `zdt1`, `dtlz2`, `wfg4`, `tsp6` (includes different encodings) |
| `zdt` | ZDT1/2/3/4/6; binary ZDT5 is selected separately |
| `dtlz`, `wfg`, `zcat`, `lz`, `lsmop` | The corresponding suites above |
| `cec` | CEC2009 UF1–10 and CF1 |
| `constrained_many` | The 10 C-/DC-DTLZ and 14 MW cases |
| `tsp` | `tsp6` |
| `tsplib` | `kroa100`, `krob100`, `kroc100`, `krod100`, `kroe100` |
| `real_world` | `ml_tuning`, `welded_beam`, `fs_real`; this preset does not include RE/RWA |

Use `--problem-set <preset>`. A shared dimension override must make sense for
every member; prefer explicit problem selections for studies that mix fixed and
scalable formulations. For a new objective function, follow
[Build your own problem](../guide/custom-problem.md).

[zdt]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/registry/families/zdt.py
[zdt5]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/zdt5.py
[dtlz]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/dtlz.py
[wfg]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/wfg.py
[lz]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/lz.py
[cec]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/cec2009.py
[lsmop]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/lsmop.py
[zcat]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/zcat/core.py
[constrained]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/constrained_many.py
[binary]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/binary.py
[integer]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/integer.py
[mixed]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/mixed.py
[tsp]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/tsp.py
[tsplib]: https://github.com/vamos-optimization/VAMOS/tree/v1.0.0/src/vamos/resources/tsplib
[re2]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/tanabe_ishibuchi_re2.py
[re3a]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/tanabe_ishibuchi_re3_a.py
[re3b]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/tanabe_ishibuchi_re3_b.py
[re4]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/tanabe_ishibuchi_re4.py
[re69]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/tanabe_ishibuchi_re6_re9.py
[rwa14]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/zapotecas_rwa_1_4.py
[rwa5]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/zapotecas_rwa_5.py
[rwa6]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/zapotecas_rwa_6.py
[rwa78]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/zapotecas_rwa_7_8.py
[rwa910]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/zapotecas_rwa_9_10.py
[fs]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/feature_selection.py
[ml]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/hyperparam.py
[beam]: https://github.com/vamos-optimization/VAMOS/blob/v1.0.0/src/vamos/foundation/problem/real_world/engineering.py
[fronts]: https://github.com/vamos-optimization/VAMOS/tree/v1.0.0/src/vamos/resources/reference_fronts
