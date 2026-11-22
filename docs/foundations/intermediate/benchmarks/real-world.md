---
description: 'Real-world benchmark problems in VAMOS (RE, RWA, CRE, RCM and its reformulations, and MODAct): what makes them different from artificial test problems, where their reference fronts come from, and VAMOS examples.'
search:
  boost: 0.5
---

# Real-world problems

Artificial test problems are built from position and distance variables, so
their fronts are known and their difficulties are chosen. Real-world problems
come from engineering models: trusses, beams, gear trains, power networks,
actuators. Nobody chose their difficulties, and nobody knows their exact
fronts. They complement the artificial suites in a study whose claims concern
practice, and they test things the artificial suites rarely have: objectives on
wildly different scales, equality constraints, discrete variables and
expensive evaluations.

In this lesson you will learn:

- the real-world suites in VAMOS and what each contains;
- what makes them different from artificial problems;
- why equality constraints defeat standard algorithms, and how the RCM
  reformulations get around them;
- how MODAct problems are evaluated, by a package outside VAMOS;
- where their reference fronts come from, and what that means for indicators.

## The suites

The catalog's [application-inspired benchmarks](../../../reference/problems.md#application-inspired-benchmarks)
list every problem with its formulation, dimensions and notes:

| Suite | Problems | Objectives | Constraints | Source |
| --- | --- | --- | --- | --- |
| RE | 16 engineering designs, `re21`–`re91` | 2 to 9 | Several add the total violation as an extra objective; no `G` | Tanabe and Ishibuchi [1] |
| RWA | 10 applications, `rwa1`–`rwa10` | 2 to 7 | Box constraints only | Zapotecas-Martínez et al. [2] |
| CRE | 8, `cre21`–`cre51` | 2 to 5 | In `G` | Tanabe and Ishibuchi [1] |
| RCM | 50, `rcm01`–`rcm50`, the CEC 2021 suite | 2 to 5 | In `G`, equalities included | Kumar et al. [3] |
| Reformulated RCM | 23, `rcm22_reformulated`, … | 2 to 4 | Equalities met by construction | VAMOS 1.5.2; Wang et al. [4] |
| MODAct | 20 actuator designs, `modact_cs1`–`modact_ctsei4` | 2 to 5 | In `G`, 7 to 10 | Picard and Schiffmann [5] |

The catalog also has three application examples of VAMOS: `welded_beam`, a
mixed-variable design, and `fs_real` and `ml_tuning`, which train a classifier
for every candidate and need scikit-learn.

## What makes them different

- **Unknown fronts.** The reference front of a real-world problem is the best
  set of solutions found so far, not the true front. Indicators measured against
  it can be beaten: a solution that dominates a reference point is possible,
  and the catalog records several such cases.
- **Objectives on different scales.** Costs in thousands next to displacements
  in thousandths. Unnormalized distances and hypervolumes are then decided by the
  largest objective alone.
- **Constraints of every kind.** Inequalities, equalities relaxed to
  |h(x)| ≤ 10⁻⁴, and variables that must be integers but are stored as reals and
  rounded inside the evaluation. RE turns the total violation into an objective,
  so the trade-off includes infeasible designs; CRE and RCM keep the
  constraints separate.
- **Expensive or external evaluation.** A MODAct design takes 10 to 30
  milliseconds, against microseconds for an analytical test problem, and calls a
  package outside VAMOS.
- **No designed structure.** No position or distance variables: separability,
  modality and bias are unknown, and can differ between problems of the same
  suite.

RE21, the design of a four-bar truss, shows the scales. Its two objectives
are a structural volume, f₁ = L (2x₁ + √2 x₂ + √x₃ + x₄), and a joint
displacement, f₂ = (FL/E)(2/x₁ + 2√2/x₂ − 2√2/x₃ + 2/x₄), with L = 200,
F = 10 and E = 2 × 10⁵: f₁ is scaled by 200 and f₂ by 0.01.

The ranges of the packaged reference fronts of a few problems:

```python
from importlib.resources import files

import numpy as np

fronts = {path.name.lower(): path for path in (files("vamos.resources") / "reference_fronts").iterdir()}


def load_front(name):
    return np.loadtxt(fronts[name].read_text().splitlines(), delimiter=",")


for name in ["re21.csv", "re33.csv", "rwa1.csv", "rcm01.csv"]:
    F = load_front(name)
    ranges = ", ".join(f"f{j + 1} [{lo:.3g}, {hi:.3g}]" for j, (lo, hi) in enumerate(zip(F.min(axis=0), F.max(axis=0))))
    print(f"{name:10s} {len(F):5d} points: {ranges}")
```

Output:

```text
re21.csv    1000 points: f1 [1.24e+03, 2.89e+03], f2 [0.00276, 0.04]
re33.csv    1500 points: f1 [-0.722, 5.31], f2 [1.14, 9.07], f3 [0, 4.32e+09]
rwa1.csv    1008 points: f1 [-305, -79.5], f2 [-0.0039, 2.29]
rcm01.csv   1000 points: f1 [61.9, 3.6e+05], f2 [-5.86e+07, -7.33e+03]
```

RE21's objectives differ by about five orders of magnitude, and RE33's third
objective, its total constraint violation, reaches 4.3 × 10⁹. RWA1's first objective is
negative because RWA1 maximizes a Nusselt number, and VAMOS minimizes its
negative; every column of `F` is minimized.

## Equality constraints and reformulations

An equality constraint leaves a feasible set of lower dimension than the search
space. A random design meets |h(x)| ≤ 10⁻⁴ with a probability close to zero, and
variation operators that move all variables at once rarely stay on it. The
share of feasible random designs shows the problem:

```python
import numpy as np

from vamos import make_problem_selection

rng = np.random.default_rng(0)
for key in ["cre21", "rcm01", "rcm15", "rcm22", "rcm36", "rcm22_reformulated", "rcm36_reformulated"]:
    problem = make_problem_selection(key).instantiate()
    X = rng.uniform(problem.xl, problem.xu, size=(2_000, problem.n_var))
    out = {}
    problem.evaluate(X, out)
    feasible = np.all(out["G"] <= 0.0, axis=1)
    print(f"{key:19s} n={problem.n_var:<3d} G columns: {out['G'].shape[1]:<3d} feasible random designs: {100 * feasible.mean():3.0f}%")
```

Output:

```text
cre21               n=3   G columns: 3   feasible random designs:   0%
rcm01               n=4   G columns: 2   feasible random designs:  59%
rcm15               n=3   G columns: 8   feasible random designs:   1%
rcm22               n=9   G columns: 6   feasible random designs:   0%
rcm36               n=28  G columns: 24  feasible random designs:   0%
rcm22_reformulated  n=5   G columns: 3   feasible random designs: 100%
rcm36_reformulated  n=4   G columns: 1   feasible random designs:  98%
```

RCM22 (Haverly's pooling problem) has four equalities, and RCM36 (sizing
distributed generators in a power network) 24. Standard evolutionary algorithms
find no feasible design of many such RCM problems. A **reformulation** searches
other variables, a *genotype*, and decodes each genotype into a complete design
of the original problem that meets the equalities by construction: a closed
formula for RCM22, a Newton power-flow solve for RCM36. The objectives and the
remaining constraints are the original's at the decoded design, and a last
constraint reports a decoding that failed. For RCM30–RCM35, whose constraints
order switching angles, the default reformulation is the cumulative-variable
one of Wang et al. [4]. The catalog's
[reformulation notes](../../../reference/problems.md#reformulated-rcm-problems)
describe each decoder.

With the same budget, NSGA-II finds no feasible design of RCM22 and only
feasible ones of its reformulation, and the decoded designs are feasible designs
of the original problem with the same objectives:

```python
import numpy as np

from vamos import make_problem_selection, optimize

for key in ["rcm22", "rcm22_reformulated"]:
    result = optimize(key, algorithm="nsgaii", pop_size=100, max_evaluations=5_000, seed=1)
    feasible = np.all(result.data["G"] <= 0.0, axis=1)
    print(f"{key}: {feasible.sum()} feasible of {len(feasible)}, searching {result.X.shape[1]} variables")

problem = make_problem_selection("rcm22_reformulated").instantiate()
designs = problem.decode(result.X)  # the nine variables of RCM22
out = {}
problem.original.evaluate(designs, out)
print("decoded designs feasible on rcm22:", bool(np.all(out["G"] <= 0.0)))
print("same objectives on rcm22:", np.allclose(out["F"], result.F))
```

Output:

```text
rcm22: 0 feasible of 100, searching 9 variables
rcm22_reformulated: 100 feasible of 100, searching 5 variables
decoded designs feasible on rcm22: True
same objectives on rcm22: True
```

A reformulated problem resolves the reference front of its original
(`rcm22.csv` here), so its results compare with results on the original. Its
genotype is not the original's decision vector, though: report which
formulation a result comes from.

## MODAct: evaluation outside VAMOS

The twenty MODAct problems [5] design an electro-mechanical actuator: a stepper
motor and three gear stages in a housing. VAMOS does not implement them. Each
evaluation calls Picard and Schiffmann's `modact` package, which is GPL-3.0
and installed separately (`python -m pip install modact`, or the `modact` extra
of VAMOS). Their names join the objectives, C (cost), T (torque excess), S
(safety factor), E (efficiency) and I (transmission ratio), to a constraint
level from 1 to 4. A selection describes a problem without evaluating it, and
the packaged fronts need no package:

```python
from importlib.resources import files

import numpy as np

from vamos import make_problem_selection

selection = make_problem_selection("modact_cs1")  # does not import modact
print(f"{selection.spec.label}: n = {selection.n_var}, M = {selection.n_obj}, evaluated by {', '.join(selection.spec.evaluator_packages)}")

fronts = {path.name.lower(): path for path in (files("vamos.resources") / "reference_fronts").iterdir()}
F = np.loadtxt(fronts["modact_cs1.csv"].read_text().splitlines(), delimiter=",")
print(f"best-known front: {len(F)} points, cost from {F[:, 0].min():.3f} to {F[:, 0].max():.3f}, "
      f"minus the safety factor from {F[:, 1].min():.2f} to {F[:, 1].max():.2f}")
```

Output:

```text
MODAct CS1 (actuator design): n = 20, M = 2, evaluated by modact, trimesh, python-fcl
best-known front: 1664 points, cost from 0.202 to 0.982, minus the safety factor from -180.61 to -24.13
```

The safety factor is maximized in MODAct, so VAMOS stores its negative. A run
records the versions of `modact` and of the packages it uses for meshes and
collision checks, and `reproduce` refuses to replay a run in an environment
with other versions. With modact 1.0.1, a run of 25,000 evaluations takes 5 to
15 minutes on a laptop, so plan studies accordingly.

## Where the reference fronts come from

The catalog's [reference-front table](../../../reference/problems.md#reference-fronts)
gives the origin of every file. For the real-world suites:

- **RE**: Tanabe and Ishibuchi's approximated fronts [1]. A dense random sample
  still dominates a few of their points: about 6% for RE61 and at most 0.5% for
  most others. RE91's is regenerated for VAMOS' deterministic formulation.
- **RWA**: samples, several regenerated by dense sampling and local refinement
  for VAMOS' formulations.
- **CRE and RCM**: jMetal's fronts, the non-dominated feasible solutions of many
  runs of several algorithms. Three are not packaged because feasible designs
  dominate some of their rows, and RCM17 has no front, since one of its
  objectives is unbounded on the feasible set.
- **MODAct**: the authors' best-known fronts, aggregated from 30 runs of NSGA-II
  (NSGA-III with three or more objectives) and 30 of C-TAEA with 300,000
  evaluations each. VAMOS packages those of the two-objective problems;
  `vamos zoo fetch-fronts` downloads the others, up to 193 MB each.

An indicator value computed against such a front measures the distance to the
best solutions known, and a new algorithm can exceed them. Normalize with fixed
bounds, report the reference front's origin, and do not read a value of zero
distance as convergence to the true front.

## Key points

- RE, RWA, CRE, RCM and MODAct come from engineering models; their fronts are
  approximations, not the true fronts.
- They differ from artificial problems in their objective scales, constraint
  types, discrete or rounded variables, evaluation cost and lack of designed
  structure.
- Equality constraints defeat standard algorithms; the 23 reformulated RCM
  problems decode a genotype into designs that meet them, and resolve their
  original's reference front.
- MODAct is evaluated by the external `modact` package, and its runs replay only
  with the same package versions.
- Report where each reference front comes from, and normalize with fixed
  bounds before computing indicators.

**Next:** [Binary, integer, permutation and mixed problems](combinatorial.md).

## References

[1] Tanabe, R., and Ishibuchi, H. (2020). [An easy-to-use real-world multi-objective optimization problem suite](https://doi.org/10.1016/j.asoc.2020.106078). *Applied Soft Computing*, 89, 106078.

[2] Zapotecas-Martínez, S., García-Nájera, A., and Menchaca-Méndez, A. (2023). [Engineering applications of multi-objective evolutionary algorithms: A test suite of box-constrained real-world problems](https://doi.org/10.1016/j.engappai.2023.106192). *Engineering Applications of Artificial Intelligence*, 123, 106192.

[3] Kumar, A., Wu, G., Ali, M. Z., Luo, Q., Mallipeddi, R., Suganthan, P. N., and Das, S. (2021). [A benchmark-suite of real-world constrained multi-objective optimization problems and some baseline results](https://doi.org/10.1016/j.swevo.2021.100961). *Swarm and Evolutionary Computation*, 67, 100961.

[4] Wang, J., Zhu, H., Nan, Y., and Ishibuchi, H. (2025). [Impact of reformulation for the real-world constrained multi-objective problems on the performance of evolutionary algorithms](https://doi.org/10.1145/3712255.3726689). *Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '25 Companion)*, 443–446.

[5] Picard, C., and Schiffmann, J. (2021). [Realistic constrained multiobjective optimization benchmark problems from design](https://doi.org/10.1109/TEVC.2020.3020046). *IEEE Transactions on Evolutionary Computation*, 25(2), 234–246.
