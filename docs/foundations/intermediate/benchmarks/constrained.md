---
description: 'The constrained benchmark suites in VAMOS (CTP, C-DTLZ, DC-DTLZ, MW, LIR-CMOP, DAS-CMOP, ZXH-CF and CEC2009 CF), the kinds of difficulty that constraints create, an interactive figure of CTP1–CTP7 and VAMOS code.'
search:
  boost: 0.5
---

# Constrained benchmarks

Lesson [Constrained optimization](../../advanced/constrained.md) explains how
algorithms compare solutions that break constraints, and the lessons after it
describe CCMO, C-TAEA, ToP and CMOEA-MS. This lesson is about the problems
used to test them. A constrained benchmark is hard not because it has
constraints, but because of where they put the feasible region: far from the
unconstrained front, in narrow tunnels, in separate pieces, or behind an
infeasible barrier. Each suite combines these difficulties in its own way.

In this lesson you will learn:

- the four kinds of difficulty that constraints create;
- how the constrained suites of VAMOS cover them, with CTP1–CTP7 as a guided
  example;
- why the share of feasible random designs does not measure difficulty;
- how an infeasible barrier stops a feasibility-first algorithm;
- which suites have tunable difficulty, and how to evaluate results on them.

## The constrained suites

The catalog's [constrained suites table](../../../reference/problems.md#explicitly-constrained-suites)
lists their keys, dimensions and numbers of constraints. In every one, a
problem writes its constraints to `G`, and `G <= 0` is feasible:

| Suite | Problems | Objectives | Source |
| --- | --- | --- | --- |
| CTP | CTP1–CTP7 | 2 | Deb, Pratap and Meyarivan [1] |
| C-DTLZ | C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C3-DTLZ1, C3-DTLZ4 | Any M | Jain and Deb [2] |
| DC-DTLZ | DC1–DC3 on DTLZ1 and DTLZ3 | Any M | Li et al. [3] |
| MW | MW1–MW14 | 2; MW4, MW8 and MW14 any M | Ma and Wang [4] |
| LIR-CMOP | LIR-CMOP1–LIR-CMOP14 | 2; 3 for 13 and 14 | Fan et al. [5] |
| DAS-CMOP | DAS-CMOP1–DAS-CMOP9 | 2; 3 for 7 to 9 | Fan et al. [6] |
| ZXH-CF | ZXH-CF1–ZXH-CF16 | Any M | Zhou, Xiang and He [7] |
| CEC2009 CF | CF1–CF10 | 2; 3 for 8 to 10 | Zhang et al. [8] |

The catalog also lists eight small classic problems (`binh2`, `constrex`,
`srinivas`, `tanaka`, `osyczka2`, `golinski`, `water` and `car_side_impact`),
and the constrained real-world suites, which lesson
[Real-world problems](real-world.md) covers.

## Four kinds of difficulty

- **The feasible front lies away from the unconstrained one.** The constraints
  make the unconstrained front infeasible, and the constrained front runs along
  the boundary of the feasible region. CTP1, C3-DTLZ1 and C3-DTLZ4 are of this
  kind: Jain and Deb's Type 3 [2].
- **The feasible region is narrow.** A small or thin feasible region is hard to
  find, and once found, hard to move along. CTP2–CTP5 end in tunnels or single
  points. In C1-DTLZ1 only a band next to the front is feasible. DAS-CMOP's
  feasibility parameter ζ narrows the feasible band of its distance function
  [6], and LIR-CMOP1–LIR-CMOP4 have small feasible regions [5].
- **The feasible region is disconnected.** The front breaks into pieces with
  infeasible gaps between them, and a population has to keep solutions in
  every piece. C2-DTLZ2 (Type 2 [2]), DC1-DTLZ1 and DC1-DTLZ3, whose single
  constraint on x₁ cuts the front into strips [3], CTP7, ZXH-CF5–ZXH-CF16, whose
  constraints exclude ranges of the front's angles [7], and DAS-CMOP's
  diversity parameter η [6] are of this kind.
- **An infeasible barrier separates the search from the front.** The front is
  feasible, but the way to it is not: a population coming from far away meets an
  infeasible region first. C1-DTLZ3 (Type 1 [2]), DC2-DTLZ1 and DC2-DTLZ3, whose
  constraints depend only on the distance function g [3], CTP6, the ellipses
  and spherical shells of LIR-CMOP5–LIR-CMOP14 [5] and DAS-CMOP's convergence
  parameter γ [6] are of this kind.

Many problems combine several: DC3 has both DC1's strips and DC2's barriers.

## CTP: the four kinds in one family

Deb, Pratap and Meyarivan [1] built CTP1–CTP7 on the same two objectives,
f₁ = x₁ and a second one computed from a Rastrigin-type g. Their constraints are
written in objective space, so the whole problem can be drawn. Every design
has g ≥ 1 and lies on or above the unconstrained front (dashed); the shaded
regions are infeasible, and the blue line or points form the constrained front.
CTP1 shows the first kind, CTP2–CTP5 the second, CTP7 the third and CTP6 the
fourth:

<div class="vf-widget" data-vf-widget="ctp-landscape"><p class="vf-fallback">Interactive figure: the objective space of CTP1–CTP7 with their infeasible regions, unconstrained front and constrained front. CTP1's front runs along two constraint boundaries; CTP2's is 13 segments at the ends of narrow tunnels; CTP3's and CTP4's are 13 isolated points; CTP5's is an arch and isolated points; CTP6 has infeasible bands across the objective space; CTP7's front is the feasible pieces of the unconstrained front. It needs JavaScript.</p></div>

The figure computes the constrained fronts from the constraints, and the
documentation tests check them against the reference fronts that VAMOS
packages for CTP1–CTP7. The catalog's
[CTP notes](../../../reference/problems.md#explicitly-constrained-suites) give the
parameters (θ, a, b, c, d, e) of each problem; `vamos.problems.CTP` builds the
family with other values.

## Feasible random designs

The share of uniformly random designs that are feasible tells how hard it is to
find the feasible region, and nothing more:

```python
import numpy as np

from vamos import make_problem_selection

rng = np.random.default_rng(0)
for key in ["ctp2", "c1dtlz1", "c1dtlz3", "c2dtlz2", "dc1dtlz1", "dc2dtlz1", "mw1", "lircmop1", "lircmop5", "dascmop1", "zxh_cf5"]:
    problem = make_problem_selection(key).instantiate()
    X = rng.uniform(problem.xl, problem.xu, size=(20_000, problem.n_var))
    out = {}
    problem.evaluate(X, out)
    feasible = np.all(out["G"] <= 0.0, axis=1)
    print(f"{key:9s} n={problem.n_var:<3d} M={problem.n_obj} G columns: {out['G'].shape[1]:<3d} feasible random designs: {100 * feasible.mean():6.2f}%")
```

Output:

```text
ctp2      n=5   M=2 G columns: 1   feasible random designs: 100.00%
c1dtlz1   n=12  M=3 G columns: 1   feasible random designs:   0.00%
c1dtlz3   n=12  M=3 G columns: 1   feasible random designs: 100.00%
c2dtlz2   n=12  M=3 G columns: 1   feasible random designs:   0.51%
dc1dtlz1  n=12  M=3 G columns: 1   feasible random designs:  10.21%
dc2dtlz1  n=12  M=3 G columns: 2   feasible random designs:   0.00%
mw1       n=15  M=2 G columns: 1   feasible random designs:   0.00%
lircmop1  n=30  M=2 G columns: 2   feasible random designs:   0.00%
lircmop5  n=30  M=2 G columns: 2   feasible random designs: 100.00%
dascmop1  n=30  M=2 G columns: 11  feasible random designs:   0.01%
zxh_cf5   n=13  M=3 G columns: 3   feasible random designs:   0.00%
```

No random design of C1-DTLZ1, DC2-DTLZ1, MW1, LIR-CMOP1 or ZXH-CF5 is feasible,
and DAS-CMOP1 has one in ten thousand: an algorithm has to be guided towards
the feasible region by the size of the violation. DC1-DTLZ1's share is close to
its exact value, arccos(0.95)/π = 10.11%, since its constraint depends on x₁
alone. But CTP2, C1-DTLZ3 and
LIR-CMOP5 have every random design feasible, and they are not easy. Random
designs lie far from the front, where these problems are feasible; their
tunnels, barriers and infeasible ellipses lie closer to it.

## An infeasible barrier

C1-DTLZ3 is DTLZ3, whose front is the unit sphere, with one constraint [2]:
with ‖f‖ the Euclidean norm of the objective vector, (‖f‖² − 16)(‖f‖² − r²) ≥ 0,
r = 9 for three objectives. Designs with 4 < ‖f‖ < 9 are infeasible: a spherical
shell between the random designs and the front. DTLZ3's local fronts give a
simple way to place designs at different distances, since setting distance
variables to 0.5 + j/10 gives g = j² per variable:

```python
import numpy as np

from vamos import make_problem_selection

problem = make_problem_selection("c1dtlz3").instantiate()  # 3 objectives, 12 variables
X = np.full((5, problem.n_var), 0.5)
X[:, :2] = 0.3  # position variables
X[1, 2] = 0.6  # g = 1
X[2, 2] = 0.7  # g = 4
X[3, 2:4] = 0.7  # g = 8
X[4, 2] = 0.8  # g = 9
out = {}
problem.evaluate(X, out)
for f, g in zip(out["F"], out["G"][:, 0]):
    print(f"norm of f {np.linalg.norm(f):5.2f}: {'feasible' if g <= 1e-9 else 'infeasible'}")
```

Output:

```text
norm of f  1.00: feasible
norm of f  2.00: feasible
norm of f  5.00: infeasible
norm of f  9.00: feasible
norm of f 10.00: feasible
```

A feasibility-first algorithm, such as NSGA-II with constrained domination,
never accepts an infeasible solution over a feasible one, so its population
stays outside the shell. With the same budget, it reaches DTLZ3's front and
stops at the outer edge of the barrier on C1-DTLZ3:

```python
import numpy as np

from vamos import optimize

for key in ["dtlz3", "c1dtlz3"]:
    result = optimize(key, algorithm="nsgaii", pop_size=100, max_evaluations=50_000, seed=1)
    norms = np.linalg.norm(result.front(), axis=1)
    print(f"{key}: norm of the returned objective vectors, smallest {norms.min():.3f}, median {np.median(norms):.3f}")
```

Output:

```text
dtlz3: norm of the returned objective vectors, smallest 1.027, median 1.035
c1dtlz3: norm of the returned objective vectors, smallest 9.015, median 9.024
```

The front of C1-DTLZ3 is DTLZ3's, at norm 1, but the run never gets below
norm 9, the outer edge of the shell. Algorithms that also use infeasible solutions,
such as the two-population CCMO or the ε-constrained methods of lesson
[Constrained optimization](../../advanced/constrained.md#beyond-feasibility-first),
are designed to cross such barriers.

## Tunable difficulty

Several suites expose the parameters that set their difficulty, so a study can
vary one kind of difficulty at a time. The registered keys use the published
defaults, which the packaged reference fronts assume:

- **DAS-CMOP** takes the triplet (η, ζ, γ), each in [0, 1]: diversity,
  feasibility and convergence hardness [6]. `vamos.problems.DASCMOP(k, eta=...,
  zeta=..., gamma=...)` builds another triplet.
- **DC-DTLZ** takes the constants a and b of its constraints; the defaults are
  those of the authors' code, and other libraries use other values (see the
  catalog's DC-DTLZ notes). `vamos.problems.DCDTLZ` builds them.
- **CTP** takes CTP1's number of constraints and the six parameters of
  CTP2–CTP7 through `vamos.problems.CTP`.
- **ZXH-CF** fixes its kind of difficulty by its number: the distance function,
  front shape, constraint type A–D and radius change from one problem to the
  next [7].

A problem built with other parameters is a different problem, and the packaged
reference front no longer applies to it.

## Evaluating results

- Compute indicators on the feasible non-dominated solutions, against a
  reference front of the **constrained** problem. VAMOS packages one for most
  constrained benchmarks; CEC2009 CF1 has none.
- Report how many runs found a feasible solution, and how soon. The rest of
  the evaluation follows lesson
  [Constrained optimization](../../advanced/constrained.md#measuring-constrained-results).
- Choose problems of all four kinds. A constraint-handling technique that
  excels at finding narrow feasible regions can fail at crossing barriers, and
  the reverse.

## Key points

- Constraints make a problem hard through where they put the feasible region:
  away from the unconstrained front, in narrow regions, in separate pieces or
  behind barriers.
- CTP1–CTP7 show the four kinds in two objectives; C-DTLZ, DC-DTLZ, MW,
  LIR-CMOP, DAS-CMOP, ZXH-CF and CEC2009 CF combine them, several in any number
  of objectives.
- The share of feasible random designs measures how hard the feasible region is
  to find, not how hard the front is to reach.
- A feasibility-first algorithm stops at an infeasible barrier, as NSGA-II does
  at the outer edge of C1-DTLZ3's shell.
- DAS-CMOP, DC-DTLZ and CTP expose their difficulty parameters; the packaged
  reference fronts hold only for the defaults.

**Next:** [Real-world problems](real-world.md).

## References

[1] Deb, K., Pratap, A., and Meyarivan, T. (2001). [Constrained test problems for multi-objective evolutionary optimization](https://doi.org/10.1007/3-540-44719-9_20). *Evolutionary Multi-Criterion Optimization – EMO 2001*, Lecture Notes in Computer Science 1993, 284–298.

[2] Jain, H., and Deb, K. (2014). [An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, Part II: Handling constraints and extending to an adaptive approach](https://doi.org/10.1109/TEVC.2013.2281534). *IEEE Transactions on Evolutionary Computation*, 18(4), 602–622.

[3] Li, K., Chen, R., Fu, G., and Yao, X. (2019). [Two-archive evolutionary algorithm for constrained multiobjective optimization](https://doi.org/10.1109/TEVC.2018.2855411). *IEEE Transactions on Evolutionary Computation*, 23(2), 303–315.

[4] Ma, Z., and Wang, Y. (2019). [Evolutionary constrained multiobjective optimization: Test suite construction and performance comparisons](https://doi.org/10.1109/TEVC.2019.2896967). *IEEE Transactions on Evolutionary Computation*, 23(6), 972–986.

[5] Fan, Z., Li, W., Cai, X., Huang, H., Fang, Y., You, Y., Mo, J., Wei, C., and Goodman, E. (2019). [An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions](https://doi.org/10.1007/s00500-019-03794-x). *Soft Computing*, 23, 12491–12510.

[6] Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K., and Goodman, E. (2020). [Difficulty adjustable and scalable constrained multiobjective test problem toolkit](https://doi.org/10.1162/evco_a_00259). *Evolutionary Computation*, 28(3), 339–378.

[7] Zhou, Y., Xiang, Y., and He, X. (2021). [Constrained multiobjective optimization: Test problem construction and performance evaluations](https://doi.org/10.1109/TEVC.2020.3011829). *IEEE Transactions on Evolutionary Computation*, 25(1), 172–186.

[8] Zhang, Q., Zhou, A., Zhao, S., Suganthan, P. N., Liu, W., and Tiwari, S. (2008). Multiobjective optimization test instances for the CEC 2009 special session and competition. Technical Report CES-487, University of Essex and Nanyang Technological University.
