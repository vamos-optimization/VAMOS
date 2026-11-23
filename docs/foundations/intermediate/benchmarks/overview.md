---
description: 'What multi-objective benchmark problems test: front shapes, multimodality, bias, deception, separability, scalability and constraints, why reference fronts matter for indicators, and how to choose a suite for a study, with an interactive gallery of fronts and VAMOS code.'
search:
  boost: 0.5
---

# What a benchmark tests

An algorithm is judged by running it on problems, and the conclusions are only
as good as the problems chosen. Benchmark problems, also called test problems,
are built so that their Pareto front is known and each of their difficulties
is controlled: the shape of the front, the number of local optima, how
uniformly the decision space maps onto the front, how the variables interact.
VAMOS registers 405 of them. The [problems catalog](../../../reference/problems.md)
lists every key with its dimensions, source and reference front; this section
of Foundations explains what the problems test and how to choose among them.

In this lesson you will learn:

- how test problems are built from position and distance variables;
- the shapes a front can take, and why each one matters to an algorithm;
- multimodality, bias, deception and separability, the main difficulties of
  the search;
- why reference fronts matter for quality indicators, and how to load the ones
  VAMOS packages;
- how to choose a suite for a study, and how not to.

## How test problems are built

Deb [1] proposed to build two-objective test problems from separate parts, and
most suites since follow the idea. The decision variables are split in two
groups:

- the **position variables** decide *where* on the front a solution lies;
- the **distance variables** decide *how far* from the front it is, through a
  function g that is smallest on the Pareto set.

ZDT [2] writes it as f₁ = x₁ and f₂ = g(x₂, …, x_n) h(f₁, g): h fixes the shape
of the front and g its difficulty. DTLZ [3] extends it to any number of
objectives, and WFG [4] builds problems from a toolkit of transformations.
Because the parts are separate, the Pareto set is known exactly. On a
connected front it is every solution with g at its minimum, whatever the
position variables. On a disconnected front, such as ZDT3's or DTLZ7's, part of
that minimum-g surface is dominated by the rest, so the Pareto set keeps only
its non-dominated part. Lesson
[ZDT, DTLZ and WFG](continuous.md) shows the definitions.

## Front shapes

With minimized objectives, a front is

- **convex** when it bulges towards the ideal point, below the straight line
  between its extremes (ZDT1);
- **concave** when it bulges away from it (ZDT2, DTLZ2);
- **linear** when it is a straight line, or a simplex with more objectives
  (DTLZ1, WFG3);
- **mixed** when it has convex and concave parts (WFG1);
- **disconnected** when it consists of separate pieces (ZDT3, DTLZ7, WFG2);
- **degenerate** when it has fewer dimensions than the M − 1 of a regular
  front, for example a curve in a three-objective space (DTLZ5, DTLZ6, WFG3 at
  M = 3).

The shape decides which mechanisms work. No weighted sum of the objectives
reaches the inner points of a concave front. Decomposition-based algorithms
spread their weight vectors evenly on a simplex, which suits a triangle-like
front such as DTLZ1's and wastes many vectors on an inverted, disconnected or
degenerate one [7]. Crowding and niching have to keep solutions on every piece
of a disconnected front. Pick a problem to see its front with two objectives,
and the properties of the problem it comes from:

<div class="vf-widget" data-vf-widget="benchmark-gallery"><p class="vf-fallback">Interactive figure: a gallery of the two-objective Pareto fronts of ZDT1–ZDT4 and ZDT6, DTLZ1–DTLZ4 and DTLZ7, IDTLZ2 and WFG1–WFG9, with each problem's front shape, what it tests, its separability, number of objectives, default number of variables and packaged reference fronts. It needs JavaScript.</p></div>

The fronts are written in closed form from VAMOS' definitions, and the
documentation tests check each against the reference front that VAMOS
packages. Several problems share a front and differ in everything else: ZDT1
and ZDT4, DTLZ2 to DTLZ4, WFG4 to WFG9. The front alone does not say how hard a
problem is.

## Multimodality, bias and deception

These properties describe the search landscape, the way from a random
solution to the front.

- **Multimodality.** A multimodal problem has many local optima, and a
  multi-objective one has local fronts: sets of solutions that no small change
  improves, but that the global front dominates. ZDT4's g has 21⁹ local fronts
  with 10 variables [2]; DTLZ1's has 11ᵏ − 1 for k distance variables [3].
  An algorithm that converges too fast stops at one of them.
- **Bias.** A biased problem maps a uniform sample of the decision space onto a
  very non-uniform set in objective space [4]. DTLZ4 raises its position
  variables to the power 100 [3], so most solutions crowd at one end of the
  front; in ZDT6, both the spread along the front and the density towards it
  are non-uniform [2]. Bias tests whether an algorithm keeps diversity when its
  variation operators keep producing solutions in the same region.
- **Deception.** A deceptive problem has a broad local optimum whose basin leads
  away from a narrow global one [4]. Following the gradient of improvement
  makes things worse. The binary ZDT5 [2] and WFG5 and WFG9 [4] are deceptive.

## Separability and variable linkage

A problem is **separable** when each variable can be optimized on its own,
independently of the others [4]. An operator that changes one variable at a
time, such as polynomial mutation with probability 1/n (lesson
[Real-coded operators](../operators/real-coded.md)), is well suited to separable
problems, so results obtained only on them can overstate how good it is. ZDT
and DTLZ are separable; WFG2, WFG3, WFG6, WFG8 and WFG9 are not.

**Variable linkage** goes further: the optimal value of a variable depends on
the values of others, so the Pareto set is a curve or a surface that is not
aligned with the coordinate axes. Li and Zhang's LZ09 problems [5] have such
complicated Pareto sets, and in LSMOP the optimal value of every distance
variable depends on x₁ [6].

Where the Pareto set lies matters too. ZDT1's distance variables are optimal at
0, the lower bound, and those of DTLZ1–DTLZ5 at 0.5, the centre of the domain
(DTLZ6's and DTLZ7's at 0). An operator
that puts extra probability on a bound, as clipping to the bounds does, finds
the first more easily, and one that is biased towards the centre the second.
Huband et al. [4] recommend avoiding both. Most WFG problems move the optimum
of their distance variables to 0.35 of the range; all of them give each
variable a different domain, [0, 2i] for variable i, and scale objective m by
2m.

## Scalability and constraints

A **scalable** problem accepts any number of variables n, any number of
objectives M, or both. DTLZ, WFG, LSMOP and ZCAT scale in M, and LSMOP is meant
for hundreds of variables. Scaling changes the difficulty in ways that lesson
[Many objectives and many variables](scalable.md) describes.

**Constraints** split the decision space into feasible and infeasible regions.
A constrained benchmark tests whether an algorithm reaches the feasible region,
crosses infeasible barriers and covers a front that the constraints may cut
into pieces; lesson [Constrained benchmarks](constrained.md) describes the
suites.

## Problems by key

Every registered problem is created by its key. `make_problem_selection`
resolves the dimensions, the same way `optimize("name")` does, and
`instantiate()` builds the problem, which evaluates a population at once:

```python
import numpy as np

from vamos import make_problem_selection

rng = np.random.default_rng(0)
for key in ["zdt1", "dtlz2", "wfg4", "c2dtlz2", "re21", "bin_knapsack"]:
    selection = make_problem_selection(key)
    problem = selection.instantiate()
    X = rng.uniform(problem.xl, problem.xu, size=(5, problem.n_var))
    out = {}
    problem.evaluate(X, out)
    n_g = out["G"].shape[1] if "G" in out else 0
    scalable = "any M" if selection.spec.allow_n_obj_override else "fixed M"
    print(f"{key:12s} n={selection.n_var:<3d} M={selection.n_obj} ({scalable}), G columns: {n_g}, encoding: {selection.spec.encoding}")
```

Output:

```text
zdt1         n=30  M=2 (fixed M), G columns: 0, encoding: real
dtlz2        n=12  M=3 (any M), G columns: 0, encoding: real
wfg4         n=24  M=3 (any M), G columns: 0, encoding: real
c2dtlz2      n=12  M=3 (any M), G columns: 1, encoding: real
re21         n=4   M=2 (fixed M), G columns: 0, encoding: continuous
bin_knapsack n=50  M=2 (fixed M), G columns: 0, encoding: binary
```

`out["F"]` holds one row of objectives per solution, and a constrained problem
also writes `out["G"]`, with `G <= 0` feasible. `make_problem_selection(key,
n_obj=5)` changes the number of objectives of a scalable problem, and a fixed
problem rejects it.

## Reference fronts and indicators

Most quality indicators compare a result with a **reference front** (lesson
[Quality indicators](../../basic/quality-indicators.md)): GD, IGD and IGD+ measure
distances to it, and the hypervolume needs a reference point, often derived
from its nadir point after normalizing with its ideal and nadir points. A
reference front is a finite sample of the true front, so

- its number of columns, its objective signs and its problem parameters must
  match the run: a three-objective DTLZ2 run needs a three-objective sample;
- two samples of the same front give different IGD values, because their
  points lie at different places;
- for most real-world problems it is an approximation, the best points that
  several long runs found, not the true front.

VAMOS packages its reference fronts as CSV files, one row per point. The
catalog's [reference-front notes](../../../reference/problems.md#reference-fronts)
say where each file comes from and for which numbers of objectives it exists.
Read them with `importlib.resources`, here to check the shape and the defining
equation of a few fronts:

```python
from importlib.resources import files

import numpy as np

folder = files("vamos.resources") / "reference_fronts"
fronts = {path.name.lower(): path for path in folder.iterdir()}


def load_front(name):
    return np.loadtxt(fronts[name].read_text().splitlines(), delimiter=",")


for name in ["zdt1.csv", "zdt3.csv", "zdt6.csv", "dtlz7.2d.csv", "wfg1.csv"]:
    F = load_front(name)
    F = F[np.argsort(F[:, 0])]
    gaps = np.diff(F[:, 0]) > 0.02 * np.ptp(F[:, 0])
    print(f"{name:13s} {F.shape[0]:5d} points, ideal {np.round(F.min(axis=0), 3)}, nadir {np.round(F.max(axis=0), 3)}, {1 + gaps.sum()} piece(s)")

dtlz1, dtlz2, dtlz5 = load_front("dtlz1.csv"), load_front("dtlz2.csv"), load_front("dtlz5.csv")
print(f"DTLZ1, {len(dtlz1)} points: f1 + f2 + f3 = 0.5 is", np.allclose(dtlz1.sum(axis=1), 0.5))
print(f"DTLZ2, {len(dtlz2)} points: f1² + f2² + f3² = 1 is", np.allclose((dtlz2**2).sum(axis=1), 1.0))
print(f"DTLZ5, {len(dtlz5)} points: f1 = f2 is", np.allclose(dtlz5[:, 0], dtlz5[:, 1]))
```

Output:

```text
zdt1.csv       1000 points, ideal [0. 0.], nadir [1. 1.], 1 piece(s)
zdt3.csv       1000 points, ideal [ 0.    -0.773], nadir [0.852 1.   ], 5 piece(s)
zdt6.csv       1000 points, ideal [0.281 0.   ], nadir [1.    0.921], 1 piece(s)
dtlz7.2d.csv   1000 points, ideal [0.    2.307], nadir [0.859 4.   ], 2 piece(s)
wfg1.csv      10000 points, ideal [0. 0.], nadir [2. 4.], 1 piece(s)
DTLZ1, 5050 points: f1 + f2 + f3 = 0.5 is True
DTLZ2, 5050 points: f1² + f2² + f3² = 1 is True
DTLZ5, 1000 points: f1 = f2 is True
```

The file names follow the lookup that VAMOS uses for runs and studies:
`<key>.<M>d.csv` for a given number of objectives, otherwise `<key>.csv`.
`dtlz1.csv` and `dtlz5.csv` hold three objectives, the default of DTLZ.
DTLZ5's three-objective front satisfies f₁ = f₂ everywhere: it is a curve, a
degenerate front. The automatic lookup never returns a file with another
number of objectives, so a default three-objective WFG run, whose packaged
samples have two columns, has no automatic reference front.

## The same front, a different problem

ZDT1 and ZDT4 share the front f₂ = 1 − √f₁, but ZDT4's distance function is
multimodal. With the same algorithm and budget, the result on ZDT4 stays far
from the front, measured here by the vertical gap f₂ − (1 − √f₁) of each
returned point:

```python
import numpy as np

from vamos import optimize

for key in ["zdt1", "zdt4"]:
    result = optimize(key, algorithm="nsgaii", pop_size=100, max_evaluations=10_000, seed=1)
    F = result.front()
    gap = F[:, 1] - (1.0 - np.sqrt(F[:, 0]))  # both fronts: f2 = 1 - sqrt(f1)
    print(f"{key}: {len(F)} non-dominated points, median gap to the front {np.median(gap):.3f}")
```

Output:

```text
zdt1: 100 non-dominated points, median gap to the front 0.024
zdt4: 12 non-dominated points, median gap to the front 0.347
```

With 20,000 evaluations NSGA-II reaches ZDT4's front as well. A comparison on
ZDT1 alone would not show how an algorithm copes with local fronts, and a
comparison at a single budget would not show how fast it gets past them.

## Choosing a suite for a study

A benchmark study supports a claim only on problems that exercise it. Some
rules:

1. **Start from the question.** A claim about many objectives needs scalable
   problems at several values of M; a claim about constraint handling needs
   constrained problems with different kinds of difficulty; a claim about
   practical use needs real-world problems
   ([Real-world problems](real-world.md)).
2. **Cover the features.** Include convex, concave, linear and disconnected
   fronts, and inverted or degenerate ones; multimodal, biased, deceptive and
   non-separable problems; objectives with different scales. WFG covers many
   of these in one suite [4]; combine it with DTLZ, ZCAT, and the other suites
   of the following lessons.
3. **Use the published settings and record them.** Dimensions and parameters
   change a problem: DTLZ1 is defined with k = 5 distance variables, DTLZ2 with
   k = 10, WFG with k position and l distance variables. The catalog notes
   where VAMOS's versions of a problem differ from earlier versions or from
   other libraries.
4. **Use matching reference fronts**, or build and document one where VAMOS
   packages none.
5. **Report every problem you ran**, not a favourable subset, and the
   statistics of several independent runs (lesson
   [Comparing algorithms rigorously](../../advanced/performance-assessment.md)).

How not to do it: compare algorithms on ZDT1–ZDT3 alone. The three problems
have two objectives, separable variables, the same unimodal distance function
and their Pareto set on a bound of the domain. An algorithm that wins there
may lose as soon as any of these changes. Ishibuchi et al. [7] showed a subtler
case: decomposition-based algorithms that do well on DTLZ1 and DTLZ2, whose
fronts have the shape of the simplex of their weight vectors, do much worse on
inverted versions of the same problems. Tanabe and Ishibuchi [8] made a similar
point about artificial problems in general, and proposed real-world problems as
a complement.

## Key points

- Test problems separate position variables, which choose the place on the
  front, from distance variables, which set the distance to it; the Pareto set
  is known exactly.
- Fronts can be convex, concave, linear, mixed, disconnected or degenerate, and
  each shape helps or hinders a different mechanism.
- Multimodality creates local fronts, bias crowds uniform samples into part of
  the front, deception leads the search away from the optimum, and
  non-separability defeats one-variable-at-a-time search.
- Indicators need a reference front with the right number of objectives and the
  right parameters; VAMOS packages samples as CSV files.
- A study should cover the features its claim depends on, with published
  settings, and report every problem.

**Next:** [ZDT, DTLZ and WFG](continuous.md).

## References

[1] Deb, K. (1999). [Multi-objective genetic algorithms: Problem difficulties and construction of test problems](https://doi.org/10.1162/evco.1999.7.3.205). *Evolutionary Computation*, 7(3), 205–230.

[2] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[3] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.

[4] Huband, S., Hingston, P., Barone, L., and While, L. (2006). [A review of multiobjective test problems and a scalable test problem toolkit](https://doi.org/10.1109/TEVC.2005.861417). *IEEE Transactions on Evolutionary Computation*, 10(5), 477–506.

[5] Li, H., and Zhang, Q. (2009). [Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II](https://doi.org/10.1109/TEVC.2008.925798). *IEEE Transactions on Evolutionary Computation*, 13(2), 284–302.

[6] Cheng, R., Jin, Y., Olhofer, M., and Sendhoff, B. (2017). [Test problems for large-scale multiobjective and many-objective optimization](https://doi.org/10.1109/TCYB.2016.2600577). *IEEE Transactions on Cybernetics*, 47(12), 4108–4121.

[7] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[8] Tanabe, R., and Ishibuchi, H. (2020). [An easy-to-use real-world multi-objective optimization problem suite](https://doi.org/10.1016/j.asoc.2020.106078). *Applied Soft Computing*, 89, 106078.
