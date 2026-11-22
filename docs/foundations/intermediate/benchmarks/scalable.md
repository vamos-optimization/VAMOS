---
description: 'Benchmark problems that scale in objectives and variables: DTLZ, WFG, IDTLZ, ZCAT and LSMOP, how their dimensions change with M, and the pitfalls of reference fronts, degenerate and inverted fronts, scaled objectives and variable linkage, with VAMOS code.'
search:
  boost: 0.5
---

# Many objectives and many variables

Lesson [Many-objective optimization](../../advanced/many-objective.md) explains
why four or more objectives change how algorithms work. To study that, you
need problems whose number of objectives M can be raised while everything else
stays comparable, and, for large-scale optimization, problems with hundreds of
variables. This lesson covers the scalable suites of VAMOS and the traps they
set for the experimenter: missing reference fronts, fronts that are not the
shape they seem, objectives on different scales and variables that depend on
each other.

In this lesson you will learn:

- which suites scale in M, and how each changes its number of variables;
- for which M VAMOS packages reference fronts;
- what degenerate and inverted fronts are, and why they matter for
  decomposition-based algorithms;
- what ZCAT adds, and why its objectives need normalizing;
- how LSMOP links its variables, and how the number of variables changes the
  difficulty.

## How the dimensions scale

DTLZ, WFG, IDTLZ, ZCAT and LSMOP accept any number of objectives, and so do
the constrained C-DTLZ, DC-DTLZ, MW4, MW8, MW14 and ZXH-CF (lesson
[Constrained benchmarks](constrained.md)). They scale differently:

- **DTLZ and IDTLZ** keep the number of distance variables k fixed (k = 5 for
  DTLZ1 and IDTLZ1, 10 for most others, 20 for DTLZ7) and use n = M + k − 1
  when n is not given [1].
- **WFG** keeps n = 24 and uses k = 2(M − 1) position-related variables for
  three or more objectives, so the distance-related variables l = n − k
  decrease as M grows [2]. With 13 objectives, n must be raised to at least 25.
- **LSMOP** keeps n = 300 and **ZCAT** n = 30, whatever M.

The catalog's notes on [dimension rules](../../../reference/problems.md#continuous-test-suites)
give the constraints, for example the even l of WFG2 and WFG3. The default
numbers of variables, and whether VAMOS packages a reference front for that
number of objectives:

```python
from importlib.resources import files

from vamos import make_problem_selection

fronts = {path.name.lower(): path for path in (files("vamos.resources") / "reference_fronts").iterdir()}


def has_front(key, m):
    # The automatic lookup: <key>.<M>d.csv, otherwise <key>.csv, and only with M columns.
    for name in (f"{key}.{m}d.csv", f"{key}.csv"):
        if name in fronts and fronts[name].read_text().split("\n", 1)[0].count(",") + 1 == m:
            return True
    return False


objectives = (2, 3, 4, 6, 8)
print(("key     " + "".join(f"M={m:<9d}" for m in objectives)).rstrip())
for key in ["dtlz2", "idtlz1", "wfg4", "zcat1", "lsmop1"]:
    cells = [f"n={make_problem_selection(key, n_obj=m).n_var:<3d} {'yes' if has_front(key, m) else 'no':4s}" for m in objectives]
    print((f"{key:8s}" + "".join(cells)).rstrip())
```

Output:

```text
key     M=2        M=3        M=4        M=6        M=8
dtlz2   n=11  yes n=12  yes n=13  yes n=15  yes n=17  yes
idtlz1  n=6   yes n=7   yes n=8   no  n=10  no  n=12  no
wfg4    n=24  yes n=24  no  n=24  no  n=24  no  n=24  no
zcat1   n=30  yes n=30  yes n=30  yes n=30  yes n=30  no
lsmop1  n=300 yes n=300 no  n=300 no  n=300 no  n=300 no
```

The default WFG and LSMOP runs have three objectives, and VAMOS packages their
fronts for two only. Where no file exists, build a reference front for your M
and document how. For DTLZ2–DTLZ4 the front is the unit sphere and for WFG4–WFG9
the ellipsoid Σ (f_m/2m)² = 1, so evenly spread points on them are easy to
generate; for the other problems, the catalog's
[reference-front notes](../../../reference/problems.md#reference-fronts)
describe how VAMOS generated its files.

## Degenerate fronts

A front with M objectives is normally an (M − 1)-dimensional surface. A
**degenerate** front has fewer dimensions: with three objectives, a curve
instead of a surface. DTLZ5, DTLZ6 [1] and WFG3 [2] were designed to have a
curve as their front for every M. WFG3 obtains it by fixing all its position
parameters but the first: on the Pareto set, its three objectives depend on a
single number and lie on a straight line.

```python
import numpy as np

from vamos import make_problem_selection

problem = make_problem_selection("wfg3", n_obj=3).instantiate()
k = 4  # position-related variables: 2 (M - 1)
rng = np.random.default_rng(2)
X = np.tile(0.35 * problem.xu, (5, 1))  # distance-related variables at their optimum
X[:, :k] = rng.uniform(0.0, 1.0, (5, k)) * problem.xu[:k]  # random position-related variables
out = {}
problem.evaluate(X, out)
F = out["F"]
print(np.round(F, 4))
print("f2 = 2 f1 for every point:", np.allclose(F[:, 1], 2 * F[:, 0]))
print("f3 = 6 (1 - f1) for every point:", np.allclose(F[:, 2], 6 * (1 - F[:, 0])))
```

Output:

```text
[[0.2801 0.5601 4.3197]
 [0.6643 1.3287 2.014 ]
 [0.4662 0.9324 3.2028]
 [0.551  1.1019 2.6942]
 [0.8253 1.6505 1.0485]]
f2 = 2 f1 for every point: True
f3 = 6 (1 - f1) for every point: True
```

These are the intended fronts, and VAMOS' packaged three-objective DTLZ5 and
DTLZ6 samples are the intended curves too. Ishibuchi, Masuda and Nojima [3]
showed that the true fronts are larger: DTLZ5 and DTLZ6 with four or more
objectives, and WFG3 with three or more, also have Pareto-optimal solutions
off the curve, a non-degenerate part. A reference front sampled from the curve
alone then misrepresents the problem, and indicator values computed with it
can rank algorithms differently from values computed with the whole front.

Degenerate fronts hurt algorithms that spread solutions with evenly spaced
reference directions: most directions point at no part of the front, and
their solutions crowd on the curve.

## Inverted fronts

Jain and Deb [4] turned the fronts of DTLZ1 and DTLZ2 over: IDTLZ1 is
f = (1 + g)/2 − (1 + g) s/2 and IDTLZ2 f = (1 + g) − (1 + g) s, where s is the
position shape of DTLZ1 or DTLZ2 and g its distance function. The front of
DTLZ1 is a simplex that points away from the origin, like the simplex of
reference directions that NSGA-III, MOEA/D and RVEA use (lesson
[NSGA-III](../../advanced/nsga-iii.md)). IDTLZ1's front points the other way.
Count how many of the 91 directions of a Das–Dennis lattice [8] with three
objectives and 12 divisions meet each front, using VAMOS' packaged samples:

```python
from importlib.resources import files
from itertools import combinations

import numpy as np

fronts = {path.name.lower(): path for path in (files("vamos.resources") / "reference_fronts").iterdir()}


def das_dennis(m, h):
    # Every way of splitting h divisions among m objectives, divided by h.
    return np.array([np.diff((-1, *bars, h + m - 1)) - 1 for bars in combinations(range(h + m - 1), m - 1)]) / h


W = das_dennis(3, 12)
directions = W / np.linalg.norm(W, axis=1, keepdims=True)
for name in ["dtlz1.csv", "idtlz1.3d.csv", "dtlz2.csv", "idtlz2.3d.csv"]:
    F = np.loadtxt(fronts[name].read_text().splitlines(), delimiter=",")
    points = F / np.linalg.norm(F, axis=1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip(directions @ points.T, -1.0, 1.0).max(axis=1)))
    print(f"{name:14s} {np.sum(angle < 1.0):2d} of {len(W)} directions meet the front (within 1 degree)")
```

Output:

```text
dtlz1.csv      91 of 91 directions meet the front (within 1 degree)
idtlz1.3d.csv  28 of 91 directions meet the front (within 1 degree)
dtlz2.csv      91 of 91 directions meet the front (within 1 degree)
idtlz2.3d.csv  40 of 91 directions meet the front (within 1 degree)
```

Every direction meets DTLZ1's and DTLZ2's fronts, but only 28 meet IDTLZ1's and
40 IDTLZ2's; the others point past them, at the empty corners. Ishibuchi et al.
[5] showed that decomposition-based algorithms that perform well on DTLZ and
WFG perform poorly on such inverted fronts, because of this mismatch. Results
on DTLZ1 and DTLZ2 alone therefore flatter these algorithms; include an
inverted problem whenever the claim concerns many objectives.

## Scaled objectives and ZCAT

When objectives have different ranges, an unnormalized distance or angle
favours the largest. WFG scales objective m by 2m, and ZCAT [7] by m²: the
four-objective front of ZCAT1 reaches about 1, 4, 9 and 16. Algorithms and
indicators need normalized objectives to treat them equally (lesson
[Quality indicators](../../basic/quality-indicators.md)).

ZCAT's twenty problems have twenty front shapes, among them degenerate ones
(ZCAT14–ZCAT16, whose Pareto set is one-dimensional) and partly degenerate
ones (ZCAT19 and ZCAT20), and distance functions chosen by options.
VAMOS accepts them by name, with the defaults of the
[catalog](../../../reference/problems.md#continuous-test-suites):

- `level`, 1 to 6, selects the distance function, from a sphere-like one at
  level 1 to multimodal ones at the higher levels;
- `bias` raises the distance terms to the power 0.05, a strong bias;
- `imbalance` gives odd and even objectives different distance functions, so
  that some objectives are harder to converge than others;
- `complicated_pareto_set` makes the Pareto set a curved manifold that
  depends on the position variables.

Pass them with `optimize("zcat3", problem_kwargs={"level": 4, "bias": True})`
and record them: each combination is a different problem.

## Many variables: LSMOP

Large-scale problems have hundreds or thousands of variables. Scaling up a
classic problem is not enough, because the classic distance variables are
separable and each of them can be optimized on its own. Cheng et al. [6]
designed LSMOP1–LSMOP9 with three features:

- **groups**: the n − M + 1 distance variables are split into M groups of
  uneven sizes, one per objective, set by a chaotic map, and each group into
  `nk = 5` subcomponents, so that variables matter to different objectives;
- **mixed separability**: each problem applies two of six basic functions
  (sphere, Schwefel's, Rastrigin's, Rosenbrock's, Ackley's and Griewank's),
  some separable and some not, to the subcomponents of alternate objectives;
- **linkage**: before that, each distance variable x_i is replaced by
  (1 + i/n) x_i − 10 x₁ in LSMOP1–LSMOP4, or by (1 + cos(πi/(2n))) x_i − 10 x₁
  in LSMOP5–LSMOP9, so its optimal value depends on x₁.

LSMOP1–LSMOP4 have DTLZ1's linear front with Σ f_i = 1, LSMOP5–LSMOP8 the unit
sphere, and LSMOP9 a disconnected front. The distance variables range over
[0, 10], and the Pareto set of LSMOP1–LSMOP4 is x_i = 10 x₁ / (1 + i/n). Move x₁
alone and the solution leaves the front:

```python
import numpy as np

from vamos import make_problem_selection

problem = make_problem_selection("lsmop1").instantiate()  # n = 300, M = 3
n, m = problem.n_var, problem.n_obj
i = np.arange(m, n + 1)  # the indices of the distance variables, 3 to 300


def pareto_optimal(position):
    x = np.empty(n)
    x[: m - 1] = position
    x[m - 1 :] = 10.0 * position[0] / (1.0 + i / n)  # LSMOP1's Pareto set
    return x


moved = pareto_optimal([0.2, 0.6])
moved[0] = 0.8  # x1 changes; the distance variables keep their optimum for x1 = 0.2
out = {}
problem.evaluate(np.vstack([pareto_optimal([0.2, 0.6]), pareto_optimal([0.8, 0.6]), moved]), out)
print("sum of the objectives (1 on the front):", np.round(out["F"].sum(axis=1), 3))
```

Output:

```text
sum of the objectives (1 on the front): [ 1.  1. 37.]
```

The optimum of every distance variable moves with x₁. An algorithm that
changes one variable at a time, or treats the distance variables as
independent, has to discover this dependence. The number of variables matters
as much: with the same budget, NSGA-II gets much closer to the front of a
30-variable LSMOP1 than to the default 300-variable one:

```python
import numpy as np

from vamos import optimize

for n_var in (30, 300):
    result = optimize("lsmop1", algorithm="nsgaii", n_var=n_var, pop_size=100, max_evaluations=20_000, seed=1)
    F = result.front()
    print(f"LSMOP1 with {n_var} variables: median sum of the objectives {np.median(F.sum(axis=1)):.3f} (1 on the front)")
```

Output:

```text
LSMOP1 with 30 variables: median sum of the objectives 1.447 (1 on the front)
LSMOP1 with 300 variables: median sum of the objectives 26.310 (1 on the front)
```

LSMOP needs enough variables for every group to keep at least one variable per
subcomponent; VAMOS rejects a smaller n when the problem is built.

## Key points

- DTLZ and IDTLZ keep k and grow n with M, WFG keeps n = 24 and shifts
  variables from distance to position, LSMOP and ZCAT keep n.
- VAMOS packages reference fronts for some values of M only: DTLZ1–DTLZ4 and
  DTLZ7 up to eight objectives, ZCAT up to six, WFG and LSMOP for two. A run with
  another M needs a supplied front.
- Degenerate fronts (DTLZ5, DTLZ6, WFG3) and inverted fronts (IDTLZ1, IDTLZ2)
  defeat evenly spread reference directions; the true fronts of DTLZ5, DTLZ6
  and WFG3 with many objectives also have non-degenerate parts.
- WFG and ZCAT scale their objectives differently, so normalize before
  comparing distances or angles.
- LSMOP groups, mixes and links its distance variables, and the same budget
  goes much further with 30 variables than with 300.

**Next:** [Constrained benchmarks](constrained.md).

## References

[1] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.

[2] Huband, S., Hingston, P., Barone, L., and While, L. (2006). [A review of multiobjective test problems and a scalable test problem toolkit](https://doi.org/10.1109/TEVC.2005.861417). *IEEE Transactions on Evolutionary Computation*, 10(5), 477–506.

[3] Ishibuchi, H., Masuda, H., and Nojima, Y. (2016). Pareto fronts of many-objective degenerate test problems. *IEEE Transactions on Evolutionary Computation*, 20(5), 807–813.

[4] Jain, H., and Deb, K. (2014). [An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, Part II: Handling constraints and extending to an adaptive approach](https://doi.org/10.1109/TEVC.2013.2281534). *IEEE Transactions on Evolutionary Computation*, 18(4), 602–622.

[5] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[6] Cheng, R., Jin, Y., Olhofer, M., and Sendhoff, B. (2017). [Test problems for large-scale multiobjective and many-objective optimization](https://doi.org/10.1109/TCYB.2016.2600577). *IEEE Transactions on Cybernetics*, 47(12), 4108–4121.

[7] Zapotecas-Martínez, S., Coello Coello, C. A., Aguirre, H. E., and Tanaka, K. (2023). [Challenging test problems for multi- and many-objective optimization](https://doi.org/10.1016/j.swevo.2023.101350). *Swarm and Evolutionary Computation*, 81, 101350.

[8] Das, I., and Dennis, J. E. (1998). [Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, 8(3), 631–657.
