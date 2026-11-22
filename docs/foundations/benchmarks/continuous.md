---
description: 'The classic continuous test suites ZDT, DTLZ and WFG: what each problem adds, position and distance variables, local fronts, bias, the WFG toolkit of transformations, an interactive figure of position bias and VAMOS code.'
search:
  boost: 0.5
---

# ZDT, DTLZ and WFG

Three suites of continuous problems appear in most papers on evolutionary
multi-objective optimization. ZDT [1] has six two-objective problems, each
adding one difficulty to a simple base. DTLZ [2] scales the number of
objectives. WFG [3] builds its nine problems from a toolkit of transformations,
so that each difficulty can be switched on separately. This lesson reads their
definitions in VAMOS and shows what each one adds; the
[continuous suites table](../../reference/problems.md#continuous-test-suites)
of the catalog gives their keys, default dimensions and reference fronts.

In this lesson you will learn:

- how ZDT builds its problems from f₁, g and h, and what each of the six adds;
- how DTLZ splits the variables into M − 1 position and k distance variables,
  and where its local fronts are;
- how bias changes what a population sees, with DTLZ4, ZDT6 and WFG1;
- how the WFG toolkit composes shifts, biases, reductions and shapes, and the
  properties of WFG1–WFG9.

## ZDT: one difficulty at a time

Each ZDT problem has the form of lesson
[What a benchmark tests](overview.md#how-test-problems-are-built):
f₁ = x₁, f₂ = g(x₂, …, x_n) h(f₁, g). For ZDT1, g = 1 + 9 · mean(x₂, …, x_n) and
h = 1 − √(f₁/g), in VAMOS:

```python
# Excerpt from src/vamos/foundation/problem/zdt1.py
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
```

g reaches its minimum, 1, when x₂ = … = x_n = 0, so the Pareto set is x₁ in
[0, 1] with every other variable at 0, and the front is f₂ = 1 − √f₁. The other
problems change one part:

| Problem | Variables | What changes | What it adds |
| --- | --- | --- | --- |
| ZDT1 | 30 in [0, 1] | The base problem | A convex front |
| ZDT2 | 30 in [0, 1] | h = 1 − (f₁/g)² | A concave front |
| ZDT3 | 30 in [0, 1] | h = 1 − √(f₁/g) − (f₁/g) sin(10π f₁) | A front in five disconnected pieces |
| ZDT4 | 10: x₁ in [0, 1], the rest in [−5, 5] | A Rastrigin-type g | 21⁹ local fronts [1] |
| ZDT5 | 80 bits | Counts of ones in bit blocks | A deceptive binary problem (lesson [Binary, integer, permutation and mixed problems](combinatorial.md)) |
| ZDT6 | 10 in [0, 1] | f₁ = 1 − exp(−4x₁) sin⁶(6πx₁), g = 1 + 9 · mean(x₂, …, x_n)^0.25 | Bias along the front and towards it, and a concave front |

ZDT4's g adds a cosine to every distance variable:

```python
# Excerpt from src/vamos/foundation/problem/zdt4.py
            x_tail = X[:, 1:]
            g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(x_tail**2 - 10.0 * np.cos(4.0 * np.pi * x_tail), axis=1)
```

In [−5, 5], cos(4πx) has 21 maxima, so each of the nine distance variables has
21 local minima of g, and their combinations form local fronts parallel to the
global one. ZDT6's f₁ is multimodal in x₁, and most values of x₁ give an f₁
close to 1; the power 0.25 in g makes values of g close to 1, near the front,
rare among random solutions [1]. Every ZDT problem is separable, and its
distance variables are optimal at a bound of their domain.

## DTLZ: any number of objectives

DTLZ [2] writes the M objectives as a position shape scaled by the distance
function. With M − 1 position variables x₁, …, x_{M−1} and k distance
variables, n = M + k − 1. DTLZ2 multiplies cosines and sines of the position
variables, so that the front is the part of the unit sphere in the positive
orthant:

<div class="vf-formula">g = Σ_{distance variables} (x_i − 0.5)²
f₁ = (1 + g) cos(x₁π/2) ⋯ cos(x_{M−2}π/2) cos(x_{M−1}π/2)
f₂ = (1 + g) cos(x₁π/2) ⋯ cos(x_{M−2}π/2) sin(x_{M−1}π/2)
⋮
f_M = (1 + g) sin(x₁π/2)</div>

```python
# Excerpt from src/vamos/foundation/problem/dtlz.py
        g = np.sum((X[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= np.cos(X[:, j] * np.pi / 2.0)
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin(X[:, idx] * np.pi / 2.0)
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
```

In DTLZ1–DTLZ5 the distance variables are optimal at 0.5, where g = 0. DTLZ6
(g = Σ x_i^0.1) and DTLZ7 (g = 1 + 9 times their mean) are smallest with them at
0. The seven problems
combine a few position shapes with a few distance functions:

| Problem | Default k | Front | What it adds |
| --- | ---: | --- | --- |
| DTLZ1 | 5 | Linear: Σ f_i = 0.5 | A multimodal g, with 11ᵏ − 1 local fronts [2] |
| DTLZ2 | 10 | Concave: Σ f_i² = 1 | The smooth reference problem |
| DTLZ3 | 10 | DTLZ2's | DTLZ1's multimodal g on DTLZ2's shape |
| DTLZ4 | 10 | DTLZ2's | Bias: each position variable raised to α = 100 |
| DTLZ5 | 10 | A curve, for M = 3 | A degenerate front |
| DTLZ6 | 10 | DTLZ5's | A biased g, Σ x_i^0.1 |
| DTLZ7 | 20 | 2^(M−1) disconnected pieces | A disconnected front in any number of objectives |

DTLZ1 and DTLZ3 share this distance function:

```python
# Excerpt from src/vamos/foundation/problem/dtlz.py
        g = 100.0 * (
            self.n_var
            - self.n_obj
            + 1
            + np.sum(
                (X[:, self.n_obj - 1 :] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, self.n_obj - 1 :] - 0.5)),
                axis=1,
            )
        )
```

cos(20π(x − 0.5)) equals 1 at x = 0, 0.1, …, 1, which gives each distance
variable 11 local minima. Setting every distance variable to 0.6 puts a
solution on a local front; moving them to 0.61 leaves it:

```python
import numpy as np

from vamos import make_problem_selection

for key in ["dtlz1", "dtlz2", "dtlz3"]:
    problem = make_problem_selection(key).instantiate()  # three objectives
    X = np.full((3, problem.n_var), 0.5)  # distance variables at 0.5: g = 0
    X[:, :2] = [0.2, 0.7]  # the two position variables
    X[1, 2:] = 0.6  # every distance variable at 0.6
    X[2, 2:] = 0.61  # and at 0.61
    out = {}
    problem.evaluate(X, out)
    F = out["F"]
    print(f"{key} (n = {problem.n_var}): sum of f {np.round(F.sum(axis=1), 3)}, norm of f {np.round(np.linalg.norm(F, axis=1), 3)}")
```

Output:

```text
dtlz1 (n = 7): sum of f [ 0.5    3.    51.271], norm of f [ 0.407  2.443 41.753]
dtlz2 (n = 12): sum of f [1.588 1.747 1.78 ], norm of f [1.    1.1   1.121]
dtlz3 (n = 12): sum of f [  1.588  17.47  324.122], norm of f [  1.     11.    204.083]
```

On the global front, DTLZ1's objectives sum to 0.5 and DTLZ2's and DTLZ3's
have norm 1. At 0.6, DTLZ1 and DTLZ3 sit on a local front, with g = 5 and
g = 10, and a small move away from it makes g much worse. DTLZ2, whose g has a
single minimum, only grows slowly. An algorithm that improves its solutions by
small steps can stop on such a local front, which is why DTLZ3 needs a much
larger budget than DTLZ2.

## Bias

A biased problem maps uniformly spread decision variables onto a crowded part
of the front [3]. DTLZ4 replaces each position variable x by x^α with α = 100:
x^100 is below 0.05 for every x below 0.97, so almost every random solution has
its angle near 0, at the end f = (1, 0, …, 0). ZDT6's f₁ crowds near 1 for the
same reason, and WFG1 applies a polynomial bias with exponent 0.02 to all its
variables, which turns 0.01 into 0.91.

The figure draws 200 Pareto-optimal solutions: position variables drawn
uniformly in [0, 1], distance variables at their optimum. The bars count how
many land in each tenth of the front's length. Set α to 1 to recover DTLZ2,
then raise it:

<div class="vf-widget" data-vf-widget="position-bias"><p class="vf-fallback">Interactive figure: 200 Pareto-optimal solutions of DTLZ4, ZDT6 or WFG1 with uniformly drawn position variables, and the share of them in each tenth of the front. With DTLZ4's α = 100, almost all of them lie in the tenth at the end f = (1, 0). It needs JavaScript.</p></div>

Bias does not change the front, only how hard it is to spread a population
along it. Variation operators produce children near their parents in decision
space, which on DTLZ4 means near the crowded end in objective space. Whether
a run recovers depends on the population it starts from:

```python
import numpy as np

from vamos import optimize

for key in ["dtlz2", "dtlz4"]:
    covered = 0
    for seed in range(1, 9):
        result = optimize(key, algorithm="nsgaii", n_obj=2, pop_size=100, max_evaluations=10_000, seed=seed)
        F = result.front()
        angles = np.degrees(np.arctan2(F[:, 1], F[:, 0]))  # 0 at f = (1, 0), 90 at f = (0, 1)
        covered += np.ptp(angles) > 80
    print(f"{key}: {covered} of 8 runs cover the front from end to end")
```

Output:

```text
dtlz2: 8 of 8 runs cover the front from end to end
dtlz4: 3 of 8 runs cover the front from end to end
```

The DTLZ4 runs that fail end with a front of one to three points at
f = (1, 0). A study on DTLZ4 therefore needs many runs per algorithm, and a
report of how many lost the front, not only a mean indicator value.

## WFG: a toolkit of transformations

Huband et al. [3] reviewed the test problems of their time against a list of
recommendations and features, and proposed a toolkit to build problems that
cover them. A WFG problem has k **position-related** and l
**distance-related** variables, z_i in [0, 2i]. After normalizing them to
[0, 1], it applies a sequence of transformations:

- **shifts** move the optimum of a variable away from the bounds and the
  centre: *linear*, *deceptive* and *multimodal* shifts, all optimal at 0.35
  in WFG1–WFG9;
- **biases** distort the mapping: *polynomial*, *flat region* and
  *parameter-dependent* biases, the last one depending on other variables;
- **reductions** combine several variables into one: a *weighted sum*, or a
  *non-separable* reduction that couples them;
- **shapes** build the front from the reduced position parameters: *linear*,
  *convex*, *concave*, *mixed* and *disconnected*.

The objectives are f_m = x_M + S_m h_m(x₁, …, x_{M−1}), where x_M is the
reduced distance parameter, 0 on the Pareto set, h_m the shape, and S_m = 2m
gives each objective a different range. Some of VAMOS' transformations and
the objective formula:

```python
# Excerpt from src/vamos/foundation/problem/wfg.py
def _transformation_shift_linear(value: np.ndarray[Any, Any], shift: float = 0.35) -> np.ndarray[Any, Any]:
    return _correct_to_01(np.fabs(value - shift) / np.fabs(np.floor(shift - value) + shift))

def _transformation_bias_poly(y: np.ndarray[Any, Any], alpha: float) -> np.ndarray[Any, Any]:
    return _correct_to_01(y**alpha)

def _shape_mixed(x: np.ndarray[Any, Any], a: float = 5.0, alpha: float = 1.0) -> np.ndarray[Any, Any]:
    aux = 2.0 * a * np.pi
    ret = np.power(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha)
    return _correct_to_01(ret)

    def _calculate(self, x: np.ndarray[Any, Any], h: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        return np.asarray(x[:, -1][:, None] + self.S * np.column_stack(h), dtype=float)
```

WFG1 chains four transformations, then a convex shape for the first M − 1
objectives and a mixed one for the last:

```python
# Excerpt from src/vamos/foundation/problem/wfg.py
    def _evaluate(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        y = x / self.xu
        y = self._t1(y, self.n_var, self.k)
        y = self._t2(y, self.n_var, self.k)
        y = self._t3(y, self.n_var)
        y = self._t4(y, self.n_obj, self.n_var, self.k)
        y = self._post(y)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_mixed(y[:, 0], alpha=1.0, a=5.0))
        return self._calculate(y, h)
```

Its steps are a linear shift and a flat-region bias of the distance variables,
the polynomial bias y^0.02 of all variables, and a weighted-sum reduction. The
nine problems combine the parts as Huband et al. [3] classify them:

| Problem | Separable | Modality | Bias | Front |
| --- | --- | --- | --- | --- |
| WFG1 | Yes | Unimodal | Polynomial, flat region | Convex, mixed |
| WFG2 | No | Unimodal; f_M multimodal | None | Convex, disconnected |
| WFG3 | No | Unimodal | None | Linear, degenerate |
| WFG4 | Yes | Multimodal | None | Concave |
| WFG5 | Yes | Deceptive | None | Concave |
| WFG6 | No | Unimodal | None | Concave |
| WFG7 | Yes | Unimodal | Parameter-dependent | Concave |
| WFG8 | No | Unimodal | Parameter-dependent | Concave |
| WFG9 | No | Multimodal, deceptive | Parameter-dependent | Concave |

VAMOS uses n = 24 by default, with k = 4 position-related variables for two
objectives and k = 2(M − 1) otherwise, and l = n − k; WFG2 and WFG3 need an
even l. WFG4–WFG9 share the front (f₁/2)² + … + (f_M/(2M))² = 1. On WFG4, every
distance-related variable at 0.35 of its range puts a solution on that front,
whatever the position-related variables:

```python
import numpy as np

from vamos import make_problem_selection

problem = make_problem_selection("wfg4", n_obj=2).instantiate()
k = 4  # position-related variables with two objectives
rng = np.random.default_rng(1)
X = np.tile(0.35 * problem.xu, (5, 1))  # distance-related variables at their optimum
X[:, :k] = rng.uniform(0.0, 1.0, (5, k)) * problem.xu[:k]  # random position-related variables
out = {}
problem.evaluate(X, out)
F = out["F"]
print("upper bounds of the first six variables:", problem.xu[:6])
print("f1:", np.round(F[:, 0], 4))
print("f2:", np.round(F[:, 1], 4))
print("(f1/2)² + (f2/4)²:", np.round((F[:, 0] / 2) ** 2 + (F[:, 1] / 4) ** 2, 6))
```

Output:

```text
upper bounds of the first six variables: [ 2.  4.  6.  8. 10. 12.]
f1: [1.4008 0.7538 1.0329 0.5373 0.725 ]
f2: [2.855  3.705  3.4253 3.8529 3.7279]
(f1/2)² + (f2/4)²: [1. 1. 1. 1. 1.]
```

WFG's reference fronts in VAMOS have two objectives only; a run with three or
more needs a reference front built for its M (lesson
[Many objectives and many variables](scalable.md)).

## Other continuous suites

The catalog lists more continuous suites. CEC2009 UF1–UF10 [4] and Li and
Zhang's LZ09 [5] have Pareto sets that are curves in decision space, not
aligned with the axes, which tests variable linkage; ZCAT and LSMOP scale in
objectives and variables and are the subject of the next lesson.

## Key points

- ZDT builds two-objective problems from f₁, g and h; each of ZDT1–ZDT6 adds one
  difficulty, and all of them are separable with the Pareto set on a bound.
- DTLZ has M − 1 position and k distance variables, optimal at 0.5 (at 0 in
  DTLZ6 and DTLZ7); DTLZ1 and
  DTLZ3 have local fronts, DTLZ4 and DTLZ6 are biased, DTLZ5 and DTLZ6 are
  degenerate and DTLZ7 is disconnected.
- Bias crowds uniform decision variables into part of the front; on DTLZ4 a
  run can lose all but one end of the front, depending on its seed.
- WFG composes shifts, biases, reductions and shapes, so that separability,
  modality, bias and front shape vary independently across WFG1–WFG9.

**Next:** [Many objectives and many variables](scalable.md).

## References

[1] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[2] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.

[3] Huband, S., Hingston, P., Barone, L., and While, L. (2006). [A review of multiobjective test problems and a scalable test problem toolkit](https://doi.org/10.1109/TEVC.2005.861417). *IEEE Transactions on Evolutionary Computation*, 10(5), 477–506.

[4] Zhang, Q., Zhou, A., Zhao, S., Suganthan, P. N., Liu, W., and Tiwari, S. (2008). Multiobjective optimization test instances for the CEC 2009 special session and competition. Technical Report CES-487, University of Essex and Nanyang Technological University.

[5] Li, H., and Zhang, Q. (2009). [Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II](https://doi.org/10.1109/TEVC.2008.925798). *IEEE Transactions on Evolutionary Computation*, 13(2), 284–302.
