---
description: 'Reference directions and weight vectors: the Das and Dennis simplex lattice and its size, why it explodes with many objectives, Deb and Jain''s two-layer lattices, weight files and their uniformity, how MOEA/D, NSGA-III and RVEA use the vectors, and how VAMOS supplies them, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# Reference directions and weight vectors

MOEA/D, NSGA-III and RVEA spread a set of vectors over the objective space
before the run and keep one part of the population near each of them. The
vectors, not distances between solutions, decide where the final solutions
lie. Lesson [Many objectives](many-objective.md#nsga-iii-in-detail) introduced
the systematic design of Das and Dennis [1]; this lesson looks at the vectors
themselves: how they are built, how many there are, how evenly they are spread,
and how each algorithm uses them.

In this lesson you will learn:

- how the Das and Dennis lattice is built, and how many vectors it has;
- why a single lattice becomes impractical with many objectives, and how Deb
  and Jain's two-layer lattice helps;
- what weight files hold, and how to measure their uniformity;
- how MOEA/D, NSGA-III and RVEA use the same vectors in different ways;
- how VAMOS takes the vectors of all three algorithms from one kind of option.

## Vectors on the unit simplex

All three algorithms take their vectors from the **unit simplex**, the points
with m non-negative coordinates that add up to 1:

<div class="vf-formula">w = (w₁, …, w_m),     w_i ≥ 0,     w₁ + … + w_m = 1</div>

With two objectives the simplex is a segment, with three a triangle, and with
m objectives an (m − 1)-dimensional simplex. The algorithms read a vector in
different ways. MOEA/D [2] calls it a **weight vector**: the weights of one
scalar subproblem. NSGA-III [3] calls it a **reference point**, and the ray
from the ideal point through it a **reference direction**. RVEA [4] calls the
normalized vector w/‖w‖ a **reference vector**. The same set can serve all
three.

## The Das and Dennis lattice

Das and Dennis [1] spread points over the simplex for their normal-boundary
intersection method. With p **divisions**, the lattice holds every point whose
coordinates are multiples of 1/p:

<div class="vf-formula">w = (k₁, …, k_m) / p,     k_i ∈ {0, 1, …, p},     k₁ + … + k_m = p
H = C(p + m − 1, m − 1)                       (number of vectors)
H_interior = C(p − 1, m − 1)  if p ≥ m, else 0   (vectors with every w_i > 0)</div>

Counting the points is counting the ways to share p units among m
coordinates: place m − 1 separators among p + m − 1 slots. The points with
every coordinate positive are the ways to give each coordinate at least one
unit, C(p − 1, m − 1), and there are none when p &lt; m. The MOEA/D paper calls
the divisions H [2]; these lessons follow Deb and Jain [3] and use p for the
divisions and H for the number of vectors.

The lattice is perfectly regular: two neighbouring points differ by 1/p in two
coordinates, so every point is √2/p from its nearest neighbour. It contains
the m corners of the simplex, the points that care about one objective only.

## Two layers

Deb and Jain [3] keep the boundary lattice of p₁ divisions and add an
**inside layer**: the lattice of p₂ divisions, shrunk towards the centre of the
simplex. VAMOS shrinks it by half, as pymoo and PlatEMO do:

<div class="vf-formula">inside layer:  w′ = w / 2 + 1 / (2m),     w in the lattice of p₂ divisions
H = C(p₁ + m − 1, m − 1) + C(p₂ + m − 1, m − 1)</div>

Every inside vector has all its coordinates at least 1/(2m), so it lies
strictly inside the simplex. With eight objectives, p₁ = 3 and p₂ = 2 give
120 + 36 = 156 vectors [3], against 6,435 for a single lattice with one
interior point. The layers must not repeat a vector: a repeated direction is a
niche that never gets an individual of its own. VAMOS rejects such layers with
a `ConfigurationError`. Deb and Jain's settings never repeat one: when p₂ &lt; m,
every inside vector has a coordinate equal to 1/(2m), which no boundary lattice
of fewer than 2m divisions has.

Move the sliders. The boundary layer is blue, the inside layer orange; the
table counts both for more objectives.

1. Set p₂ = 0 and p₁ = 2: the six directions all lie on the edges of the
   triangle, and the middle of the front has none.
2. Keep p₁ = 2 and set p₂ = 1: three inside directions fill the middle, for
   nine in all; a single lattice needs p = 3 and ten directions for one
   interior point.
3. Set p₁ = 6 and p₂ = 1: the inside directions coincide with boundary ones,
   circled, and VAMOS would reject the layers.

<div class="vf-widget" data-vf-widget="lattice-layers"><p class="vf-fallback">Interactive figure: the Das and Dennis lattice of p₁ divisions and the inside layer of p₂ divisions on the simplex of three objectives, the number of directions of each layer, and the counts for 3 to 15 objectives. It needs JavaScript.</p></div>

## Why one lattice explodes

For a fixed p, H grows like m^p / p!, polynomially in m. The trouble is that a
small p puts every vector on the boundary of the simplex, and an interior
vector needs p ≥ m, which costs at least C(2m − 1, m − 1) vectors, a number
that roughly quadruples with each extra objective. The table gives H and, in
parentheses, the number of interior vectors:

```python
from math import comb


def lattice(m, p):
    """Number of Das and Dennis directions: m coordinates that are multiples of 1/p."""
    return comb(p + m - 1, m - 1)


def interior(m, p):
    """How many of them have every coordinate above zero."""
    return comb(p - 1, m - 1) if p >= m else 0


print(" m       p = 3           p = 6            p = 12")
for m in (3, 5, 8, 10, 15):
    cells = [f"{lattice(m, p):>9,} ({interior(m, p):>3})" for p in (3, 6, 12)]
    print(f"{m:>2}  " + "  ".join(cells))

for m, p1, p2 in ((8, 3, 2), (10, 3, 2), (15, 2, 1)):
    print(f"two layers, m = {m}: {lattice(m, p1)} + {lattice(m, p2)} = {lattice(m, p1) + lattice(m, p2)} directions")
```

Output:

```text
 m       p = 3           p = 6            p = 12
 3         10 (  1)         28 ( 10)         91 ( 55)
 5         35 (  0)        210 (  5)      1,820 (330)
 8        120 (  0)      1,716 (  0)     50,388 (330)
10        220 (  0)      5,005 (  0)    293,930 ( 55)
15        680 (  0)     38,760 (  0)  9,657,700 (  0)
two layers, m = 8: 120 + 36 = 156 directions
two layers, m = 10: 220 + 55 = 275 directions
two layers, m = 15: 120 + 15 = 135 directions
```

With ten objectives, 5,005 directions still leave the middle of the simplex
empty. Two layers of few divisions keep the number of vectors, and with it the
population, in the hundreds. Every vector with a zero coordinate asks for a
solution that ignores an objective, so a lattice that is all boundary pushes
the population to the edges of the front.

## Weight files and uniformity

jMetal distributes weight vectors as text files named `W<m>D_<n>.dat`: n
vectors for m objectives, one per line. VAMOS ships 31 of them in
`vamos/resources/weights`. Some are lattices (`W3D_91.dat` is the lattice of
12 divisions); others have sizes that no lattice has, such as `W3D_100.dat`
and `W4D_100.dat`, so MOEA/D can run 100 individuals with three or four
objectives.

A useful measure of uniformity is each vector's distance to its nearest other
vector: all equal for a lattice, spread out for an irregular set, and 0 for a
repeated vector. The example reads three of the files and compares them with
lattices built in a few lines:

```python
from importlib import resources
from itertools import product

import numpy as np

weights = resources.files("vamos.resources") / "weights"  # the jMetal files that VAMOS ships


def read(name):
    with (weights / name).open() as handle:
        return np.loadtxt(handle)


def das_dennis(m, p):
    """Every point of the simplex whose m coordinates are multiples of 1/p."""
    return np.array([k for k in product(range(p + 1), repeat=m) if sum(k) == p], dtype=float) / p


def describe(label, W):
    distances = np.linalg.norm(W[:, None] - W[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)  # each vector's distance to its closest other vector
    distinct = len(np.unique(W.round(9), axis=0))
    print(f"{label:<22} {len(W):>3} rows, {distinct:>3} distinct, nearest neighbour {nearest.min():.3f} to {nearest.max():.3f}")


describe("lattice m=3, p=12", das_dennis(3, 12))
describe("W3D_91.dat", read("W3D_91.dat"))
describe("W3D_100.dat", read("W3D_100.dat"))
describe("W8D_156.dat", read("W8D_156.dat"))
describe("two layers m=8, 3 + 2", np.vstack([das_dennis(8, 3), das_dennis(8, 2) / 2 + 1 / 16]))
```

Output:

```text
lattice m=3, p=12       91 rows,  91 distinct, nearest neighbour 0.118 to 0.118
W3D_91.dat              91 rows,  91 distinct, nearest neighbour 0.118 to 0.118
W3D_100.dat            100 rows, 100 distinct, nearest neighbour 0.101 to 0.119
W8D_156.dat            156 rows, 148 distinct, nearest neighbour 0.000 to 0.408
two layers m=8, 3 + 2  156 rows, 156 distinct, nearest neighbour 0.306 to 0.468
```

- `W3D_91.dat` is the lattice: every nearest-neighbour distance is
  √2/12 ≈ 0.118.
- `W3D_100.dat` is nearly as even, with distances from 0.101 to 0.119.
- `W8D_156.dat` has the size of Deb and Jain's set for eight objectives, but
  its second layer is the lattice of 2 divisions without shrinking. Its eight
  corners repeat corners of the first layer, so only 148 of its 156 rows are
  distinct. MOEA/D in VAMOS runs this file when it gets 8 objectives and 156
  individuals without weight options; `weight_vectors(divisions=3,
  inner_divisions=2)` gives the 156 distinct vectors instead.

Lattices exist only for the sizes C(p + m − 1, m − 1). Methods that place any
number of well-spaced points on the simplex produce sets that VAMOS can read
from a file. One of them, the Riesz s-energy method of Blank et al. [7], spreads
the points by minimizing the sum of 1 / d^s over all pairs, which grows without
bound as two points meet; VAMOS generates its sets for MOEA/D, PPS, NSGA-III and
RVEA with `method="riesz_energy"` and a `seed`, one vector per individual (see
[reference and weight vectors](../../reference/algorithms.md#reference-and-weight-vectors)).
With 100 vectors and three objectives its closest pair is about 0.10 apart, as
in `W3D_100.dat`. Even spacing
on the simplex is not even spacing on the front: where the solutions end up
also depends on the shape of the front and on how each algorithm uses its
vectors, as Ishibuchi et al. showed [5].

## How the three algorithms use the vectors

| | MOEA/D | NSGA-III | RVEA |
| --- | --- | --- | --- |
| A vector is | the weights of a scalar subproblem g(x \| w, z\*) | a ray from the ideal point through w, in normalized objectives | the unit vector w/‖w‖, a ray from the ideal point |
| Solutions meet it by | the subproblem keeping its best solution; neighbourhoods are the T closest weight vectors | association at the smallest perpendicular distance, then niche counts | association at the smallest angle, then the angle-penalized distance |
| Population | exactly one solution per vector | at least one per direction, N ≥ H | exactly one per vector; it shrinks when vectors are empty |
| During the run | fixed | fixed; the objectives are normalized each generation | stretched to the ranges of the objectives every f_r · t_max generations |

The table hides one difference that matters. NSGA-III and RVEA look for
solutions **along** each vector. MOEA/D with the Tchebycheff function does not:
the optimum of max_k w_k |f_k − z\*_k| lies where every w_k (f_k − z\*_k) is the
same, on the ray from z\* with direction (1/w₁, …, 1/w_m) (lesson
[MOEA/D](../intermediate/moead.md#aggregation-functions)). Evenly spaced weights
therefore ask for unevenly spaced solutions. VAMOS's
`aggregation("inverse_weight_tchebycheff")`, max_k |f_k − z\*_k| / w_k [6],
moves the optimum onto the ray along w itself.

The lessons [MOEA/D](../intermediate/moead.md), [NSGA-III](nsga-iii.md) and
[RVEA](rvea.md) explain each mechanism and how to configure it in VAMOS.

## In VAMOS: one source of vectors

Since VAMOS 1.5.0, MOEA/D's `weight_vectors`, NSGA-III's
`reference_directions` and RVEA's `reference_vectors` take the same options,
as builder methods of the same names:

| Options | Vectors |
| --- | --- |
| `path=` a file | One vector per line, components separated by whitespace or commas. Rows must be non-negative and sum to 1 within jMetal's rounding (10⁻⁵); they are not renormalized. |
| `path=` a directory | The file `W<m>D_<N>.dat` of the directory, for m objectives and a population of N, such as the directory of jMetal files that VAMOS ships. |
| `divisions=p` | The complete Das and Dennis lattice of p divisions. A lattice is never truncated. |
| `divisions=p₁, inner_divisions=p₂` | Deb and Jain's two-layer lattice, boundary rows first. It excludes `path`. |

The vectors are the population: MOEA/D and RVEA need exactly one individual per
vector, NSGA-III at least one. Without options, each algorithm keeps its own
rule. MOEA/D uses N evenly spaced vectors with two objectives and, with more,
the lattice of exactly N vectors or the jMetal file of that size;
`vamos.algorithms.moead_populations(m)` lists the populations it accepts.
NSGA-III takes the largest complete lattice that fits in its population, and
RVEA the lattice of `n_partitions` divisions (12 by default), which must have
exactly N vectors.

Given the same options, the three algorithms start from the same vectors.
Here they all read `W3D_100.dat` from the directory of shipped files, on
DTLZ2 [8], whose front lies on the unit sphere:

```python
from importlib import resources

import numpy as np

from vamos import optimize
from vamos.algorithms import MOEADConfig, NSGAIIIConfig, RVEAConfig
from vamos.problems import DTLZ2

weights = str(resources.files("vamos.resources") / "weights")  # a directory of jMetal's W<m>D_<n>.dat files
configs = {
    "moead": MOEADConfig.builder()
    .pop_size(100)
    .neighbor_size(20)
    .delta(0.9)
    .replace_limit(2)
    .crossover("sbx", prob=1.0, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .aggregation("pbi", theta=5.0)
    .weight_vectors(path=weights)
    .result_mode("population")
    .build(),
    "nsgaiii": NSGAIIIConfig.builder()
    .pop_size(100)
    .crossover("sbx", prob=1.0, eta=30.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("random")
    .reference_directions(path=weights)
    .result_mode("population")
    .build(),
    "rvea": RVEAConfig.builder()
    .pop_size(100)
    .alpha(2.0)
    .adapt_freq(0.1)
    .crossover("sbx", prob=1.0, eta=30.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .reference_vectors(path=weights)
    .result_mode("population")
    .build(),
}
problem = DTLZ2(n_var=12, n_obj=3)
for name, config in configs.items():
    result = optimize(problem, algorithm=name, algorithm_config=config, max_evaluations=5_000, seed=1)
    radius = np.linalg.norm(result.F, axis=1)  # 1 on DTLZ2's front
    print(f"{name:<8} {len(result.F)} solutions, median distance to the origin {np.median(radius):.3f}")
```

Output:

```text
moead    100 solutions, median distance to the origin 1.023
nsgaiii  100 solutions, median distance to the origin 1.017
rvea     100 solutions, median distance to the origin 1.017
```

When a run names a path, VAMOS records the file it read and the SHA-256 digest
of its vectors. Verification needs neither the file nor the machine it was on,
and a replay refuses, before any evaluation, a file whose vectors have changed.

The two-layer lattice needs no file. NSGA-III with eight objectives and
p₁ = 3, p₂ = 2 needs at least 156 individuals, and VAMOS checks this before the
first evaluation:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIIConfig, moead_populations
from vamos.problems import DTLZ2

print("MOEA/D runs these populations with 8 objectives:", moead_populations(8, max_pop_size=400))

problem = DTLZ2(n_var=17, n_obj=8)
for pop_size in (150, 156):
    config = (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=1.0, eta=30.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("random")
        .reference_directions(divisions=3, inner_divisions=2)  # 120 + 36 = 156 directions
        .result_mode("population")
        .build()
    )
    try:
        result = optimize(problem, algorithm="nsgaiii", algorithm_config=config, max_evaluations=7_800, seed=1)
    except ValueError as error:
        print(error)
        continue
    radius = np.linalg.norm(result.F, axis=1)  # 1 on DTLZ2's front
    print(f"pop_size={pop_size}: {len(result.F)} solutions, median distance to the origin {np.median(radius):.3f}")
```

Output:

```text
MOEA/D runs these populations with 8 objectives: (8, 36, 120, 156, 330)
NSGA-III reference directions for divisions=3 and inner_divisions=2 (156) need pop_size >= 156 (pop_size=150). Increase pop_size or choose fewer reference directions.
pop_size=156: 156 solutions, median distance to the origin 1.104
```

MOEA/D's list holds the lattices of 1 to 4 divisions (8, 36, 120 and 330
vectors) and 156, the size of the jMetal file. Other sizes need explicit
weights, a file or two layers.

## Key points

- A Das and Dennis lattice with p divisions has C(p + m − 1, m − 1) vectors,
  all √2/p from their nearest neighbours; only C(p − 1, m − 1) of them, none
  when p &lt; m, have every coordinate positive.
- With many objectives a lattice of few divisions lies on the boundary of the
  simplex, and one with interior vectors is huge. Deb and Jain add an inside
  layer of few divisions, shrunk towards the centre: 156 vectors for eight
  objectives with p₁ = 3 and p₂ = 2.
- Weight files fill sizes that lattices lack; check their uniformity, for
  example with nearest-neighbour distances, and watch for repeated rows.
- MOEA/D scalarizes with each vector, NSGA-III counts solutions per direction,
  and RVEA penalizes angles to each vector; with the Tchebycheff function,
  MOEA/D's solutions lie along 1/w rather than w.
- In VAMOS the three algorithms take their vectors from a file, a directory,
  a lattice or two layers, through the same options.

**Next:** [NSGA-III: reference-point niching](nsga-iii.md).

## References

[1] Das, I., and Dennis, J. E. (1998). [Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, 8(3), 631–657.

[2] Zhang, Q., and Li, H. (2007). [MOEA/D: A multiobjective evolutionary algorithm based on decomposition](https://doi.org/10.1109/TEVC.2007.892759). *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

[3] Deb, K., and Jain, H. (2014). [An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints](https://doi.org/10.1109/TEVC.2013.2281535). *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.

[4] Cheng, R., Jin, Y., Olhofer, M., and Sendhoff, B. (2016). [A reference vector guided evolutionary algorithm for many-objective optimization](https://doi.org/10.1109/TEVC.2016.2519378). *IEEE Transactions on Evolutionary Computation*, 20(5), 773–791.

[5] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[6] Qi, Y., Ma, X., Liu, F., Jiao, L., Sun, J., and Wu, J. (2014). [MOEA/D with adaptive weight adjustment](https://doi.org/10.1162/EVCO_a_00109). *Evolutionary Computation*, 22(2), 231–264.

[7] Blank, J., Deb, K., Dhebar, Y., Bandaru, S., and Seada, H. (2021). [Generating well-spaced points on a unit simplex for evolutionary many-objective optimization](https://doi.org/10.1109/TEVC.2020.2992387). *IEEE Transactions on Evolutionary Computation*, 25(1), 48–60.

[8] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.
