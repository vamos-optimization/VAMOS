---
description: 'A3. RVEA, the reference vector guided evolutionary algorithm: unit reference vectors, translation by the ideal point, association by angle, the angle-penalized distance and its schedule, and the adaptation of the vectors to the scales of the objectives, as VAMOS implements them, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# A3. RVEA: angle-penalized distance

NSGA-III (lesson [A2](nsga-iii.md)) still sorts the population into fronts and
uses its directions only to cut the last one. RVEA, the reference vector
guided evolutionary algorithm of Cheng, Jin, Olhofer and Sendhoff [1], drops
dominance altogether. Each reference vector keeps one solution, chosen by a
single number, the **angle-penalized distance** (APD): the distance to the
ideal point measures convergence, the angle to the vector measures diversity.
The weight of the angle grows during the run, so RVEA first pushes the
population towards the front and then spreads it along the vectors.

A second idea replaces NSGA-III's hyperplane: from time to time the reference
vectors themselves are stretched to the ranges of the objectives, so that
objectives of different scales still receive evenly spread solutions.

In this lesson you will learn:

- how RVEA builds its unit reference vectors, and why VAMOS requires exactly
  one individual per vector;
- how solutions are translated by the ideal point and associated with vectors
  by angle;
- how the angle-penalized distance works, and what its parameters α and γ do;
- how and when the vectors adapt to the scales of the objectives;
- where VAMOS follows the article and where it follows pymoo.

## From niches to subproblems

Decomposition-based algorithms divide the objective space among a set of
vectors and look for one good solution per vector. MOEA/D [3] gives each
vector a scalarizing function and lets neighbouring vectors exchange
solutions (lesson
[I6](../intermediate/beyond-nsga-ii.md#decomposition-based-algorithms));
NSGA-III [2] keeps dominance and uses the vectors to count. Cheng et al.
observed that an angle between a solution and a vector is a natural measure of
diversity that does not depend on how far the solution is from the front, and
combined it with the distance to the ideal point into one criterion that
changes its emphasis over time [1].

The article also shows that vectors restricted to a region of interest express
a decision maker's preferences, proposes a variant that regenerates the
vectors left without solutions on irregular fronts (RVEA\*), and extends the
algorithm to constraints [1]. VAMOS implements the basic RVEA with vector
adaptation.

## Reference vectors

RVEA starts from the same Das and Dennis lattice as NSGA-III [4] and
normalizes each point to a unit vector. Each vector also gets its own angular
spacing γ, the angle to its nearest neighbour:

<div class="vf-formula">vᵢ = wᵢ / ‖wᵢ‖                         (w a Das and Dennis point, ‖vᵢ‖ = 1)
γᵢ = min_(j ≠ i) arccos( vᵢᵀ vⱼ )       (smallest angle to another vector)</div>

With p divisions the lattice has H = C(p + m − 1, m − 1) vectors, and RVEA
keeps at most one solution per vector, so in VAMOS the population must have
**exactly** H individuals. `n_partitions` sets p (12 by default). A run whose
`pop_size` differs from H stops before the first evaluation with a
`ValueError`. `RVEAConfig.default(pop_size=N, n_obj=m)` looks for the lattice
with exactly N vectors and, when there is none, raises an error naming the two
nearest valid sizes. Without a population size it uses p = 99 for two
objectives, 12 for three and 6 for four or more.

## Translation and association by angle

At each generation the merged parents and offspring are translated so that the
ideal point becomes the origin. Each solution then joins the vector at the
smallest angle, forming one subpopulation per vector:

<div class="vf-formula">z_min,i = min f_i over the solutions seen so far        (ideal point)
f′(x) = f(x) − z_min
cos θ(x, vⱼ) = f′(x)ᵀ vⱼ / ‖f′(x)‖
P̄ⱼ = { x : vⱼ has the largest cos θ(x, ·) }              (subpopulation of vⱼ)</div>

Unlike NSGA-III, RVEA does not rescale the objectives here: the angle is
measured on translated, unscaled objectives, and scales are handled by
adapting the vectors (below). VAMOS keeps the smallest value of each objective
seen during the run, as pymoo does; the article computes the ideal point from
the current population [1].

## The angle-penalized distance

Within each subpopulation, RVEA keeps the solution with the smallest APD:

<div class="vf-formula">APD(x, vⱼ) = ( 1 + P(θ) ) · ‖f′(x)‖
P(θ) = M · ( t / t_max )^α · θ(x, vⱼ) / γⱼ
survivor of vⱼ = argmin over x ∈ P̄ⱼ of APD(x, vⱼ)       (for every non-empty P̄ⱼ)</div>

- **‖f′(x)‖**, the distance to the ideal point, measures convergence.
- **θ / γⱼ** is the angle to the vector relative to the local spacing of the
  vectors: the same angle weighs more where the vectors are dense.
- **M**, the number of objectives, scales the penalty because solutions spread
  over more angles in higher dimension.
- **(t / t_max)^α** is the schedule. At the start (t ≈ 0) the penalty vanishes
  and each vector keeps the solution closest to the ideal point, whatever its
  angle. At the end it keeps a solution well aligned with the vector. The
  exponent α (2 by default, as in the article) decides how long RVEA favours
  convergence: the larger α, the later the angle starts to count.

A worked example: two solutions in the cone of a vector with γ = 18.4°, a at
distance 0.80 and 9.6° off the vector, b at distance 0.95 and 0.4° off. With
M = 1 and α = 2, a has the smaller APD until 0.80·(1 + s·9.6/18.4) exceeds
0.95·(1 + s·0.4/18.4), that is until s = (t/t_max)² ≈ 0.38, or
t/t_max ≈ 0.62. From then on b survives.

A vector whose subpopulation is empty keeps nothing, so the population can
become smaller than N, as in the article, PlatEMO and pymoo. The next
generation still creates N offspring. Parents are drawn at random: VAMOS takes
them from random permutations of the population, without replacement, as
pymoo does.

VAMOS details and deviations:

- **M = 1 with two objectives.** VAMOS, like pymoo, uses M = 1 instead of 2 when
  m = 2, which halves the penalty; with three or more objectives M = m.
- **The schedule.** The first survival uses t = 1, and t_max is the number of
  generations the budget allows, ⌈(max_evaluations − N) / N⌉: 219 for 20,000
  evaluations with N = 91.
- **Ties.** On equal APD, the solution listed first wins.
- **VAMOS 1.5.0: two variants.** `variant("pymoo")`, the default,
  makes the choices above. `variant("platemo")` follows the authors' code in
  PlatEMO, which jMetal also follows: M = m with two objectives too, the
  objectives are translated by the minimum of each generation's candidates
  instead of the accumulated ideal point, and the vectors adapt to the range of
  the population (below).

## Explore the penalty

The figure has five reference vectors (p = 4) drawn from the ideal point of
twelve solutions. Vector v3 has no solution in its cone. Move t/t_max from 0
to 1 with α = 2: at first every vector keeps its solution closest to the ideal
point, then v1, v2 and v4 switch, one after the other, to a better-aligned
solution; v3 keeps nothing, so four solutions survive. Change α to 1 and to 5
to move the switches earlier and later. Finally tick the article's M = 2: the
penalty doubles, and even v5 abandons its near solution 11 for the distant but
perfectly aligned 12.

<div class="vf-widget" data-vf-widget="rvea-apd"><p class="vf-fallback">Interactive figure: RVEA's angle-penalized distance for five reference vectors and twelve solutions, with sliders for the progress t/t_max and the exponent α, the solution each vector selects, and a table of the APD values. It needs JavaScript.</p></div>

## Adapting the reference vectors

If one objective ranges over [0, 1] and another over [0, 100], evenly spread
unit vectors crowd the solutions into a small part of the front. Every
f_r · t_max generations RVEA stretches the initial vectors to the current
ranges and normalizes them again:

<div class="vf-formula">vᵢ ← ( v⁰ᵢ ∘ (z_max − z_min) ) / ‖ v⁰ᵢ ∘ (z_max − z_min) ‖     every ⌈f_r · t_max⌉ generations</div>

Here ∘ multiplies element by element, v⁰ᵢ are the initial unit vectors, z_max
is the largest value of each objective in the current population and z_min is
the ideal point. The spacings γᵢ are recomputed from the new vectors. The
article recommends f_r = 0.1, so the vectors adapt ten times per run [1]:
adapting at every generation would make them chase a population that is still
far from the front.

In VAMOS, `adapt_freq(f_r)` sets the frequency (0.1 by default) and
`adapt_freq(None)` disables adaptation. Adaptation happens after the survival
of every generation t that is a multiple of ⌈f_r · t_max⌉, with z_max taken
from the survivors. With constraints only feasible solutions define z_min and
z_max, and the vectors do not adapt until a feasible solution exists. The
`variant("platemo")` scales instead by the range of the current population,
its largest minus its smallest value of each objective.

## In VAMOS

The example first asks `RVEAConfig.default` for a population that matches no
lattice. It then runs RVEA on WFG4 [6], whose front
(f₁/2)² + (f₂/4)² + (f₃/6)² = 1 has very different scales, with and without
adaptation, and counts how many of the 91 Das and Dennis directions are the
nearest direction of at least one solution once the front is mapped onto the
unit sphere:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import RVEAConfig
from vamos.problems import WFG4

try:
    RVEAConfig.default(pop_size=100, n_obj=3)
except ValueError as error:
    print(error)

problem = WFG4(n_var=12, n_obj=3)  # front: (f₁/2)² + (f₂/4)² + (f₃/6)² = 1
W = np.array([(i, j, 12 - i - j) for i in range(13) for j in range(13 - i)], dtype=float)
W /= np.linalg.norm(W, axis=1, keepdims=True)  # the 91 Das and Dennis directions, p = 12


def occupied(Z):
    """Number of directions that are the nearest direction of at least one point."""
    cosine = (Z / np.linalg.norm(Z, axis=1, keepdims=True)) @ W.T
    return np.unique(np.argmax(cosine, axis=1)).size


for adapt_freq in (0.1, None):
    config = (
        RVEAConfig.builder()
        .pop_size(91)
        .n_partitions(12)
        .alpha(2.0)
        .adapt_freq(adapt_freq)
        .crossover("sbx", prob=1.0, eta=30.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .result_mode("population")
        .build()
    )
    result = optimize(problem, algorithm="rvea", algorithm_config=config, max_evaluations=20_000, engine="numpy", seed=1)
    scaled = result.F / np.array([2.0, 4.0, 6.0])  # maps the true front onto the unit sphere
    print(f"adapt_freq={adapt_freq}: {len(result.F)} solutions, {occupied(scaled)} of 91 directions occupied")
```

Output:

```text
RVEA needs one individual per reference vector, and no Das-Dennis lattice for n_obj=3 has 100 vectors. Use pop_size=91 (12 partitions) or pop_size=105 (13 partitions).
adapt_freq=0.1: 91 solutions, 90 of 91 directions occupied
adapt_freq=None: 91 solutions, 65 of 91 directions occupied
```

With adaptation the solutions cover almost every direction of the scaled
front. Without it the vectors stay evenly spread in the raw objective space,
and the solutions follow them, leaving a large part of the stretched front
empty.

The vectors and their spacings come from two small functions. `_calc_gamma`
sorts each row of the cosine matrix in decreasing order; the first entry is
the vector itself (cosine 1), so the second is its nearest neighbour:

```python
# Excerpt from src/vamos/engine/algorithm/rvea/rvea.py
def _calc_V(ref_dirs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(ref_dirs / norms, dtype=float)


def _calc_gamma(V: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    cosine = V @ V.T
    gamma = np.arccos((-np.sort(-1.0 * cosine, axis=1))[:, 1])
    gamma = np.maximum(gamma, 1e-64)
    return np.asarray(gamma, dtype=float)
```

The survival follows the formulas. `angle_penalized_distances` translates the
objectives by an origin, associates each solution with the vector at the
smallest angle (`niches`), and computes the penalty, with M = 1 for two
objectives in the default variant, and the APD. `apd_survival` passes the
accumulated ideal point as the origin. `np.lexsort` orders the solutions by
vector, then by APD, then by position, so the first solution of each vector in
that order is its survivor; vectors without solutions simply do not appear:

```python
# Excerpt from src/vamos/engine/algorithm/rvea/survival.py
    n_obj = F.shape[1]
    F_shift = F - origin
    dist_to_origin = np.linalg.norm(F_shift, axis=1)
    dist_to_origin[dist_to_origin < 1e-64] = 1e-64
    F_prime = F_shift / dist_to_origin[:, None]

    cos_theta = np.clip(F_prime @ V.T, -1.0, 1.0)
    acute_angle = np.arccos(cos_theta)
    niches = acute_angle.argmin(axis=1)

    M = float(n_obj) if n_obj > 2 or variant == "platemo" else 1.0
    progress = (n_gen / n_max_gen) ** alpha
    theta = acute_angle[np.arange(F.shape[0]), niches]
    penalty = M * progress * (theta / gamma[niches])
    return niches, penalty, dist_to_origin * (1.0 + penalty)


def apd_survival(
    ideal = np.minimum(F.min(axis=0), ideal)
    origin = F.min(axis=0) if variant == "platemo" else ideal
    niches, _penalty, apd = angle_penalized_distances(F, origin, V, gamma, n_gen, n_max_gen, alpha, variant)

    order = np.lexsort((np.arange(F.shape[0]), apd, niches))
    sorted_niches = niches[order]
    first_in_niche = np.empty(sorted_niches.shape[0], dtype=bool)
    first_in_niche[0] = True
    first_in_niche[1:] = sorted_niches[1:] != sorted_niches[:-1]
    survivors = order[first_in_niche][: min(int(n_survive), F.shape[0])]

    nadir = F[survivors].max(axis=0) if survivors.size else None
    return survivors, ideal, nadir
```

The last lines return the largest objective values of the survivors, the
z_max of the adaptation. The run computes t_max from the budget, builds one
reference vector per individual (`_reference_vectors` raises when the lattice
does not match `pop_size`), sets the adaptation interval, and stretches the
vectors at each multiple of it:

```python
# Excerpt from src/vamos/engine/algorithm/rvea/rvea.py
        # The APD penalty grows with generation / max_gen; n_gen gives max_gen directly.
        if termination[0] == "n_gen":
            max_gen = max_evals // pop_size
        else:
            max_gen = max(1, int(math.ceil((max_evals - pop_size) / pop_size)))

        ref_dirs = _reference_vectors(self.cfg.get("reference_vectors"), pop_size, n_obj, n_partitions)
        adapt_interval = None
        if adapt_freq is not None:
            adapt_interval = max(1, int(math.ceil(max_gen * float(adapt_freq))))
        # Periodic reference-vector adaptation
        if st.adapt_interval is not None and st.generation % st.adapt_interval == 0:
            scale: np.ndarray[Any, Any] | None = None
            if st.variant == "platemo":
                scale = np.ptp(st.F, axis=0)
            elif st.nadir is not None:
                scale = st.nadir - st.ideal
            if scale is not None:
                st.V = _calc_V(_calc_V(st.ref_dirs) * np.maximum(scale, 1e-64))
                st.gamma = _calc_gamma(st.V)
```

**VAMOS 1.5.0:** the population no longer has to be a complete
lattice. `reference_vectors(path=...)` reads the vectors from a file, such as
jMetal's `W3D_100.dat` that VAMOS ships, and
`reference_vectors(divisions=p₁, inner_divisions=p₂)` builds Deb and Jain's
two-layer lattice (lesson [A2](nsga-iii.md#reference-directions)).

The [RVEA lesson](../../algorithms/rvea.md#interpret-the-result-and-its-limits)
records a run on DTLZ2 and the changes of VAMOS 1.2.0; the
[configuration reference](../../reference/api/algorithms/rvea.md) lists every
builder method. With constraints, RVEA keeps feasible solutions first and
applies the APD among them; `constraint_handling` accepts the techniques that
judge one solution at a time (lesson [A5](constrained.md)).

## RVEA, NSGA-III and MOEA/D

| | NSGA-III | RVEA | MOEA/D |
| --- | --- | --- | --- |
| Convergence | Non-dominated rank | Distance to the ideal point | Scalarizing function |
| Diversity | Niche counts per direction | Angle to the vector, penalized more over time | One subproblem per vector |
| Scales | Normalization every generation | Vectors stretched every f_r · t_max generations | None in the original algorithm |
| Population | N ≥ H | N = H; can shrink when vectors are empty | N = number of vectors |

**Strengths.** Selection needs no non-dominated sorting, so it stays cheap and
keeps its selection pressure when almost every solution is non-dominated. The
schedule makes the balance between convergence and diversity explicit, and the
vectors can encode preferences.

**Limits.** The schedule needs t_max, so the budget must be known in advance.
Empty vectors shrink the population, and the returned set can be smaller than
`pop_size`. RVEA never checks dominance, so a survivor can be dominated by
another one (`result_mode("non_dominated")`, the default, filters the
returned set). Like every method with fixed vectors, it suffers on fronts
whose shape does not match the simplex: degenerate, disconnected or inverted
fronts [5]. Lesson [A4](agemoea.md) takes the opposite approach and adapts to
the shape of the front instead.

## Key points

- RVEA keeps one solution per unit reference vector, chosen by the
  angle-penalized distance APD = (1 + P(θ))·‖f′‖; VAMOS requires
  `pop_size` to equal the number of vectors.
- The penalty P(θ) = M·(t/t_max)^α·θ/γ grows during the run: convergence first,
  alignment with the vectors later; α sets how late.
- γ, the smallest angle between a vector and its neighbours, makes the penalty
  relative to the local spacing of the vectors.
- Every f_r · t_max generations the vectors are stretched to the ranges of the
  population, which lets RVEA handle objectives of different scales.
- By default VAMOS follows pymoo where the article differs: M = 1 with two
  objectives and an ideal point that accumulates across generations;
  `variant("platemo")` follows the authors' code instead.

**Next:** [A4. AGE-MOEA: adapting to the geometry of the front](agemoea.md).

## References

[1] Cheng, R., Jin, Y., Olhofer, M., and Sendhoff, B. (2016). [A reference vector guided evolutionary algorithm for many-objective optimization](https://doi.org/10.1109/TEVC.2016.2519378). *IEEE Transactions on Evolutionary Computation*, 20(5), 773–791.

[2] Deb, K., and Jain, H. (2014). [An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints](https://doi.org/10.1109/TEVC.2013.2281535). *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.

[3] Zhang, Q., and Li, H. (2007). [MOEA/D: A multiobjective evolutionary algorithm based on decomposition](https://doi.org/10.1109/TEVC.2007.892759). *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

[4] Das, I., and Dennis, J. E. (1998). [Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, 8(3), 631–657.

[5] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[6] Huband, S., Hingston, P., Barone, L., and While, L. (2006). [A review of multiobjective test problems and a scalable test problem toolkit](https://doi.org/10.1109/TEVC.2005.861417). *IEEE Transactions on Evolutionary Computation*, 10(5), 477–506.
