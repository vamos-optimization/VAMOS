---
description: 'A2. NSGA-III step by step as VAMOS implements it: Das and Dennis reference directions and the population rule, adaptive normalization with extreme points and intercepts, association by perpendicular distance, niche counts and the niche-preservation loop, mating selection and constraints, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# A2. NSGA-III: reference-point niching

Lesson [A1](many-objective.md#nsga-iii-in-detail) introduced NSGA-III [1] as
NSGA-II with a different rule for the last front: reference directions instead
of crowding distance. This lesson follows that rule into the VAMOS code, one
mechanism at a time, and shows where VAMOS makes a choice that the paper
leaves open.

NSGA-III was proposed by Deb and Jain in 2014 for problems with four or more
objectives; a companion paper extended it to constraints and to reference
points that move during the run [2]. Merging parents and offspring,
non-dominated sorting and filling whole fronts are inherited unchanged from
NSGA-II (lessons [I2](../intermediate/non-dominated-sorting.md) to
[I5](../intermediate/elitist-survival.md)). What changes is how the last
front is cut, and how parents are chosen.

In this lesson you will learn:

- how VAMOS builds the reference directions, and why it refuses a population
  smaller than their number;
- how adaptive normalization finds the ideal point, the extreme points and the
  intercepts, and what happens when the hyperplane is degenerate;
- how solutions are associated with directions, and how niche counts and the
  niche-preservation loop fill the last places;
- how VAMOS chooses parents and treats constraints.

## Why reference points

NSGA-II measures density *between solutions*: the crowding distance sums the
gaps to the neighbours in each objective (lesson
[I3](../intermediate/crowding-distance.md)). With many objectives nearly every
solution is extreme in some objective, and the gaps stop describing how
crowded a region is. Deb and Jain measure density *against a fixed set of
directions* supplied before the run [1]. The directions come from
decomposition: MOEA/D [4] solves one scalar subproblem per weight vector, and
the systematic design of Das and Dennis [3] spreads the weights evenly.
NSGA-III keeps the directions but not the scalarization: a direction never
ranks solutions, it only counts how many survivors it already has.

## Reference directions

A Das and Dennis lattice with p *divisions* contains every point of the unit
simplex whose m coordinates are multiples of 1/p. Each point w defines a ray
from the origin, a reference direction:

<div class="vf-formula">w = (k₁, …, k_m) / p,     k_i ∈ {0, 1, …, p},     k₁ + … + k_m = p
H = C(p + m − 1, m − 1)        (number of directions)</div>

| m | p | H |
| --- | --- | --- |
| 2 | 99 | 100 |
| 3 | 12 | 91 |
| 5 | 6 | 210 |
| 8 | 3 | 120 |

VAMOS generates the complete lattice and never truncates it. The builder
method `reference_directions(divisions=p)` fixes p; without it, VAMOS takes the
largest complete lattice that fits in the population. `NSGAIIIConfig.default`
uses p = 99 for two objectives, 12 for three and 6 for four or more, with
N = H.

With eight or more objectives a single lattice either has too many directions
or puts them all on the boundary of the simplex. Deb and Jain then combine two
layers: an outer lattice with p₁ divisions and an inner one with p₂ divisions
shrunk towards the centre, for example 120 + 36 = 156 directions for m = 8
with p₁ = 3 and p₂ = 2 [1]. **VAMOS 1.5.0:**
`reference_directions(divisions=p₁, inner_divisions=p₂)` generates that
two-layer set, the inner lattice shrunk by half towards the centre,
w/2 + 1/(2m), as pymoo does; VAMOS 1.4.0 generated one layer only. Any other
set can be read from a file with `reference_directions(path=...)`: rows that
are non-negative, sum to one and have one column per objective, separated by
commas or, since the same development, by whitespace, as in jMetal's weight
files.

### The population rule

Every direction must be able to keep a survivor, so the population must hold
at least one solution per direction:

<div class="vf-formula">N ≥ H</div>

Deb and Jain used the smallest multiple of four that is at least H, for
example N = 92 for H = 91 [1]. VAMOS accepts any N ≥ H and checks the rule
before the first evaluation. A smaller population stops with a `ValueError`
that names H and asks for a larger population or fewer directions; VAMOS
neither enlarges the population nor drops directions.

## Adaptive normalization

The directions live on the unit simplex, but objectives can have any scale.
Before associating solutions with directions, NSGA-III maps the merged
population onto a space where the front's corners lie on the unit axes:

<div class="vf-formula">z*_i = min f_i over the solutions evaluated so far            (ideal point)
f′_i(x) = f_i(x) − z*_i                                       (translation)
ASF(x, wʲ) = max_i  f′_i(x) / wʲ_i,    wʲ_i = 1 if i = j, 10⁻⁶ otherwise
zʲ = the solution that minimizes ASF(·, wʲ)                   (extreme point of axis j)
a₁, …, a_m = intercepts of the hyperplane through z¹, …, z^m
fⁿ_i(x) = f′_i(x) / a_i                                       (normalized objectives)</div>

The achievement scalarizing function (ASF) of Wierzbicki [6] with a weight of
10⁻⁶ on every other objective is minimized by the solution that is smallest in
all the other objectives, the corner of the front on axis j. The m extreme
points span a hyperplane, and its intercepts with the axes estimate how far
the front extends in each objective: the nadir point is estimated as z* + a.
Dividing by the intercepts maps the extreme points onto the unit axes.

VAMOS makes these choices:

- The ideal point is the smallest value of each objective seen during the
  whole run, not only in the current population.
- Translated values below 10⁻³ are set to 0 before the ASF, and the extreme
  points are searched among the first front **and** the previous generation's
  extreme points, so a good corner is not lost when it leaves the population.
- The hyperplane is rejected when its linear system is singular or not
  satisfied, when an intercept is not above 10⁻⁶, or when the estimated nadir
  exceeds, in some objective, the worst value seen so far. VAMOS then uses the
  worst point of the first front for every objective. An objective whose range
  is still not above 10⁻⁶ takes the worst value of the merged population.

The paper does not fix the fallback, and libraries differ: pymoo 0.6 replaces
only the objectives whose intercept is out of range. The
[NSGA-III lesson](../../algorithms/nsgaiii.md#variants-and-their-effect)
records the difference.

## Association by perpendicular distance

Each normalized solution is attached to the direction whose ray passes
closest to it:

<div class="vf-formula">d⊥(x, w) = ‖ fⁿ(x) − ( wᵀfⁿ(x) / ‖w‖² ) · w ‖ = ‖fⁿ(x)‖ · sin θ(x, w)
π(x) = argmin_w  d⊥(x, w)                       (the direction of x)</div>

For one solution ‖fⁿ(x)‖ is fixed, so the closest ray is also the one at the
smallest angle θ. VAMOS uses that: it takes the direction with the largest
cosine and computes the distance as ‖fⁿ‖·√(1 − cos²θ). The distance d⊥ is a
second, local measure of quality: a solution close to its ray represents its
direction well.

## Niche counts and niche preservation

Let F_l be the last front, the first one that does not fit. The niche count
ρⱼ of direction j counts the solutions of the earlier fronts, all of which
survive, associated with it. The free places are then filled from F_l:

<div class="vf-formula">ρⱼ ← number of solutions of F₁ ∪ … ∪ F_(l−1) associated with wⱼ
while fewer than N solutions are selected:
    J ← the directions with at least one unselected member of F_l
    J_min ← the directions of J with the smallest ρⱼ, in random order
    for each j in J_min, while places are left:
        if ρⱼ = 0:   add the member of F_l on wⱼ with the smallest d⊥
        else:        add a random member of F_l on wⱼ
        ρⱼ ← ρⱼ + 1</div>

The least crowded direction is served first. An empty direction receives its
best representative, the closest candidate; a direction that already has
survivors receives a random one, because closeness to the ray no longer
matters once the direction is represented. A direction without candidates in
F_l is ignored for the rest of the generation, so it can stay empty.

Deb and Jain serve one direction at a time and break ties between directions at
random [1]. VAMOS, like pymoo, serves all the tied directions of J_min in one
shuffled round. The outcome has the same distribution: once a direction has
been served its count exceeds that of the others in the tie, so the
one-at-a-time rule would choose among the remaining ones anyway.

## Step through the niche loop

The figure shows one generation with two objectives, already normalized (the
ideal point is the origin and both intercepts are 1), six directions
(p = 5) and N = 9. Front 1 fits whole and fills five places; front 2 is the
last front and has four; front 3 is never examined. Press **Step**: round 1
serves the two empty directions w2 and w5 with their closest candidates, G
and K. Round 2 serves w2 and w3, now at count 1, with random candidates: press
**Reshuffle** to see the other choice on w3. Notice F, almost on ray w1 but
never chosen, because w1 already holds two solutions; set N = 10 to let it
compete. With N = 6 a single place is disputed by two empty directions.

<div class="vf-widget" data-vf-widget="nsga3-niching"><p class="vf-fallback">Interactive figure: NSGA-III niching in a normalized two-objective space, with six reference directions, the solutions already selected, the last-front candidates, their perpendicular distances and niche counts, stepped until the population is full. It needs JavaScript.</p></div>

## Mating selection and constraints

**Parents.** Deb and Jain choose parents at random, since niching already
keeps the population diverse [1]; their constrained version runs a binary
tournament on constraint violation [2]. VAMOS offers two selections, and
neither uses the crowding distance, which is not part of NSGA-III:

- `selection("random")`, the default, draws the parents at random, as Deb and
  Jain do. With constraints, each parent is the winner of a binary tournament
  on violation: a feasible solution beats an infeasible one, the smaller
  violation wins, and a tie, such as two feasible solutions, is drawn at
  random [2]. pymoo's NSGA-III does the same.
- `selection("tournament", size=k)` is jMetal's NSGA-III tournament: k
  distinct solutions (2 by default) are drawn at random and the one with the
  lowest non-dominated rank wins, with ties drawn at random.

**VAMOS 1.5.0:** VAMOS 1.4.0 and earlier chose NSGA-III's parents
by NSGA-II's tournament on rank and crowding distance
(lesson [I3](../intermediate/crowding-distance.md)), whatever the configured
selection, so the same seed gives other results than with VAMOS 1.4.0.

**Constraints.** With constraint values, VAMOS puts feasible solutions first,
following Jain and Deb [2]. When at most N merged solutions are feasible, all
of them survive and the least violated infeasible ones fill the remaining
places, with no niching. Otherwise the infeasible ones are discarded and
sorting, normalization and niching run on the feasible solutions only. The
rank-based tournament ranks every infeasible solution after the feasible ones
and compares two infeasible solutions by violation. Since VAMOS 1.2.0,
`constraint_handling` replaces this rule by another technique, whose fronts
take the place of the non-dominated ones in the survival and in the
rank-based tournament, while the random selection then ignores constraints (lesson
[A5](constrained.md) and the
[constraint reference](../../reference/constraints.md#choose-a-constraint-handling-technique)).

## In VAMOS

WFG4 [7] has a concave front with very different scales,
(f₁/2)² + (f₂/4)² + (f₃/6)² = 1. The example first asks for a population
smaller than H, then runs NSGA-III with N = H = 91 and counts how many of the
91 directions are the nearest direction of at least one returned solution,
once on objectives normalized by their range and once on the raw objectives:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIIConfig
from vamos.problems import WFG4

problem = WFG4(n_var=12, n_obj=3)  # front: (f₁/2)² + (f₂/4)² + (f₃/6)² = 1


def config(pop_size):
    return (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=1.0, eta=30.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("random")
        .reference_directions(divisions=12)  # H = C(14, 2) = 91 directions
        .result_mode("population")
        .build()
    )


try:
    optimize(problem, algorithm="nsgaiii", algorithm_config=config(80), max_evaluations=20_000, engine="numpy", seed=1)
except ValueError as error:
    print(error)

result = optimize(problem, algorithm="nsgaiii", algorithm_config=config(91), max_evaluations=20_000, engine="numpy", seed=1)
F = result.F

W = np.array([(i, j, 12 - i - j) for i in range(13) for j in range(13 - i)], dtype=float)
W /= np.linalg.norm(W, axis=1, keepdims=True)  # the 91 directions as unit vectors


def occupied(Z):
    """Number of directions that are the nearest direction of at least one point."""
    cosine = (Z / np.linalg.norm(Z, axis=1, keepdims=True)) @ W.T
    return np.unique(np.argmax(cosine, axis=1)).size


ideal, nadir = F.min(axis=0), F.max(axis=0)
print("objective maxima:", nadir.round(1))
print("directions with a solution, normalized objectives:", occupied((F - ideal) / (nadir - ideal)), "of 91")
print("directions with a solution, raw objectives:", occupied(F - ideal), "of 91")
```

Output:

```text
NSGA-III reference directions for divisions=12 (91) need pop_size >= 91 (pop_size=80). Increase pop_size or choose fewer reference directions.
objective maxima: [2. 4. 6.]
directions with a solution, normalized objectives: 91 of 91
directions with a solution, raw objectives: 65 of 91
```

In normalized space every direction holds a solution. On the raw objectives
more than a quarter of the directions look empty, because the front is
stretched along f₃:
normalization is what lets one set of directions serve objectives of different
scales.

The population rule is checked from the binomial count, before any lattice is
built:

```python
# Excerpt from src/vamos/engine/algorithm/nsgaiii/initialization.py
    elif ref_divisions is not None:
        lattice = f"divisions={ref_divisions}" if ref_inner is None else f"divisions={ref_divisions} and inner_divisions={ref_inner}"
        source = f"NSGA-III reference directions for {lattice}"
        if n_obj >= 2:
            # Reject from the binomial count: an oversized lattice can need gigabytes.
            n_directions = (
                lattice_size(n_obj, int(ref_divisions)) if ref_inner is None else two_layer_size(n_obj, int(ref_divisions), int(ref_inner))
            )
            if n_directions > pop_size:
                raise ValueError(
                    f"{source} ({n_directions}) need pop_size >= {n_directions} (pop_size={pop_size}). "
                    "Increase pop_size or choose fewer reference directions."
                )
        ref_dirs = load_or_generate_weight_vectors(
            pop_size, n_obj, divisions=int(ref_divisions), inner_divisions=None if ref_inner is None else int(ref_inner)
        )
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
```

The last line turns each lattice point into a unit vector, which the
association below needs.

The extreme points and the nadir estimate come from two helpers. In
`get_extreme_points`, `base` stacks the previous extreme points on top of the
first front; `asf` is the identity matrix with 10⁶ off the diagonal, so
row j of `vals` is the ASF of every candidate for axis j, and `argmin` picks
the extreme point of each axis. In `get_nadir_point`, `np.linalg.solve` finds
the hyperplane through the translated extreme points; any of the four
rejection tests raises `LinAlgError`, and the `except` branch falls back to
the worst point of the first front; the `mask` lines repair objectives that
still have no range:

```python
# Excerpt from src/vamos/engine/algorithm/nsgaiii/helpers.py
    base = F
    if extreme_points is not None and extreme_points.size:
        base = np.vstack([extreme_points, F])

    shifted = base - ideal_point
    shifted[shifted < 1e-3] = 0.0

    asf = np.eye(n_obj)
    asf[asf == 0] = 1e6
    vals = np.max(shifted * asf[:, None, :], axis=2)
    idx = np.argmin(vals, axis=1)
    return np.asarray(base[idx], dtype=float)


def get_nadir_point(
    try:
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)
        intercepts = 1.0 / plane
        nadir = ideal_point + intercepts
        if not np.allclose(M @ plane, b) or np.any(intercepts <= 1e-6) or np.any(nadir > worst_point):
            raise np.linalg.LinAlgError
    except Exception:
        nadir = worst_of_front

    mask = nadir - ideal_point <= 1e-6
    nadir = nadir.copy()
    nadir[mask] = worst_of_population[mask]
    return np.asarray(nadir)
```

Association and niching follow the formulas line by line. `associate` takes
the largest cosine and converts it to a perpendicular distance. In
`niche_selection`, `unique_refs` is J, `candidate_refs` is J_min after the
shuffle, cut to the places left; inside the loop, `candidates` is shuffled
before the choice, so `argmin` breaks ties in distance at random and
`candidates[0]` is a random member:

```python
# Excerpt from src/vamos/engine/algorithm/nsgaiii/helpers.py
    norms = np.linalg.norm(normalized_F, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1e-12)
    normalized_vectors = normalized_F / norms
    cosine = normalized_vectors @ ref_dirs_norm.T
    cosine = np.clip(cosine, -1.0, 1.0)
    associations = np.argmax(cosine, axis=1)
    cos_selected = cosine[np.arange(cosine.shape[0]), associations]
    distances = norms.flatten() * np.sqrt(1.0 - np.square(cos_selected))
    return associations, distances


def niche_selection(
    mask = np.ones(front.size, dtype=bool)
    while len(selected) < n_remaining:
        remaining = n_remaining - len(selected)
        active_pos = np.where(mask)[0]
        if active_pos.size == 0:
            break

        assoc_front = associations[front[active_pos]]
        unique_refs = np.unique(assoc_front)
        ref_counts = niche_counts[unique_refs]
        min_count = np.min(ref_counts)
        candidate_refs = unique_refs[ref_counts == min_count]
        rng.shuffle(candidate_refs)
        candidate_refs = candidate_refs[:remaining]

        for ref_choice in candidate_refs:
            # Never empty: every candidate reference is associated with an active row.
            candidates = active_pos[assoc_front == ref_choice]
            rng.shuffle(candidates)
            if niche_counts[ref_choice] == 0:
                cand_dist = distances[front[candidates]]
                pick = candidates[int(np.argmin(cand_dist))]
            else:
                pick = candidates[0]
            mask[pick] = False
            selected.append(int(front[pick]))
            niche_counts[ref_choice] += 1
```

The [NSGA-III lesson](../../algorithms/nsgaiii.md) runs the DTLZ2 example of
lesson A1, draws its result and records the configuration choices; the
[configuration reference](../../reference/api/algorithms/nsgaiii.md) lists
every builder method.

## NSGA-III among its relatives

- **NSGA-II.** The two share everything except the cut of the last front and
  the parent selection. With two or three objectives they often perform
  similarly; NSGA-III then mainly adds the requirement N ≥ H.
- **MOEA/D** (lesson [I6](../intermediate/beyond-nsga-ii.md#decomposition-based-algorithms)).
  Both spread weight vectors with the Das and Dennis design. MOEA/D compares
  solutions by a scalarizing function within neighbourhoods of vectors;
  NSGA-III compares them by dominance and uses the directions only to count.
  It needs no scalarizing function and no neighbourhood size.
- **RVEA** (lesson [A3](rvea.md)) drops dominance and chooses each direction's
  survivor by an angle-penalized distance; **AGE-MOEA** (lesson
  [A4](agemoea.md)) drops the directions and adapts its distances to the shape
  of the front.

NSGA-III works best when the front covers the simplex, as on DTLZ2 or WFG4.
Fronts that are degenerate, disconnected or inverted leave many directions
without solutions and pile the survivors onto the others: Ishibuchi et al. [5]
showed that the performance of reference-direction algorithms depends strongly
on the shape of the front. Normalization can also be misled by poor extreme
points early in the run. And H grows combinatorially with m, which fixes a
minimum population whether or not the problem needs it.

## Key points

- NSGA-III keeps NSGA-II's structure and replaces the crowding distance by
  niching around H = C(p + m − 1, m − 1) Das and Dennis directions; VAMOS
  requires N ≥ H and raises a `ValueError` otherwise.
- Adaptive normalization translates by the ideal point and divides by the
  intercepts of the hyperplane through the ASF extreme points; when the
  hyperplane is degenerate VAMOS uses the worst point of the first front.
- Each solution is associated with the direction at the smallest
  perpendicular distance.
- The last front fills the least crowded directions first: the closest
  candidate for an empty direction, a random one otherwise, with random
  tie-breaks.
- VAMOS selects parents at random by default, as Deb and Jain, or by a
  tournament on rank, never by crowding distance, and treats constraints
  feasibility first, unless `constraint_handling` names another technique.

**Next:** [A3. RVEA: angle-penalized distance](rvea.md).

## References

[1] Deb, K., and Jain, H. (2014). [An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints](https://doi.org/10.1109/TEVC.2013.2281535). *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.

[2] Jain, H., and Deb, K. (2014). [An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, Part II: Handling constraints and extending to an adaptive approach](https://doi.org/10.1109/TEVC.2013.2281534). *IEEE Transactions on Evolutionary Computation*, 18(4), 602–622.

[3] Das, I., and Dennis, J. E. (1998). [Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, 8(3), 631–657.

[4] Zhang, Q., and Li, H. (2007). [MOEA/D: A multiobjective evolutionary algorithm based on decomposition](https://doi.org/10.1109/TEVC.2007.892759). *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

[5] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[6] Wierzbicki, A. P. (1980). The use of reference objectives in multiobjective optimization. In G. Fandel and T. Gal (Eds.), *Multiple Criteria Decision Making Theory and Application*, Lecture Notes in Economics and Mathematical Systems 177, 468–486. Springer.

[7] Huband, S., Hingston, P., Barone, L., and While, L. (2006). [A review of multiobjective test problems and a scalable test problem toolkit](https://doi.org/10.1109/TEVC.2005.861417). *IEEE Transactions on Evolutionary Computation*, 10(5), 477–506.
