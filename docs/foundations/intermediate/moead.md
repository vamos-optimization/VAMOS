---
description: 'I8. MOEA/D in depth: weight vectors, neighbourhoods, mating with probability δ, differential evolution, the ideal point, replacement limited to n_r solutions and the aggregation functions of VAMOS, with an interactive figure and the VAMOS code.'
search:
  boost: 0.5
---

# I8. MOEA/D: decomposition and neighbourhoods

NSGA-II compares whole solutions by dominance and crowding. MOEA/D, proposed
by Zhang and Li in 2007 [1], never sorts its population. It *decomposes* the
problem into N single-objective subproblems, one per weight vector, and solves
them all at once. Each subproblem keeps exactly one solution, and a new
solution competes only with the solutions of nearby subproblems, each judged
by the scalarizing function of **its own** subproblem. Lesson
[I6](beyond-nsga-ii.md#decomposition-based-algorithms) introduced the idea;
this lesson takes it apart.

Two things set MOEA/D apart from NSGA-II. Diversity does not come from a
density estimate but from the weight vectors, which are fixed before the run.
And the work is local: an offspring is bred from neighbours and compared with
T neighbours, so one step costs O(T·m) aggregation terms for m objectives
instead of a sort of the whole population. VAMOS runs MOEA/D steady-state, one
offspring at a time, with the differential-evolution operator of MOEA/D-DE, the
variant of Li and Zhang [2].

In this lesson you will learn:

- how weight vectors define the subproblems, and how VAMOS generates them for
  two and for three or more objectives;
- what a neighbourhood is, and how mating (probability δ) and replacement
  (limit n_r) use it;
- how the ideal point and the aggregation function decide a replacement, and
  which aggregation functions VAMOS offers;
- how VAMOS implements and configures MOEA/D, and when to choose it.

## From one weighted problem to many

A scalarization turns a multi-objective problem into a single-objective one
(lesson [B1](../basic/multi-objective-problems.md#turning-many-objectives-into-one-scalarization)).
Solving it once gives one Pareto-optimal point; sweeping the weights and
solving again gives others, one run per point. Zhang and Li [1] observed that
the optimal solutions of two subproblems with **close weight vectors are
close** in the objective space, so what one subproblem has found is useful to
its neighbours. MOEA/D therefore solves all the subproblems together and lets
neighbours share parents and offspring.

The original algorithm let a good offspring replace every neighbour it
improved. Two years later, MOEA/D-DE [2] replaced the genetic operators with
differential evolution [3] and added two measures that protect diversity: with
probability 1 − δ the parents and the candidates for replacement come from the
whole population instead of the neighbourhood, and one offspring may replace at
most n_r solutions. VAMOS implements this version, after jMetal.

## Weight vectors and subproblems

Each subproblem i has a weight vector wⁱ = (wⁱ₁, …, wⁱₘ) with non-negative
components that add up to 1, and an aggregation function g(x | wⁱ, z\*) to
minimize, where z\* is the ideal point (below). There is one solution per
subproblem, so **the population size is the number of weight vectors**.
VAMOS generates them as jMetal does:

- **Two objectives.** N evenly spaced vectors, for any N:

<div class="vf-formula">wᵏ = ( k / (N − 1),  1 − k / (N − 1) ),     k = 0, 1, …, N − 1</div>

- **Three or more objectives.** The Das and Dennis simplex lattice [4]
  explained in lesson [A1](../advanced/many-objective.md#nsga-iii-in-detail):
  every vector whose components are multiples of 1/p, for p divisions, which
  gives H = C(p + m − 1, m − 1) vectors. The population must have exactly H
  members, for example 91 for three objectives and p = 12. When no complete
  lattice has exactly the requested size, VAMOS uses the weight file that jMetal
  distributes for that size if it ships one (`W3D_100.dat` gives 100 individuals
  with three objectives), and otherwise stops with an error: a lattice is never
  truncated, and the population size is never changed behind your back.

`MOEADConfig.default(...)` chooses the lattice for you when you give no
population size: 100 vectors with two objectives, 91 with three and the lattice
with p = 6 beyond that. `weight_vectors(path=...)` reads your own vectors from
a file instead.

## Neighbourhoods

The neighbourhood B(i) of subproblem i holds the indices of the T weight
vectors closest to wⁱ in Euclidean distance, i itself included:

<div class="vf-formula">B(i) = { the T indices j with the smallest ‖wʲ − wⁱ‖ }          i ∈ B(i)</div>

The neighbourhoods are computed once, before the run. VAMOS's default is
T = 20 (or N if the population is smaller). Lattice weights have many
equidistant neighbours; VAMOS breaks those ties by index, so a run gives the
same neighbourhoods, and the same results, on every platform.

T sets how far information travels. With a small T, subproblems cooperate only
with their immediate neighbours: the search is local and good genes spread
slowly along the front. With a large T, an offspring competes with many
subproblems, and a single good solution can take over a large part of the
population, which is what the limit n_r prevents.

## One step of MOEA/D

VAMOS creates and places one offspring per step:

1. **Choose a subproblem i.** The subproblems are visited in a random
   permutation, a new one in each cycle of N steps (`subproblem_order`, whose
   other value, `"cyclic"`, visits 0, 1, …, N − 1).
2. **Choose the mating pool.** With probability δ (default 0.9) the pool P is
   the neighbourhood B(i); otherwise it is the whole population.
3. **Breed.** Draw two distinct parents r₁ and r₂ from P and create one
   offspring y from them and the current solution xⁱ.
4. **Evaluate y and update the ideal point** z\*.
5. **Replace.** Visit the members of P in a random order, and let y replace
   the solutions it improves, at most n_r of them.

For step 3, VAMOS applies DE rand/1/bin [3] with the current solution xⁱ as the
base vector, as MOEA/D-DE does, then polynomial mutation (lesson
[I4](variation-operators.md#polynomial-mutation)):

<div class="vf-formula">v = xⁱ + F · ( x^r₁ − x^r₂ )
y_j = v_j    if rand_j &lt; CR or j = j_rand,     otherwise y_j = xⁱ_j
y ← polynomial mutation of y   (p_m = 1/n, η_m = 20),   then clipped to the bounds</div>

With VAMOS's defaults F = 0.5 and CR = 1.0, every variable comes from v. In
step 5, whenever y is strictly better than the current solution of
subproblem j **in subproblem j**, it takes its place, and the loop stops after
n_r replacements (default 2):

<div class="vf-formula">for j in P, in random order:
    if g(y | wʲ, z*) &lt; g(xʲ | wʲ, z*):    xʲ ← y
    stop after n_r replacements</div>

Notice that the offspring is compared with each neighbour's solution through
the neighbour's weight vector, not through wⁱ. An offspring that is a poor
answer to its own subproblem can still be a good answer to the next one.

## The ideal point

The Tchebycheff and PBI functions measure distances from the **ideal point**
z\*, the best value of each objective (lesson
[B2](../basic/pareto-dominance.md#ideal-and-nadir-points)). The true ideal
point is unknown, so MOEA/D estimates it with the best values found so far:

<div class="vf-formula">z*_k ← min( z*_k ,  f_k(y) )     for every objective k</div>

VAMOS starts from the minimum of the initial population and updates z\* with
each offspring **before** the comparisons of step 5. Every move of z\* shifts
all the subproblems at once; early in the run, when z\* is a poor estimate, the
subproblems point at a different part of the front than they will at the end.

## Aggregation functions

The aggregation function decides what "better in subproblem j" means. VAMOS's
three classic functions are:

<div class="vf-formula">Tchebycheff:      g(x | w, z*) = max_k  w_k · | f_k(x) − z*_k |           (a zero weight counts as 10⁻⁴)
Weighted sum:     g(x | w)     = Σ_k  w_k · f_k(x)
PBI:              g(x | w, z*) = d₁ + θ · d₂
                  d₁ = | (f(x) − z*) · w | / ‖w‖,     d₂ = ‖ f(x) − z* − d₁ · w / ‖w‖ ‖</div>

They have different geometries:

- **Tchebycheff** [5] has corner-shaped contour lines. On a continuous
  two-objective front, the optimum of w lies where w₁(f₁ − z\*₁) = w₂(f₂ − z\*₂),
  on the ray from z\* with direction (1/w₁, 1/w₂). It reaches every
  Pareto-optimal point for some weight, whatever the shape of the front.
- **Weighted sum** has straight contour lines perpendicular to w. It cannot
  reach the non-convex parts of a front (lesson
  [I6](beyond-nsga-ii.md#decomposition-based-algorithms)).
- **PBI** (penalty-based boundary intersection), proposed with MOEA/D [1],
  measures the progress d₁ along w and penalizes the distance d₂ from the ray
  through z\* along w, with weight θ (VAMOS: 5). A large θ keeps each subproblem
  close to its own ray.

The same weight vector therefore asks for different points. With
w = (0.2, 0.8), Tchebycheff wants f₁ − z\*₁ = 4 (f₂ − z\*₂), towards the
lower right of a front, and PBI wants f₂ − z\*₂ = 4 (f₁ − z\*₁), towards the upper
left. VAMOS also offers four more functions, with jMetal's definitions:

| `aggregation(...)` | Definition | Parameter |
| --- | --- | --- |
| `"tchebycheff"` | max_k w_k \|f_k − z\*_k\| | none |
| `"weighted_sum"` | Σ_k w_k f_k | none |
| `"pbi"` | d₁ + θ d₂ from the ideal point | `theta`, default 5 |
| `"modified_tchebycheff"` | Tchebycheff + ρ Σ_k w_k \|f_k − z\*_k\| | `rho`, default 10⁻³ |
| `"augmented_tchebycheff"` | the same formula, jMetal's default [5] | `rho`, default 10⁻⁴ |
| `"inverse_weight_tchebycheff"` | max_k \|f_k − z\*_k\| / w_k [6] | none |
| `"inverted_pbi"` | θ d₂ − d₁ measured from the nadir point [7] | `theta`, default 0.1 |

Every function accepts `normalize=True`, which divides each f_k − z\*_k by the
current range z^nad_k − z\*_k (plus a small ε), with the nadir point estimated
from the non-dominated solutions found so far. Without it, an objective
measured in larger units dominates every aggregation. The [MOEA/D
lesson](../../algorithms/moead.md#what-the-configuration-means) and the
[configuration reference](../../reference/api/algorithms/moead.md) give the
details.

## Constraints

With constraints (g_j(x) ≤ 0, lesson [A5](../advanced/constrained.md)), VAMOS's
replacement follows the feasibility rules by default: a feasible offspring
replaces an infeasible solution; between two infeasible solutions the smaller
constraint violation wins; only two feasible solutions are compared by the
aggregation function. `subproblem_update_criterion(...)` replaces this rule by
an improved ε method or an adaptive violation threshold, as described in
[subproblem update criteria](../../reference/constraints.md#moead-subproblem-update-criteria).

## See a replacement

The figure shows 11 subproblems with two objectives, their weight vectors drawn
as rays from the ideal point, and the current solution of each subproblem.
Select a subproblem to highlight its T neighbours, then drag the offspring. The
table compares it with every candidate in the drawn order. Try:

- the default position: the offspring improves three subproblems, but with
  n_r = 2 it replaces only the first two it meets; shuffle the order, then raise
  n_r;
- the same offspring under PBI and under the weighted sum;
- dragging the offspring left of every solution, so that z\* moves.

<div class="vf-widget" data-vf-widget="moead-neighbourhood"><p class="vf-fallback">Interactive figure: eleven weight vectors as rays from the ideal point, the neighbourhood of a selected subproblem, and which neighbours a draggable offspring replaces under Tchebycheff, PBI or the weighted sum with the limit n_r. It needs JavaScript.</p></div>

## In VAMOS

`MOEADConfig.default(...)` gives VAMOS's default configuration; `variant="de"`
gives MOEA/D-DE as jMetal configures it, with the Tchebycheff function. The
default factory also chooses the population from the lattice:

```python
from vamos.algorithms import MOEADConfig

config = MOEADConfig.default(pop_size=100, n_var=30, n_obj=2)
print("neighbourhood T:", config.neighbor_size)
print("delta:", config.delta, " n_r:", config.replace_limit)
print("crossover:", config.crossover)
print("mutation:", config.mutation[0], "eta", config.mutation[1]["eta"])
print("aggregation:", config.aggregation)
print("variant 'de':", MOEADConfig.default(pop_size=100, n_var=30, n_obj=2, variant="de").aggregation)

for n_obj in (2, 3, 5):
    default = MOEADConfig.default(n_var=12, n_obj=n_obj)
    print(f"{n_obj} objectives: pop_size {default.pop_size}, weight vectors {default.weight_vectors}")
```

Output:

```text
neighbourhood T: 20
delta: 0.9  n_r: 2
crossover: ('de', {'cr': 1.0, 'f': 0.5})
mutation: pm eta 20.0
aggregation: ('pbi', {'theta': 5.0})
variant 'de': ('tchebycheff', {})
2 objectives: pop_size 100, weight vectors {'path': None, 'divisions': 99}
3 objectives: pop_size 91, weight vectors {'path': None, 'divisions': 12}
5 objectives: pop_size 210, weight vectors {'path': None, 'divisions': 6}
```

Without `variant`, VAMOS pairs MOEA/D-DE's operators with PBI, its own choice;
the [MOEA/D lesson](../../algorithms/moead.md#what-the-configuration-means)
records this and the planned change of default.

The weight vectors are built by a function that follows jMetal. With two
objectives it is a `linspace`; with more, it tries the packaged jMetal file
when no complete lattice has the requested size, and otherwise generates the
lattice with exactly `pop_size` vectors (`strict=True`), or raises an error:

```python
# Excerpt from src/vamos/engine/algorithm/components/weight_vectors.py
def _load_or_generate_jmetalpy_weights(pop_size: int, n_obj: int, path: str | None, divisions: int | None) -> FloatArray:
    if n_obj == 2:
        vals = np.linspace(0.0, 1.0, pop_size, dtype=float)
        return np.column_stack([vals, 1.0 - vals])

    weight_path = _resolve_jmetalpy_weight_path(path, n_obj, pop_size)
    if weight_path is None and divisions is None and _exact_divisions(pop_size, n_obj) is None:
        # No complete lattice has pop_size vectors: use the jMetal weights VAMOS ships for it, if any.
        packaged = _packaged_weights(n_obj, pop_size)
        if packaged is not None:
            return packaged
    weights = _generate_weights(pop_size, n_obj, divisions=divisions, strict=True)
```

The neighbourhoods are the first T columns of the sorted distance matrix
between weight vectors. The stable sort puts the subproblem itself first
(distance 0) and breaks ties by index:

```python
# Excerpt from src/vamos/engine/algorithm/moead/neighborhood.py
def compute_neighbors(weights: np.ndarray[Any, Any], neighbor_size: int) -> np.ndarray[Any, Any]:
    """Compute neighborhood indices based on weight vector distances."""
    dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    # Lattice weights have many equidistant neighbours; a stable sort breaks those ties by index, so the
    # neighbourhoods, and the trajectory, are the same on every platform.
    order = np.argsort(dist, axis=1, kind="stable")
    return order[:, :neighbor_size]
```

`ask()` performs steps 1 to 3. `use_neighbors` is the draw against δ; the
parents of the DE crossover are the random parents followed by the current
solution of the subproblem, `st.X[active]`, which serves as the base vector
and as the source of the variables that the crossover does not take from the
mutant:

```python
# Excerpt from src/vamos/engine/algorithm/moead/moead.py
        active = self._next_active_indices(st, batch_size)
        use_neighbors = st.rng.random(batch_size) < st.delta

        parent_pairs = self._sample_parent_pairs(st, active, use_neighbors)

        n_var = st.X.shape[1]
        if self._cross_is_de:
            # (random parents..., current solution): jMetal's MOEADDEBuilder, or Evolver's pool and the current one.
            width = parent_pairs.shape[1]
            parents = np.empty((batch_size, width + 1, n_var), dtype=st.X.dtype)
            for slot in range(width):
                parents[:, slot, :] = st.X[parent_pairs[:, slot]]
            parents[:, width, :] = st.X[active]
            offspring = st.crossover_fn(parents, st.rng)
            children = offspring[:, 0, :].copy()
        else:
            parents = st.X[parent_pairs.reshape(-1)].reshape(batch_size, 2, n_var)
            offspring = st.crossover_fn(parents, st.rng)
            children = offspring[:, 0, :].copy()

        children = st.mutation_fn(children, st.rng)
```

`batch_size` is 1 by default, which makes the algorithm steady-state. `tell()`
then updates the ideal point with `st.ideal = np.minimum(st.ideal, F_child.min(axis=0))`
and draws the order of the candidates, `st.rng.permutation(st.neighbors[active[0]])`
or `st.rng.permutation(pop_size)`, depending on the same δ draw. The replacement
loop is the heart of MOEA/D. VAMOS runs a compiled copy of it when Numba is
installed; this is the pure-Python version, which performs the same comparisons:

```python
# Excerpt from src/vamos/engine/algorithm/moead/neighborhood_kernels.py
    replacements = 0
    for idx in range(candidate_length):
        k = int(candidate_order[idx])

        diff_current = np.abs(F[k] - ideal)
        diff_child = np.abs(child_f - ideal)

        if normalize == 1 or agg_id in {AGG_INVERTED_PBI, AGG_INVERSE_WEIGHT_TCHEBYCHEFF}:
            args = (k, weights, weights_safe, weights_unit, ideal, nadir, agg_id, agg_theta, agg_rho, agg_epsilon, normalize)
            current_val = _scalar_aggregation(F[k], *args)
            child_val = _scalar_aggregation(child_f, *args)
        elif agg_id == AGG_TCHEBYCHEFF:
            current_val = float(np.max(weights_safe[k] * diff_current))
            child_val = float(np.max(weights_safe[k] * diff_child))
        elif agg_id == AGG_WEIGHTED_SUM:
            current_val = float(np.dot(weights[k], F[k]))
            child_val = float(np.dot(weights[k], child_f))

        replace = False
        if use_constraints == 1:
            current_cv = cv[k]
            feas_child = child_cv <= 0.0
            feas_curr = current_cv <= 0.0
            if (not feas_curr) and feas_child:
                replace = True
            elif feas_child and feas_curr:
                replace = child_val < current_val
            else:
                replace = child_cv < current_cv
        else:
            replace = child_val < current_val

        if not replace:
            continue

        X[k] = child
        F[k] = child_f
        if use_constraints == 1:
            G[k] = child_g
            cv[k] = child_cv
        replacements += 1
        if replacements >= replace_limit:
            break
```

Line by line: `k` is the next candidate in the random order; both values are
computed with the weights of `k`. `weights_safe` is the weight matrix with
every zero replaced by 10⁻⁴, the rule for zero weights in the Tchebycheff
formula. The branches for PBI and the modified Tchebycheff, skipped here,
follow the formulas above. Without constraints a replacement needs a strictly
smaller value (`child_val < current_val`); with constraints the feasibility
rules come first. The counter stops the loop at `replace_limit`, the n_r of
the formulas.

The aggregation function matters most when the front is not convex. ZDT2 [8]
has the concave front f₂ = 1 − f₁². With the same operators and budget,
Tchebycheff spreads the 50 subproblems along the front, while the weighted sum
can only reach its two ends:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import MOEADConfig
from vamos.problems import ZDT2

problem = ZDT2(n_var=10)  # concave Pareto front: f2 = 1 - f1^2
for aggregation in ("tchebycheff", "weighted_sum"):
    config = (
        MOEADConfig.builder()
        .pop_size(50)
        .neighbor_size(10)
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation(aggregation)
        .result_mode("population")
        .build()
    )
    result = optimize(problem, algorithm="moead", algorithm_config=config, max_evaluations=4000, engine="numpy", seed=1)
    distinct = np.unique(np.round(result.F, 2), axis=0)
    print(f"{aggregation:<13} {result.data['evaluations']} evaluations, {len(result.F)} subproblems, {len(distinct)} distinct objective vectors")
```

Output:

```text
tchebycheff   4000 evaluations, 50 subproblems, 50 distinct objective vectors
weighted_sum  4000 evaluations, 50 subproblems, 3 distinct objective vectors
```

`result_mode("population")` returns one solution per subproblem. With the
weighted sum, many subproblems hold the same solution: nothing in MOEA/D
prevents two subproblems from sharing one, and on a concave front the weighted
sum gives them every reason to.

## MOEA/D, NSGA-II and the decomposition family

| | NSGA-II | MOEA/D |
| --- | --- | --- |
| Selection | Rank and crowding over parents and offspring | Scalar comparison within a neighbourhood |
| Source of diversity | Crowding distance, measured each generation | Weight vectors, fixed before the run |
| Offspring | N per generation | One per step (steady-state) |
| Work per offspring | A share of the sort of 2N solutions | T aggregation values |
| Population size | Free | The number of weight vectors |

**Strengths.** Each step is cheap and local. Any single-objective operator,
such as DE, plugs in directly, and Li and Zhang [2] showed that MOEA/D-DE
handles problems whose Pareto sets have complicated shapes. Selection does not
depend on dominance, so it keeps its pressure when most solutions are mutually
non-dominated, which matters with many objectives (lesson
[A1](../advanced/many-objective.md)).

**Limits.** The weight vectors fix the distribution of the final solutions,
and evenly spaced weights give evenly spaced solutions only when the front has
the matching shape; Ishibuchi et al. [9] showed how strongly performance
depends on that match. Objectives with different scales need `normalize=True`.
Subproblems can share a solution, as the example shows, and with three or more
objectives the population size is tied to the lattice.

**When to use it.** MOEA/D-DE is a strong choice for two or three objectives
with continuous variables, especially when the Pareto set is complicated; its
descendants NSGA-III and RVEA (lessons [A2](../advanced/nsga-iii.md) and
[A3](../advanced/rvea.md)) keep the idea of fixed
directions for many objectives.

## Key points

- MOEA/D solves N scalar subproblems at once, one weight vector and one
  solution each; the population size is the number of weight vectors.
- Neighbourhoods are the T closest weight vectors. With probability δ mating
  and replacement use the neighbourhood, otherwise the whole population.
- An offspring replaces a solution only if it is better **in that solution's
  subproblem**, and at most n_r solutions per offspring.
- Tchebycheff, weighted sum and PBI define different subproblems for the same
  weight vector; the weighted sum cannot reach non-convex parts of a front.
- VAMOS's default uses DE and polynomial mutation, T = 20, δ = 0.9, n_r = 2
  and PBI; `variant="de"` gives MOEA/D-DE with Tchebycheff.

**Next:** [I9. SMS-EMOA: selection by hypervolume contribution](smsemoa.md).

## References

[1] Zhang, Q., and Li, H. (2007). [MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition](https://doi.org/10.1109/TEVC.2007.892759). *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

[2] Li, H., and Zhang, Q. (2009). [Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II](https://doi.org/10.1109/TEVC.2008.925798). *IEEE Transactions on Evolutionary Computation*, 13(2), 284–302.

[3] Storn, R., and Price, K. (1997). [Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces](https://doi.org/10.1023/A:1008202821328). *Journal of Global Optimization*, 11(4), 341–359.

[4] Das, I., and Dennis, J. E. (1998). [Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, 8(3), 631–657.

[5] Steuer, R. E., and Choo, E.-U. (1983). [An interactive weighted Tchebycheff procedure for multiple objective programming](https://doi.org/10.1007/BF02591870). *Mathematical Programming*, 26(3), 326–344.

[6] Qi, Y., Ma, X., Liu, F., Jiao, L., Sun, J., and Wu, J. (2014). [MOEA/D with Adaptive Weight Adjustment](https://doi.org/10.1162/EVCO_a_00109). *Evolutionary Computation*, 22(2), 231–264.

[7] Sato, H. (2014). [Inverted PBI in MOEA/D and its impact on the search performance on multi and many-objective optimization](https://doi.org/10.1145/2576768.2598297). *Proceedings of the 2014 Annual Conference on Genetic and Evolutionary Computation (GECCO '14)*, 645–652.

[8] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[9] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.
