---
description: 'SMPSO in depth: particle swarm basics, the velocity update with inertia, C1, C2, r1 and r2, the constriction coefficient and the speed constraint, the leaders archive with crowding distance, personal bests and polynomial-mutation turbulence, with an interactive figure and the VAMOS code.'
search:
  boost: 0.5
---

# SMPSO: multi-objective particle swarm optimization

Every algorithm so far breeds children from parents with crossover and
mutation. Particle swarm optimization (PSO), introduced by Kennedy and
Eberhart in 1995 [2], works differently: a *swarm* of particles flies through
the decision space, and each particle moves according to its own best position
so far and the best position known to the swarm. Nothing is recombined; every
particle is updated in place.

SMPSO, the *speed-constrained multi-objective PSO* of Nebro et al. [1], adapts
this idea to several objectives. The best position known to the swarm is
replaced by an archive of non-dominated **leaders**, from which every particle
picks its guide, and the velocity of each particle is bounded so that it
cannot jump erratically between the bounds of the variables. Unlike NSGA-II,
the swarm itself is not elitist: the archive keeps the best solutions, and it
is what VAMOS returns.

In this lesson you will learn:

- how a particle moves: position, velocity, personal best and leader;
- the velocity update with the inertia w and the coefficients C₁, C₂, r₁ and r₂;
- the constriction coefficient χ and the speed constraint that define SMPSO;
- how the leaders archive, leader selection, personal bests and turbulence
  work, and how VAMOS implements and configures them.

## From bird flocks to multi-objective swarms

Kennedy and Eberhart [2] modelled a population that shares information like a
flock of birds: each particle keeps its position x, its velocity v and the best
position it has visited, its *personal best* p, and it is also attracted by the
best position found by the swarm, the *leader* g. Two later additions became
standard: an **inertia weight** w that scales the previous velocity, proposed by
Shi and Eberhart [3], and a **constriction coefficient** χ that scales the
whole update, derived by Clerc and Kennedy [4] from a stability analysis of the
particle's trajectory.

With several objectives there is no single best position. Multi-objective PSOs
keep the non-dominated positions found so far in an archive and let each
particle choose its leader there; how the archive is bounded and how a leader
is chosen decide the spread of the result. Nebro et al. [1] observed that in
some problems the velocities grow so large that particles move erratically,
from one bound of a variable to the other, and they proposed SMPSO with a
**speed constraint** on every variable.

## A particle and its velocity

At each iteration every particle computes a new velocity from three terms and
moves:

<div class="vf-formula">v ← w · v  +  C₁ r₁ (p − x)  +  C₂ r₂ (g − x)
x ← x + v</div>

- **Inertia** w · v keeps part of the previous movement. VAMOS uses w = 0.1, a
  small value: the particle has little momentum.
- **The cognitive term** C₁ r₁ (p − x) pulls the particle towards its own
  personal best.
- **The social term** C₂ r₂ (g − x) pulls it towards its leader.

r₁ and r₂ are random numbers in [0, 1] that vary the strength of the two pulls;
C₁ and C₂ set their scale. VAMOS, like jMetal, draws one r₁, one r₂, one C₁ and
one C₂ per particle and iteration, shared by all its variables, with C₁ and C₂
uniform in [c1, c1 + 1] and [c2, c2 + 1]: [1.5, 2.5] with the defaults. The
pulls change length from one iteration to the next, never direction.

## Constriction and the speed constraint

SMPSO then scales the new velocity by the constriction coefficient, which
depends on φ = C₁ + C₂:

<div class="vf-formula">φ = C₁ + C₂
χ = 2 / ( 2 − φ − √(φ² − 4φ) )      if φ > 4,          χ = 1      otherwise
v ← χ · ( w · v + C₁ r₁ (p − x) + C₂ r₂ (g − x) )</div>

and clamps every component of the velocity to a fraction of the range of its
variable, the **speed constraint** that gives SMPSO its name:

<div class="vf-formula">δ_j = vmax_fraction · ( x_j^U − x_j^L )           (VAMOS default: vmax_fraction = 0.5)
v_j ← min( max( v_j, −δ_j ), δ_j )
x_j ← x_j + v_j
if x_j &lt; x_j^L:  x_j ← x_j^L,  v_j ← −v_j          if x_j > x_j^U:  x_j ← x_j^U,  v_j ← −v_j</div>

With the default, no step can be longer than half the range of a variable. A
particle that crosses a bound stops on it, and that component of its velocity
changes sign.

Look closely at the sign of χ. For φ > 4 the denominator 2 − φ − √(φ² − 4φ) is
negative, and VAMOS evaluates the expression as written, as in jMetal (see the
[SMPSO lesson](../../algorithms/smpso.md#understand-the-configuration)), so
χ lies between −1 and 0: about −0.73 for φ = 4.1 and −0.38 for φ = 5. Clerc and
Kennedy's coefficient [4] is the absolute value of this number, 0.73 for
φ = 4.1. With the default draws, φ is the sum of two uniform values in
[1.5, 2.5], so φ > 4 in half of the updates; in those, the step points in the
opposite direction of the combined pull, before the speed constraint bounds it.
In the other half, χ = 1 and the speed constraint alone limits the step.

## See one velocity update

The figure shows one particle in a two-variable decision space with bounds
[0, 1], its current velocity, its personal best and its leader. The arrows are
the three terms of the update; their sum is the unconstrained new velocity,
which the figure then constricts and clamps. Try:

- the default C₁ = 2.0 and C₂ = 1.9: φ = 3.9, χ = 1, and the speed constraint
  cuts the first component of the step from 0.95 to 0.5;
- C₁ + C₂ above 4, for example 2.2 and 2.3: χ = −0.5, the step reverses and the
  particle stops on the lower bounds;
- resampling r₁ and r₂ several times.

<div class="vf-widget" data-vf-widget="pso-velocity"><p class="vf-fallback">Interactive figure: one particle with its velocity, personal best and leader, the inertia, cognitive and social terms of the SMPSO velocity update, the constriction coefficient, the speed constraint and the new position, for chosen C1, C2, r1 and r2. It needs JavaScript.</p></div>

## Leaders, personal bests and turbulence

**The leaders archive.** SMPSO keeps the non-dominated positions found so far
in an archive of bounded size, by default the swarm size. VAMOS inserts the
particles one at a time, as jMetal does: a particle that a leader dominates, or
equals, is rejected; leaders that the particle dominates leave; and when the
archive overflows, the leader with the smallest crowding distance (lesson
[Crowding and tournaments](crowding-distance.md)) is removed. Crowding distance [5] keeps the leaders
spread along the front, with the boundary leaders always kept.

**Choosing a leader.** For each particle, two distinct leaders are drawn at
random and the one with the larger crowding distance wins; a tie is decided by
a coin. Leaders in sparse regions are chosen more often, so the swarm is drawn
towards the gaps of the front. While the archive holds at most two leaders,
VAMOS gives every particle the first one.

**The personal best.** After evaluation, each particle compares its new
position with its personal best. The new position replaces the personal best
**unless the personal best dominates it**:

<div class="vf-formula">p ← x     unless  f(p) ≺ f(x)</div>

So a new position that is better, or merely incomparable, becomes the new
personal best. With constraints, a feasible personal best is kept against an
infeasible position, and between two infeasible ones the smaller violation
wins.

**Turbulence.** PSO has no mutation of its own. SMPSO adds polynomial mutation
(lesson [Real-coded operators](../operators/real-coded.md#polynomial-mutation), [6]) to part of the
swarm, a *turbulence* that helps particles leave local optima. VAMOS mutates
every sixth particle (indices 0, 6, 12, …) after it moves, with p_m = 1/n and
η_m = 20 by default.

One iteration of VAMOS's SMPSO therefore runs: choose a leader for every
particle, update velocities and positions, apply the turbulence, evaluate the
swarm, update the personal bests, and insert the new positions into the
leaders archive. The swarm starts at random positions with zero velocity, and
each personal best starts at its particle's initial position.

## In VAMOS

`SMPSOConfig.default(...)` gives the default configuration:

```python
from vamos.algorithms import SMPSOConfig

config = SMPSOConfig.default(n_var=30)
name, params = config.mutation
print("swarm:", config.pop_size, " leaders archive:", config.archive_size)
print("inertia w:", config.inertia)
print("C1 drawn from", [config.c1, config.c1 + 1.0], " C2 drawn from", [config.c2, config.c2 + 1.0])
print("speed constraint: vmax_fraction", config.vmax_fraction)
print("turbulence:", name, "prob", round(params["prob"], 4), "eta", params["eta"])
print("result:", config.result_mode)
```

Output:

```text
swarm: 100  leaders archive: 100
inertia w: 0.1
C1 drawn from [1.5, 2.5]  C2 drawn from [1.5, 2.5]
speed constraint: vmax_fraction 0.5
turbulence: polynomial prob 0.0333 eta 20.0
result: non_dominated
```

`result_mode` `non_dominated` returns the leaders archive; `population`
returns the final swarm. The velocity update in `SMPSO.ask()` follows the
formulas above, for the whole swarm at once:

```python
# Excerpt from src/vamos/engine/algorithm/smpso/smpso.py
        # PSO velocity update (SMPSO with constriction)
        pop_size, n_var = st.X.shape
        # Drawn per particle and used as drawn, as in jMetal (jMetalPy rounds them to one decimal).
        r1 = st.rng.uniform(st.r1_min, st.r1_max, size=(pop_size, 1))
        r2 = st.rng.uniform(st.r2_min, st.r2_max, size=(pop_size, 1))
        c1 = st.rng.uniform(st.c1_min, st.c1_max, size=(pop_size, 1))
        c2 = st.rng.uniform(st.c2_min, st.c2_max, size=(pop_size, 1))
        rho = c1 + c2
        disc = np.maximum(rho * rho - 4.0 * rho, 0.0)
        constriction = np.where(
            rho <= 4.0,
            1.0,
            2.0 / (2.0 - rho - np.sqrt(disc)),
        )
        velocity = constriction * (st.max_weight * st.velocity + c1 * r1 * (st.pbest_X - st.X) + c2 * r2 * (leaders - st.X))
        velocity = np.clip(velocity, st.delta_min, st.delta_max)

        # Position update
        X_new = st.X + velocity
        low_mask = X_new < st.xl
        if np.any(low_mask):
            X_new = np.where(low_mask, st.xl, X_new)
            velocity[low_mask] *= st.change_velocity1
        high_mask = X_new > st.xu
        if np.any(high_mask):
            X_new = np.where(high_mask, st.xu, X_new)
            velocity[high_mask] *= st.change_velocity2

        # Apply mutation (turbulence)
        if st.mutation_op is not None:
            if st.mutation_every <= 1:
                X_new = st.mutation_op(X_new, st.rng)
            else:
                mask = (np.arange(pop_size) % st.mutation_every) == 0
                if np.any(mask):
                    X_new[mask] = st.mutation_op(X_new[mask], st.rng)
```

Line by line: the four random factors have shape `(pop_size, 1)`, one value
per particle that broadcasts over its variables. `rho` is φ, and `disc` guards
the square root when φ ≤ 4, where `np.where` returns 1. `st.max_weight` is the
inertia w, and `leaders` holds one leader per particle. `st.delta_min` and
`st.delta_max` are ∓δ, computed once from the bounds and `vmax_fraction`.
`change_velocity1` and `change_velocity2` are −1, the reversal at the bounds.
The last block is the turbulence: `mutation_every` is 6.

The leader of each particle comes from a binary tournament on the crowding
distances of the archive, computed once per iteration:

```python
# Excerpt from src/vamos/engine/algorithm/smpso/smpso.py
    size = arch_X.shape[0]
    if size == 0:
        raise ValueError("Archive is empty.")
    if size <= 2:
        leaders = np.repeat(arch_X[:1], n_select, axis=0)
        leader_idx = np.zeros(n_select, dtype=int)
        return leaders, leader_idx

    first = rng.integers(0, size, size=n_select, dtype=int)
    second = rng.integers(0, size - 1, size=n_select, dtype=int)
    second += second >= first
    crowding_first = np.nan_to_num(crowding[first], nan=-np.inf)
    crowding_second = np.nan_to_num(crowding[second], nan=-np.inf)

    leader_idx = np.empty(n_select, dtype=int)
    better_first = crowding_first > crowding_second
    better_second = crowding_second > crowding_first
    leader_idx[better_first] = first[better_first]
    leader_idx[better_second] = second[better_second]

    tied = ~(better_first | better_second)
    if tied.any():
        flip = rng.integers(0, 2, size=int(tied.sum()), dtype=int)
        leader_idx[tied] = np.where(flip == 0, first[tied], second[tied])

    leaders = arch_X[leader_idx]
    return leaders, leader_idx
```

The second draw skips the first index, so the two contestants always differ,
the same trick as the parent tournament of NSGA-II (lesson
[Crowding and tournaments](crowding-distance.md#in-vamos)). The personal bests are updated after
evaluation:

```python
# Excerpt from src/vamos/engine/algorithm/smpso/helpers.py
    if technique is not None and constraint_mode != "none" and G is not None and pbest_G is not None:
        from vamos.engine.algorithm.components.constraint_handling import preferred

        update_idx = ~preferred(technique, pbest_F, pbest_G, F, G)
    elif constraint_mode and constraint_mode != "none" and G is not None and pbest_G is not None:
        feas_new = is_feasible(G)
        feas_old = is_feasible(pbest_G)
        cv_new = compute_violation(G)
        cv_old = compute_violation(pbest_G)

        both_infeasible = ~feas_new & ~feas_old
        same_violation = (feas_new & feas_old) | (both_infeasible & (cv_new == cv_old))
        old_better = (feas_old & ~feas_new) | (both_infeasible & (cv_old < cv_new)) | (same_violation & dominates(pbest_F, F))
        update_idx = ~old_better
    else:
        # Without constraints, update if current is not dominated by personal best
        update_idx = ~dominates(pbest_F, F)

    pbest_X[update_idx] = X[update_idx]
    pbest_F[update_idx] = F[update_idx]
    if pbest_G is not None and G is not None:
        pbest_G[update_idx] = G[update_idx]
```

The first branch applies a constraint-handling technique chosen with
`constraint_handling(...)` (see the [constraint
reference](../../reference/constraints.md#choose-a-constraint-handling-technique)).
The second is the default with constraints: the personal best survives only
when it is better, that is feasible against infeasible, less violated, or
dominating at equal violation. Without constraints only dominance counts.

The speed constraint is what makes SMPSO work on multimodal problems. ZDT4 [7]
has many local Pareto fronts; its distance function g equals 1 only on the
global front. With the default constraint, almost every leader reaches the
global front within 10,000 evaluations; with `vmax_fraction(10.0)`, which lets
a step span ten times the range of a variable, none does:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import SMPSOConfig
from vamos.problems import ZDT4

problem = ZDT4(n_var=10)  # x1 in [0, 1], x2..x10 in [-5, 5]
for vmax_fraction in (0.5, 10.0):
    config = (
        SMPSOConfig.builder()
        .pop_size(100)
        .archive_size(100)
        .mutation("pm", prob="1/n", eta=20.0)
        .vmax_fraction(vmax_fraction)
        .build()
    )
    result = optimize(problem, algorithm="smpso", algorithm_config=config, max_evaluations=10_000, engine="numpy", seed=2)
    X = result.X  # the leaders archive
    g = 1 + 10 * (problem.n_var - 1) + np.sum(X[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * X[:, 1:]), axis=1)
    print(f"vmax_fraction {vmax_fraction}: {len(X)} leaders, {int(np.sum(g < 1.01))} on the global front (g < 1.01)")
```

Output:

```text
vmax_fraction 0.5: 100 leaders, 96 on the global front (g < 1.01)
vmax_fraction 10.0: 12 leaders, 0 on the global front (g < 1.01)
```

This is one seed, but not a lucky one: over seeds 1 to 10 the default put
between 91 and 100 leaders on the global front, and `vmax_fraction` 1.0 or
more put none. The [SMPSO lesson](../../algorithms/smpso.md) records a run
with a smaller budget and the changes VAMOS 1.2.0 made to follow jMetal.

## SMPSO compared with NSGA-II and MOEA/D

| | NSGA-II | MOEA/D | SMPSO |
| --- | --- | --- | --- |
| New solutions | SBX and polynomial mutation | DE and polynomial mutation | Velocity update, turbulence on every sixth particle |
| Memory | The population | One solution per subproblem | Personal bests and the leaders archive |
| Elitism | Survival of the best N of 2N | Replacement only by better solutions | Only in the archive; the swarm always moves |
| Diversity | Crowding distance | Weight vectors | Crowding distance in the archive and in leader selection |
| Encodings in VAMOS | All | All, with suitable operators | Real and mixed |

**Strengths.** The velocity update is cheap and vectorized, and the particles
follow directed trajectories with memory instead of random recombination. The
speed constraint makes SMPSO robust on multimodal problems such as ZDT4, as
the example shows.

**Limits.** SMPSO needs a velocity, so VAMOS rejects binary, integer and
permutation problems. The archive is bounded with crowding distance, which
degrades with many objectives (lesson [Many objectives](../advanced/many-objective.md)).
The swarm is not elitist: the final swarm can be far worse than the archive,
which is why the archive is the default result.

**When to use it.** Continuous problems with two or three objectives,
especially multimodal ones, and cheap evaluations where many iterations are
affordable.

## Key points

- A particle moves with a velocity that combines inertia, a pull towards its
  personal best and a pull towards a leader, scaled by the random C₁ r₁ and
  C₂ r₂.
- SMPSO scales the velocity by the constriction coefficient χ when
  C₁ + C₂ > 4, then clamps each component to vmax_fraction of the variable's
  range: the speed constraint. In VAMOS, χ is negative when C₁ + C₂ > 4.
- Leaders come from a bounded archive of non-dominated positions, truncated by
  crowding distance and sampled by a crowding tournament; the archive is the
  result.
- A personal best is replaced unless it dominates the new position; polynomial
  mutation on every sixth particle adds turbulence.
- VAMOS's defaults: w = 0.1, C₁ and C₂ in [1.5, 2.5], vmax_fraction 0.5,
  archive and swarm of 100.

**Next section:** [Real-coded operators](../operators/real-coded.md).

## References

[1] Nebro, A. J., Durillo, J. J., García-Nieto, J., Coello Coello, C. A., Luna, F., and Alba, E. (2009). [SMPSO: A new PSO-based metaheuristic for multi-objective optimization](https://doi.org/10.1109/MCDM.2009.4938830). *IEEE Symposium on Computational Intelligence in Multi-Criteria Decision-Making*, 66–73.

[2] Kennedy, J., and Eberhart, R. (1995). [Particle swarm optimization](https://doi.org/10.1109/ICNN.1995.488968). *Proceedings of ICNN'95 – International Conference on Neural Networks*, 4, 1942–1948.

[3] Shi, Y., and Eberhart, R. (1998). [A modified particle swarm optimizer](https://doi.org/10.1109/ICEC.1998.699146). *Proceedings of the 1998 IEEE International Conference on Evolutionary Computation*, 69–73.

[4] Clerc, M., and Kennedy, J. (2002). [The particle swarm – explosion, stability, and convergence in a multidimensional complex space](https://doi.org/10.1109/4235.985692). *IEEE Transactions on Evolutionary Computation*, 6(1), 58–73.

[5] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[6] Deb, K., and Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for engineering design. *Computer Science and Informatics*, 26(4), 30–45.

[7] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.
