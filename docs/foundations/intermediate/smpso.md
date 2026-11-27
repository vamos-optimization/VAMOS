---
description: 'SMPSO in depth: particle swarm basics, the velocity update with inertia, C1, C2, r1 and r2, the constriction coefficient and the speed constraint, the leaders archive with crowding distance, personal bests and polynomial-mutation turbulence, with an interactive figure and VAMOS examples.'
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
  work, and how to configure them in VAMOS.

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
(lesson [Real-coded operators](operators/real-coded.md#polynomial-mutation), [6]) to part of the
swarm, a *turbulence* that helps particles leave local optima. VAMOS mutates
every sixth particle (indices 0, 6, 12, …) after it moves, with p_m = 1/n and
η_m = 20 by default.

One iteration of VAMOS's SMPSO therefore runs: choose a leader for every
particle, update velocities and positions, apply the turbulence, evaluate the
swarm, update the personal bests, and insert the new positions into the
leaders archive. The swarm starts at random positions with zero velocity, and
each personal best starts at its particle's initial position.

### Try it: one iteration step by step

Step through one iteration on six particles of a problem with two variables
whose Pareto front is ZDT1's, f₂ = 1 − √f₁. The left panel shows the
positions, velocities, personal bests and leaders, and the right one their
objective vectors. The settings are VAMOS' defaults, with a swarm and a leaders
archive of six. The figure starts after one iteration from random positions,
so that the particles already have velocities and some personal bests differ
from their positions.

<div class="vf-widget" data-vf-widget="smpso-steps"><p class="vf-fallback">Interactive figure: one SMPSO iteration in nine steps on six particles: the swarm and the leaders archive with its crowding distances, the leader tournaments, the inertia, cognitive and social terms of the velocity update, the constriction coefficient and the speed limit, the move and the bounds, the turbulence, the evaluation, the personal bests and the insertion of the new positions into the leaders archive. It needs JavaScript.</p></div>

Things to try:

1. Step 2: particle 1 draws F and G, the two extremes of the archive, both
   with crowding distance ∞: a full tie, decided by a coin. Particle 4 draws D
   and F, and F wins with ∞ against 1.00.
2. Steps 3 to 5: only particle 4 has a personal best away from its position,
   so only it is pulled towards p, by (−0.08, −0.87). Its φ = 4.53 gives
   χ = −0.49, which turns the sum (−0.06, −0.95) into the step (0.03, 0.47),
   away from both guides. The step crosses x₂ = 1, so particle 4 stops on that
   bound and v₂ becomes −0.47.
3. Step 4: particle 1 has φ = 3.87, so χ = 1, and the speed limit cuts its
   sum (0.51, −1.63) to (0.50, −0.50).
4. Step 8: particles 4 and 6 keep their personal bests, which dominate their
   new positions; 1, 2, 3 and 5 move to incomparable positions, which replace
   their personal bests.
5. Step 9: particle 2 enters the archive as H and leaves at once, the most
   crowded of seven leaders (0.43); 3 enters as I and pushes out B (0.57); 5
   enters as J and removes G and I, which it dominates. Press **New
   population** for another swarm.

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
returns the final swarm. In a builder, `.inertia(...)` sets w, `.c1(...)` and
`.c2(...)` the lower ends of the ranges of C₁ and C₂, `.vmax_fraction(...)`
the speed constraint and `.mutation(...)` the turbulence. With constraints,
`.constraint_handling(...)` replaces the default rule for the personal bests by
another technique (see the [constraint
reference](../../reference/constraints.md#choose-a-constraint-handling-technique)).

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

**Next section:** [Variation operators: an overview](operators/overview.md).

## References

[1] Nebro, A. J., Durillo, J. J., García-Nieto, J., Coello Coello, C. A., Luna, F., and Alba, E. (2009). [SMPSO: A new PSO-based metaheuristic for multi-objective optimization](https://doi.org/10.1109/MCDM.2009.4938830). *IEEE Symposium on Computational Intelligence in Multi-Criteria Decision-Making*, 66–73.

[2] Kennedy, J., and Eberhart, R. (1995). [Particle swarm optimization](https://doi.org/10.1109/ICNN.1995.488968). *Proceedings of ICNN'95 – International Conference on Neural Networks*, 4, 1942–1948.

[3] Shi, Y., and Eberhart, R. (1998). [A modified particle swarm optimizer](https://doi.org/10.1109/ICEC.1998.699146). *Proceedings of the 1998 IEEE International Conference on Evolutionary Computation*, 69–73.

[4] Clerc, M., and Kennedy, J. (2002). [The particle swarm – explosion, stability, and convergence in a multidimensional complex space](https://doi.org/10.1109/4235.985692). *IEEE Transactions on Evolutionary Computation*, 6(1), 58–73.

[5] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[6] Deb, K., and Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for engineering design. *Computer Science and Informatics*, 26(4), 30–45.

[7] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.
