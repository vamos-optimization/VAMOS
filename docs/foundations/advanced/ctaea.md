---
description: 'C-TAEA in depth: weight vectors and subregions, the convergence archive and its three update cases, the diversity archive that ignores the constraints, restricted mating selection, and where VAMOS follows PlatEMO and jMetal instead of the paper, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# C-TAEA: a convergence archive and a diversity archive

CCMO (lesson [CCMO](ccmo.md)) helps a constrained population with a second
population that ignores the constraints. The **two-archive evolutionary
algorithm for constrained multi-objective optimization**, C-TAEA, of Li, Chen,
Fu and Yao [1] keeps the same two roles in a different structure: two archives
of N solutions, both fed by one stream of offspring, and a mating rule that
decides, generation by generation, which archive the parents come from.

The **convergence archive (CA)** is driven by feasibility and convergence, and
it is the result of the run. The **diversity archive (DA)** ignores the
constraints and keeps solutions where the CA has few, so that the search can
cross infeasible regions towards the parts of the front that the CA has not
reached. Both archives are organized by **weight vectors**, as in
decomposition-based algorithms (lesson [MOEA/D](../intermediate/moead.md)): each
vector defines a subregion of the objective space, and "where the CA has few"
means "in the subregions with few CA members".

Unlike NSGA-II, C-TAEA has no population that is sorted and truncated as a
whole. Feasibility decides the CA, density in subregions decides both
archives, and the Tchebycheff function chooses within a subregion.

In this lesson you will learn:

- how weight vectors divide the objective space into subregions;
- the three cases of the CA update, and how it removes solutions from the most
  crowded subregion;
- how the DA fills the subregions that the CA neglects, without looking at the
  constraints;
- how the restricted mating selection chooses each parent's archive, and where
  VAMOS follows PlatEMO and jMetal instead of the paper.

## Why two archives

Feasibility-first selection (lesson [Constraints](constrained.md#beyond-feasibility-first))
throws away infeasible solutions as soon as enough feasible ones exist, and with
them the paths across infeasible barriers. Li et al. [1] separate the two jobs.
The CA does what a feasibility-first algorithm does: it keeps feasible
solutions and pushes them towards the constrained front. The DA does what
feasibility first forbids: it keeps good solutions *whatever their
feasibility*, but only in the subregions where the CA is thin, so that it adds
diversity instead of duplicating the CA.

The two archives do not evolve separately. Each generation creates N offspring
from parents taken from either archive, and both archives are updated with the
same offspring. Which archive supplies the parents depends on which one is
currently more useful, measured by its share of non-dominated solutions.

## Weight vectors and subregions

C-TAEA spreads N weight vectors over the unit simplex with Das and Dennis'
lattice [2] (lesson [Many objectives](many-objective.md)), as MOEA/D [3]
does. A solution belongs to the subregion of the vector closest to it in angle,
and inside a subregion solutions are compared by a Tchebycheff function with
the subregion's own vector:

<div class="vf-formula">subregion(x) = argmax_k  cos( f(x), wᵏ )                    (the largest cosine similarity)
g(x | w, z) = max_i  | f_i(x) − z_i | / w_i                 (z: the ideal point of the set compared)</div>

VAMOS takes the requested population size as the requested number of vectors:
the lattice with the most divisions that gives at most that many, with an inner
layer when it has fewer divisions than objectives, and every component raised
to at least 10⁻⁶ so that g never divides by zero. Their number is the size of
both archives and of each generation: 100 requested vectors give 100 for two
objectives and 91 for three.

## The convergence archive

The candidates of the CA update are the CA and the new offspring, 2N
solutions. The update has three cases, decided by the number of feasible
candidates:

<div class="vf-formula">exactly N feasible:   they are the new CA
more than N:          add the fronts of the feasible candidates (non-dominated sorting) until ≥ N,
                      then, while more than N:
                         normalize with the ideal and nadir points of the selection,
                         take the most crowded subregion,
                         among its members at the smallest distance from another one,
                         remove the one with the largest Tchebycheff value
fewer than N:         keep every feasible candidate, then rank the infeasible ones by
                      non-dominated sorting on ( CV, Tchebycheff value ),
                      add fronts until ≥ N and drop the largest violations of the last front</div>

The second case is where C-TAEA differs most from NSGA-II. Removal is driven by
crowding in subregions, not by rank: a dominated solution alone in its
subregion can survive, and a non-dominated solution in a crowded subregion can
go. The CA may therefore hold dominated solutions. In the third case the
infeasible candidates compete on two criteria at once, their violation and
their convergence in their subregion, so a slightly more violated solution that
is much closer to the front can enter the CA.

## The diversity archive

The candidates of the DA update are the DA and the offspring, and the
constraints play no part. The update counts the members of the **new** CA in
each subregion and fills the subregions where the CA is thin first:

<div class="vf-formula">c_k = number of CA members in subregion k
for itr = 1, 2, 3, ...:
    for each subregion k in order:
        add up to  itr − c_k  candidates of k, one at a time, each time the one that is
        non-dominated among the remaining candidates of k and has the smallest Tchebycheff value
stop when the DA has N members or no candidate is left</div>

In the first iteration, every subregion without a CA member receives one
candidate; in the second, every subregion with fewer than two CA members
receives up to 2 − c_k more, and so on. The picks of earlier iterations do not
reduce the quota, so a subregion that the CA leaves empty receives one
candidate in the first iteration and up to two more in the second, if it has
them. Because feasibility is ignored, the DA collects the best
solutions of the neglected subregions even when they are infeasible, much as
population 2 of CCMO does along the whole front.

## Restricted mating selection

Each offspring has two parents, and each parent is chosen from one archive:

<div class="vf-formula">ρ_c, ρ_d = numbers of non-dominated solutions of CA ∪ DA that lie in the CA, in the DA, divided by |CA ∪ DA|
first parent:   from the CA if ρ_c > ρ_d, otherwise from the DA        (the same for the whole generation)
second parent:  from the CA with probability P_C = share of non-dominated solutions in the CA alone,
                otherwise from the DA</div>

While the CA is behind (few of its solutions are non-dominated when compared
with the DA), mating draws on the DA; as the CA catches up, it supplies more
parents. Within an archive each parent wins a binary tournament: a feasible
solution beats an infeasible one; of two feasible ones, the first wins unless
the second dominates it; of two infeasible ones, a third solution drawn at
random wins. The offspring is the first child of SBX [4], mutated: one
offspring per weight vector and generation.

## Try it: the two updates on the same candidates

The figure shows the 16 CTP2 candidates of lesson [CCMO](ccmo.md#try-it-one-set-of-candidates-two-rules),
with N = 8 weight vectors drawn as rays from the origin. In a run, the CA update
receives the CA and the offspring, and the DA update the DA and the offspring;
here both receive the same 16 candidates, so that only the rules differ. Start
with the CA update, then switch to the DA update and compare the subregions
each archive covers.

<div class="vf-widget" data-vf-widget="constrained-selection" data-rule="ctaea"><p class="vf-fallback">Interactive figure: sixteen candidate solutions of CTP2, the eight weight vectors of C-TAEA, and the eight solutions kept by the convergence archive update and by the diversity archive update. It needs JavaScript.</p></div>

Nine candidates are feasible, one more than the CA holds, so the second case
applies. The first two fronts of the feasible candidates hold all nine; after
normalization, subregions 2 and 4 hold two each, and the first, subregion 2,
loses O2, whose Tchebycheff value (0.379) is worse than that of P3 (0.070). P2
and P6, both dominated, stay because their subregions are not crowded. The DA
then starts where the CA has nothing: subregion 6 receives O7 in the first
iteration; subregions 2, 4 and 6 receive O3, O5 and P8 in the second; and the
third iteration adds O1, P3, P4 and O4. Six of its eight members are infeasible.

## In VAMOS

C-TAEA on CTP2 [5] with 50 requested weight vectors and 10,000 evaluations.
The script counts, for each archive, the feasible members and the subregions
they occupy, using the weight vectors that the result stores:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import CTAEAConfig
from vamos.problems import CTP

problem = CTP(2)
config = CTAEAConfig.default(pop_size=50, n_var=problem.n_var)
result = optimize(problem, algorithm="ctaea", algorithm_config=config, max_evaluations=10_000, engine="numpy", seed=1)
W = result.data["weights"]
print("weight vectors:", W.shape)
print("non-dominated feasible solutions of the CA:", result.F.shape[0])
for name in ("population", "diversity_archive"):
    F, G = result.data[name]["F"], result.data[name]["G"]
    feasible = int(np.all(G <= 0.0, axis=1).sum())
    cosine = (F @ W.T) / np.outer(np.linalg.norm(F, axis=1), np.linalg.norm(W, axis=1))
    occupied = len(np.unique(np.argmax(cosine, axis=1)))
    print(f"{name}: {feasible} of {len(F)} feasible, {occupied} subregions, mean f1 + f2 = {F.sum(axis=1).mean():.3f}")
```

Output:

```text
weight vectors: (50, 2)
non-dominated feasible solutions of the CA: 25
population: 50 of 50 feasible, 11 subregions, mean f1 + f2 = 1.801
diversity_archive: 3 of 50 feasible, 34 subregions, mean f1 + f2 = 1.054
```

`result.data["population"]` is the whole CA, the result of PlatEMO's
implementation and of jMetal; `result.F` keeps its feasible non-dominated
members, here 25 of 50: the CA holds dominated solutions, as described above.
On CTP2 the sum f₁ + f₂ is the distance function g(x), equal to 1 on the
unconstrained front. After this short run the CA is feasible but concentrated
in 11 subregions, while the DA, almost entirely infeasible, lies close to the
unconstrained front and covers 34.

## Where VAMOS differs from the paper

VAMOS follows jMetal's port, which follows PlatEMO's implementation where it
differs from the paper; the [C-TAEA lesson](../../algorithms/ctaea.md#where-vamos-departs-from-jmetals-port)
lists the checks against jMetal's classes. Relative to the paper:

- the second parent comes from the CA with the share of non-dominated
  solutions of the CA alone, P_C, not with the share that chooses the first
  parent's archive;
- solutions are associated with subregions on their raw objectives, except in
  the truncation of the CA's feasible solutions, while the paper normalizes
  before every association;
- the ideal point of each Tchebycheff function is that of the set it is
  applied to;
- ties go to the solution that comes first instead of a random one, and a
  tournament between two infeasible solutions picks a third one at random,
  which gives the winner the same distribution.

Relative to jMetal's port:

- **Budget:** exact; the last generation evaluates only the first offspring,
  in the order of the weight vectors, while jMetal can exceed the budget by up
  to N − 1 evaluations.
- **Parent order:** jMetal's SBX gives a variable that it does not cross to the
  other child, and VAMOS's leaves it with its own parent, so VAMOS passes the
  second parent first; the kept child then has jMetal's distribution.
- **Result:** jMetal returns the whole CA, which is `result.data["population"]`;
  `result.X` and `result.F` follow the configured `result_mode`.

The [C-TAEA lesson](../../algorithms/ctaea.md#agreement-with-jmetals-port-and-the-published-results)
records the agreement with jMetal and with the published results.

## C-TAEA and CCMO

| | CCMO | C-TAEA |
| --- | --- | --- |
| Structure | Two populations, each breeding its own offspring | Two archives, one stream of offspring |
| Constraint-free part | Population 2, over the whole front | DA, only where the CA is thin |
| Diversity | SPEA2 density and truncation | Subregions of N weight vectors |
| Who breeds | Each population from itself | Parents from the CA or the DA, by non-dominated shares |
| Result | Population 1 | The CA |

Strengths:

- **Targeted exploration.** The DA spends its places in the subregions that
  the CA neglects, not along the whole unconstrained front.
- **Adaptive mating.** The share of non-dominated solutions decides which
  archive breeds, without a schedule or a threshold.
- **Two criteria for infeasible solutions.** While the CA has too few feasible
  solutions, violation and convergence decide together which infeasible ones
  enter it.

Limits:

- **Weight vectors fix the archive size and assume a front that they cover
  well.** The number of solutions follows the lattice (91 for 100 requested
  with three objectives), and, as for MOEA/D, a front whose shape does not match
  the vectors leaves subregions empty.
- **The CA can hold dominated solutions,** as the example shows; the result
  needs the non-dominated filter that `result.F` applies.
- **Cost.** The updates work subregion by subregion and solution by solution;
  in VAMOS a C-TAEA generation is slower than one of NSGA-II or CCMO with the
  same number of evaluations.

## Key points

- C-TAEA keeps a convergence archive and a diversity archive of N solutions,
  one per weight vector, and updates both with the same offspring.
- The CA keeps feasible solutions; with too many, it removes the worst
  Tchebycheff solution of the most crowded subregion, whatever its rank; with
  too few, it ranks the infeasible ones on violation and Tchebycheff value.
- The DA ignores the constraints and fills the subregions with the fewest CA
  members first.
- The first parent's archive is the one with the larger share of non-dominated
  solutions of CA ∪ DA; the second parent comes from the CA with probability
  equal to the CA's own non-dominated share.
- VAMOS follows PlatEMO's implementation through jMetal's port, with an exact
  budget.

**Next:** [ToP: two phases](top.md).

## References

[1] Li, K., Chen, R., Fu, G., and Yao, X. (2019). [Two-archive evolutionary algorithm for constrained multiobjective optimization](https://doi.org/10.1109/TEVC.2018.2855411). *IEEE Transactions on Evolutionary Computation*, 23(2), 303–315.

[2] Das, I., and Dennis, J. E. (1998). [Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, 8(3), 631–657.

[3] Zhang, Q., and Li, H. (2007). [MOEA/D: A multiobjective evolutionary algorithm based on decomposition](https://doi.org/10.1109/TEVC.2007.892759). *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

[4] Deb, K., and Agrawal, R. B. (1995). Simulated binary crossover for continuous search space. *Complex Systems*, 9(2), 115–148.

[5] Deb, K., Pratap, A., and Meyarivan, T. (2001). [Constrained test problems for multi-objective evolutionary optimization](https://doi.org/10.1007/3-540-44719-9_20). *Evolutionary Multi-Criterion Optimization – EMO 2001*, Lecture Notes in Computer Science 1993, 284–298.
