---
description: 'Variation operators for permutations: order, partially mapped, cycle, position-based, edge recombination and alternating edges crossover, swap, insert, scramble, inversion, displacement and 2-opt mutation, with interactive step-throughs and a tour.'
search:
  boost: 0.5
---

# Permutation operators

A permutation lists n items in an order, each exactly once: the cities of a
tour, the jobs of a schedule, the rooms of a layout. VAMOS encodes it as a row
holding 0, 1, …, n − 1 in some order. The constraint "each exactly once" rules
out the operators of the previous lessons: a one-point crossover of two tours
duplicates some cities and loses others. Permutation operators are built to
keep the constraint, and they differ in what they pass on from the parents:
the **absolute positions** of the items, their **relative order**, or their
**adjacencies**, the edges of a tour.

In this lesson you will learn:

- how order crossover, partially mapped crossover and cycle crossover build
  valid children, step by step;
- how position-based, edge recombination and alternating edges crossover
  work, and what each one preserves;
- how swap, insert, scramble, inversion, displacement and 2-opt mutation move
  items, and why an inversion is the 2-opt move of routing;
- how to select them in VAMOS.

## Three classic crossovers

- **Order crossover (OX)** [1] copies a segment of P1 into C1 in place and
  fills the other positions with the missing items in the order they appear in
  P2, starting after the segment and wrapping around. It preserves the
  **relative order** of P2's items. In VAMOS the segment runs from a start
  position to an end position, excluded, and the items are taken from P2 from
  its first position on.
- **Partially mapped crossover (PMX)** [2] copies a section of P2 into C1, in
  place, and copies P1 everywhere else. An item of P1 that already appears in
  the section is replaced through the **mapping** that the section defines
  (P2's item ↔ P1's item at the same position), repeatedly, until the item is
  new. It preserves **absolute positions** from both parents. In VAMOS the
  section runs between two distinct positions, both included.
- **Cycle crossover (CX)** [3] partitions the positions into cycles: start at a
  position, look at P2's item there, go to the position of that item in P1, and
  repeat until the start comes back. The positions of a cycle hold the same set
  of items in both parents, so C1 can take them from P1 and everything else
  from P2. Every item keeps a position it had in one of the parents. VAMOS
  takes one cycle, from a random start, from P1, and the rest from P2.

### Try it: OX, PMX and CX step by step

Choose an operator, set the segment or the start position, and step through
it. Cells are coloured by the parent the item came from; the segment, the
section or the cycle is marked in the parents.

<div class="vf-widget" data-vf-widget="permutation-steps"><p class="vf-fallback">Interactive figure: two permutations of eight items and, step by step, how PMX, order crossover, cycle crossover or position-based crossover builds the two children, with adjustable segment, start position or kept positions. It needs JavaScript.</p></div>

Things to try:

1. PMX with the section from 2 to 5: the mapping is 4↔2, 0↔3, 5↔4 and 7↔5,
   and C1 = [3 1 4 0 5 7 6 2]. At position 7, P1's item 7 is already in C1's
   section, and the chain 7 → 5 → 4 → 2 takes three steps to reach an item that
   is still free.
2. CX with start 1: the cycle is positions 1 and 6 only, because P2 holds item
   6 at position 1 and P1 holds it at position 6, and vice versa for item 1.
   Start 0 gives a cycle of six positions.
3. OX with a segment of one item: almost everything comes from P2, in P2's
   order, so C1 is nearly P2 rotated.

In the figure, OX with the segment from 2 to 5 copies P1's 2, 3, 4, 5 and
fills with P2's 6, 0, 7, 1, giving C1 = [7 1 2 3 4 5 6 0]; CX from start 1
exchanges only positions 1 and 6.

## Position-based, edge recombination and alternating edges

- **Position-based crossover** [4] keeps a random set of positions from P1 in
  C1 and fills the free positions, left to right, with P2's other items in
  P2's order. VAMOS draws the number of kept positions uniformly from 1 to n,
  so a child can equal its parent.
- **Edge recombination (ERX)** [5] treats the parents as tours and passes on
  their **edges**. It builds the table of each item's neighbours in either
  parent, starts from the first item of P1 or P2, chosen at random, and always
  continues to the neighbour that has the fewest neighbours left, breaking ties
  at random; only when the current item has no unused neighbour does it jump to
  a random unvisited item. Most edges of the child come from a parent, which
  suits routing problems, but in VAMOS it is much slower than the others.
- **Alternating edges (AEX)** [6] starts at the first item of P1 and takes its
  successor in P1, then the successor of that item in P2, then in P1 again,
  alternating; when the successor is already in the child it takes the other
  parent's, and when both are, a random unvisited item. In VAMOS the second
  child follows the same rule from the same start, so the two children differ
  only where a random item was needed.

## Mutations

Each permutation mutation changes a whole child at once, so its `prob` is the
probability that a **child** is mutated, not each position. All of them draw two
distinct positions, here called i and j, or a segment between two positions:

- **swap** exchanges the items at i and j;
- **insert** moves the item at i to position j and shifts the items in between;
- **scramble** shuffles the items of the segment, both ends included;
- **inversion** reverses the segment, both ends included;
- **displacement** cuts the segment, from its start to its end excluded, and
  reinserts it at a random position of the rest;
- **2-opt** is the same operator as inversion, with the same random draws;
  the name is the one routing uses.

On a tour, the edges inside a reversed segment stay the same in a symmetric
problem; only the two edges at its ends change. An inversion is therefore the
**2-opt move** that Croes introduced for the travelling salesman problem [8]:
remove two edges and reconnect the two paths the other way. When the two
removed edges cross, the move removes the crossing and shortens the tour. A
swap changes up to four edges and an insertion up to three [7].

### Try it: a mutation on a tour

The tour below visits eight cities and has one crossing. Choose a mutation and
the positions i and j; the dashed line is the current tour, the solid line the
tour after the move, and the orange edges are the new ones. **Best inversion**
finds the inversion that shortens the tour the most, and **Apply the move**
makes the result the current tour.

<div class="vf-widget" data-vf-widget="tour-moves"><p class="vf-fallback">Interactive figure: a tour of eight cities with a crossing, and the tour after an inversion (2-opt move), swap or insertion at chosen positions, with the tour length, the number of crossings and the number of edges that change. It needs JavaScript.</p></div>

Press **Best inversion**: it reverses positions 2 to 6, removes the crossing
and shortens the tour from 3.445 to 3.068. Try a swap of the same positions to
see four edges change instead of two.

A displacement replaces up to three edges of a tour, like an insertion of a
whole segment.

## In VAMOS

The builder names are `ox` (also `order`, `oxd`), `pmx`, `cx` (`cycle`),
`position_based` (`position`, `pos`), `erx` (`edge`, `edge_recombination`) and
`aex` (`alternating_edges`) for the crossovers, and `swap`, `insert`,
`scramble`, `inversion`, `displacement` and `two_opt` for the mutations.
`NSGAIIConfig.default(encoding="permutation")` uses OX and swap; as lesson
[Variation operators](overview.md#two-kinds-of-probability) explains, leaving
out `prob` leaves the probabilities to the algorithm's fallback, so state them.
Six pairs on `kroa100`, the 100 cities of the TSPLIB instance with two
objectives, the tour length f1 and the longest edge f2:

```python
from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig, available_crossover_methods, available_mutation_methods

print("crossovers:", ", ".join(available_crossover_methods("permutation")))
print("mutations:", ", ".join(available_mutation_methods("permutation")))

problem = make_problem_selection("kroa100").instantiate()
pairs = [("ox", "swap"), ("ox", "insert"), ("ox", "displacement"), ("ox", "inversion"), ("pmx", "inversion"), ("cx", "inversion")]
for crossover, mutation in pairs:
    config = (
        NSGAIIConfig.builder()
        .pop_size(100)
        .crossover(crossover, prob=0.9)
        .mutation(mutation, prob=0.2)
        .selection("tournament", size=2)
        .build()
    )
    result = optimize(problem, algorithm="nsgaii", algorithm_config=config, max_evaluations=10_000, seed=1)
    print(f"{crossover:<3} + {mutation:<12} shortest tour in f1: {result.F[:, 0].min():,.0f}")
```

Output:

```text
crossovers: aex, alternating_edges, cx, cycle, edge, edge_recombination, erx, order, ox, oxd, pmx, pos, position, position_based
mutations: displacement, insert, inversion, scramble, swap, two_opt
ox  + swap         shortest tour in f1: 80,266
ox  + insert       shortest tour in f1: 71,142
ox  + displacement shortest tour in f1: 75,446
ox  + inversion    shortest tour in f1: 73,109
pmx + inversion    shortest tour in f1: 78,806
cx  + inversion    shortest tour in f1: 94,591
```

10,000 evaluations are far too few for 100 cities (TSPLIB's optimal kroA100
tour has length 21,282 with its rounded distances), and one run ranks nothing.
In this run, with OX, the swap, which changes up to four edges, gave the
longest tour, and the moves that change two or three edges did better. Cycle crossover passes on absolute positions, which carry no
meaning in a tour, where only the adjacencies count.

## Key points

- Permutation operators keep each item exactly once. OX passes on relative
  order, PMX and CX absolute positions, ERX and AEX the edges of a tour.
- PMX repairs duplicates through the mapping of its section; CX copies whole
  cycles of positions, so every item keeps a position from a parent.
- Permutation mutations apply their probability per child. Swap, insert,
  scramble, inversion and displacement move one or more items; 2-opt is
  VAMOS's name for the inversion on a tour.
- On a symmetric tour an inversion changes exactly two edges and can remove a
  crossing; a swap changes up to four.
- VAMOS's permutation default is OX with swap; state `prob` explicitly.

## References

[1] Davis, L. (1985). Applying adaptive algorithms to epistatic domains. *Proceedings of the Ninth International Joint Conference on Artificial Intelligence (IJCAI)*.

[2] Goldberg, D. E., and Lingle, R. (1985). Alleles, loci, and the traveling salesman problem. *Proceedings of the First International Conference on Genetic Algorithms and Their Applications*.

[3] Oliver, I. M., Smith, D. J., and Holland, J. R. C. (1987). A study of permutation crossover operators on the traveling salesman problem. *Proceedings of the Second International Conference on Genetic Algorithms*.

[4] Syswerda, G. (1991). Schedule optimization using genetic algorithms. In L. Davis (Ed.), *Handbook of Genetic Algorithms*. Van Nostrand Reinhold.

[5] Whitley, D., Starkweather, T., and Fuquay, D. (1989). Scheduling problems and traveling salesmen: The genetic edge recombination operator. *Proceedings of the Third International Conference on Genetic Algorithms*.

[6] Grefenstette, J., Gopal, R., Rosmaita, B., and Van Gucht, D. (1985). Genetic algorithms for the traveling salesman problem. *Proceedings of the First International Conference on Genetic Algorithms and Their Applications*.

[7] Larrañaga, P., Kuijpers, C. M. H., Murga, R. H., Inza, I., and Dizdarevic, S. (1999). Genetic algorithms for the travelling salesman problem: A review of representations and operators. *Artificial Intelligence Review*, 13(2), 129–170.

[8] Croes, G. A. (1958). A method for solving traveling-salesman problems. *Operations Research*, 6(6), 791–812.

**Next:** [Mixed operators](mixed.md).
