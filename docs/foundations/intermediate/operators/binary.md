---
description: 'Variation operators for bit strings: one-point, two-point, uniform and HUX crossover, bit-flip and segment inversion mutation, positional bias, an interactive step-through and the VAMOS implementation.'
search:
  boost: 0.5
---

# Binary operators

A binary decision vector is a string of n bits: an item is selected or not, a
feature is used or not, a switch is on or off. Binary crossovers build children
by **copying** bits from the parents, so a position where both parents agree
keeps its value in every child; only mutation can change it. The operators
differ in which bits travel together, and that matters when neighbouring bits
mean related things.

In this lesson you will learn:

- how one-point, two-point, uniform and half-uniform (HUX) crossover choose the
  bits that the children exchange;
- what positional bias is, and which operators have it;
- how bit-flip and segment inversion mutation change a string;
- how VAMOS implements them, and how to select them.

## Crossovers

All four crossovers take a pair of parents and return two complementary
children: wherever the first child takes P2's bit, the second takes P1's. Each
pair is crossed with the crossover probability (0.9 by default) and copied
otherwise.

- **One-point crossover** [1] draws a cut point c from 1 to n − 1. The children
  exchange their tails: C1 = P1[0:c] + P2[c:n].
- **Two-point crossover** [2] draws two positions and exchanges the segment
  between them. In VAMOS both are drawn from 0 to n − 1; the smaller is the
  start, the larger (at least start + 1) the end, excluded.
- **Uniform crossover** [3] decides every position independently: the children
  exchange each bit with probability 0.5.
- **HUX**, half-uniform crossover, comes from Eshelman's CHC algorithm [5]. It
  looks only at the positions where the parents differ and exchanges exactly
  half of them, rounded down (at least one), chosen at random. Each child then
  lies halfway between the parents in Hamming distance, exactly so when the
  number of differing positions is even.

## Try it: building the children step by step

The figure shows two 12-bit parents. Choose an operator and step through it:
first the parents, then the cut points or the mask that the operator draws, then
the children, with every bit coloured by the parent it came from. Click a
parent bit to change it, or focus it with <kbd>Tab</kbd>, move with the arrow
keys and press <kbd>Enter</kbd>. **New draw** draws other cut points or another
mask with VAMOS's rules.

<div class="vf-widget" data-vf-widget="binary-steps"><p class="vf-fallback">Interactive figure: two editable 12-bit parents and, step by step, the cut points or mask that one-point, two-point, uniform or HUX crossover draws, or the bits that bit-flip or segment inversion mutation changes, and the resulting children. It needs JavaScript.</p></div>

Things to try:

1. With one-point crossover, note that bits 0 and 11 never travel together:
   every cut separates them. Bits 4 and 5 are separated only when the cut falls
   exactly between them, with probability 1/11.
2. Make the parents equal in their first six bits. Every operator now gives
   children that share those six bits, whatever it draws.
3. With HUX, count the differing positions in step 1 and the exchanged ones in
   step 2: always half, rounded down.

## Positional bias

One-point crossover keeps adjacent bits together far more often than distant
ones: two bits at distance k are separated with probability k/(n − 1). It has a
**positional bias** [4]: it preserves blocks of neighbouring bits and breaks
long-range combinations. Two-point crossover exchanges a segment instead of a
tail, so the first and last bits are no longer always separated; it reduces the
bias but does not remove it. Uniform crossover and HUX have no
positional bias; instead they exchange many bits at once, a strong
**distributional** disruption [4].

Neither property is good or bad in itself. When related decisions sit next to
each other, as in ZDT5, whose objectives count the ones in consecutive blocks
of bits [6], keeping blocks together helps; when the order of the bits is
arbitrary, it is an accident of the encoding.

## Mutations

- **Bit-flip mutation** flips each bit independently with probability p_m. With
  p_m = 1/n, the default `"1/n"`, a child has one flipped bit on average and
  about a third of the children have none, (1 − 1/n)ⁿ, which tends to
  1/e ≈ 37% for long strings [7].
- **Segment inversion** chooses a random segment, from position a to position b
  with a &lt; b, both included, and flips every bit in it. Its probability is
  per **child**: `prob` is the share of children that receive such a segment.
  (The name refers to inverting the bits, not to reversing their order.)

The core of the four operators in VAMOS, starting with one-point crossover and
HUX:

```python
# Excerpt from src/vamos/engine/operators/impl/binary.py
    cuts = rng.integers(1, D, size=idx.size)
    for row, cut in zip(idx, cuts):
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = np.concatenate([p1[:cut], p2[cut:]])
        child2 = np.concatenate([p2[:cut], p1[cut:]])
        diff_mask = p1 != p2
        if not np.any(diff_mask):
            continue
        diff_idx = np.flatnonzero(diff_mask)
        swap_count = max(1, diff_idx.size // 2)
        chosen = rng.choice(diff_idx, size=swap_count, replace=False)
```

and the two mutations:

```python
# Excerpt from src/vamos/engine/operators/impl/binary.py
    mask = rng.random(X.shape) <= prob
    active = np.flatnonzero(rng.random(rows) <= prob)
    for row in active:
        a = int(rng.integers(0, cols))
        b = int(rng.integers(0, cols - 1))
        if b >= a:
            b += 1
        if a > b:
            a, b = b, a
        X[row, a : b + 1] = 1 - X[row, a : b + 1]
```

## In VAMOS

The operators live in `src/vamos/engine/operators/impl/binary.py`, and the
figure above follows their draws; a test checks its children against them. In
the figure's default parents, P1 = 110100111010 and P2 = 011011010001 differ in
nine positions, so HUX always exchanges four of them, and bits 1, 7 and 9,
equal in both parents, keep their value in every child. Bit-flip with 1/n
makes small, scattered changes and leaves about a third of the children
untouched; a segment inversion changes a contiguous block of at least two bits,
a large step, so give it a small probability.

### Selecting them in a configuration

The builder names are `one_point` (also `single_point`, `1point`, `spx`),
`two_point` (`2point`), `uniform` and `hux`, and `bitflip` (`bit_flip`) and
`segment_inversion`. `NSGAIIConfig.default(encoding="binary")` uses uniform
crossover with probability 0.9 and bit-flip with 1/n. The four crossovers on
ZDT5, 80 bits, with bit-flip mutation, and one-point crossover with segment
inversion for one child in ten:

```python
import numpy as np

from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig, available_crossover_methods, available_mutation_methods

print("crossovers:", ", ".join(available_crossover_methods("binary")))
print("mutations:", ", ".join(available_mutation_methods("binary")))

problem = make_problem_selection("zdt5").instantiate()
pairs = [("one_point", "bitflip", "1/n"), ("two_point", "bitflip", "1/n"), ("uniform", "bitflip", "1/n"),
         ("hux", "bitflip", "1/n"), ("one_point", "segment_inversion", 0.1)]
for crossover, mutation, prob in pairs:
    config = (
        NSGAIIConfig.builder()
        .pop_size(100)
        .crossover(crossover, prob=0.9)
        .mutation(mutation, prob=prob)
        .selection("tournament", size=2)
        .build()
    )
    result = optimize(problem, algorithm="nsgaii", algorithm_config=config, max_evaluations=10_000, seed=1)
    distinct = len(np.unique(result.F, axis=0))
    print(f"{crossover:<10} {mutation:<17} lowest f2: {result.F[:, 1].min():.3f}, distinct objective vectors: {distinct}")
```

Output:

```text
crossovers: 1point, 2point, hux, one_point, single_point, spx, two_point, uniform
mutations: bit_flip, bitflip, segment_inversion
one_point  bitflip           lowest f2: 0.414, distinct objective vectors: 29
two_point  bitflip           lowest f2: 0.517, distinct objective vectors: 28
uniform    bitflip           lowest f2: 0.826, distinct objective vectors: 23
hux        bitflip           lowest f2: 0.704, distinct objective vectors: 26
one_point  segment_inversion lowest f2: 0.464, distinct objective vectors: 26
```

In this single run the crossovers with positional bias reached the lowest f2,
which is consistent with ZDT5's blocks of consecutive bits; several seeds would
be needed to conclude anything. The encodings tutorial runs a binary problem
with uniform crossover end to end: [Decision representations](../../../tutorials/encodings.md#binary-masks).

## Key points

- Binary crossovers copy bits; a position where both parents agree keeps its
  value in every child.
- One-point and two-point crossover exchange contiguous blocks and keep nearby
  bits together (positional bias); uniform crossover and HUX decide each
  position independently, and HUX exchanges exactly half of the differing bits.
- Bit-flip mutation flips each bit with probability p_m, 1/n by default;
  segment inversion flips a whole random segment of a child.
- VAMOS's binary default is uniform crossover (0.9) with bit-flip (1/n).

## References

[1] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

[2] De Jong, K. A. (1975). *An Analysis of the Behavior of a Class of Genetic Adaptive Systems*. PhD thesis, University of Michigan.

[3] Syswerda, G. (1989). Uniform crossover in genetic algorithms. *Proceedings of the Third International Conference on Genetic Algorithms*.

[4] Eshelman, L. J., Caruana, R. A., and Schaffer, J. D. (1989). Biases in the crossover landscape. *Proceedings of the Third International Conference on Genetic Algorithms*.

[5] Eshelman, L. J. (1991). The CHC adaptive search algorithm: How to have safe search when engaging in nontraditional genetic recombination. In *Foundations of Genetic Algorithms*. Morgan Kaufmann.

[6] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[7] Bäck, T. (1993). Optimal mutation rates in genetic search. *Proceedings of the Fifth International Conference on Genetic Algorithms*, 2–8.

**Next:** [Integer operators](integer.md).
