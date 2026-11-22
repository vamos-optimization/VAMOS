---
description: 'Learn multi-objective optimization from the ground up: Pareto dominance, quality indicators, NSGA-II step by step, many-objective and constrained problems, and rigorous comparison, with interactive figures and VAMOS code.'
---

# Foundations of multi-objective optimization

This learning path explains the ideas behind VAMOS rather than its interface.
It starts from zero (what it means to optimize several objectives at once)
and ends with the questions researchers face when they compare algorithms.
Every lesson has interactive figures you can manipulate, runnable VAMOS code,
and references to the papers where each idea was introduced.

You need no previous knowledge of optimization. Basic mathematical notation
(vectors, sums, minimum and maximum) is enough to follow the text; Python is
only needed to run the code.

## How the path is organized

Three levels build on each other, and the topics that every algorithm shares
have their own section.

| Section | Lessons | After it you can |
| --- | --- | --- |
| **Basic** | [What is a multi-objective problem?](basic/multi-objective-problems.md) · [Pareto dominance and the Pareto front](basic/pareto-dominance.md) · [What makes a good approximation?](basic/quality-aspects.md) · [Quality indicators](basic/quality-indicators.md) · [Families of multi-objective algorithms](basic/algorithm-families.md) | Formulate a problem with several objectives, reason with Pareto dominance, judge an approximation set, compute the standard quality indicators and tell the families of algorithms apart. |
| **Intermediate** | *Dominance-based:* NSGA-II ([at a glance](intermediate/nsga-ii.md) · [non-dominated sorting](intermediate/non-dominated-sorting.md) · [crowding distance and tournaments](intermediate/crowding-distance.md) · [elitist survival](intermediate/elitist-survival.md)) · [SPEA2](intermediate/spea2.md). *Decomposition-based:* [MOEA/D](intermediate/moead.md). *Indicator-based:* [SMS-EMOA](intermediate/smsemoa.md) · [IBEA](intermediate/ibea.md). *Swarm-based:* [SMPSO](intermediate/smpso.md). | Explain every step of NSGA-II and the code that implements it in VAMOS, and explain in the same depth one or two algorithms of each family. |
| **Operators** | [Real-coded operators: SBX and polynomial mutation](operators/real-coded.md) | Explain how crossover and mutation create offspring, and what their parameters control. |
| **Benchmarks** | [What a benchmark tests](benchmarks/overview.md) · [ZDT, DTLZ and WFG](benchmarks/continuous.md) · [Many objectives and many variables](benchmarks/scalable.md) · [Constrained benchmarks](benchmarks/constrained.md) · [Real-world problems](benchmarks/real-world.md) · [Binary, integer, permutation and mixed problems](benchmarks/combinatorial.md) | Explain what a benchmark problem tests, recognize the difficulties of the continuous, scalable, constrained, real-world and discrete suites of VAMOS, read their reference fronts, and choose problems for a study. |
| **Advanced** | *Many objectives:* [many-objective optimization](advanced/many-objective.md) · [NSGA-III](advanced/nsga-iii.md) · [RVEA](advanced/rvea.md) · [AGE-MOEA](advanced/agemoea.md). *Constraints:* [constrained optimization](advanced/constrained.md) · [CCMO](advanced/ccmo.md) · [C-TAEA](advanced/ctaea.md) · [ToP](advanced/top.md) · [CMOEA-MS](advanced/cmoeams.md). [Comparing algorithms rigorously](advanced/performance-assessment.md). | Recognize why four or more objectives and constraints change the problem, explain the many-objective and constrained algorithms of VAMOS, and design a comparison whose conclusions survive scrutiny. |

Read the levels in order the first time; each lesson builds on the previous one
and ends with a link to the next. Within the Intermediate level, the NSGA-II
lessons come first, because the other algorithms are explained by how they
differ from it. The Operators section can be read whenever a lesson mentions
crossover or mutation, and the Benchmarks section once the Basic level is
done.

## How the interactive figures work

The figures run in your browser. You can drag the points, or focus one with
<kbd>Tab</kbd> and move it with the arrow keys (hold <kbd>Shift</kbd> for
larger steps). Sliders and buttons change the data, and a readout next to
each figure states the numbers that the picture shows.

The figures use a small JavaScript port of the same algorithms that VAMOS
implements in Python: non-dominated sorting, crowding distance, simulated
binary crossover, polynomial mutation and the two-objective quality indicators.
Their results are checked against VAMOS and MooCore, but they exist to teach;
use VAMOS itself for real experiments. Without JavaScript the figures do not
appear, and the text of every lesson remains complete.

## Run the code

Each lesson ends with a short program that uses the public VAMOS interface.
Install VAMOS first, following the [installation guide](../guide/installation.md):

```bash
python -m pip install "vamos-optimization==1.5.2"
```

The quality-indicator examples also use [MooCore](https://multi-objective.github.io/moocore/python/),
the indicator library that the [quality and budget tutorial](../tutorials/t07-quality-and-budget.md)
uses as well:

```bash
python -m pip install moocore
```

The outputs printed in the lessons come from VAMOS 1.5.2 with the NumPy
engine. Values computed by an optimization run can differ in their last digits
in another environment; a seed makes a run repeatable in the same
environment, not bitwise identical everywhere.

## Where to go next

After the Basic level you can already read a result: [T01. Your first
meaningful result](../tutorials/t01-first-result.md) applies Pareto dominance
to a real run. After the Intermediate level, the [algorithm lessons](../tutorials/index.md#learn-an-algorithm)
show each built-in algorithm with its exact configuration. The Advanced level
leads to the [study tutorials](../tutorials/t06-fair-study.md), where
comparisons are planned, measured and reported with saved evidence.

## Further reading

These books and tutorials cover the whole path in more depth. Each lesson
cites the original papers for its own topic.

- Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley.
- Coello Coello, C. A., Lamont, G. B., and Van Veldhuizen, D. A. (2007). [*Evolutionary Algorithms for Solving Multi-Objective Problems*](https://doi.org/10.1007/978-0-387-36797-2) (2nd ed.). Springer.
- Miettinen, K. (1999). [*Nonlinear Multiobjective Optimization*](https://doi.org/10.1007/978-1-4615-5563-6). Kluwer Academic Publishers.
- Emmerich, M. T. M., and Deutz, A. H. (2018). [A tutorial on multiobjective optimization: fundamentals and evolutionary methods](https://doi.org/10.1007/s11047-018-9685-y). *Natural Computing*, 17(3), 585–609.
