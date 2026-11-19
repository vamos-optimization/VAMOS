---
description: 'A10. Comparing stochastic multi-objective algorithms rigorously: experimental design, empirical attainment functions, statistical tests and effect sizes, anytime performance and common pitfalls, with VAMOS and MooCore code.'
search:
  boost: 0.5
---

# A10. Comparing algorithms rigorously

"Algorithm A is better than algorithm B" is the most common claim in the
field, and the easiest to get wrong. Evolutionary algorithms are randomized,
their output is a set, and their performance depends on the problem, the
budget and the settings. This lesson turns the tools of the previous lessons
into a method for comparisons that survive scrutiny.

In this lesson you will learn:

- why one run is one random sample, and what that implies;
- what to fix before running an experiment;
- how attainment functions show the distribution of whole approximation sets;
- how to test differences in an indicator and report their size;
- how to run such a comparison with VAMOS.

## A run is a sample

Run an algorithm twice with different seeds and you get two different
approximation sets. The outcome of a run is a **random set**, and the
algorithm's performance is the distribution of that set [1]. Consequently:

- one run per algorithm proves nothing, however good the plot looks;
- runs must be **independent**: different seeds, recorded in advance;
- every summary (a median hypervolume, a plot of one run) must say which runs
  it summarizes and how much they vary.

## Design the experiment first

Decide the following **before** the first run [2, 3]:

1. **The question.** "Is A better than B on bi-objective problems with
   disconnected fronts within 10,000 evaluations?" is testable; "Is A better?"
   is not.
2. **The problems.** Choose problems that represent the question, with
   different front shapes and difficulties; avoid tuning the problem set until
   your algorithm wins.
3. **The budget**, in function evaluations, not in time. Several budgets
   reveal whether a ranking changes as runs get longer.
4. **The settings.** Give every algorithm the same care: published defaults for
   all, or the same tuning effort for all. Tuning only your own algorithm
   biases the comparison.
5. **The number of runs** and their seeds. A few dozen independent runs per
   algorithm and problem is common; decide the number in advance.
6. **The reference data**: reference points, reference sets and normalization
   bounds (lesson [B4](../basic/quality-indicators.md#choosing-and-using-indicators)).
7. **The indicators and the statistical procedure**, including the correction
   for multiple comparisons.

VAMOS supports this discipline: a durable study fixes the matrix of
algorithms, problems and seeds before it runs and saves every run with its
resolved configuration. [T06. Run a fair comparative
study](../../tutorials/t06-fair-study.md) applies the checklist step by step.

## Attainment functions: the distribution of whole sets

An indicator reduces each run to one number. Attainment functions keep the
sets [1, 4]. A point z of objective space is **attained** by a run if the
run's approximation set weakly dominates it. The **empirical attainment
function** (EAF) gives, for every z, the fraction of runs that attain it. Its
level sets are the **k%-attainment surfaces**:

- the *best* surface bounds what at least one run attained;
- the *median* (50%) surface bounds what half of the runs attained, a typical
  run;
- the *worst* surface bounds what every run attained.

Comparing the median surfaces of two algorithms, or plotting the difference of
their EAFs, shows **where** in objective space each one is better [5]. The
figure uses synthetic runs of two algorithms: A covers the whole front but
converges less; B converges closer, but only on part of the front.

<div class="vf-widget" data-vf-widget="attainment"><p class="vf-fallback">Interactive figure: median, best and worst attainment surfaces of two synthetic algorithms and the hypervolume of each of their runs. It needs JavaScript.</p></div>

The median surfaces usually cross: B is better in the region its runs
converged to, and A where B's runs rarely reach. The hypervolume weights those
regions in its own way and declares one winner; the attainment surfaces show
what that verdict is made of. Press **New runs** a few times to see how much
the picture varies with only eleven runs each.

## Test the differences

With one indicator value per run, two algorithms on one problem give two
samples. Indicator values are rarely normally distributed, so use
non-parametric tests [6, 7]:

- **Per problem**, compare the independent runs of two algorithms with the
  Mann–Whitney U test, also called the Wilcoxon rank-sum test [8, 9].
- **Across problems**, summarize each algorithm on each problem by one
  pre-defined statistic (for example the median hypervolume), and compare the
  algorithms with the Friedman test followed by post-hoc tests [10, 6].
- **Correct for multiple comparisons**, for example with Holm's procedure [11]:
  with twenty comparisons at 5%, one "significant" result is expected by chance.
- **Report the effect size**, not only the p-value. The Vargha–Delaney A₁₂
  statistic [12] estimates the probability that a run of A beats a run of B:
  0.5 means no difference, 1 means A always wins. A tiny difference can be
  "significant" with enough runs, and a large one can be "not significant"
  with too few.

A p-value says how surprising the data would be if the algorithms performed
the same; it says nothing about how much better one is, or whether the
difference matters to anyone. VAMOS' experimental analysis helpers include the
Friedman test and pairwise Wilcoxon tests for the across-problem case; see
[analysis and visualization](../../topics/analysis.md#statistical-comparison).

## Anytime performance

A single budget gives a snapshot. Plotting the indicator against the number of
evaluations, as the live figure of lesson
[I1](../intermediate/nsga-ii.md#watch-it-converge) does, shows which algorithm
is better early and which is better late. When the budget of a real application
is uncertain, that is often the relevant comparison. [T07. Measure quality and
progress with budget](../../tutorials/t07-quality-and-budget.md) measures saved
runs at several budgets.

## Common pitfalls

- Reporting one run, or the best of several runs.
- Different budgets, population sizes or numbers of runs for different
  algorithms, or a budget counted in generations when the population sizes
  differ.
- Choosing the reference point, the reference set or the normalization after
  seeing the results.
- Using an indicator that is not Pareto compliant as the only evidence
  (lesson [B4](../basic/quality-indicators.md#pareto-compliance)).
- Generalizing beyond the problems tested. No algorithm is best on all
  problems [13]; a comparison supports a claim only about problems like those
  in it.
- Discarding failed or infeasible runs from the summary without reporting them.

## In VAMOS

Eleven independent runs each of NSGA-II and SPEA2 on ZDT1, with the hypervolume
of every run, a Mann–Whitney test, the Vargha–Delaney effect size, and the
median attainment surface of NSGA-II computed by MooCore. Passing a list of
seeds to `optimize` runs one independent run per seed.

```python
import moocore
import numpy as np
from scipy.stats import mannwhitneyu

from vamos import optimize

seeds = list(range(1, 12))  # 11 independent runs per algorithm, fixed in advance
reference_point = [1.1, 1.1]
hv = {}
fronts = {}
for algorithm in ("nsgaii", "spea2"):
    runs = optimize("zdt1", algorithm=algorithm, pop_size=100, max_evaluations=5_000, engine="numpy", seed=seeds)
    fronts[algorithm] = [run.front() for run in runs]
    hv[algorithm] = np.array([moocore.hypervolume(front, ref=reference_point) for front in fronts[algorithm]])
    q1, median, q3 = np.percentile(hv[algorithm], [25, 50, 75])
    print(f"{algorithm:<7} runs {len(runs)}  median HV {median:.4f}  IQR {q3 - q1:.4f}")

a, b = hv["nsgaii"], hv["spea2"]
test = mannwhitneyu(a, b, alternative="two-sided")
a12 = (np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b))
print(f"Mann-Whitney U {test.statistic:.1f}, p {test.pvalue:.4f}, A12 {a12:.2f}")

labelled = np.vstack([np.column_stack([front, np.full(len(front), k)]) for k, front in enumerate(fronts["nsgaii"], start=1)])
median_surface = moocore.eaf(labelled[:, :2], sets=labelled[:, 2], percentiles=[50])
print("median attainment surface of NSGA-II:", median_surface.shape[0], "points")
```

Output:

```text
nsgaii  runs 11  median HV 0.6636  IQR 0.0822
spea2   runs 11  median HV 0.6411  IQR 0.0642
Mann-Whitney U 74.0, p 0.3933, A12 0.61
median attainment surface of NSGA-II: 109 points
```

The medians differ by less than the spread between runs, and with eleven
runs each the test cannot distinguish the two algorithms (p ≈ 0.39). The
effect size says that a random NSGA-II run beats a random SPEA2 run in about
61% of the pairs: a modest difference that more runs might confirm or not.
Read the result for what it is: one problem, one budget, default settings. It
concerns NSGA-II and SPEA2 with these settings on ZDT1 at 5,000 evaluations,
and nothing more general. A durable
[study](../../guide/studies.md) runs the same design for many problems,
algorithms and seeds, records every run and its configuration, and
[T08](../../tutorials/t08-study-report.md) turns it into a report.

## Key points

- An algorithm's output is a random set: compare distributions over many
  independent runs, never single runs.
- Fix the question, problems, budget, settings, runs, reference data,
  indicators and tests before running.
- Attainment surfaces show where in objective space each algorithm is
  better; indicators summarize it in one number.
- Use non-parametric tests with corrections for multiple comparisons, and
  report effect sizes together with p-values.
- Conclusions hold for the problems, budgets and settings tested.

**You have completed the Foundations path.** Continue with the
[study tutorials](../../tutorials/t06-fair-study.md) to apply it with saved,
verifiable evidence.

## References

[1] Fonseca, C. M., and Fleming, P. J. (1996). [On the performance assessment and comparison of stochastic multiobjective optimizers](https://doi.org/10.1007/3-540-61723-X_1022). *Parallel Problem Solving from Nature – PPSN IV*, Lecture Notes in Computer Science 1141, 584–593.

[2] Knowles, J., Thiele, L., and Zitzler, E. (2006). *A tutorial on the performance assessment of stochastic multiobjective optimizers*. TIK Report 214, Computer Engineering and Networks Laboratory, ETH Zurich.

[3] Bartz-Beielstein, T., et al. (2020). [Benchmarking in optimization: Best practice and open issues](https://arxiv.org/abs/2007.03488). arXiv:2007.03488.

[4] Grunert da Fonseca, V., Fonseca, C. M., and Hall, A. O. (2001). [Inferential performance assessment of stochastic optimisers and the attainment function](https://doi.org/10.1007/3-540-44719-9_15). *Evolutionary Multi-Criterion Optimization – EMO 2001*, Lecture Notes in Computer Science 1993, 213–225.

[5] López-Ibáñez, M., Paquete, L., and Stützle, T. (2010). [Exploratory analysis of stochastic local search algorithms in biobjective optimization](https://doi.org/10.1007/978-3-642-02538-9_9). In T. Bartz-Beielstein et al. (Eds.), *Experimental Methods for the Analysis of Optimization Algorithms*, 209–222. Springer.

[6] Derrac, J., García, S., Molina, D., and Herrera, F. (2011). [A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms](https://doi.org/10.1016/j.swevo.2011.02.002). *Swarm and Evolutionary Computation*, 1(1), 3–18.

[7] Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.

[8] Wilcoxon, F. (1945). [Individual comparisons by ranking methods](https://doi.org/10.2307/3001968). *Biometrics Bulletin*, 1(6), 80–83.

[9] Mann, H. B., and Whitney, D. R. (1947). [On a test of whether one of two random variables is stochastically larger than the other](https://doi.org/10.1214/aoms/1177730491). *The Annals of Mathematical Statistics*, 18(1), 50–60.

[10] Friedman, M. (1937). [The use of ranks to avoid the assumption of normality implicit in the analysis of variance](https://doi.org/10.1080/01621459.1937.10503522). *Journal of the American Statistical Association*, 32(200), 675–701.

[11] Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.

[12] Vargha, A., and Delaney, H. D. (2000). [A critique and improvement of the CL common language effect size statistics of McGraw and Wong](https://doi.org/10.3102/10769986025002101). *Journal of Educational and Behavioral Statistics*, 25(2), 101–132.

[13] Wolpert, D. H., and Macready, W. G. (1997). [No free lunch theorems for optimization](https://doi.org/10.1109/4235.585893). *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82.
