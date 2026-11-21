---
description: 'Model-based algorithm configuration: Bayesian optimization with surrogate models (random forests in SMAC, TPE in Optuna), successive halving, Hyperband and BOHB, pruning, and how the optuna, bohb_optuna, smac3 and bohb backends of VAMOS map budget levels to instance/seed pairs, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# Model-based tuning: SMAC3, BOHB and Optuna

Racing (lesson [Racing and irace](tuning-racing.md)) spends its runs well
among the candidates it has, but it chooses the candidates with simple
sampling rules. **Model-based** tuners learn from every configuration they
have evaluated: they fit a **surrogate model** that predicts the score of a
configuration, and use it to choose the next configuration to try. A second
idea, **multi-fidelity**, makes evaluations cheaper: most candidates are
judged on short runs, and only the promising ones get the full budget.

In this lesson you will learn:

- sequential model-based optimization: surrogate models and acquisition
  functions;
- the random forests of SMAC and the tree-structured Parzen estimator (TPE)
  of Optuna;
- successive halving, Hyperband and BOHB, and pruning;
- how the `optuna`, `bohb_optuna`, `smac3` and `bohb` backends of `vamos tune`
  use them, and how VAMOS turns a budget level into runs on instances and
  seeds.

## Sequential model-based optimization

Sequential model-based optimization repeats four steps [1]:

1. evaluate a few configurations, the **initial design**;
2. fit a surrogate model m̂(θ) of the score on all the configurations
   evaluated so far;
3. choose the next configuration by maximizing an **acquisition function**,
   which balances a high predicted score (exploitation) against a high
   uncertainty (exploration); the most common is the expected improvement
   over the best score so far;
4. evaluate it, add it to the history, and go back to step 2.

Each evaluation of a configuration is expensive, many runs of an MOEA, while
fitting the model and maximizing the acquisition function are cheap, so it
pays to think before every evaluation. Configuration spaces add two
difficulties for the model: categorical parameters, and conditional
parameters that exist only in part of the space (lesson
[Automatic algorithm configuration](tuning.md#configuration-spaces)). The two
models below handle both.

### Random forests: SMAC

**SMAC** [1] fits a **random forest** to the history. A forest of regression
trees splits naturally on categorical values and ignores inactive
parameters, and the spread of its trees' predictions gives the uncertainty
that the expected improvement needs. SMAC also *intensifies*: a new
configuration is compared with the current best on more and more instances
and seeds before it can replace it, as a race of two. SMAC3 [2] is the
current Python implementation; its facade for hyperparameter optimization,
which VAMOS uses, models the score with a random forest.

### Tree-structured Parzen estimators: Optuna

The **tree-structured Parzen estimator** (TPE) [3] turns the problem around.
It splits the evaluated configurations into the best fraction γ and the rest,
and estimates two densities over the space: l(θ) from the good
configurations and g(θ) from the others. It proposes the configuration that
maximizes l(θ)/g(θ), which maximizes the expected improvement under its
model. The densities are built parameter by parameter along the tree of the
space, so conditional parameters are modelled only where they exist.
**Optuna** [4] is a framework whose default sampler is TPE; its
*define-by-run* interface asks for each parameter as the code reaches it, so
a conditional space is written as ordinary branching code.

## Multi-fidelity: short runs first

The fidelity of an evaluation is how much it costs: here, the evaluations of
each run, and possibly the number of instances and seeds. A configuration
that is poor after 1,000 evaluations is often still poor after 9,000, so
short runs can discard most candidates cheaply.

**Successive halving** [5] starts n configurations at a small budget r, keeps
the best 1/η of them, multiplies the budget by η, and repeats until one
configuration has the full budget. With η = 3, nine configurations at 300
evaluations, three at 900 and one at 2,700 cost as much as three
configurations at the full budget would.

The risk is that short runs mislead. A configuration that converges fast can
lead at 300 evaluations and fall behind at 2,700, and successive halving
cannot tell in advance how aggressive it should be. **Hyperband** [6] hedges:
it runs several successive-halving **brackets**, from an aggressive one that
starts many configurations at the smallest budget to a cautious one that
starts a few at the full budget. **BOHB** [7] is Hyperband whose new
configurations come from TPE-like density models fitted to the results of
the largest budget with enough data, instead of uniform sampling.

**Pruning** applies the same idea inside a trial: the tuner evaluates a
configuration at increasing budgets and stops it as soon as an intermediate
score is bad. Optuna's median pruner stops a trial whose intermediate score
is worse than the median of the earlier trials at the same step; its
Hyperband pruner assigns trials to brackets of successive halving.

### Try it: successive halving with VAMOS

Successive halving written out on the six NSGA-II configurations of the
race in lesson [Racing and irace](tuning-racing.md#a-race-of-nsga-ii-configurations),
plus three with the distribution index 20, at 300, 900 and 2,700 evaluations
with η = 3, on ZDT1, ZDT2 and ZDT3 with one seed:

```python
import numpy as np

from vamos import optimize
from vamos.engine.tuning.grammar import builtin_parameter_space
from vamos.engine.tuning.racing.bridge import config_from_assignment

space = builtin_parameter_space("nsgaii")
candidates = [
    dict(space.default_values(), sbx_distribution_index=eta, mutation_probability_factor=factor)
    for eta in (5.0, 20.0, 40.0)
    for factor in (0.25, 1.0, 3.0)
]


def hypervolume_2d(F, ref):
    volume, level = 0.0, ref[1]
    for f1, f2 in F[np.lexsort((F[:, 1], F[:, 0]))]:
        if f1 < ref[0] and f2 < level:
            volume += (ref[0] - f1) * (level - f2)
            level = f2
    return volume


def score(assignment, budget):
    config = config_from_assignment("nsgaii", dict(assignment, pop_size=20))
    runs = [optimize(name, algorithm="nsgaii", algorithm_config=config, max_evaluations=budget, seed=1) for name in ("zdt1", "zdt2", "zdt3")]
    return np.mean([hypervolume_2d(run.F, (10.0, 10.0)) for run in runs])


alive, evaluations = candidates, 0
for budget in (300, 900, 2700):
    scores = [score(c, budget) for c in alive]
    evaluations += 3 * budget * len(alive)
    order = np.argsort(scores)[::-1]
    alive = [alive[i] for i in order[: max(1, len(alive) // 3)]]
    kept = ", ".join(f"eta {c['sbx_distribution_index']:.0f}/factor {c['mutation_probability_factor']:.2f} ({scores[i]:.2f})" for c, i in zip(alive, order))
    print(f"{budget:4d} evaluations, candidates {len(scores)}: kept {kept}")
print(f"{evaluations} evaluations, against {3 * 2700 * len(candidates)} for every configuration at 2700")
```

Output:

```text
 300 evaluations, candidates 9: kept eta 20/factor 0.25 (85.30), eta 40/factor 1.00 (84.84), eta 20/factor 3.00 (84.53)
 900 evaluations, candidates 3: kept eta 20/factor 3.00 (91.81)
2700 evaluations, candidates 1: kept eta 20/factor 3.00 (100.47)
24300 evaluations, against 72900 for every configuration at 2700
```

Successive halving finds the configuration with the largest mutation factor,
the one the race of the previous lesson kept, with a third of the
evaluations. It also shows the risk: at 300 evaluations the best
configuration has the *smallest* mutation factor, 0.25, and the eventual
winner is only third, one place from being discarded. With η = 9, keeping one
configuration of nine, it would have been lost. Hyperband's cautious brackets
exist for this case.

## The model-based backends of VAMOS

`vamos tune` has four model-based backends. They need the optional tuning
dependencies, `python -m pip install "vamos-optimization[tuning]==1.5.2"`
(Optuna, SMAC3, ConfigSpace and HpBandSter); `--backend-fallback random` or
`racing` runs a built-in backend instead when they are missing.

| Backend | Proposes configurations with | Multi-fidelity, with `--fidelity-levels` | `--tune-budget` counts |
| --- | --- | --- | --- |
| `optuna` (the default backend) | Optuna's TPE | each level in turn, median pruner | trials |
| `bohb_optuna` | Optuna's TPE | each level in turn, Hyperband pruner | trials |
| `smac3` | SMAC3, random forest | Hyperband between the smallest and the largest level, η = 3 | runs, one per configuration and budget |
| `bohb` | HpBandSter's BOHB | Hyperband between the smallest and the largest level, η = `--bohb-reduction-factor` | configurations |

All four start from the space's grammar. Optuna's sampler walks the tree and
suggests each active parameter, and the first trial of a new study is the
space's default configuration, which runs as the algorithm's default. SMAC3
and BOHB receive a ConfigSpace space with the grammar's defaults and
conditions that follow its tree; SMAC3 keeps its own initial design. VAMOS
seeds every backend with `--seed`, and with one worker (`--n-jobs 1`) a search
is reproducible.

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend optuna \
  --budget 5000 \
  --tune-budget 100 \
  --optuna-storage sqlite:///results/tune.db \
  --optuna-study-name nsgaii_zdt
```

With `--optuna-storage`, the study is saved and the same command resumes it,
running only the trials that `--tune-budget` still allows. VAMOS appends the
scoring contract and the first 16 hexadecimal digits of the space's SHA-256
to the study name, so a study searched over another space, or scored
otherwise, is never mixed with this one.

### One fidelity unless given levels

Without `--fidelity-levels`, the four backends evaluate every candidate once,
at `--budget` evaluations, on every training instance and seed: a trial
costs one run per instance/seed pair, and nothing is pruned. With
`--fidelity-levels`, the largest level is the full budget:

- `optuna` and `bohb_optuna` evaluate a trial at each level in turn, each
  from scratch, so a trial that is never pruned costs the sum of the levels
  per instance/seed pair. The median pruner (`optuna`) or the Hyperband
  pruner with reduction factor `--bohb-reduction-factor` (`bohb_optuna`,
  3 by default) can stop a trial after any level but the last. A pruned trial
  stays in the history with its last score, but it is never returned as the
  best configuration.
- `smac3` and `bohb` run Hyperband between the smallest and the largest
  level. Its intermediate budgets are the largest level divided by powers of
  η, not the intermediate levels that were given. Both return the best
  configuration evaluated at the full budget, and fail with an error when the
  search ends before any configuration reaches it.

The pruner is chosen from the number of levels:

```python
# Excerpt from src/vamos/engine/tuning/_model_backend_runners.py
    levels = resolve_budget_levels(tuner)
    sampler = _build_optuna_sampler(tuner.optuna_sampler, seed=int(tuner.seed))
    # A trial reports the score of each level but the last at steps 1, 2, ...; the pruner may
    # stop it after any of them. With a single level there is nothing to prune.
    pruner: Any
    if len(levels) == 1:
        pruner = optuna.pruners.NopPruner()
    elif bohb_mode:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=len(levels),
            reduction_factor=max(2, int(tuner.bohb_reduction_factor)),
        )
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, min(40, int(tuner.max_trials // 4) if tuner.max_trials > 0 else 5)),
            n_warmup_steps=0,
            interval_steps=1,
        )
```

The median pruner prunes nothing during the first max(5, min(40, ⌊trials/4⌋))
trials, where trials is `--tune-budget`.

### From a budget level to instance/seed pairs

A low fidelity can also mean fewer runs: VAMOS can evaluate a candidate at a
low budget on part of the training instances and seeds. The share grows
linearly with the budget, from the smallest level to the largest:

<div class="vf-formula">f = (b − b_min) / (b_max − b_min)                  (0 at the smallest level, 1 at the full budget)
instances(b) = round( n · ( φ + f (1 − φ) ) )         φ = --fidelity-min-instance-frac  (1.0: always every instance)
seeds(b)     = round( s_min + f (s_max − s_min) )     --fidelity-min-seed-count, --fidelity-max-seed-count (0: every seed)</div>

```python
# Excerpt from src/vamos/engine/tuning/_model_backend_fidelity.py
    frac = budget_fraction(tuner, b)
    min_inst_frac = float(tuner.fidelity_min_instance_frac)
    inst_frac = min_inst_frac + frac * (1.0 - min_inst_frac)
    target_instances = int(max(1, min(len(all_instances), int(round(float(len(all_instances)) * inst_frac)))))
    min_seed_count, max_seed_count = resolve_seed_bounds(tuner)
    target_seed_count = int(round(float(min_seed_count) + frac * float(max_seed_count - min_seed_count)))
    target_seed_count = max(1, min(len(all_seeds), target_seed_count))
    selected_seeds = list(all_seeds[:target_seed_count]) if target_seed_count < len(all_seeds) else list(all_seeds)
```

The seeds are the first ones of the training list; the instances are a
random subset, drawn once per budget with `--fidelity-selection-seed` (by
default `--seed`). By default, φ = 1 and every
seed is used, so **every budget runs every instance/seed pair**: only the run
length changes.

The figure shows the budgets that each family of backends evaluates for a
set of levels, and the pairs that each budget runs, for six training
instances and five seeds (the default `--n-seeds`). In the Hyperband mode it
also lists BOHB's brackets: how many configurations each bracket runs at each
budget, for the number of configurations that `--tune-budget` allows.

<div class="vf-widget" data-vf-widget="fidelity-budgets"><p class="vf-fallback">Interactive figure: the budgets that the model-based backends of VAMOS evaluate and the instance/seed pairs that each budget runs. With levels 1,000, 3,000 and 9,000, six instances, five seeds, an instance fraction of 0.5 and at least one seed, a run at 1,000 evaluations uses 3 instances and 1 seed, one at 3,000 uses 4 instances and 2 seeds, and one at 9,000 all 6 instances and 5 seeds. It needs JavaScript.</p></div>

With the levels 1,000, 3,000 and 9,000, `--fidelity-min-instance-frac 0.5` and
`--fidelity-min-seed-count 1`, a candidate evaluated at 1,000 evaluations runs
on 3 instances with 1 seed, 3 pairs, and one at the full budget on all 30
pairs: the smallest budget costs 3,000 evaluations per configuration instead
of 30,000. Hyperband with η = 3 between 1,000 and 9,000 evaluates the same
three budgets; between 500 and 5,000 it evaluates 556, 1,667 and 5,000
(5,000 divided by 9, 3 and 1), not the levels. A smaller share of pairs makes a low-fidelity score noisier, as
well as shorter.

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend bohb \
  --budget 9000 \
  --fidelity-levels 1000,3000,9000 \
  --bohb-reduction-factor 3 \
  --fidelity-min-instance-frac 0.5 \
  --fidelity-min-seed-count 1 \
  --tune-budget 27
```

`--backend smac3` takes the same options except `--bohb-reduction-factor`,
and `--backend bohb_optuna` evaluates the three levels themselves.

### Failures

A run that raises, or whose score is not finite, is a failed evaluation in
every backend: it scores `--failure-score` (0 by default), so the models
never receive a NaN, and `tuning_summary.json` counts the failures. When every
evaluation fails, `vamos tune` stops with exit code 1.

## Racing or a model?

| | Racing (`racing`, `irace`) | Model-based (`optuna`, `smac3`, `bohb`, …) |
| --- | --- | --- |
| Chooses candidates | by sampling, around elites in irace | by a surrogate model and an acquisition function |
| Compares candidates | on the same blocks, with rank-based tests | by an aggregated score per trial |
| Saves runs by | eliminating candidates early | pruning or short runs (multi-fidelity) |
| Needs | the core installation, or R for irace | the `tuning` extra |

Racing uses the paired structure of the blocks and ranks, which suits
problems on different scales; model-based tuners learn which regions of the
space are good, which suits large spaces and small budgets. Whichever is
used, the tuned configuration is validated on held-out instances and seeds
(lesson [Automatic algorithm configuration](tuning.md#training-validation-and-test-instances)).

## Key points

- Model-based tuners fit a surrogate of the score to every evaluated
  configuration and choose the next one by an acquisition function: random
  forests in SMAC, TPE's two densities in Optuna.
- Successive halving discards most candidates on short runs; Hyperband hedges
  with several brackets; BOHB adds density models to Hyperband; pruners stop
  bad trials early.
- In VAMOS, the four model-based backends evaluate one fidelity unless
  `--fidelity-levels` is given; Optuna's backends then evaluate the levels in
  turn, and SMAC3 and BOHB run Hyperband between the smallest and the largest
  level.
- A budget can also run fewer instances and seeds, linearly in the budget,
  but only when `--fidelity-min-instance-frac` or the seed counts ask for it;
  by default every budget runs every pair.

**Next:** [Evolver and meta-optimization](tuning-evolver.md).

## References

[1] Hutter, F., Hoos, H. H., and Leyton-Brown, K. (2011). [Sequential model-based optimization for general algorithm configuration](https://doi.org/10.1007/978-3-642-25566-3_40). In *Learning and Intelligent Optimization (LION 5)*, Lecture Notes in Computer Science 6683, 507–523. Springer.

[2] Lindauer, M., Eggensperger, K., Feurer, M., Biedenkapp, A., Deng, D., Benjamins, C., Ruhkopf, T., Sass, R., and Hutter, F. (2022). [SMAC3: A versatile Bayesian optimization package for hyperparameter optimization](https://jmlr.org/papers/v23/21-0888.html). *Journal of Machine Learning Research*, 23(54), 1–9.

[3] Bergstra, J., Bardenet, R., Bengio, Y., and Kégl, B. (2011). [Algorithms for hyper-parameter optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization). In *Advances in Neural Information Processing Systems 24 (NIPS 2011)*, 2546–2554.

[4] Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). [Optuna: A next-generation hyperparameter optimization framework](https://doi.org/10.1145/3292500.3330701). In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.

[5] Jamieson, K., and Talwalkar, A. (2016). [Non-stochastic best arm identification and hyperparameter optimization](https://proceedings.mlr.press/v51/jamieson16.html). In *Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS 2016)*, PMLR 51, 240–248.

[6] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. (2018). [Hyperband: A novel bandit-based approach to hyperparameter optimization](https://jmlr.org/papers/v18/16-558.html). *Journal of Machine Learning Research*, 18(185), 1–52.

[7] Falkner, S., Klein, A., and Hutter, F. (2018). [BOHB: Robust and efficient hyperparameter optimization at scale](https://proceedings.mlr.press/v80/falkner18a.html). In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*, PMLR 80, 1437–1446.
