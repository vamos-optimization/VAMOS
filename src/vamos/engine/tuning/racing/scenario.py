from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    """
    Racing scenario settings, inspired by irace's scenario file.

    This class collects how the racing tuner should behave: total experiment
    budget, how aggressively to eliminate configurations, how to traverse
    instances and seeds, etc.

    It does NOT hold the param space or instances themselves; those live
    in TuningTask. Scenario only controls the "racing strategy".
    """

    max_experiments: int
    """
    Maximum total number of configuration evaluations (config x instance x seed).
    Once this limit is reached, the race stops.
    """

    min_survivors: int = 1
    """
    Minimum number of configurations that must survive until the end of the race.
    The racing tuner will not eliminate configs if doing so would go below this
    number.
    """

    elimination_fraction: float = 0.25
    """
    Fraction of configurations to eliminate in each elimination step.
    For example, 0.5 means "drop roughly the worst 50% after each stage",
    subject to the min_survivors constraint.
    Must be in (0.0, 1.0].
    """

    instance_order_random: bool = True
    """
    Whether to randomize the order of instances when scheduling evaluations.
    If False, instances are taken in the order given by TuningTask.instances.
    """

    seed_order_random: bool = True
    """
    Whether to randomize the order of seeds when scheduling evaluations.
    If False, seeds are taken in the order given by TuningTask.seeds.
    """

    start_instances: int = 1
    """
    Number of instances to consider at the very beginning of the race.
    For example, 1 means that the first stages will use only the first instance
    (or a random one). As the race progresses, more instances can be added.
    """

    verbose: bool = True
    """
    Whether the racing tuner should print progress messages.
    """

    max_stages: int | None = None
    """
    Optional maximum number of racing stages. Each stage corresponds to adding
    at least one (instance, seed) combination and performing an elimination
    step. If None, the race continues until instances/seeds/experiment budget
    are exhausted.
    """

    use_statistical_tests: bool = True
    """
    If True, eliminate configurations during racing with F-race tests: Friedman on
    the ranks of each block followed by Conover's post-hoc comparison, or a paired
    Wilcoxon signed-rank test with two configurations alive. If False, fall back to
    a simple rank-based elimination.
    """

    alpha: float = 0.05
    """
    Significance level for statistical tests. Typical values are 0.05, 0.10, 0.01.
    This is used when use_statistical_tests=True.
    """

    min_blocks_before_elimination: int = 3
    """
    Minimum number of blocks (instance x seed combinations) that must be evaluated
    before elimination is allowed. In statistical mode this gates hypothesis tests;
    in rank-only mode it acts as a grace period to reduce noisy early pruning.
    """

    first_test: int = 5
    """
    Number of blocks before the first statistical test, as irace's ``firstTest``.
    With ``use_statistical_tests=True``, a race tests only when every alive
    configuration has at least ``max(first_test, min_blocks_before_elimination)``
    blocks.
    """

    use_adaptive_budget: bool = False
    """
    If True, the racing tuner will adjust the evaluation budget per stage instead
    of always using task.budget_per_run. Earlier stages use a smaller budget,
    and surviving configurations receive increasing budgets in later stages.
    """

    initial_budget_per_run: int | None = None
    """
    Initial evaluation budget per block (config x instance x seed) used in the
    first stage of the race when use_adaptive_budget=True. If None, fall back
    to task.budget_per_run.
    """

    max_budget_per_run: int | None = None
    """
    Maximum evaluation budget per block allowed when use_adaptive_budget=True.
    If None, no explicit upper bound is applied (besides task.budget_per_run).
    """

    budget_growth_factor: float = 2.0
    """
    Multiplicative factor used to increase the budget per stage when
    use_adaptive_budget=True. For example, with initial_budget_per_run=5000
    and budget_growth_factor=2.0, stages will use budgets approximately:
        5000, 10000, 20000, ...
    until capped by max_budget_per_run or task.budget_per_run.
    """

    use_elitist_restarts: bool = False
    """
    If True, the racing tuner will maintain an elite archive and, after each
    elimination step, may spawn new configurations based on elite configs
    (local search) to refill the population up to a target size.
    """

    target_population_size: int | None = None
    """
    Target number of alive configurations to maintain during the race. If None,
    the racing tuner will use max_initial_configs as the target population size.

    After elimination, if the number of alive configs is below this target,
    new configs will be generated (some as local neighbors of elites, some
    using the sampler).
    """

    elite_fraction: float = 0.3
    """
    Fraction of top configurations (among current alive ones) to consider as
    "elites" when updating the elite archive.
    """

    max_elite_archive_size: int = 20
    """
    Maximum number of elite configurations kept in the archive. When the
    archive grows beyond this size, only the best ones are kept.
    """

    neighbor_fraction: float = 0.5
    """
    Fraction of newly spawned configurations that should be generated as
    local neighbors of elite configs (local search). The remaining fraction
    is generated by sampling from the sampler (exploration).
    """

    n_jobs: int = 1
    """
    Number of parallel jobs to use for configuration evaluation.
    - 1: Sequential execution (default).
    - -1: Use all available CPU cores.
    - >1: Use exact number of cores.
    """

    convergence_window: int = 0
    """
    Number of consecutive stages without improvement before stopping early.
    Set to 0 to disable convergence-based early stopping.
    """

    convergence_threshold: float = 0.01
    """
    Relative improvement threshold for convergence detection.
    If the best score improves by less than this fraction over convergence_window
    stages, the race is considered converged.
    """

    # =========================================================================
    # Multi-Fidelity (Hyperband-style)
    # =========================================================================

    use_multi_fidelity: bool = False
    """
    If True, enables Hyperband-style multi-fidelity evaluation.
    Configurations are first evaluated with low budget, and only the best
    are promoted to higher fidelity levels.
    """

    fidelity_levels: tuple[int, ...] = (1000, 3000, 10000)
    """
    Sequence of evaluation budgets for each fidelity level.
    Must be strictly increasing. Example: (1000, 3000, 10000) means:
    - Level 0: evaluate with budget=1000
    - Level 1: survivors evaluated with budget=3000
    - Level 2: final survivors evaluated with budget=10000
    """

    fidelity_promotion_ratio: float = 0.3
    """
    Fraction of configurations promoted to the next fidelity level.
    For example, 0.3 means the top 30% of configs at each level advance.
    """

    fidelity_min_configs: int = 3
    """
    Minimum number of configurations to keep at each fidelity level.
    Ensures we don't eliminate too aggressively between fidelity levels.
    """

    fidelity_warm_start: bool = True
    """
    If True, pass checkpoints between fidelity levels so the algorithm can
    continue from where it left off instead of starting from scratch.

    The eval_fn receives `ctx.checkpoint` with the previous state and should
    return a tuple `(score, new_checkpoint)` instead of just `score`.
    """

    def __post_init__(self) -> None:
        if self.max_experiments <= 0:
            raise ValueError("max_experiments must be > 0")
        if not (0.0 < self.elimination_fraction <= 1.0):
            raise ValueError("elimination_fraction must be in (0.0, 1.0]")
        if self.min_survivors < 1:
            raise ValueError("min_survivors must be >= 1")
        if self.start_instances < 1:
            raise ValueError("start_instances must be >= 1")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if self.min_blocks_before_elimination < 1:
            raise ValueError("min_blocks_before_elimination must be >= 1")
        if self.first_test < 1:
            raise ValueError("first_test must be >= 1")
        if self.use_adaptive_budget and self.initial_budget_per_run is not None and self.initial_budget_per_run <= 0:
            raise ValueError("initial_budget_per_run must be > 0 when use_adaptive_budget is True")
        if self.max_budget_per_run is not None and self.max_budget_per_run <= 0:
            raise ValueError("max_budget_per_run must be > 0 when provided")
        if self.budget_growth_factor < 1.0:
            raise ValueError("budget_growth_factor must be >= 1.0")
        if not (0.0 < self.elite_fraction <= 1.0):
            raise ValueError("elite_fraction must be in (0, 1]")
        if not (0.0 <= self.neighbor_fraction <= 1.0):
            raise ValueError("neighbor_fraction must be in [0, 1]")
        if self.max_elite_archive_size < 1:
            raise ValueError("max_elite_archive_size must be >= 1")
        if self.target_population_size is not None and self.target_population_size < 1:
            raise ValueError("target_population_size must be >= 1 when provided")
        if self.n_jobs < 1 and self.n_jobs != -1:
            raise ValueError("n_jobs must be >= 1 or -1")
        if self.convergence_window < 0:
            raise ValueError("convergence_window must be >= 0")
        if not (0.0 < self.convergence_threshold <= 1.0):
            raise ValueError("convergence_threshold must be in (0, 1]")
        # Multi-fidelity validation
        if self.use_multi_fidelity:
            if len(self.fidelity_levels) < 2:
                raise ValueError("fidelity_levels must have at least 2 levels")
            if not all(self.fidelity_levels[i] < self.fidelity_levels[i + 1] for i in range(len(self.fidelity_levels) - 1)):
                raise ValueError("fidelity_levels must be strictly increasing")
            if not (0.0 < self.fidelity_promotion_ratio <= 1.0):
                raise ValueError("fidelity_promotion_ratio must be in (0, 1]")
            if self.fidelity_min_configs < 1:
                raise ValueError("fidelity_min_configs must be >= 1")


__all__ = ["Scenario"]
