from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vamos.foundation.exceptions import _suggest_names

_BENCH_DOCS = "docs/guide/cli.md"
_TROUBLESHOOTING_DOCS = "docs/guide/troubleshooting.md"


def _format_unknown_suite(name: str, options: list[str]) -> str:
    parts = [f"Unknown benchmark suite '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_BENCH_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


@dataclass
class BenchmarkExperiment:
    problem_name: str
    problem_params: dict[str, Any] = field(default_factory=dict)
    evaluation_budget: int | None = None
    max_generations: int | None = None
    seeds: list[int] = field(default_factory=list)

    def resolved_budget(self, population_size: int | None = None) -> int:
        """
        Resolve the evaluation budget for this experiment.
        Prefers explicit evaluation_budget, otherwise derives from generations * population_size.
        """
        if self.evaluation_budget is not None:
            return int(self.evaluation_budget)
        if self.max_generations is not None:
            pop = population_size if population_size is not None else 100
            return int(pop * self.max_generations)
        raise ValueError("BenchmarkExperiment requires evaluation_budget or max_generations.")


@dataclass
class BenchmarkSuite:
    name: str
    experiments: list[BenchmarkExperiment]
    default_algorithms: list[str]
    default_metrics: list[str]
    description: str = ""
    default_seeds: list[int] = field(default_factory=lambda: list(range(5)))


_SUITES: dict[str, BenchmarkSuite] = {}


def _register_suite(suite: BenchmarkSuite) -> None:
    _SUITES[suite.name] = suite


def _ensure_default_suites() -> None:
    if _SUITES:
        return
    _init_default_suites()


def get_benchmark_suite(name: str) -> BenchmarkSuite:
    _ensure_default_suites()
    try:
        return _SUITES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(_format_unknown_suite(name, list_benchmark_suites())) from exc


def list_benchmark_suites() -> list[str]:
    _ensure_default_suites()
    return sorted(_SUITES.keys())


def _init_default_suites() -> None:
    zdt_experiments = [
        BenchmarkExperiment("zdt1", {"n_var": 30}, evaluation_budget=25000),
        BenchmarkExperiment("zdt2", {"n_var": 30}, evaluation_budget=25000),
        BenchmarkExperiment("zdt3", {"n_var": 30}, evaluation_budget=25000),
        BenchmarkExperiment("zdt4", {"n_var": 10}, evaluation_budget=30000),
        BenchmarkExperiment("zdt6", {"n_var": 10}, evaluation_budget=30000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="ZDT_small",
            experiments=zdt_experiments,
            default_algorithms=["nsgaii", "moead"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="ZDT1-4,6 with modest budgets for quick comparisons.",
            default_seeds=list(range(5)),
        )
    )

    dtlz_experiments = [
        BenchmarkExperiment("dtlz1", {"n_var": 12, "n_obj": 3}, evaluation_budget=40000),
        BenchmarkExperiment("dtlz2", {"n_var": 12, "n_obj": 3}, evaluation_budget=40000),
        BenchmarkExperiment("dtlz3", {"n_var": 12, "n_obj": 3}, evaluation_budget=40000),
        BenchmarkExperiment("dtlz4", {"n_var": 12, "n_obj": 3}, evaluation_budget=40000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="DTLZ_medium",
            experiments=dtlz_experiments,
            default_algorithms=["nsgaii", "moead", "nsgaiii"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="DTLZ1-4 at 3 objectives with medium budgets.",
            default_seeds=list(range(5)),
        )
    )

    wfg_experiments = [
        BenchmarkExperiment("wfg1", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
        BenchmarkExperiment("wfg2", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
        BenchmarkExperiment("wfg3", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
        BenchmarkExperiment("wfg4", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
        BenchmarkExperiment("wfg5", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
        BenchmarkExperiment("wfg6", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
        BenchmarkExperiment("wfg7", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
        BenchmarkExperiment("wfg8", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
        BenchmarkExperiment("wfg9", {"n_var": 24, "n_obj": 3}, evaluation_budget=60000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="WFG_heavy",
            experiments=wfg_experiments,
            default_algorithms=["nsgaii", "moead", "nsgaiii"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="Full WFG suite (requires pymoo) with larger budgets.",
            default_seeds=list(range(5)),
        )
    )

    cec_curved_experiments = [
        BenchmarkExperiment("cec2009_uf1", {"n_var": 30}, evaluation_budget=50000),
        BenchmarkExperiment("cec2009_uf2", {"n_var": 30}, evaluation_budget=50000),
        BenchmarkExperiment("cec2009_uf3", {"n_var": 30}, evaluation_budget=50000),
        BenchmarkExperiment("cec2009_uf4", {"n_var": 30}, evaluation_budget=50000),
        BenchmarkExperiment("cec2009_uf5", {"n_var": 30}, evaluation_budget=50000),
        BenchmarkExperiment("cec2009_uf6", {"n_var": 30}, evaluation_budget=50000),
        BenchmarkExperiment("cec2009_uf7", {"n_var": 30}, evaluation_budget=50000),
        BenchmarkExperiment("cec2009_cf1", {"n_var": 10}, evaluation_budget=50000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="CEC2009_UF_CF_curved",
            experiments=cec_curved_experiments,
            default_algorithms=["nsgaii", "moead", "smsemoa"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="CEC2009 UF/CF curved decision-space landscapes (bi-objective).",
            default_seeds=list(range(5)),
        )
    )

    lsmop_experiments = [
        BenchmarkExperiment("lsmop1", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
        BenchmarkExperiment("lsmop2", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
        BenchmarkExperiment("lsmop3", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
        BenchmarkExperiment("lsmop4", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
        BenchmarkExperiment("lsmop5", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
        BenchmarkExperiment("lsmop6", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
        BenchmarkExperiment("lsmop7", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
        BenchmarkExperiment("lsmop8", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
        BenchmarkExperiment("lsmop9", {"n_var": 300, "n_obj": 2}, evaluation_budget=120000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="LSMOP_large",
            experiments=lsmop_experiments,
            default_algorithms=["nsgaii", "moead"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="Large-scale many-objective benchmarks (LSMOP1-9) projected to 2 objectives.",
            default_seeds=list(range(3)),
        )
    )

    constrained_experiments = [
        BenchmarkExperiment("c1dtlz1", {"n_var": 12, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("c1dtlz3", {"n_var": 12, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("c2dtlz2", {"n_var": 12, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("dc1dtlz1", {"n_var": 12, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("dc1dtlz3", {"n_var": 12, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("dc2dtlz1", {"n_var": 12, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("dc2dtlz3", {"n_var": 12, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("mw1", {"n_var": 15, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("mw2", {"n_var": 15, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("mw3", {"n_var": 15, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("mw5", {"n_var": 15, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("mw6", {"n_var": 15, "n_obj": 2}, evaluation_budget=60000),
        BenchmarkExperiment("mw7", {"n_var": 15, "n_obj": 2}, evaluation_budget=60000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="Constrained_CDTLZ_MW_DCDTLZ",
            experiments=constrained_experiments,
            default_algorithms=["nsgaii", "moead", "smsemoa"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="Constrained MO benchmarks combining C-DTLZ, DC-DTLZ, and MW (bi-objective).",
            default_seeds=list(range(5)),
        )
    )

    perm_experiments = [
        BenchmarkExperiment("tsp6", {"n_var": 6}, evaluation_budget=5000),
        BenchmarkExperiment("kroa100", {"n_var": 100}, evaluation_budget=40000),
        BenchmarkExperiment("krob100", {"n_var": 100}, evaluation_budget=40000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="Permutation_TSP",
            experiments=perm_experiments,
            default_algorithms=["nsgaii", "spea2"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="Permutation-encoded TSP/TSPLIB problems.",
            default_seeds=list(range(3)),
        )
    )

    discrete_experiments = [
        BenchmarkExperiment("bin_knapsack", {"n_var": 50}, evaluation_budget=20000),
        BenchmarkExperiment("bin_qubo", {"n_var": 30}, evaluation_budget=15000),
        BenchmarkExperiment("int_jobs", {"n_var": 30}, evaluation_budget=20000),
        BenchmarkExperiment("int_alloc", {"n_var": 20}, evaluation_budget=15000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="Discrete_BIN_INT",
            experiments=discrete_experiments,
            default_algorithms=["nsgaii", "moead", "spea2"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="Binary/integer benchmarks exercising non-continuous operators.",
            default_seeds=list(range(3)),
        )
    )

    mixed_experiments = [
        BenchmarkExperiment("mixed_design", {"n_var": 9}, evaluation_budget=15000),
        BenchmarkExperiment("welded_beam", {}, evaluation_budget=20000),
        BenchmarkExperiment("ml_tuning", {}, evaluation_budget=10000),
    ]
    _register_suite(
        BenchmarkSuite(
            name="Mixed_real",
            experiments=mixed_experiments,
            default_algorithms=["nsgaii"],
            default_metrics=["hv", "igd_plus", "epsilon_additive"],
            description="Mixed and real-world examples (requires examples/analysis extras for some problems).",
            default_seeds=list(range(3)),
        )
    )
