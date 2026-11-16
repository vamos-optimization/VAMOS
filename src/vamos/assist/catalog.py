from __future__ import annotations

from vamos.engine.algorithm.registry import get_algorithms_registry
from vamos.engine.variation.helpers import operator_names
from vamos.experiment.cli.quickstart import available_templates
from vamos.foundation.kernel.registry import KERNELS


def build_catalog(problem_type: str = "real") -> dict[str, list[str]]:
    """Build a deterministic catalog of available optimization building blocks."""
    return {
        "algorithms": sorted(get_algorithms_registry().keys()),
        "kernels": sorted(KERNELS.keys()),
        "crossover_methods": sorted(operator_names("crossover", problem_type)),
        "mutation_methods": sorted(operator_names("mutation", problem_type)),
        "templates": sorted(available_templates()),
    }


__all__ = ["build_catalog"]
