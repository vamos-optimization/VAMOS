"""
UX/analysis utilities; not required for core optimization runtime.
"""

from __future__ import annotations

from vamos.foundation.quality_indicators.moocore_indicators import (
    HV_APPROX_METHODS,
    HVApproxIndicator,
    HVIndicator,
    IndicatorResult,
)
from vamos.ux.analysis.core_objective_reduction import (
    ObjectiveReducer,
    ObjectiveReductionConfig,
    reduce_objectives,
)
from vamos.ux.analysis.mcdm import (
    MCDMResult,
    knee_point_scores,
    reference_point_scores,
    tchebycheff_scores,
    weighted_sum_scores,
)
from vamos.ux.analysis.stats import (
    FriedmanResult,
    WilcoxonResult,
    compute_ranks,
    friedman_test,
    pairwise_wilcoxon,
    plot_critical_distance,
)
from vamos.ux.results import (
    explore_result_front,
    log_result_summary,
    plot_result_front,
    result_summary_text,
    result_to_dataframe,
    result_to_latex,
)
from vamos.ux.visualization import (
    plot_hv_convergence,
    plot_parallel_coordinates,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
)

__all__ = [
    # Development after 1.4.0: quality indicators, the exact and the approximate hypervolume
    "HVIndicator",
    "HVApproxIndicator",
    "HV_APPROX_METHODS",
    "IndicatorResult",
    # Objective reduction
    "ObjectiveReductionConfig",
    "ObjectiveReducer",
    "reduce_objectives",
    # MCDM
    "MCDMResult",
    "weighted_sum_scores",
    "tchebycheff_scores",
    "reference_point_scores",
    "knee_point_scores",
    # Stats
    "FriedmanResult",
    "WilcoxonResult",
    "compute_ranks",
    "friedman_test",
    "pairwise_wilcoxon",
    "plot_critical_distance",
    # Visualization helpers
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_parallel_coordinates",
    "plot_hv_convergence",
    # Result helpers
    "result_summary_text",
    "log_result_summary",
    "plot_result_front",
    "explore_result_front",
    "result_to_dataframe",
    "result_to_latex",
]
