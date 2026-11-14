"""C-TAEA: the two-archive evolutionary algorithm for constrained multi-objective optimization (Li et al., 2019)."""

from .archives import (
    associate,
    convergence_archive_update,
    diversity_archive_update,
    front_ranks,
    tchebycheff_in_own_subregion,
    worst_of_the_most_crowded_subregion,
)
from .ctaea import CTAEA
from .mating import MatingProportions, ctaea_tournament, mating_proportions, tournament_winners
from .state import CTAEAState, build_ctaea_result
from .weights import ctaea_weight_vectors, weight_vector_count

__all__ = [
    "CTAEA",
    "CTAEAState",
    "MatingProportions",
    "associate",
    "build_ctaea_result",
    "convergence_archive_update",
    "ctaea_tournament",
    "ctaea_weight_vectors",
    "diversity_archive_update",
    "front_ranks",
    "mating_proportions",
    "tchebycheff_in_own_subregion",
    "tournament_winners",
    "weight_vector_count",
    "worst_of_the_most_crowded_subregion",
]
