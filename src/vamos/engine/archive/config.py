from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

CanonicalPrunePolicy = Literal["crowding", "hv", "mc_hv", "knn", "maxmin", "ref_dirs"]
PrunePolicy = CanonicalPrunePolicy
DeduplicateIn = Literal["objective", "decision", "both"]

_VALID_PRUNE_POLICIES = {"crowding", "hv", "mc_hv", "knn", "maxmin", "ref_dirs"}


def normalize_prune_policy(policy: str) -> PrunePolicy:
    if policy not in _VALID_PRUNE_POLICIES:
        valid = ", ".join(sorted(_VALID_PRUNE_POLICIES))
        raise ValueError(f"Unsupported pruning '{policy}'. Expected one of: {valid}.")
    return cast(PrunePolicy, policy)


@dataclass(frozen=True)
class ExternalArchiveConfig:
    """Configuration for the single external result archive path.

    ``objective_tolerance`` is the Chebyshev distance under which two objective
    vectors count as duplicates. It is absolute, so the default is 0: only identical
    vectors are merged, whatever the scale of the objectives.
    """

    capacity: int | None = None
    pruning: PrunePolicy = "crowding"
    hv_ref_point: list[float] | None = None
    rng_seed: int = 0
    objective_tolerance: float = 0.0
    truncate_size: int | None = None
    deduplicate_in: DeduplicateIn = "objective"
    decision_tolerance: float = 1e-32

    def __post_init__(self) -> None:
        object.__setattr__(self, "pruning", normalize_prune_policy(self.pruning))
        if self.capacity is not None and self.capacity <= 0:
            raise ValueError("capacity must be > 0 when provided.")
        if self.truncate_size is not None:
            if self.capacity is None:
                raise ValueError("truncate_size requires a finite capacity.")
            if self.truncate_size <= 0:
                raise ValueError("truncate_size must be > 0.")
            if self.truncate_size > self.capacity:
                raise ValueError("truncate_size must be <= capacity.")
        if self.objective_tolerance < 0.0:
            raise ValueError("objective_tolerance must be >= 0.")
        if self.decision_tolerance < 0.0:
            raise ValueError("decision_tolerance must be >= 0.")


__all__ = [
    "CanonicalPrunePolicy",
    "DeduplicateIn",
    "ExternalArchiveConfig",
    "PrunePolicy",
    "normalize_prune_policy",
]
