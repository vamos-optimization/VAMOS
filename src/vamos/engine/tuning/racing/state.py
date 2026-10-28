from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfigState:
    """
    Internal structure to keep track of a single configuration during racing.

    For multi-fidelity warm-starting:
    - `checkpoint`: Stores algorithm state from last evaluation for warm-starting
    - `last_budget`: Budget used in the last evaluation (for cumulative budgets)
    """

    config_id: int
    config: dict[str, Any]
    alive: bool = True
    # Scores are stored in the same order as the evaluation schedule.
    scores: list[float] = field(default_factory=list)
    # Internal: the (instance index, seed index) block of each entry in `scores`,
    # so the final choice can compare survivors on the blocks they share.
    blocks: list[tuple[int, int]] = field(default_factory=list)
    # Warm-start support
    checkpoint: Any | None = None
    last_budget: int = 0
    checkpoint_map: dict[tuple[int, int], Any] = field(default_factory=dict)
    last_budget_map: dict[tuple[int, int], int] = field(default_factory=dict)
    # Multi-fidelity score tracking
    fidelity_scores: dict[int, list[float]] = field(default_factory=dict)


@dataclass
class EliteEntry:
    """
    Elite archive entry: configuration and its aggregated score.
    """

    config: dict[str, Any]
    score: float


__all__ = ["ConfigState", "EliteEntry"]
