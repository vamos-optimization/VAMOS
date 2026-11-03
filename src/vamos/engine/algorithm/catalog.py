from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ALGORITHM = "nsgaii"

ENABLED_ALGORITHMS = (
    "nsgaii",
    "moead",
    "smsemoa",
    "nsgaiii",
    "spea2",
    "ibea",
    "smpso",
    "agemoea",
    "rvea",
)

OPTIONAL_ALGORITHMS: tuple[str, ...] = ()

# The smallest population and number of objectives each built-in algorithm runs with.
# NSGA-II, IBEA and AGE-MOEA select parents by tournament without replacement (two
# individuals); MOEA/D needs two weight vectors and NSGA-III and RVEA two reference
# directions, which also need two objectives. SMS-EMOA's hypervolume contributions and
# AGE-MOEA's front geometry are defined for two or more objectives. NSGA-III and RVEA
# check their exact lattice size when their configuration is built.
ALGORITHM_MINIMUMS: dict[str, dict[str, int]] = {
    "nsgaii": {"pop_size": 2, "n_obj": 1},
    "moead": {"pop_size": 2, "n_obj": 1},
    "smsemoa": {"pop_size": 1, "n_obj": 2},
    "nsgaiii": {"pop_size": 2, "n_obj": 2},
    "spea2": {"pop_size": 1, "n_obj": 1},
    "ibea": {"pop_size": 2, "n_obj": 1},
    "smpso": {"pop_size": 1, "n_obj": 1},
    "agemoea": {"pop_size": 2, "n_obj": 2},
    "rvea": {"pop_size": 2, "n_obj": 2},
}

_LABELS = {
    "nsgaii": "NSGA-II",
    "moead": "MOEA/D",
    "smsemoa": "SMS-EMOA",
    "nsgaiii": "NSGA-III",
    "spea2": "SPEA2",
    "ibea": "IBEA",
    "smpso": "SMPSO",
    "agemoea": "AGE-MOEA",
    "rvea": "RVEA",
}


@dataclass(frozen=True)
class MinimumViolation:
    """A population or objective count below what a built-in algorithm runs with."""

    field: str
    minimum: int
    actual: int
    message: str


def minimum_violation(algorithm: str, *, pop_size: int | None = None, n_obj: int | None = None) -> MinimumViolation | None:
    """Return the first of ``n_obj`` and ``pop_size`` below ``algorithm``'s minimum, or ``None``.

    Algorithms outside ``ALGORITHM_MINIMUMS``, such as plugins, are not checked, and a
    ``None`` value is not checked.
    """
    key = algorithm.strip().lower()
    minimums = ALGORITHM_MINIMUMS.get(key)
    if minimums is None:
        return None
    label = _LABELS.get(key, algorithm)
    if n_obj is not None and n_obj < minimums["n_obj"]:
        return MinimumViolation(
            "n_obj",
            minimums["n_obj"],
            n_obj,
            f"{label} needs at least {minimums['n_obj']} objectives; the problem has {n_obj}.",
        )
    if pop_size is not None and pop_size < minimums["pop_size"]:
        return MinimumViolation(
            "pop_size",
            minimums["pop_size"],
            pop_size,
            f"{label} needs pop_size >= {minimums['pop_size']}; got {pop_size}.",
        )
    return None


__all__ = [
    "ALGORITHM_MINIMUMS",
    "DEFAULT_ALGORITHM",
    "ENABLED_ALGORITHMS",
    "MinimumViolation",
    "OPTIONAL_ALGORITHMS",
    "minimum_violation",
]
