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
    "ccmo",
    "ctaea",
)

OPTIONAL_ALGORITHMS: tuple[str, ...] = ()

# The smallest population and number of objectives each built-in algorithm runs with.
# NSGA-II, IBEA and AGE-MOEA select parents by tournament without replacement (two
# individuals by default; for NSGA-II and IBEA the configured selection decides, see minimum_violation); MOEA/D needs two weight vectors and NSGA-III and RVEA two reference
# directions, which also need two objectives. SMS-EMOA's hypervolume contributions and
# AGE-MOEA's front geometry are defined for two or more objectives. NSGA-III and RVEA
# check their exact lattice size when their configuration is built. CCMO pairs the parents
# of each population two by two, so each population needs two individuals. C-TAEA draws its
# parents with replacement from archives of at least two weight vectors, which need two objectives.
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
    "ccmo": {"pop_size": 2, "n_obj": 1},
    "ctaea": {"pop_size": 1, "n_obj": 2},
}

# Built-in algorithms that evaluate several populations of pop_size before their first
# generation: CCMO initializes its two populations.
INITIAL_POPULATIONS: dict[str, int] = {"ccmo": 2}

# Built-in algorithms whose pop_size is a requested number of weight vectors: the initial
# population has one individual per vector generated for the problem's objectives (C-TAEA).
WEIGHT_VECTOR_POPULATIONS: tuple[str, ...] = ("ctaea",)

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
    "ccmo": "CCMO",
    "ctaea": "C-TAEA",
}


# Algorithms whose population minimum comes from their configurable parent selection.
_SELECTION_DEPENDENT_POPULATION = ("nsgaii", "ibea")
# Selections that draw parents from any number of individuals, one included.
_SINGLETON_SELECTIONS = ("random", "boltzmann", "ranking", "sus")


def _selection_population_minimum(selection: object) -> int | None:
    """The smallest population ``selection`` works with, or ``None`` when it is not given or unknown.

    A tournament of size ``k`` draws ``k`` distinct candidates (2 by default); the other
    selections work with a single individual.
    """
    if isinstance(selection, str):
        name, params = selection, {}
    elif isinstance(selection, (tuple, list)) and selection and isinstance(selection[0], str):
        name = selection[0]
        params = selection[1] if len(selection) > 1 and isinstance(selection[1], dict) else {}
    else:
        return None
    name = name.strip().lower()
    if name in _SINGLETON_SELECTIONS:
        return 1
    if name == "tournament":
        size = params.get("size", 2)
        return size if isinstance(size, int) and not isinstance(size, bool) and size >= 1 else None
    return None


@dataclass(frozen=True)
class MinimumViolation:
    """A population or objective count below what a built-in algorithm runs with."""

    field: str
    minimum: int
    actual: int
    message: str


def initial_evaluations(algorithm: str, pop_size: int, n_obj: int | None = None) -> int:
    """Evaluations that ``algorithm`` spends on its initial population(s) of ``pop_size``.

    ``2 * pop_size`` for CCMO, which initializes two populations; for C-TAEA, whose ``pop_size``
    is the requested number of weight vectors, the number of vectors it generates for ``n_obj``
    objectives (91 for 100 and three objectives), or ``pop_size`` when ``n_obj`` is not given;
    and ``pop_size`` for the other algorithms. The budget must cover them.
    """
    key = algorithm.strip().lower()
    if key in WEIGHT_VECTOR_POPULATIONS and n_obj is not None and n_obj >= 2 and pop_size >= 1:
        from vamos.engine.algorithm.ctaea.weights import weight_vector_count

        return weight_vector_count(int(pop_size), int(n_obj))
    return INITIAL_POPULATIONS.get(key, 1) * int(pop_size)


def minimum_violation(
    algorithm: str, *, pop_size: int | None = None, n_obj: int | None = None, selection: object = None
) -> MinimumViolation | None:
    """Return the first of ``n_obj`` and ``pop_size`` below ``algorithm``'s minimum, or ``None``.

    ``selection`` is the configured parent selection (a name or a ``(name, params)`` pair).
    For NSGA-II and IBEA it sets the population minimum: a tournament needs as many individuals
    as its size, and random, Boltzmann, ranking and SUS selection one. Algorithms outside
    ``ALGORITHM_MINIMUMS``, such as plugins, are not checked, and a ``None`` value is not checked.
    """
    key = algorithm.strip().lower()
    minimums = ALGORITHM_MINIMUMS.get(key)
    if minimums is None:
        return None
    selection_minimum = _selection_population_minimum(selection) if key in _SELECTION_DEPENDENT_POPULATION else None
    if selection_minimum is not None:
        minimums = {**minimums, "pop_size": selection_minimum}
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
    "INITIAL_POPULATIONS",
    "MinimumViolation",
    "OPTIONAL_ALGORITHMS",
    "WEIGHT_VECTOR_POPULATIONS",
    "initial_evaluations",
    "minimum_violation",
]
