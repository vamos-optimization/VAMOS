# operators/policies/moead.py
"""Operator building for MOEA/D.

**Development after 1.3.0:** the ``de`` crossover takes a ``scheme``, one of jMetal's
``DifferentialEvolutionCrossover`` variants (jMetal 7.8; Evolver's ``differentialEvolutionCrossover``):
``"rand_1_bin"`` (the default, VAMOS's only scheme before), ``"rand_1_exp"``, ``"rand_2_bin"`` and
``"rand_2_exp"``. The child starts as the current solution of the subproblem, the last parent; the
mutant is ``x + F (x_r1 - x_r2)`` with rand/1 and ``x + F (x_r1 - x_r2) + F (x_r3 - x_r4)`` with
rand/2, whose base is that current solution, as in jMetal's ``MOEADDEBuilder`` (which appends it to
the mating pool) and in MOEA/D-DE. The binomial crossover takes each variable from the mutant when
its draw is lower than CR, and one variable at random always; the exponential crossover takes a
block of consecutive variables (:func:`~vamos.engine.operators.impl.real.crossover_differential.exponential_crossover`).

**Development after 1.3.0:** the ``base`` of the ``de`` crossover chooses the mutant's base vector:
``"current"`` (the default), the current solution as above, or ``"random"``, one more random parent,
distinct from those of the difference vectors, as in Evolver's MOEA/D, whose
``populationAndNeighborhoodMatingPoolSelection`` passes ``selectCurrentSolution = false``: the mating
pool holds three (rand/1) or five (rand/2) random parents, jMetal's ``DifferentialEvolutionCrossover``
takes the last of them as the base, and the child still starts as the current solution. The parents
are then ``(x_r1, x_r2, [x_r3, x_r4,] x_base, x)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.impl.real import ClampRepair, PolynomialMutation, SBXCrossover, VariationWorkspace
from vamos.engine.operators.impl.real.crossover_differential import exponential_crossover
from vamos.engine.operators.policies._discrete_variation_builder import (
    VariationFn,
    build_discrete_variation_operators,
)
from vamos.engine.operators.policies._real_operator_builder import instantiate_operator
from vamos.engine.operators.policies.discrete_operator_maps import (
    BINARY_CROSSOVER_COMMON,
    BINARY_MUTATION_COMMON,
    INT_CROSSOVER_COMMON,
    INT_MUTATION_COMMON,
    PERM_CROSSOVER_COMMON,
    PERM_MUTATION_COMMON,
)
from vamos.engine.operators.policies.real_repair import apply_policy_repair, resolve_policy_repair
from vamos.engine.variation.helpers import resolve_prob_expression, validate_pairwise_real_operators
from vamos.engine.variation.protocol import RepairConfigValue, RepairOperator
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.exceptions import ConfigurationError

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels

VariationCrossoverFn = VariationFn
VariationMutationFn = VariationFn

BINARY_CROSSOVER: dict[str, Any] = {**BINARY_CROSSOVER_COMMON}
BINARY_MUTATION = {**BINARY_MUTATION_COMMON}
INT_CROSSOVER = {**INT_CROSSOVER_COMMON}
INT_MUTATION = {**INT_MUTATION_COMMON}
PERM_CROSSOVER = {**PERM_CROSSOVER_COMMON}
PERM_MUTATION = {**PERM_MUTATION_COMMON}

#: The schemes of the ``de`` crossover; the first is the default.
DE_SCHEMES: tuple[str, ...] = ("rand_1_bin", "rand_1_exp", "rand_2_bin", "rand_2_exp")


def normalize_de_scheme(value: object) -> str:
    """``value`` in lower case, or ``ConfigurationError`` unless it is one of :data:`DE_SCHEMES`."""
    name = value.strip().lower() if isinstance(value, str) else None
    if name not in DE_SCHEMES:
        raise ConfigurationError(f"DE crossover scheme must be one of {', '.join(map(repr, DE_SCHEMES))}; got {value!r}.")
    return name


#: The base vectors of the ``de`` crossover's mutant; the first is the default.
DE_BASES: tuple[str, ...] = ("current", "random")


def normalize_de_base(value: object) -> str:
    """``value`` in lower case, or ``ConfigurationError`` unless it is one of :data:`DE_BASES`."""
    name = value.strip().lower() if isinstance(value, str) else None
    if name not in DE_BASES:
        raise ConfigurationError(f"DE crossover base must be one of {', '.join(map(repr, DE_BASES))}; got {value!r}.")
    return name


def de_random_parents(scheme: str, base: str = "current") -> int:
    """Random parents of a scheme besides the current solution: 2 for rand/1, 4 for rand/2, one more with a random base."""
    return (4 if scheme.startswith("rand_2") else 2) + (1 if base == "random" else 0)


def check_de_neighbourhood(scheme: str, base: str, neighbor_size: int) -> int:
    """The random parents of the scheme and base; ``ConfigurationError`` when the neighbourhood holds fewer."""
    parents = de_random_parents(scheme, base)
    if neighbor_size < parents:
        with_base = " with base='random'" if base == "random" else ""
        raise ConfigurationError(
            f"The DE crossover scheme {scheme}{with_base} needs neighbor_size >= {parents} "
            f"({parents} distinct random parents); got {neighbor_size}."
        )
    return parents


def build_variation_operators(
    cfg: dict[str, Any],
    encoding: EncodingLike,
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    mixed_spec: dict[str, NDArray[np.generic]] | None = None,
    *,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build crossover and mutation operators for MOEA/D."""
    return build_discrete_variation_operators(
        config=cfg,
        algorithm_label="MOEA/D",
        encoding=encoding,
        n_var=n_var,
        xl=xl,
        xu=xu,
        rng=rng,
        real_builder=_build_continuous_operators,
        mixed_spec=mixed_spec,
        binary_crossover=BINARY_CROSSOVER,
        binary_mutation=BINARY_MUTATION,
        integer_crossover=INT_CROSSOVER,
        integer_mutation=INT_MUTATION,
        permutation_crossover=PERM_CROSSOVER,
        permutation_mutation=PERM_MUTATION,
        variation_kernels=variation_kernels,
    )


def _build_continuous_operators(
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    repair_cfg: RepairConfigValue,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build variation operators for continuous/real encoding."""
    method = (cross_method or "sbx").lower()
    de_aliases = {"de", "differential", "differential_evolution"}
    # MOEA/D feeds DE three parents (the subproblem and two neighbours); other crossovers get pairs.
    validate_pairwise_real_operators(
        "de" if method in de_aliases else method,
        mut_method or "polynomial",
        algorithm_label="MOEA/D",
        three_parent_allowed={"de"},
    )
    workspace = VariationWorkspace()
    repair_operator = resolve_policy_repair("real", repair_cfg)
    assert repair_operator is not None

    crossover: DifferentialCrossoverStep | RepairedOperatorStep
    for key in ("scheme", "base"):
        if method not in de_aliases and key in cross_params:
            raise ConfigurationError(f"The {key} applies to the DE crossover ('de'), not to crossover {cross_method!r}.")
    if method in de_aliases:
        crossover = DifferentialCrossoverStep(
            f=float(cross_params.get("f", cross_params.get("F", 0.5))),
            cr=float(cross_params.get("cr", cross_params.get("CR", 1.0))),
            scheme=normalize_de_scheme(cross_params.get("scheme", "rand_1_bin")),
            base=normalize_de_base(cross_params.get("base", "current")),
            repair=repair_operator,
            xl=xl,
            xu=xu,
            rng=rng,
            variation_kernels=variation_kernels,
        )
    else:
        cross_kwargs = dict(cross_params)
        prob = cross_kwargs.pop("prob", None)
        cross_kwargs.setdefault("prob_crossover", 0.9 if prob is None else float(prob))
        cross_kwargs.setdefault("allow_inplace", True)
        cross_kwargs.setdefault("lower", xl)
        cross_kwargs.setdefault("upper", xu)
        cross_kwargs.setdefault("workspace", workspace)
        crossover_operator = instantiate_operator(method, cross_kwargs, label="crossover", variation_kernels=variation_kernels)
        crossover = RepairedOperatorStep(crossover_operator, repair_operator, xl, xu, rng)

    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    mut_name = (mut_method or "polynomial").lower()

    mut_kwargs = dict(mut_params)
    mut_kwargs.pop("prob", None)
    mut_kwargs.setdefault("prob_mutation", mut_prob)
    mut_kwargs.setdefault("lower", xl)
    mut_kwargs.setdefault("upper", xu)
    mut_kwargs.setdefault("workspace", workspace)
    mutation_operator = instantiate_operator(mut_name, mut_kwargs, label="mutation", variation_kernels=variation_kernels)
    return crossover, RepairedOperatorStep(mutation_operator, repair_operator, xl, xu, rng)


class DifferentialCrossoverStep:
    """DE children of ``(p1, p2, base)`` parents, ``(p1, p2, p3, p4, base)`` with rand/2, then the policy repair.

    rand/1/bin (the default) is ``base + f * (p1 - p2)`` where a draw is lower than ``cr``, and at one
    variable at random; see the module for the other schemes.
    """

    def __init__(
        self,
        *,
        f: float,
        cr: float,
        repair: RepairOperator,
        xl: NDArray[np.generic],
        xu: NDArray[np.generic],
        rng: np.random.Generator,
        variation_kernels: VariationKernels | None,
        scheme: str = "rand_1_bin",
        base: str = "current",
    ) -> None:
        self.f = f
        self.cr = cr
        self.scheme = scheme
        self.base = base
        self.repair = repair
        self.xl = xl
        self.xu = xu
        self.rng = rng
        self.variation_kernels = variation_kernels

    def __call__(self, parents: NDArray[np.floating[Any]], _rng: np.random.Generator | None = None) -> NDArray[np.generic]:
        rng = self.rng if _rng is None else _rng
        parents_arr = np.asarray(parents)
        width = 1 + de_random_parents(self.scheme, self.base)
        if parents_arr.ndim != 3 or parents_arr.shape[1] != width:
            raise ValueError(f"DE crossover {self.scheme} expects parents shaped (n_pairs, {width}, n_var).")
        n_pairs, _, n_vars = parents_arr.shape
        f, cr = self.f, self.cr
        if self.scheme != "rand_1_bin" or self.base != "current":
            child = _scheme_child(self.scheme, parents_arr, f, cr, rng, random_base=self.base == "random")
        elif self.variation_kernels is not None and parents_arr.dtype == np.float64:
            # Same draws (uniform grid, then one forced index per child) and arithmetic.
            rand = rng.random((n_pairs, n_vars))
            j_rand = rng.integers(0, n_vars, size=n_pairs)
            child = self.variation_kernels.differential_crossover(parents_arr, f, cr, rand, j_rand)
        else:
            base = parents_arr[:, 2, :]
            p1 = parents_arr[:, 0, :]
            p2 = parents_arr[:, 1, :]
            mutant = base + f * (p1 - p2)
            rand = rng.random((n_pairs, n_vars))
            mask = rand < cr
            j_rand = rng.integers(0, n_vars, size=n_pairs)
            mask[np.arange(n_pairs), j_rand] = True
            child = np.where(mask, mutant, base)
        offspring = child[:, None, :]
        return apply_policy_repair(self.repair, offspring, self.xl, self.xu, rng)


def _scheme_child(
    scheme: str, parents: NDArray[Any], f: float, cr: float, rng: np.random.Generator, *, random_base: bool = False
) -> NDArray[np.float64]:
    """The children of the schemes and bases other than rand/1/bin from the current solution, with jMetal's operations in jMetal's order.

    The last parent is the current solution, which the child starts from; the mutant's base is that
    solution, or with ``random_base`` the parent before it.
    """
    current = np.asarray(parents[:, -1, :], dtype=float)
    base = np.asarray(parents[:, -2, :], dtype=float) if random_base else current
    mutant = base + f * (parents[:, 0, :] - parents[:, 1, :])
    if scheme.startswith("rand_2"):
        mutant = mutant + f * (parents[:, 2, :] - parents[:, 3, :])
    n_pairs, n_vars = current.shape
    if scheme.endswith("_bin"):
        # The draws of rand/1/bin: the uniform grid, then the variable that always comes from the mutant.
        mask = rng.random((n_pairs, n_vars)) < cr
        mask[np.arange(n_pairs), rng.integers(0, n_vars, size=n_pairs)] = True
        return np.asarray(np.where(mask, mutant, current), dtype=float)
    # The draws of CMOEA-MS's rand/1/exp: the first variable of each block, then the uniform grid.
    start = rng.integers(0, n_vars, size=n_pairs)
    return exponential_crossover(current, mutant, start, rng.random((n_pairs, n_vars)), cr)


class RepairedOperatorStep:
    """A registry crossover or mutation operator followed by the policy repair."""

    def __init__(
        self,
        operator: Callable[..., object],
        repair: RepairOperator,
        xl: NDArray[np.generic],
        xu: NDArray[np.generic],
        rng: np.random.Generator,
    ) -> None:
        self.operator = operator
        self.repair = repair
        self.xl = xl
        self.xu = xu
        self.rng = rng

    def __call__(self, values: NDArray[np.generic], _rng: np.random.Generator | None = None) -> NDArray[np.generic]:
        rng = self.rng if _rng is None else _rng
        result = np.asarray(self.operator(values, rng))
        return apply_policy_repair(self.repair, result, self.xl, self.xu, rng)


@dataclass(frozen=True)
class OneOffspringRealVariation:
    """Resolved parameters of a real DE or SBX, polynomial mutation and clip pipeline.

    The values are read from the operators that :func:`build_variation_operators`
    returned, so compiled code that consumes the same draws reproduces them exactly.
    ``crossover`` is ``"de"`` or ``"sbx"``; the parameters of the other crossover are unused.
    """

    crossover: str
    de_f: float
    de_cr: float
    sbx_prob: float
    sbx_eta: float
    sbx_prob_var: float
    sbx_lower: NDArray[np.float64]
    sbx_upper: NDArray[np.float64]
    pm_prob: float
    pm_eta: float
    pm_lower: NDArray[np.float64]
    pm_upper: NDArray[np.float64]
    clip_lower: NDArray[np.float64]
    clip_upper: NDArray[np.float64]


def one_offspring_real_variation(crossover: object, mutation: object, n_var: int) -> OneOffspringRealVariation | None:
    """Describe operators built here when a compiled one-offspring step can reproduce them.

    That needs DE or SBX crossover and polynomial mutation that both use compiled
    variation kernels, the workspace-backed mutation draws, the clip repair, and
    float bounds shaped ``(n_var,)`` with ``lower <= upper``. Any other pipeline
    returns ``None``; callers then keep the operators themselves.
    """
    if type(mutation) is not RepairedOperatorStep or type(mutation.operator) is not PolynomialMutation:
        return None
    pm = mutation.operator
    if pm.variation_kernels is None or pm.workspace is None:
        return None
    sbx: SBXCrossover | None = None
    if type(crossover) is DifferentialCrossoverStep:
        if crossover.variation_kernels is None or crossover.scheme != "rand_1_bin" or crossover.base != "current":
            return None
    elif type(crossover) is RepairedOperatorStep and type(crossover.operator) is SBXCrossover:
        sbx = crossover.operator
        if sbx.variation_kernels is None:
            return None
    else:
        return None
    if not (_exact_type(crossover.repair, ClampRepair) and _exact_type(mutation.repair, ClampRepair)):
        return None
    if crossover.xl is not mutation.xl or crossover.xu is not mutation.xu:
        return None
    clip_lower = np.asarray(crossover.xl, dtype=float)
    clip_upper = np.asarray(crossover.xu, dtype=float)
    bounds = [(clip_lower, clip_upper), (pm.lower, pm.upper)]
    if sbx is not None:
        bounds.append((sbx.lower, sbx.upper))
    for lower, upper in bounds:
        if lower.shape != (n_var,) or upper.shape != (n_var,) or lower.dtype != np.float64 or upper.dtype != np.float64:
            return None
        if bool(np.any(lower > upper)):
            return None
    de = crossover if type(crossover) is DifferentialCrossoverStep else None
    return OneOffspringRealVariation(
        crossover="sbx" if sbx is not None else "de",
        de_f=de.f if de is not None else 0.0,
        de_cr=de.cr if de is not None else 0.0,
        sbx_prob=sbx.prob if sbx is not None else 0.0,
        sbx_eta=sbx.eta if sbx is not None else 0.0,
        sbx_prob_var=sbx.prob_var if sbx is not None else 0.0,
        sbx_lower=sbx.lower if sbx is not None else clip_lower,
        sbx_upper=sbx.upper if sbx is not None else clip_upper,
        pm_prob=pm.prob,
        pm_eta=pm.eta,
        pm_lower=pm.lower,
        pm_upper=pm.upper,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )


def _exact_type(value: object, cls: type[object]) -> bool:
    # Subclasses may change the arithmetic, so only the exact operator class is described.
    return type(value) is cls


__all__ = [
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "DE_BASES",
    "DE_SCHEMES",
    "DifferentialCrossoverStep",
    "OneOffspringRealVariation",
    "RepairedOperatorStep",
    "build_variation_operators",
    "check_de_neighbourhood",
    "de_random_parents",
    "normalize_de_base",
    "normalize_de_scheme",
    "one_offspring_real_variation",
]
