"""Names of the tuning parameters: the spaces' canonical names, VAMOS's earlier names and Evolver's.

The built-in spaces name their parameters after Evolver's, in snake_case (``sbx_distribution_index``
for Evolver's ``sbxDistributionIndex``), and some values after Evolver's (``undc`` for ``undx``). A value
is never renamed to another operator's name: ``uniform_reset`` keeps its name, because ``random_reset`` is
the integer reset mutation. The names the spaces
used up to VAMOS 1.2.1 stay aliases: :func:`bridge_assignment` turns an assignment written with either
into the names that the configuration bridges read, so stored assignments keep building the same
configurations. :func:`to_evolver` and :func:`from_evolver` exchange assignments with Evolver's
camelCase parameters and values. The fields of the configuration classes do not change.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .param_space import Categorical, Condition, ConditionalBlock, ParamType

# (name the bridges read and the spaces used up to VAMOS 1.2.1, canonical name, Evolver's name or None).
PARAMETER_NAMES: tuple[tuple[str, str, str | None], ...] = (
    ("pop_size", "pop_size", "populationSize"),
    ("crossover_prob", "crossover_probability", "crossoverProbability"),
    ("crossover_eta", "sbx_distribution_index", "sbxDistributionIndex"),
    ("mutation_eta", "polynomial_mutation_distribution_index", "polynomialMutationDistributionIndex"),
    ("mutation_prob_factor", "mutation_probability_factor", "mutationProbabilityFactor"),
    ("neighbor_size", "neighborhood_size", "neighborhoodSize"),
    ("delta", "neighborhood_selection_probability", "neighborhoodSelectionProbability"),
    ("replace_limit", "maximum_number_of_replaced_solutions", "maximumNumberOfReplacedSolutions"),
    ("aggregation", "aggregation_function", "aggregationFunction"),
    ("pbi_theta", "pbi_theta", "pbiTheta"),
    ("de_cr", "de_crossover_rate", "CR"),
    ("de_f", "de_scale_factor", "F"),
    ("initializer", "initial_solutions", "createInitialSolutions"),
    ("selection_pressure", "tournament_size", "selectionTournamentSize"),
    ("selection_size", "tournament_size", "selectionTournamentSize"),
    ("crossover", "crossover", "crossover"),
    ("mutation", "mutation", "mutation"),
    ("selection", "selection", "selection"),
)

# The values of categorical parameters, by the parameter's earlier name, in the same three forms.
VALUE_NAMES: dict[str, tuple[tuple[str, str, str | None], ...]] = {
    "crossover": (
        ("sbx", "sbx", "SBX"),
        ("blx_alpha", "blx_alpha", "blxAlpha"),
        ("blx_alpha_beta", "blx_alpha_beta", "blxAlphaBeta"),
        ("arithmetic", "arithmetic", "arithmetic"),
        ("whole_arithmetic", "whole_arithmetic", "wholeArithmetic"),
        ("laplace", "laplace", "laplace"),
        ("fuzzy", "fuzzy", "fuzzyRecombination"),
        ("pcx", "pcx", "PCX"),
        ("undx", "undc", "UNDC"),
    ),
    "mutation": (
        ("polynomial", "polynomial", "polynomial"),
        # The integer spaces' polynomial mutation is "pm" in the integer operator registry, so "pm" is not
        # renamed; the real-coded spaces name it "polynomial" (bridge_space_parts_discrete).
        ("pm", "pm", "polynomial"),
        ("linked_polynomial", "linked_polynomial", "linkedPolynomial"),
        ("non_uniform", "non_uniform", "nonUniform"),
        ("uniform", "uniform", "uniform"),
        ("levy_flight", "levy_flight", "levyFlight"),
        ("power_law", "power_law", "powerLaw"),
    ),
    "aggregation": (
        ("tchebycheff", "tchebycheff", "tschebyscheff"),
        ("weighted_sum", "weighted_sum", "weightedSum"),
        ("pbi", "pbi", "penaltyBoundaryIntersection"),
        # Development after 1.3.0: Evolver's modifiedTschebyscheff is jMetal's ModifiedTschebyscheff,
        # max |f - z*| / w; VAMOS's modified_tchebycheff (the augmented formula, rho = 1e-3) has no Evolver name.
        ("modified_tchebycheff", "modified_tchebycheff", None),
        ("inverse_weight_tchebycheff", "inverse_weight_tchebycheff", "modifiedTschebyscheff"),
        ("augmented_tchebycheff", "augmented_tchebycheff", "augmentedTschebyscheff"),
    ),
    "initializer": (
        ("random", "random", "default"),
        ("lhs", "lhs", "latinHypercubeSampling"),
        ("scatter", "scatter", "scatterSearch"),
        ("sobol", "sobol", "sobol"),
        ("obl", "obl", "oppositionBased"),
    ),
    "selection": (
        ("tournament", "tournament", "tournament"),
        ("random", "random", "random"),
        ("boltzmann", "boltzmann", "boltzmann"),
        ("ranking", "ranking", "ranking"),
        ("sus", "sus", "stochasticUniversalSampling"),
    ),
}


# Values that Evolver does not have and that :func:`to_evolver` refuses, with the reason, by the
# parameter's earlier name: Evolver has a value of that name for another function.
NO_EVOLVER_EQUIVALENT: dict[str, dict[str, str]] = {
    "aggregation": {
        "modified_tchebycheff": (
            "VAMOS's modified_tchebycheff is the augmented Tchebycheff formula with rho = 1e-3, which Evolver does not offer; "
            "Evolver's modifiedTschebyscheff is VAMOS's inverse_weight_tchebycheff"
        ),
    },
}


def canonical_parameter_name(name: str) -> str:
    """The space's name of the parameter that the bridges call ``name``; other names are returned as they are."""
    return next((canonical for vamos, canonical, _ in PARAMETER_NAMES if vamos == name), name)


def bridge_parameter_names(name: str) -> tuple[str, ...]:
    """The names the bridges read for the parameter ``name`` (canonical or earlier), ``(name,)`` for another."""
    names = tuple(vamos for vamos, canonical, _ in PARAMETER_NAMES if name in {vamos, canonical})
    return names or (name,)


def _bridge_parameter(name: str) -> str:
    return bridge_parameter_names(name)[0]


def canonical_value(parameter: str, value: object) -> object:
    """The space's form of ``value`` of the parameter ``parameter`` (canonical or earlier name)."""
    for vamos, canonical, _ in VALUE_NAMES.get(_bridge_parameter(parameter), ()):
        if value == vamos:
            return canonical
    return value


def _bridge_value(parameter: str, value: object) -> object:
    for vamos, canonical, _ in VALUE_NAMES.get(_bridge_parameter(parameter), ()):
        if value == canonical:
            return vamos
    return value


def bridge_assignment(assignment: Mapping[str, Any]) -> dict[str, Any]:
    """``assignment`` with the parameter names and values that the configuration bridges read.

    Canonical names and the earlier ones are both accepted, and when an assignment names a parameter
    both ways the later key wins, as in a dictionary update. A canonical name that several earlier
    parameters share (``tournament_size``) gives each of them, for whichever the algorithm's bridge
    reads. Other keys are kept as they are.
    """
    result: dict[str, Any] = {}
    for name, value in assignment.items():
        for target in bridge_parameter_names(name):
            result[target] = _bridge_value(target, value)
    return result


def canonical_assignment(assignment: Mapping[str, Any]) -> dict[str, Any]:
    """``assignment`` with the spaces' canonical names and values."""
    return {canonical_parameter_name(name): canonical_value(name, value) for name, value in assignment.items()}


def to_evolver(assignment: Mapping[str, Any]) -> dict[str, Any]:
    """``assignment`` (canonical or earlier names) with Evolver's camelCase parameters and values.

    Parameters and values that Evolver does not have keep their canonical names, except those of
    :data:`NO_EVOLVER_EQUIVALENT`, which raise ``ValueError``.
    """
    _check_evolver_equivalents(assignment)
    result: dict[str, Any] = {}
    for name, value in canonical_assignment(assignment).items():
        bridge = _bridge_parameter(name)
        evolver_name = next((evolver for vamos, _, evolver in PARAMETER_NAMES if vamos == bridge and evolver), name)
        evolver_value = next((evolver for _, canonical, evolver in VALUE_NAMES.get(bridge, ()) if value == canonical and evolver), value)
        result[evolver_name] = evolver_value
    return result


def _check_evolver_equivalents(assignment: Mapping[str, Any]) -> None:
    for name, value in assignment.items():
        reason = NO_EVOLVER_EQUIVALENT.get(_bridge_parameter(name), {}).get(str(value))
        if reason is not None:
            raise ValueError(f"{value} has no Evolver equivalent ({name}): {reason}.")


def from_evolver(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """An assignment with the spaces' canonical names from Evolver's camelCase parameters and values.

    Evolver's ``--name value`` pairs of a ``VAR_CONF.txt`` line give this mapping; keys that are not
    Evolver names are taken as canonical or earlier VAMOS names.
    """
    result: dict[str, Any] = {}
    for name, value in configuration.items():
        row = next((row for row in PARAMETER_NAMES if row[2] == name), None)
        bridge, canonical = (row[0], row[1]) if row is not None else (_bridge_parameter(name), canonical_parameter_name(name))
        canonical_value_ = next((c for _, c, evolver in VALUE_NAMES.get(bridge, ()) if value == evolver), canonical_value(bridge, value))
        result[canonical] = canonical_value_
    return result


def evolver_value_candidates(parameter: str, value: object) -> tuple[object, ...]:
    """Every canonical value of ``parameter`` (canonical, earlier or Evolver name) that Evolver writes as ``value``.

    Evolver has one name for values that VAMOS keeps apart, such as ``polynomial`` for the real-coded
    ``polynomial`` and the integer ``pm``; a reader picks the one its space offers.
    """
    row = next((row for row in PARAMETER_NAMES if row[2] == parameter), None)
    bridge = row[0] if row is not None else _bridge_parameter(parameter)
    return tuple(canonical for _, canonical, evolver in VALUE_NAMES.get(bridge, ()) if evolver is not None and value == evolver)


# A comparison of a parameter with a quoted value in a condition, and a parameter on its own.
_CONDITION_TERM = r"cfg\['(\w+)'\](\s*[!=]=\s*)'([^']*)'"
_CONDITION_NAME = r"cfg\['(\w+)'\]"


def _canonical_expression(expr: str) -> str:
    def term(match: re.Match[str]) -> str:
        name, operator, value = match.groups()
        return f"cfg['{canonical_parameter_name(name)}']{operator}'{canonical_value(name, value)}'"

    renamed = re.sub(_CONDITION_TERM, term, expr)
    return re.sub(_CONDITION_NAME, lambda match: f"cfg['{canonical_parameter_name(match.group(1))}']", renamed)


def canonical_parameter(param: ParamType) -> ParamType:
    """``param`` under its canonical name, with canonical values for a categorical one."""
    renamed = replace(param, name=canonical_parameter_name(param.name))
    if isinstance(renamed, Categorical):
        choices: list[Any] = []
        for choice in renamed.choices:
            value = canonical_value(param.name, choice)
            if value not in choices:
                choices.append(value)
        renamed = replace(renamed, choices=choices)
    return renamed


def canonical_conditional(block: ConditionalBlock) -> ConditionalBlock:
    return ConditionalBlock(
        canonical_parameter_name(block.parent_name),
        canonical_value(block.parent_name, block.parent_value),
        [canonical_parameter(param) for param in block.params],
    )


def canonical_condition(condition: Condition) -> Condition:
    return Condition(canonical_parameter_name(condition.param_name), _canonical_expression(condition.expr))


__all__ = [
    "NO_EVOLVER_EQUIVALENT",
    "PARAMETER_NAMES",
    "VALUE_NAMES",
    "bridge_assignment",
    "bridge_parameter_names",
    "canonical_assignment",
    "canonical_condition",
    "canonical_conditional",
    "canonical_parameter",
    "canonical_parameter_name",
    "canonical_value",
    "evolver_value_candidates",
    "from_evolver",
    "to_evolver",
]
