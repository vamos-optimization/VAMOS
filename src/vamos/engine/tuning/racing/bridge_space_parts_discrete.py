from __future__ import annotations

from .config_space import SpacePart
from .param_space import Boolean, Categorical, Condition, ConditionalBlock, Int, ParamType, Real


def real_operator_part_medium(
    *,
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
    include_initializer: bool = True,
    include_repair: bool = True,
) -> SpacePart:
    params: list[ParamType] = []
    if include_initializer:
        params.append(Categorical("initializer", ["random", "lhs", "scatter"], role="structural"))
    params.extend(
        [
            Categorical("crossover", ["sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex"], role="operator"),
            Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1], role="operator_rate"),
            Categorical(
                "mutation",
                ["pm", "linked_polynomial", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform"],
                role="operator",
            ),
            Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1], role="operator_rate"),
            Real("mutation_eta", 5.0, 40.0, role="operator_rate"),
        ]
    )
    if include_repair:
        params.append(Categorical("repair", ["clip", "reflect", "random", "round"], role="structural"))

    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0, role="operator_rate")]),
        ConditionalBlock(
            "crossover",
            "blx_alpha",
            [
                Real("crossover_alpha", 0.0, 1.0, role="operator_rate"),
                Categorical("blx_repair", ["clip", "random", "reflect", "round"], role="structural"),
            ],
        ),
        ConditionalBlock(
            "crossover",
            "pcx",
            [
                Real("pcx_sigma_eta", 0.01, 0.5, role="operator_rate"),
                Real("pcx_sigma_zeta", 0.01, 0.5, role="operator_rate"),
            ],
        ),
        ConditionalBlock(
            "crossover",
            "undx",
            [
                Real("undx_zeta", 0.1, 1.0, role="operator_rate"),
                Real("undx_eta", 0.1, 1.0, role="operator_rate"),
            ],
        ),
        ConditionalBlock("crossover", "simplex", [Real("simplex_epsilon", 0.1, 1.0, role="operator_rate")]),
        ConditionalBlock("mutation", "non_uniform", [Real("nonuniform_perturbation", 0.05, 0.5, role="operator_rate")]),
        ConditionalBlock("mutation", "gaussian", [Real("gaussian_sigma", 0.001, 0.5, role="operator_rate")]),
        ConditionalBlock("mutation", "cauchy", [Real("cauchy_gamma", 0.001, 0.5, role="operator_rate")]),
        ConditionalBlock("mutation", "uniform", [Real("uniform_perturb", 0.01, 0.5, role="operator_rate")]),
    ]
    if include_initializer:
        conditionals.append(
            ConditionalBlock(
                "initializer",
                "scatter",
                [Categorical("scatter_base_size_factor", [0.1, 0.2, 0.3, 0.5, 0.75, 1.0], role="structural")],
            ),
        )
    return params, conditionals, [Condition("mutation_eta", "cfg['mutation'] == 'pm' or cfg['mutation'] == 'linked_polynomial'")]


def permutation_operator_part_full(
    *,
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
) -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["ox", "pmx", "edge", "cycle", "position", "aex"], role="operator"),
        Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1], role="operator_rate"),
        Categorical("mutation", ["swap", "insert", "scramble", "inversion", "displacement", "two_opt"], role="operator"),
        Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1], role="operator_rate"),
    ]
    return params, [], []


def binary_operator_part_full(
    *,
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
) -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["hux", "uniform", "one_point", "two_point"], role="operator"),
        Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1], role="operator_rate"),
        Categorical("mutation", ["bitflip", "segment_inversion"], role="operator"),
        Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1], role="operator_rate"),
    ]
    return params, [], []


def integer_operator_part_full(
    *,
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
) -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["uniform", "arithmetic", "sbx"], role="operator"),
        Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1], role="operator_rate"),
        Categorical("mutation", ["reset", "creep", "pm", "gaussian", "boundary"], role="operator"),
        Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1], role="operator_rate"),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0, role="operator_rate")]),
        ConditionalBlock("mutation", "pm", [Real("mutation_eta", 5.0, 40.0, role="operator_rate")]),
        ConditionalBlock("mutation", "creep", [Int("creep_step", 1, 5, role="operator_rate")]),
        ConditionalBlock("mutation", "gaussian", [Real("gaussian_sigma", 0.1, 5.0, role="operator_rate")]),
    ]
    return params, conditionals, []


def mixed_operator_part(
    *,
    crossover_choices: tuple[str, ...] = ("mixed",),
    mutation_choices: tuple[str, ...] = ("mixed",),
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
) -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", list(crossover_choices), role="operator"),
        Categorical("mutation", list(mutation_choices), role="operator"),
        Categorical("perm_crossover", ["ox", "pmx", "edge", "cycle", "position", "aex"], role="operator"),
        Real("perm_crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1], role="operator_rate"),
        Categorical("perm_mutation", ["swap", "insert", "scramble", "inversion", "displacement", "two_opt"], role="operator"),
        Real("perm_mutation_prob", mutation_prob_bounds[0], mutation_prob_bounds[1], role="operator_rate"),
        Categorical("real_crossover", ["arithmetic", "mean"], role="operator"),
        Real("real_crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1], role="operator_rate"),
        Categorical("real_mutation", ["gaussian", "uniform_reset", "polynomial"], role="operator"),
        Real("real_mutation_prob", mutation_prob_bounds[0], mutation_prob_bounds[1], role="operator_rate"),
        Categorical("int_crossover", ["uniform", "arithmetic", "sbx"], role="operator"),
        Real("int_crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1], role="operator_rate"),
        Categorical("int_mutation", ["reset", "creep", "polynomial"], role="operator"),
        Real("int_mutation_prob", mutation_prob_bounds[0], mutation_prob_bounds[1], role="operator_rate"),
        Categorical("cat_crossover", ["uniform"], role="operator"),
        Real("cat_crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1], role="operator_rate"),
        Categorical("cat_mutation", ["reset"], role="operator"),
        Real("cat_mutation_prob", mutation_prob_bounds[0], mutation_prob_bounds[1], role="operator_rate"),
    ]
    conditionals = [
        ConditionalBlock("real_mutation", "gaussian", [Real("real_mutation_sigma_factor", 0.01, 0.5, role="operator_rate")]),
        ConditionalBlock("real_mutation", "polynomial", [Real("real_mutation_eta", 5.0, 40.0, role="operator_rate")]),
        ConditionalBlock("int_crossover", "sbx", [Real("int_crossover_eta", 5.0, 40.0, role="operator_rate")]),
        ConditionalBlock("int_mutation", "creep", [Int("int_mutation_step", 1, 5, role="operator_rate")]),
        ConditionalBlock("int_mutation", "polynomial", [Real("int_mutation_eta", 5.0, 40.0, role="operator_rate")]),
    ]
    return params, conditionals, []


def external_archive_part() -> SpacePart:
    """External-archive params shared by all algorithms."""
    params: list[ParamType] = [
        Boolean("use_external_archive", role="adaptive"),
    ]
    archive_unbounded_param = Boolean("archive_unbounded", role="adaptive")
    archive_prune_policy_param = Categorical(
        "archive_prune_policy", ["crowding", "hv", "mc_hv", "knn", "maxmin", "ref_dirs"], role="adaptive"
    )
    conditionals = [
        ConditionalBlock(
            "use_external_archive",
            True,
            [archive_unbounded_param, archive_prune_policy_param],
        ),
    ]
    conditions = [
        Condition("archive_prune_policy", "cfg['archive_unbounded'] == False"),
    ]
    return params, conditionals, conditions


__all__ = [
    "binary_operator_part_full",
    "external_archive_part",
    "integer_operator_part_full",
    "mixed_operator_part",
    "permutation_operator_part_full",
    "real_operator_part_medium",
]
