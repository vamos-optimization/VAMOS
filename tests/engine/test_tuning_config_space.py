import numpy as np
import pytest

from vamos.engine.tuning.racing.bridge import (
    build_agemoea_binary_config_space,
    build_agemoea_config_space,
    build_agemoea_integer_config_space,
    build_agemoea_permutation_config_space,
    build_ibea_binary_config_space,
    build_ibea_integer_config_space,
    build_ibea_permutation_config_space,
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_integer_config_space,
    build_moead_permutation_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_integer_config_space,
    build_nsgaiii_permutation_config_space,
    build_rvea_binary_config_space,
    build_rvea_config_space,
    build_rvea_integer_config_space,
    build_rvea_permutation_config_space,
    build_smpso_mixed_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_permutation_config_space,
    build_spea2_binary_config_space,
    build_spea2_integer_config_space,
    build_spea2_permutation_config_space,
    config_from_assignment,
)
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.param_space import (
    Boolean,
    Categorical,
    Int,
    Real,
)


def test_param_round_trip_sampling():
    rng = np.random.default_rng(0)
    cat = Categorical("color", ["red", "blue", "green"])
    intval = Int("k", 1, 9)
    floatval = Real("p", 0.1, 0.9)
    boolp = Boolean("flag")
    cati = Categorical("size", [16, 32, 64])

    assignment = {
        "color": cat.sample(rng),
        "k": intval.sample(rng),
        "p": floatval.sample(rng),
        "flag": boolp.sample(rng),
        "size": cati.sample(rng),
    }
    vec = [
        cat.to_unit(assignment["color"]),
        intval.to_unit(assignment["k"]),
        floatval.to_unit(assignment["p"]),
        boolp.to_unit(assignment["flag"]),
        cati.to_unit(assignment["size"]),
    ]
    assert cat.from_unit(vec[0]) in cat.choices
    assert isinstance(intval.from_unit(vec[1]), int)
    assert 0.1 <= floatval.from_unit(vec[2]) <= 0.9
    assert isinstance(boolp.from_unit(vec[3]), bool)
    assert cati.from_unit(vec[4]) in cati.choices


def test_algorithm_config_space_round_trip():
    rng = np.random.default_rng(1)
    params = [Categorical("a", ["x", "y"]), Int("b", 1, 3)]
    space = AlgorithmConfigSpace("toy", params, [])
    assignment = space.sample(rng)
    vec = space.to_unit_vector(assignment)
    decoded = space.from_unit_vector(vec)
    assert decoded["a"] in ["x", "y"]
    assert 1 <= decoded["b"] <= 3


@pytest.mark.parametrize("builder", [build_nsgaii_config_space, build_moead_config_space, build_agemoea_config_space])
def test_mutation_eta_is_active_only_for_polynomial_mutations(builder):
    space = builder()
    param_space = space.to_param_space()
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(80):
        config = space.sample(rng)
        active = config["mutation"] in {"pm", "polynomial", "linked_polynomial"}
        assert ("mutation_eta" in config) == active
        assert {param.name for param in space.flatten(config)} == set(config)
        param_space.validate(config)
        vector = space.to_unit_vector(config)
        decoded = space.from_unit_vector(vector)
        assert set(decoded) == set(config)
        np.testing.assert_allclose(space.to_unit_vector(decoded), vector)
        seen.add(active)
    assert seen == {False, True}


def test_nsgaii_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(2)
    space = build_nsgaii_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("nsgaii", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in (
        "sbx",
        "blx_alpha",
        "blx_alpha_beta",
        "arithmetic",
        "whole_arithmetic",
        "laplace",
        "fuzzy",
        "pcx",
        "undx",
        "simplex",
    )
    assert cfg.mutation[0] in (
        "polynomial",
        "linked_polynomial",
        "non_uniform",
        "gaussian",
        "uniform_reset",
        "cauchy",
        "uniform",
        "levy_flight",
        "power_law",
    )


@pytest.mark.parametrize("initializer", ["random", "lhs", "scatter", "sobol", "halton", "obl"])
def test_nsgaii_config_space_supports_full_real_initializer_catalog(initializer: str):
    assignment = {
        "pop_size": 32,
        "offspring_ratio": 1.0,
        "selection": "tournament",
        "selection_size": 2,
        "use_external_archive": False,
        "archive_unbounded": False,
        "initializer": initializer,
        "crossover": "sbx",
        "crossover_prob": 0.9,
        "crossover_eta": 15.0,
        "mutation": "pm",
        "mutation_prob_factor": 1.0,
        "mutation_eta": 20.0,
        "repair": "clip",
    }
    if initializer == "scatter":
        assignment["scatter_base_size_factor"] = 0.5

    cfg = config_from_assignment("nsgaii", assignment)

    assert cfg.initializer is not None
    assert cfg.initializer["type"] == initializer


@pytest.mark.parametrize("selection", ["tournament", "random", "boltzmann", "ranking", "sus"])
def test_nsgaii_config_space_supports_full_selection_catalog(selection: str):
    assignment = {
        "pop_size": 32,
        "offspring_ratio": 1.0,
        "selection": selection,
        "use_external_archive": False,
        "archive_unbounded": False,
        "initializer": "random",
        "crossover": "sbx",
        "crossover_prob": 0.9,
        "crossover_eta": 15.0,
        "mutation": "polynomial",
        "mutation_prob_factor": 1.0,
        "mutation_eta": 20.0,
        "repair": "clip",
    }
    if selection == "tournament":
        assignment["selection_size"] = 2

    cfg = config_from_assignment("nsgaii", assignment)

    assert cfg.selection[0] == selection
    if selection == "tournament":
        assert cfg.selection[1]["size"] == 2
    else:
        assert cfg.selection[1] == {}


def test_nsgaii_archive_unbounded_disables_archive_params():
    space = build_nsgaii_config_space()
    param_space = space.to_param_space()
    assert "selection_size" in param_space.params
    cfg_disabled = {"use_external_archive": False}
    assert not param_space.is_active("selection_size", cfg_disabled)
    assert not param_space.is_active("archive_unbounded", cfg_disabled)
    assert not param_space.is_active("archive_prune_policy", cfg_disabled)
    cfg_random = {"selection": "random", "use_external_archive": False}
    assert not param_space.is_active("selection_size", cfg_random)
    cfg_tournament = {"selection": "tournament", "use_external_archive": False}
    assert param_space.is_active("selection_size", cfg_tournament)
    cfg = {"use_external_archive": True, "archive_unbounded": True}
    assert param_space.is_active("archive_unbounded", cfg)
    assert not param_space.is_active("archive_prune_policy", cfg)
    cfg_bounded = {"use_external_archive": True, "archive_unbounded": False}
    assert param_space.is_active("archive_unbounded", cfg_bounded)
    assert param_space.is_active("archive_prune_policy", cfg_bounded)


def test_real_tuning_spaces_do_not_expose_none_repair():
    nsgaii_repair = build_nsgaii_config_space().to_param_space().params["repair"]
    assert isinstance(nsgaii_repair, Categorical)
    assert "none" not in nsgaii_repair.choices


def test_nsgaii_permutation_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(3)
    space = build_nsgaii_permutation_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("nsgaii_permutation", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in ("ox", "pmx", "edge", "cycle", "position", "aex")
    assert cfg.mutation[0] in ("swap", "insert", "scramble", "inversion", "displacement", "two_opt")


def test_nsgaii_mixed_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(4)
    space = build_nsgaii_mixed_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("nsgaii_mixed", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] == "mixed"
    assert cfg.mutation[0] == "mixed"
    assert "crossover_prob" not in assignment
    assert "mutation_prob_factor" not in assignment
    assert "perm_crossover" in assignment
    assert "perm_mutation" in assignment
    assert "int_crossover" in assignment
    assert "int_mutation" in assignment
    assert cfg.crossover[1]["perm_crossover"] == assignment["perm_crossover"]
    assert cfg.crossover[1]["int_crossover"] == assignment["int_crossover"]
    assert cfg.mutation[1]["perm_mutation"] == assignment["perm_mutation"]
    assert cfg.mutation[1]["int_mutation"] == assignment["int_mutation"]
    assert "prob" not in cfg.crossover[1]
    assert "prob" not in cfg.mutation[1]


def test_nsgaii_mixed_config_accepts_segment_operator_assignment():
    assignment = {
        "pop_size": 32,
        "offspring_ratio": 1.0,
        "selection": "tournament",
        "selection_size": 2,
        "use_external_archive": False,
        "archive_unbounded": False,
        "crossover": "mixed",
        "mutation": "mixed",
        "perm_crossover": "position",
        "perm_crossover_prob": 0.8,
        "perm_mutation": "swap",
        "perm_mutation_prob": 0.2,
        "real_crossover": "mean",
        "real_crossover_prob": 0.7,
        "real_mutation": "gaussian",
        "real_mutation_prob": 0.1,
        "real_mutation_sigma_factor": 0.15,
        "int_crossover": "sbx",
        "int_crossover_prob": 0.6,
        "int_crossover_eta": 20.0,
        "int_mutation": "polynomial",
        "int_mutation_prob": 0.05,
        "int_mutation_eta": 30.0,
        "cat_crossover": "uniform",
        "cat_crossover_prob": 0.5,
        "cat_mutation": "reset",
        "cat_mutation_prob": 0.25,
    }

    cfg = config_from_assignment("nsgaii_mixed", assignment)

    assert cfg.crossover[0] == "mixed"
    assert cfg.mutation[0] == "mixed"
    assert cfg.crossover[1]["perm_crossover"] == "position"
    assert cfg.crossover[1]["int_crossover"] == "sbx"
    assert cfg.crossover[1]["int_crossover_eta"] == 20.0
    assert cfg.mutation[1]["perm_mutation"] == "swap"
    assert cfg.mutation[1]["int_mutation"] == "polynomial"
    assert cfg.mutation[1]["int_mutation_eta"] == 30.0
    assert cfg.mutation[1]["real_mutation"] == "gaussian"
    assert cfg.mutation[1]["real_mutation_sigma_factor"] == 0.15
    assert "prob" not in cfg.crossover[1]
    assert "prob" not in cfg.mutation[1]


def test_moead_permutation_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(5)
    space = build_moead_permutation_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("moead_permutation", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in ("ox", "pmx", "edge", "cycle", "position", "aex")
    assert cfg.mutation[0] in ("swap", "insert", "scramble", "inversion", "displacement", "two_opt")


def test_agemoea_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(6)
    space = build_agemoea_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("agemoea", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in ("sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex")
    assert cfg.mutation[0] in (
        "pm",
        "linked_polynomial",
        "non_uniform",
        "gaussian",
        "uniform_reset",
        "cauchy",
        "uniform",
    )


def test_rvea_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(7)
    space = build_rvea_config_space()
    assignment = space.sample(rng)
    assignment["n_obj"] = 3
    cfg = config_from_assignment("rvea", assignment)
    assert cfg.pop_size > 0
    assert cfg.n_partitions == int(assignment["n_partitions"])
    assert cfg.crossover[0] in ("sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex")
    assert cfg.mutation[0] in (
        "pm",
        "linked_polynomial",
        "non_uniform",
        "gaussian",
        "uniform_reset",
        "cauchy",
        "uniform",
    )


def test_agemoea_external_archive_config():
    rng = np.random.default_rng(8)
    space = build_agemoea_config_space()
    assignment = space.sample(rng)
    assignment.update(
        {
            "use_external_archive": True,
            "archive_unbounded": False,
            "archive_prune_policy": "crowding",
        }
    )
    cfg = config_from_assignment("agemoea", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity == cfg.pop_size
    assert cfg.result_mode == "non_dominated"


def test_rvea_external_archive_config():
    rng = np.random.default_rng(9)
    space = build_rvea_config_space()
    assignment = space.sample(rng)
    assignment.update(
        {
            "n_obj": 3,
            "use_external_archive": True,
            "archive_unbounded": False,
            "archive_prune_policy": "crowding",
        }
    )
    cfg = config_from_assignment("rvea", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity == cfg.pop_size
    assert cfg.result_mode == "non_dominated"


def test_external_archive_config_uses_hv_policy_names():
    assignment = {
        "pop_size": 32,
        "offspring_size": 32,
        "selection": "tournament",
        "selection_size": 2,
        "crossover": "sbx",
        "crossover_prob": 0.9,
        "crossover_eta": 15.0,
        "mutation": "pm",
        "mutation_prob": "1/n",
        "mutation_eta": 20.0,
        "use_external_archive": True,
        "archive_unbounded": False,
        "archive_prune_policy": "hv",
    }
    cfg = config_from_assignment("nsgaii", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity == cfg.pop_size
    assert cfg.external_archive.pruning == "hv"


def test_unbounded_external_archive_omits_pruning_from_serialized_config():
    assignment = {
        "pop_size": 32,
        "offspring_size": 32,
        "selection": "tournament",
        "selection_size": 2,
        "crossover": "sbx",
        "crossover_prob": 0.9,
        "crossover_eta": 15.0,
        "mutation": "pm",
        "mutation_prob": "1/n",
        "mutation_eta": 20.0,
        "use_external_archive": True,
        "archive_unbounded": True,
        "archive_prune_policy": "mc_hv",
    }
    cfg = config_from_assignment("nsgaii", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity is None
    serialized = cfg.to_dict()
    assert serialized["external_archive"]["capacity"] is None
    assert "pruning" not in serialized["external_archive"]


def test_external_archive_config_uses_knn_policy_name():
    assignment = {
        "pop_size": 32,
        "offspring_size": 32,
        "selection": "tournament",
        "selection_size": 2,
        "crossover": "sbx",
        "crossover_prob": 0.9,
        "crossover_eta": 15.0,
        "mutation": "pm",
        "mutation_prob": "1/n",
        "mutation_eta": 20.0,
        "use_external_archive": True,
        "archive_unbounded": False,
        "archive_prune_policy": "knn",
    }
    cfg = config_from_assignment("nsgaii", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity == cfg.pop_size
    assert cfg.external_archive.pruning == "knn"


def test_external_archive_config_uses_maxmin_policy_name():
    assignment = {
        "pop_size": 32,
        "offspring_size": 32,
        "selection": "tournament",
        "selection_size": 2,
        "crossover": "sbx",
        "crossover_prob": 0.9,
        "crossover_eta": 15.0,
        "mutation": "pm",
        "mutation_prob": "1/n",
        "mutation_eta": 20.0,
        "use_external_archive": True,
        "archive_unbounded": False,
        "archive_prune_policy": "maxmin",
    }
    cfg = config_from_assignment("nsgaii", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity == cfg.pop_size
    assert cfg.external_archive.pruning == "maxmin"


def test_external_archive_config_uses_ref_dirs_policy_name():
    assignment = {
        "pop_size": 32,
        "offspring_size": 32,
        "selection": "tournament",
        "selection_size": 2,
        "crossover": "sbx",
        "crossover_prob": 0.9,
        "crossover_eta": 15.0,
        "mutation": "pm",
        "mutation_prob": "1/n",
        "mutation_eta": 20.0,
        "use_external_archive": True,
        "archive_unbounded": False,
        "archive_prune_policy": "ref_dirs",
    }
    cfg = config_from_assignment("nsgaii", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity == cfg.pop_size
    assert cfg.external_archive.pruning == "ref_dirs"


def test_external_archive_config_rejects_unknown_pruning_policy():
    assignment = {
        "pop_size": 32,
        "offspring_size": 32,
        "selection": "tournament",
        "selection_size": 2,
        "crossover": "sbx",
        "crossover_prob": 0.9,
        "crossover_eta": 15.0,
        "mutation": "pm",
        "mutation_prob": "1/n",
        "mutation_eta": 20.0,
        "use_external_archive": True,
        "archive_unbounded": False,
        "archive_prune_policy": "invalid-policy",
    }
    with pytest.raises(ValueError, match="Unsupported pruning 'invalid-policy'"):
        config_from_assignment("nsgaii", assignment)


def test_binary_integer_config_spaces_build_and_construct_config():
    cases = [
        (
            build_nsgaii_binary_config_space,
            "nsgaii_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
        ),
        (
            build_nsgaii_integer_config_space,
            "nsgaii_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
        ),
        (
            build_moead_binary_config_space,
            "moead_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
        ),
        (
            build_moead_integer_config_space,
            "moead_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
        ),
        (
            build_nsgaiii_binary_config_space,
            "nsgaiii_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
        ),
        (
            build_nsgaiii_integer_config_space,
            "nsgaiii_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
        ),
        (
            build_smsemoa_binary_config_space,
            "smsemoa_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
        ),
        (
            build_smsemoa_integer_config_space,
            "smsemoa_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
        ),
        (
            build_ibea_binary_config_space,
            "ibea_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
        ),
        (
            build_ibea_integer_config_space,
            "ibea_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
        ),
    ]

    for idx, (builder, algo, cross_set, mut_set) in enumerate(cases):
        rng = np.random.default_rng(100 + idx)
        space = builder()
        assignment = space.sample(rng)
        cfg = config_from_assignment(algo, assignment)
        assert cfg.crossover[0] in cross_set
        assert cfg.mutation[0] in mut_set


def test_new_permutation_binary_integer_builders_construct_configs():
    cases = [
        (
            build_agemoea_permutation_config_space,
            "agemoea_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
        (
            build_agemoea_binary_config_space,
            "agemoea_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
            False,
        ),
        (
            build_agemoea_integer_config_space,
            "agemoea_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
            False,
        ),
        (
            build_rvea_permutation_config_space,
            "rvea_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            True,
        ),
        (
            build_rvea_binary_config_space,
            "rvea_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
            True,
        ),
        (
            build_rvea_integer_config_space,
            "rvea_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
            True,
        ),
        (
            build_spea2_permutation_config_space,
            "spea2_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
        (
            build_spea2_binary_config_space,
            "spea2_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
            False,
        ),
        (
            build_spea2_integer_config_space,
            "spea2_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
            False,
        ),
        (
            build_ibea_permutation_config_space,
            "ibea_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
        (
            build_smsemoa_permutation_config_space,
            "smsemoa_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
        (
            build_nsgaiii_permutation_config_space,
            "nsgaiii_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
    ]
    for idx, (builder, algo, cross_set, mut_set, needs_n_obj) in enumerate(cases):
        rng = np.random.default_rng(200 + idx)
        space = builder()
        assignment = space.sample(rng)
        if needs_n_obj:
            assignment["n_obj"] = 3
        cfg = config_from_assignment(algo, assignment)
        assert cfg.crossover[0] in cross_set
        assert cfg.mutation[0] in mut_set


def test_smpso_mixed_builder_and_assignment():
    rng = np.random.default_rng(300)
    space = build_smpso_mixed_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("smpso_mixed", assignment)
    assert cfg.mutation[0] == "mixed"
    assert cfg.mutation[1]["perm_mutation"] == assignment["perm_mutation"]
    assert cfg.mutation[1]["int_mutation"] == assignment["int_mutation"]


def test_new_integer_parameter_mapping_in_assignment():
    ag_space = build_agemoea_integer_config_space()
    ag_assignment = ag_space.sample(np.random.default_rng(400))
    ag_assignment["mutation"] = "gaussian"
    ag_assignment["gaussian_sigma"] = 2.5
    ag_cfg = config_from_assignment("agemoea_integer", ag_assignment)
    assert ag_cfg.mutation[0] == "gaussian"
    assert float(ag_cfg.mutation[1]["sigma"]) == 2.5

    sp_space = build_spea2_integer_config_space()
    sp_assignment = sp_space.sample(np.random.default_rng(401))
    sp_assignment["mutation"] = "creep"
    sp_assignment["creep_step"] = 3
    sp_cfg = config_from_assignment("spea2_integer", sp_assignment)
    assert sp_cfg.mutation[0] == "creep"
    assert int(sp_cfg.mutation[1]["step"]) == 3
