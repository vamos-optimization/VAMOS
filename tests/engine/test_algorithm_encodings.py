from unittest.mock import MagicMock

import numpy as np
import pytest

from vamos.engine.algorithm.builders import (
    build_agemoea_algorithm,
    build_moead_algorithm,
    build_nsgaiii_algorithm,
    build_rvea_algorithm,
    build_smsemoa_algorithm,
)
from vamos.foundation.kernel.backend import KernelBackend


class MockBinaryProblem:
    encoding = "binary"
    n_var = 10
    n_obj = 2
    xl = [0] * 10
    xu = [1] * 10


class MockPermutationProblem:
    encoding = "permutation"
    n_var = 10
    n_obj = 2
    xl = [0] * 10
    xu = [9] * 10


@pytest.fixture
def mock_kernel():
    return MagicMock(spec=KernelBackend)


# ... (Keeping existing tests) ...
def test_moead_binary_encoding_defaults(mock_kernel):
    problem = MockBinaryProblem()
    algo, config = build_moead_algorithm(
        kernel=mock_kernel,
        problem=problem,
        pop_size=100,
        moead_variation=None,
    )

    cfg_dict = config.to_dict()
    mutation = cfg_dict.get("mutation")
    assert mutation is not None
    assert mutation[0] == "bitflip"
    assert mutation[1] == {"prob": "1/n"}


def test_smsemoa_binary_encoding_defaults(mock_kernel):
    problem = MockBinaryProblem()
    algo, config = build_smsemoa_algorithm(
        kernel=mock_kernel,
        problem=problem,
        pop_size=100,
        smsemoa_variation=None,
    )

    cfg_dict = config.to_dict()
    mutation = cfg_dict.get("mutation")
    assert mutation is not None
    assert mutation[0] == "bitflip"


def test_nsgaiii_permutation_encoding_defaults(mock_kernel):
    problem = MockPermutationProblem()
    algo, config = build_nsgaiii_algorithm(
        kernel=mock_kernel,
        problem=problem,
        pop_size=100,
        nsgaiii_variation=None,
        selection_pressure=2,
    )

    cfg_dict = config.to_dict()
    mutation = cfg_dict.get("mutation")
    assert mutation is not None
    assert mutation[0] == "swap"


def test_repair_override_rejected_for_binary_encoding(mock_kernel):
    problem = MockBinaryProblem()
    with pytest.raises(ValueError, match="Repair operators are only supported for real encoding"):
        build_moead_algorithm(
            kernel=mock_kernel,
            problem=problem,
            pop_size=100,
            moead_variation={"repair": {"method": "clip"}},
        )


def test_agemoea_builder_accepts_mapping_variation(mock_kernel):
    problem = MockBinaryProblem()
    variation = {"crossover": {"method": "hux", "prob": 0.5}}
    algo, config = build_agemoea_algorithm(
        kernel=mock_kernel,
        problem=problem,
        pop_size=100,
        agemoea_variation=variation,
    )

    cfg_dict = config.to_dict()
    assert cfg_dict["pop_size"] == 100
    assert cfg_dict["crossover"] == ("hux", {"prob": 0.5})


def test_rvea_builder_accepts_mapping_variation(mock_kernel):
    problem = MockBinaryProblem()
    variation = {"mutation": {"method": "bitflip", "prob": 0.1}}
    algo, config = build_rvea_algorithm(
        kernel=mock_kernel,
        problem=problem,
        pop_size=100,
        rvea_variation=variation,
    )

    cfg_dict = config.to_dict()
    assert cfg_dict["pop_size"] == 100
    assert cfg_dict["mutation"] == ("bitflip", {"prob": 0.1})


# We can also keep the pipeline tests if we want to ensure run() still works
def test_agemoea_pipeline_run_integration(mock_kernel):
    problem = MockBinaryProblem()
    # Build via builder
    algo, _ = build_agemoea_algorithm(
        kernel=mock_kernel,
        problem=problem,
        pop_size=100,
        agemoea_variation=None,  # Uses defaults
    )

    from unittest.mock import patch

    from vamos.foundation.eval.backends import SerialEvalBackend

    # Mock backend
    mock_backend = MagicMock(spec=SerialEvalBackend)
    mock_backend.evaluate.return_value.F = np.full((100, 2), 0.1)
    mock_kernel.nsga2_ranking.return_value = (np.zeros(100, dtype=int), np.zeros(100, dtype=float))

    with patch("vamos.engine.algorithm.agemoea.agemoea.VariationPipeline") as MockPipeline:
        algo.run(problem, ("n_gen", 1), seed=0, eval_strategy=mock_backend)
        _, kwargs = MockPipeline.call_args
        # Should default to binary operators
        assert kwargs["encoding"] == "binary"
        assert kwargs["mut_method"] == "bitflip"


_ALGORITHMS = ("nsgaii", "moead", "smsemoa", "nsgaiii", "spea2", "ibea", "smpso", "agemoea", "rvea")
_DISCRETE_PROBLEMS = ("bin_knapsack", "tsp6", "int_alloc", "mixed_design")


@pytest.mark.parametrize(
    ("algorithm", "problem_name"),
    [
        (algorithm, problem_name)
        for problem_name in _DISCRETE_PROBLEMS
        for algorithm in _ALGORITHMS
        # SMPSO supports real and mixed encodings only.
        if algorithm != "smpso" or problem_name == "mixed_design"
    ],
)
def test_default_config_follows_problem_encoding(algorithm, problem_name):
    import vamos

    problem = vamos.make_problem_selection(problem_name).instantiate()

    result = vamos.optimize(problem, algorithm=algorithm, max_evaluations=300, seed=1, engine="numpy")

    assert result.data["evaluations"] == 300
    assert len(result) > 0
