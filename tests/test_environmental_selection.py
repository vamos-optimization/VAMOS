"""Regression tests for the public environmental-selection API."""

from __future__ import annotations

import math
import unittest

import numpy as np

import vamos
from vamos import EnvironmentalSelectionResult, select_survivors
from vamos.api import (
    EnvironmentalSelectionResult as ApiEnvironmentalSelectionResult,
)
from vamos.api import select_survivors as api_select_survivors
from vamos.foundation.kernel.numpy_backend import NumPyKernel


class PublicApiTests(unittest.TestCase):
    """Validate facade exports and version metadata."""

    def test_top_level_public_imports(self) -> None:
        self.assertIs(select_survivors, api_select_survivors)
        self.assertIs(
            EnvironmentalSelectionResult,
            ApiEnvironmentalSelectionResult,
        )

    def test_version_metadata_is_1_0_1(self) -> None:
        self.assertEqual(vamos.__version__, "1.0.1")


class SelectionBehaviorTests(unittest.TestCase):
    """Validate deterministic NSGA-II ranking and survivor ordering."""

    def test_simple_all_nondominated_front(self) -> None:
        result = select_survivors([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], 3)
        self.assertEqual(result.ranks, (0, 0, 0))

    def test_dominated_candidates_receive_higher_ranks(self) -> None:
        result = select_survivors([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], 2)
        self.assertEqual(result.ranks, (0, 1, 2))

    def test_selected_count_is_exact(self) -> None:
        result = select_survivors([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], 2)
        self.assertEqual(len(result.selected_indices), 2)

    def test_selected_indices_are_unique(self) -> None:
        result = select_survivors([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], 3)
        self.assertEqual(len(set(result.selected_indices)), 3)

    def test_lower_rank_is_preferred(self) -> None:
        result = select_survivors([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], 2)
        self.assertEqual(result.selected_indices, (0, 1))

    def test_higher_crowding_is_preferred_within_front(self) -> None:
        objectives = [[0.0, 10.0], [1.0, 9.0], [2.0, 1.0], [10.0, 0.0]]
        result = select_survivors(objectives, 3)
        self.assertEqual(result.selected_indices, (0, 3, 2))
        self.assertGreater(result.crowding_distances[2], result.crowding_distances[1])

    def test_original_index_breaks_exact_ties(self) -> None:
        result = select_survivors([[1.0, 1.0]] * 4, 1)
        self.assertEqual(result.selected_indices, (0,))

    def test_boundary_solutions_receive_positive_infinity(self) -> None:
        result = select_survivors([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], 3)
        self.assertTrue(math.isinf(result.crowding_distances[0]))
        self.assertTrue(math.isinf(result.crowding_distances[2]))
        self.assertGreater(result.crowding_distances[0], 0.0)

    def test_duplicate_objective_rows_are_deterministic(self) -> None:
        objectives = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        first = select_survivors(objectives, 3)
        second = select_survivors(objectives, 3)
        self.assertEqual(first, second)

    def test_repeated_calls_are_identical(self) -> None:
        objectives = [[0.0, 4.0], [1.0, 1.0], [4.0, 0.0], [3.0, 3.0]]
        self.assertEqual(
            select_survivors(objectives, 3),
            select_survivors(objectives, 3),
        )

    def test_input_numpy_array_is_not_mutated(self) -> None:
        objectives = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        original = objectives.copy()
        select_survivors(objectives, 2)
        np.testing.assert_array_equal(objectives, original)

    def test_nested_list_input_is_accepted(self) -> None:
        result = select_survivors([[0, 2], [1, 1], [2, 0]], 2)
        self.assertIsInstance(result, EnvironmentalSelectionResult)

    def test_ranks_cover_every_candidate(self) -> None:
        result = select_survivors([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], 2)
        self.assertEqual(len(result.ranks), 3)

    def test_crowding_values_cover_every_candidate(self) -> None:
        result = select_survivors([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], 2)
        self.assertEqual(len(result.crowding_distances), 3)

    def test_result_fields_are_built_in_tuples(self) -> None:
        result = select_survivors([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], 2)
        self.assertIs(type(result.selected_indices), tuple)
        self.assertIs(type(result.ranks), tuple)
        self.assertIs(type(result.crowding_distances), tuple)

    def test_agrees_with_existing_nsga2_survival_on_fixed_matrix(self) -> None:
        objectives = np.array(
            [[0.0, 4.0], [1.0, 1.0], [4.0, 0.0], [3.0, 3.0]],
            dtype=float,
        )
        decisions = np.arange(4, dtype=float).reshape(-1, 1)
        _, _, expected = NumPyKernel().nsga2_survival(
            decisions[:2],
            objectives[:2],
            decisions[2:],
            objectives[2:],
            3,
            return_indices=True,
        )
        result = select_survivors(objectives, 3)
        self.assertEqual(set(result.selected_indices), set(expected.tolist()))


class InputValidationTests(unittest.TestCase):
    """Validate public input rejection without running an optimization loop."""

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([], 1)

    def test_one_dimensional_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([0.0, 1.0], 1)

    def test_one_objective_matrix_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([[0.0], [1.0]], 1)

    def test_ragged_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([[0.0, 1.0], [2.0]], 1)

    def test_nan_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([[0.0, np.nan], [1.0, 1.0]], 1)

    def test_positive_infinite_objective_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([[0.0, np.inf], [1.0, 1.0]], 1)

    def test_negative_infinite_objective_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([[0.0, -np.inf], [1.0, 1.0]], 1)

    def test_zero_survivor_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([[0.0, 1.0]], 0)

    def test_survivor_count_above_population_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_survivors([[0.0, 1.0]], 2)

    def test_bool_survivor_count_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            select_survivors([[0.0, 1.0]], True)


class ResultValidationTests(unittest.TestCase):
    """Validate immutable result construction independently of selection."""

    def test_result_is_frozen(self) -> None:
        result = EnvironmentalSelectionResult((0,), (0,), (math.inf,))
        with self.assertRaises((AttributeError, TypeError)):
            result.selected_indices = (1,)  # type: ignore[misc]

    def test_result_requires_tuple_fields(self) -> None:
        with self.assertRaises(TypeError):
            EnvironmentalSelectionResult([0], (0,), (math.inf,))  # type: ignore[arg-type]

    def test_duplicate_selected_indices_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentalSelectionResult((0, 0), (0, 0), (math.inf, math.inf))

    def test_invalid_selected_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentalSelectionResult((1,), (0,), (math.inf,))

    def test_negative_rank_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentalSelectionResult((0,), (-1,), (math.inf,))

    def test_bool_rank_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            EnvironmentalSelectionResult((0,), (False,), (math.inf,))

    def test_nan_crowding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentalSelectionResult((0,), (0,), (math.nan,))

    def test_negative_crowding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentalSelectionResult((0,), (0,), (-1.0,))

    def test_negative_infinite_crowding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentalSelectionResult((0,), (0,), (-math.inf,))

    def test_positive_infinite_crowding_is_accepted(self) -> None:
        result = EnvironmentalSelectionResult((0,), (0,), (math.inf,))
        self.assertEqual(result.crowding_distances, (math.inf,))

    def test_rank_and_crowding_lengths_must_match(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentalSelectionResult((0,), (0, 1), (math.inf,))


if __name__ == "__main__":
    unittest.main()
