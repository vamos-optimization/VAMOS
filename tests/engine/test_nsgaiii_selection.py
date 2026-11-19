"""NSGA-III's parent selection: random parents (Deb and Jain, 2014) or jMetal's tournament on rank.

Development after 1.4.0: VAMOS 1.4.0 and earlier ran a binary tournament on rank and crowding
distance, as NSGA-II, whatever the configured selection. These tests use a front where crowding
distance would pick the two extremes (their distance is infinite) whenever they enter a tournament.
"""

from __future__ import annotations

import numpy as np
import pytest

from vamos import optimize
from vamos.algorithms import NSGAIIIConfig
from vamos.engine.algorithm.components.constraint_handling import ConstraintHandlingTechnique, run_technique
from vamos.engine.algorithm.config.nsgaiii import nsgaiii_selection
from vamos.engine.algorithm.factory import build_algorithm
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.nsgaiii.helpers import nsgaiii_parents
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.grammar import builtin_parameter_space, builtin_parameter_space_names
from vamos.engine.tuning.racing.parameter_names import from_evolver, to_evolver
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem

DRAWS = 20_000


def _line_front(n: int) -> np.ndarray:
    """``n`` mutually non-dominated points on f1 + f2 = 1: crowding distance favours the two ends."""
    f1 = np.linspace(0.0, 1.0, n)
    return np.column_stack([f1, 1.0 - f1])


def _frequencies(parents: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(parents, minlength=n) / parents.size


def _parents(
    method: str,
    F: np.ndarray,
    *,
    G: np.ndarray | None = None,
    seed: int = 0,
    pressure: int = 2,
    kernel: KernelBackend | None = None,
    constraint_mode: str = "feasibility",
    technique: ConstraintHandlingTechnique | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return nsgaiii_parents(method, pressure, kernel or NumPyKernel(), F, G, constraint_mode, technique, rng, DRAWS)


def _config(**selection: object) -> NSGAIIIConfig:
    method = str(selection.pop("method", "random"))
    return (
        NSGAIIIConfig.builder()
        .pop_size(10)
        .crossover("sbx", prob=0.9, eta=30.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection(method, **selection)
        .reference_directions(divisions=3)
        .build()
    )


class _NoTournamentKernel(NumPyKernel):
    def tournament_selection(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("random selection of an unconstrained run ran a tournament")


# ---- random -------------------------------------------------------------------


def test_random_parents_do_not_depend_on_ranks_or_crowding() -> None:
    n = 10
    front = _line_front(n)
    # The same rows dominated in a chain: ranks 0 to 9, and crowding distance 0 inside each front.
    chain = np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float)])

    parents = _parents("random", front, seed=3)

    assert np.array_equal(parents, _parents("random", chain, seed=3))
    assert np.array_equal(parents, np.random.default_rng(3).integers(0, n, size=DRAWS, dtype=int))


def test_random_parents_are_uniform_where_crowding_would_pick_the_extremes() -> None:
    n = 10
    frequencies = _frequencies(_parents("random", _line_front(n)), n)

    # Each individual 1/n (the standard error is 0.002); crowding gave each extreme about twice that.
    assert np.allclose(frequencies, 1.0 / n, atol=0.01)


def test_a_random_run_never_runs_a_tournament() -> None:
    problem = DTLZ2Problem(n_var=12, n_obj=3)

    result = NSGAIII(_config().to_dict(), kernel=_NoTournamentKernel()).run(problem, termination=("max_evaluations", 60), seed=1)

    assert result["evaluations"] == 60


def test_random_parents_with_constraints_win_a_tournament_on_the_violation() -> None:
    # Jain and Deb (2014) and pymoo: feasible beats infeasible, the smaller violation wins, and a tie is drawn.
    n = 10
    G = np.array([[-1.0]] * 5 + [[0.5], [1.0], [2.0], [3.0], [4.0]])

    frequencies = _frequencies(_parents("random", _line_front(n), G=G), n)

    assert frequencies[9] == 0.0  # the most violated row loses every tournament it enters
    # A feasible row wins unless it meets a feasible one and loses the draw: 2/n - 4/(n (n - 1)) each.
    assert np.allclose(frequencies[:5], 2 / n - 4 / (n * (n - 1)), atol=0.01)
    assert np.all(np.diff(frequencies[5:]) < 0)


def test_random_parents_ignore_the_constraints_without_feasibility_first() -> None:
    n = 10
    G = np.array([[-1.0]] * 5 + [[0.5], [1.0], [2.0], [3.0], [4.0]])
    technique = run_technique(("violated_constraints", {}), "feasibility")
    assert technique is not None

    ignored = _frequencies(_parents("random", _line_front(n), G=G, constraint_mode="none"), n)
    with_technique = _frequencies(_parents("random", _line_front(n), G=G, technique=technique), n)

    assert np.allclose(ignored, 1.0 / n, atol=0.01)
    assert np.allclose(with_technique, 1.0 / n, atol=0.01)


# ---- tournament ---------------------------------------------------------------


@pytest.mark.parametrize("kernel", [NumPyKernel(), NumbaKernel()], ids=["numpy", "numba"])
def test_the_tournament_compares_ranks_and_not_crowding(kernel: KernelBackend) -> None:
    n = 10
    # A front of five on the line, where crowding would pick rows 0 and 4, and five dominated rows.
    F = np.vstack([_line_front(5), _line_front(5) + 1.0])

    frequencies = _frequencies(_parents("tournament", F, kernel=kernel), n)

    # A dominated row wins only against another dominated row: (5/n) (4/(n - 1)) of the tournaments.
    assert frequencies[5:].sum() == pytest.approx(20 / (n * (n - 1)), abs=0.01)
    # The ties are drawn, so the extremes of each front are not preferred.
    assert np.allclose(frequencies[:5], frequencies[:5].mean(), atol=0.01)
    assert np.allclose(frequencies[5:], frequencies[5:].mean(), atol=0.01)


def test_the_numpy_and_numba_tournaments_select_the_same_parents() -> None:
    F = np.vstack([_line_front(5), _line_front(5) + 1.0])

    for pressure in (2, 3):
        numpy_parents = _parents("tournament", F, kernel=NumPyKernel(), pressure=pressure, seed=5)
        numba_parents = _parents("tournament", F, kernel=NumbaKernel(), pressure=pressure, seed=5)
        assert np.array_equal(numpy_parents, numba_parents)


def test_the_tournament_puts_feasible_solutions_first() -> None:
    n = 10
    G = np.array([[-1.0]] * 5 + [[0.5], [1.0], [2.0], [3.0], [4.0]])

    frequencies = _frequencies(_parents("tournament", _line_front(n), G=G, pressure=3), n)

    # With three candidates, the most violated row and the second most violated never win.
    assert frequencies[8] == 0.0 and frequencies[9] == 0.0
    assert frequencies[:5].min() > frequencies[5:].max()


def test_the_tournament_size_sets_the_pressure() -> None:
    n = 10
    F = np.vstack([_line_front(5), _line_front(5) + 1.0])

    binary = _frequencies(_parents("tournament", F), n)[5:].sum()
    larger = _frequencies(_parents("tournament", F, pressure=4), n)[5:].sum()

    # Four distinct candidates are all dominated in (5/10)(4/9)(3/8)(2/7) of the tournaments.
    assert larger == pytest.approx(120 / 5040, abs=0.01)
    assert larger < binary


# ---- configuration --------------------------------------------------------------


def test_defaults_resolve_to_random_parents() -> None:
    resolved = optimize(DTLZ2Problem(n_var=12, n_obj=3), algorithm="nsgaiii", pop_size=10, max_evaluations=30, seed=0, engine="numpy")

    assert NSGAIIIConfig.default().selection == ("random", {})
    assert NSGAIIIConfig.default(pop_size=100, n_obj=5).selection == ("random", {})
    assert resolved.explain_defaults()["resolved_spec"]["operators"]["selection"]["component_id"].startswith("vamos.operator:random@")


def test_the_command_line_builder_selects_random_parents_unless_its_block_sets_a_tournament() -> None:
    problem = DTLZ2Problem(n_var=12, n_obj=3)
    tournament = {"nsgaiii": {"selection": {"method": "tournament", "size": 3}}}

    _, default = build_algorithm("nsgaiii", "numpy", problem, ExperimentConfig(population_size=10), selection_pressure=4)
    _, configured = build_algorithm("nsgaiii", "numpy", problem, ExperimentConfig(population_size=10), variations=tournament)

    assert default.to_dict()["selection"] == ("random", {})
    assert configured.to_dict()["selection"] == ("tournament", {"size": 3})


@pytest.mark.parametrize("method", ["boltzmann", "ranking", "sus", "roulette"])
def test_an_unsupported_selection_is_rejected(method: str) -> None:
    with pytest.raises(ConfigurationError, match="NSGA-III does not support selection"):
        NSGAIIIConfig.builder().selection(method)
    valid = _config().to_dict()
    with pytest.raises(ConfigurationError, match="NSGA-III does not support selection"):
        NSGAIIIConfig.from_dict({**valid, "selection": [method, {}]})
    # A configuration mapping that bypasses NSGAIIIConfig fails before the first evaluation.
    with pytest.raises(ConfigurationError, match="NSGA-III does not support selection"):
        NSGAIII({**valid, "selection": (method, {})}, kernel=NumPyKernel()).run(
            DTLZ2Problem(n_var=12, n_obj=3), termination=("max_evaluations", 30), seed=0
        )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (("random", {"size": 2}), "random selection takes no options"),
        (("tournament", {"pressure": 2}), "tournament selection takes only 'size'"),
        (("tournament", {"size": 0}), "positive integer"),
        (("tournament", {"size": True}), "positive integer"),
        ("random", "pair"),
    ],
)
def test_invalid_selection_options_are_rejected(selection: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        nsgaiii_selection(selection)


def test_the_supported_selections_build() -> None:
    assert nsgaiii_selection(("random", {})) == ("random", 2)
    assert nsgaiii_selection(("tournament", {})) == ("tournament", 2)
    assert nsgaiii_selection(["Tournament", {"size": 4}]) == ("tournament", 4)
    assert _config(method="tournament", size=3).selection == ("tournament", {"size": 3})


# ---- tuning spaces ----------------------------------------------------------------


NSGAIII_SPACES = sorted(name for name in builtin_parameter_space_names() if builtin_parameter_space(name).algorithm == "nsgaiii")


@pytest.mark.parametrize("name", NSGAIII_SPACES)
def test_the_spaces_choose_random_parents_or_a_tournament(name: str) -> None:
    space = builtin_parameter_space(name)
    selection = space.parameter("selection")

    assert selection.default == "random" and selection.values == ("random", "tournament")
    assert selection.production("random") == ()
    assert [parameter.name for parameter in selection.production("tournament")] == ["tournament_size"]
    assert config_from_assignment(name, {**space.default_values(), "pop_size": 100}).to_dict()["selection"] == ("random", {})


@pytest.mark.parametrize("name", ["nsgaiii", "nsgaiii_constrained"])
def test_the_bridge_builds_the_chosen_selection(name: str) -> None:
    base = {"pop_size": 100, "crossover": "sbx", "mutation": "polynomial"}

    def selection(**assignment: object) -> object:
        return config_from_assignment(name, {**base, **assignment}).to_dict()["selection"]

    assert selection(selection="random") == ("random", {})
    assert selection(selection="tournament", tournament_size=4) == ("tournament", {"size": 4})
    # An assignment saved from an earlier space names only the tournament size, and keeps its tournament.
    assert selection(tournament_size=5) == ("tournament", {"size": 5})
    assert selection() == ("random", {})


def test_the_selection_is_exchanged_with_evolver() -> None:
    assignment = {"selection": "tournament", "tournament_size": 3}

    assert to_evolver(assignment) == {"selection": "tournament", "selectionTournamentSize": 3}
    assert from_evolver({"selection": "random"}) == {"selection": "random"}
