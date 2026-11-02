import typing

import pytest

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.exceptions import ConfigurationError, InvalidAlgorithmError
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_unknown_algorithm_name_errors():
    with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
        optimize(ZDT1Problem(n_var=4), algorithm="does_not_exist", max_evaluations=4, pop_size=4, seed=0)


def test_unknown_problem_selection_errors():
    with pytest.raises(Exception):
        optimize("nonexistent_problem", algorithm="nsgaii", max_evaluations=4, pop_size=4, seed=0)


def test_invalid_max_evaluations_errors():
    with pytest.raises(ConfigurationError, match="max_evaluations must be a positive integer"):
        optimize(ZDT1Problem(n_var=4), algorithm="nsgaii", max_evaluations=0, pop_size=4, seed=0)


def test_invalid_pop_size_errors():
    with pytest.raises(ConfigurationError, match="pop_size must be a positive integer"):
        optimize(ZDT1Problem(n_var=4), algorithm="nsgaii", max_evaluations=4, pop_size=-1, seed=0)


def test_invalid_algorithm_config_pop_size_errors():
    # The builder rejects a non-positive population as soon as it is set.
    with pytest.raises(ConfigurationError, match="pop_size must be a positive integer"):
        NSGAIIConfig.builder().pop_size(-4)


def test_invalid_eval_strategy_errors():
    with pytest.raises(ConfigurationError, match="eval_strategy must be one of"):
        optimize(
            ZDT1Problem(n_var=4),
            algorithm="nsgaii",
            max_evaluations=4,
            pop_size=4,
            seed=0,
            eval_strategy="parallel",
        )


_ALL_ALGORITHMS = ["nsgaii", "nsgaiii", "moead", "smsemoa", "spea2", "ibea", "smpso", "agemoea", "rvea"]


class _RecordingCallback:
    """A live callback that records each generation it receives and can fail at one of them."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.generations: list[int] = []
        self._fail_at = fail_at

    def on_start(self, ctx: object = None) -> None:
        return None

    def on_generation(self, generation: int, F: object = None, X: object = None, stats: object = None) -> None:
        self.generations.append(int(generation))
        if generation == self._fail_at:
            raise ValueError("user callback failed")

    def on_end(self, final_F: object = None, final_stats: object = None) -> None:
        return None


@pytest.mark.parametrize("algorithm", _ALL_ALGORITHMS)
def test_live_callback_exception_propagates_in_all_algorithms(algorithm: str) -> None:
    callback = _RecordingCallback(fail_at=1)
    with pytest.raises(ValueError, match="user callback failed"):
        optimize("zdt1", algorithm=algorithm, engine="numpy", max_evaluations=1500, seed=1, live_viz=callback)
    assert callback.generations.count(1) == 1


@pytest.mark.parametrize("algorithm", _ALL_ALGORITHMS)
def test_first_notification_is_generation_zero(algorithm: str) -> None:
    callback = _RecordingCallback()
    optimize("zdt1", algorithm=algorithm, engine="numpy", max_evaluations=600, seed=1, live_viz=callback)
    assert callback.generations[:2] == [0, 1]


def _nsgaii_builder():  # noqa: ANN202 - builder type is private
    return (
        NSGAIIConfig.builder()
        .pop_size(10)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
    )


def test_live_callback_mode_type_matches_accepted_values() -> None:
    from vamos.engine.algorithm.config.base import LiveCallbackMode
    from vamos.engine.algorithm.config.validation import _LIVE_CALLBACK_MODES

    assert set(typing.get_args(LiveCallbackMode)) == set(_LIVE_CALLBACK_MODES)
    for mode in _LIVE_CALLBACK_MODES:
        assert _nsgaii_builder().live_callback_mode(mode).build().live_callback_mode == mode
    with pytest.raises(ConfigurationError, match="live_callback_mode"):
        _nsgaii_builder().live_callback_mode("all").build()
