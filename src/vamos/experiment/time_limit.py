"""The ``("max_time", seconds)`` termination of ``optimize``: a wall-clock limit checked once per generation."""

from __future__ import annotations

import math
import numbers
import time
from collections.abc import Mapping
from typing import Any

from vamos.engine.algorithm.components.hooks import live_should_stop
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.observer import RunContext

#: The evaluation budget of a time limit without ``max_evaluations``: no run reaches it.
UNBOUNDED_EVALUATIONS = 2**62

# Read through the module so that tests can replace the clock.
_clock = time.monotonic


def parse_time_limit(value: object) -> tuple[float, int | None]:
    """``(seconds, max_evaluations)`` of a ``max_time`` termination value.

    The value is a positive finite number of seconds, or a mapping with ``seconds`` and an
    optional positive integer ``max_evaluations`` that also stops the run.
    """
    cap: object = None
    seconds: object = value
    if isinstance(value, Mapping):
        unknown = sorted(str(key) for key in value if key not in ("seconds", "max_evaluations"))
        if unknown or "seconds" not in value:
            raise ConfigurationError(
                f"The max_time termination takes seconds and an optional max_evaluations; got {sorted(map(str, value))}.",
                suggestion="Use ('max_time', 60) or ('max_time', {'seconds': 60, 'max_evaluations': 100000}).",
            )
        seconds, cap = value["seconds"], value.get("max_evaluations")
    if isinstance(seconds, bool) or not isinstance(seconds, numbers.Real) or not math.isfinite(float(seconds)) or float(seconds) <= 0:
        raise ConfigurationError(f"The max_time termination needs a positive finite number of seconds; got {seconds!r}.")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, numbers.Integral) or int(cap) <= 0):
        raise ConfigurationError(f"The max_evaluations of a max_time termination must be a positive integer; got {cap!r}.")
    return float(seconds), None if cap is None else int(cap)


class TimeLimit:
    """A live callback that asks the run to stop once its wall-clock time has passed.

    It forwards every call to the user's callback, whose ``should_stop`` is honored first;
    a subclass of ``NoOpLiveVisualization`` is a callback like any other.
    Algorithms check ``should_stop`` after each generation (MOEA/D and SMS-EMOA after each
    ``pop_size`` evaluations or step), so a run can overrun the limit by one generation.
    """

    def __init__(self, seconds: float, inner: LiveVisualization | None = None) -> None:
        self._deadline = _clock() + seconds
        self._inner = inner
        self.reached = False

    def on_start(self, ctx: RunContext | None = None) -> None:
        if self._inner is not None:
            self._inner.on_start(ctx)

    def on_generation(self, generation: int, F: Any = None, X: Any = None, stats: dict[str, Any] | None = None) -> None:
        if self._inner is not None:
            self._inner.on_generation(generation, F=F, X=X, stats=stats)

    def on_end(self, final_F: Any = None, final_stats: dict[str, Any] | None = None) -> None:
        if self._inner is not None:
            self._inner.on_end(final_F=final_F, final_stats=final_stats)

    def should_stop(self) -> bool:
        if self._inner is not None and live_should_stop(self._inner):
            return True
        if not self.reached and _clock() >= self._deadline:
            self.reached = True
        return self.reached


__all__ = ["UNBOUNDED_EVALUATIONS", "TimeLimit", "parse_time_limit"]
