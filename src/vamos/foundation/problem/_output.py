"""The objective output that ``evaluate(X, out)`` writes."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
from numpy.typing import NDArray


def objective_output(out: MutableMapping[str, Any], n_rows: int, n_obj: int) -> NDArray[np.float64]:
    """Return ``out["F"]`` as an ``(n_rows, n_obj)`` array, creating it when needed.

    The algorithms allocate ``out["F"]`` before calling ``evaluate``, and the problem
    fills it in place. A direct call such as ``problem.evaluate(X, {})``, or one whose
    ``out["F"]`` has another shape, gets a new float array stored in ``out["F"]``.
    """
    F = out.get("F")
    if not isinstance(F, np.ndarray) or F.shape != (n_rows, n_obj):
        F = np.empty((n_rows, n_obj), dtype=float)
        out["F"] = F
    return F


__all__ = ["objective_output"]
