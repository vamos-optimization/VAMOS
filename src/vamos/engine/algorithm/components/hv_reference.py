"""Hypervolume reference points relative to the objective range of a point set."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def relative_reference_point(F: ArrayLike, offset: float) -> NDArray[np.float64]:
    """``nadir + offset * (nadir - ideal)`` of the rows of ``F``, objective by objective.

    The point is the reference ``1 + offset`` of the objectives normalized to
    ``[0, 1]`` by the ideal and the nadir of ``F``, expressed in the original units.
    Rescaling an objective rescales the reference with it, so the order of the
    hypervolume contributions, and any selection that follows it, does not depend
    on the objective units. An objective with no spread uses a range of 1: all
    points then share that factor of their contributions.
    """
    values = np.asarray(F, dtype=float)
    top = np.max(values, axis=0)
    span = top - np.min(values, axis=0)
    span = np.where(span > 0.0, span, 1.0)
    return np.asarray(top + offset * span, dtype=np.float64)


__all__ = ["relative_reference_point"]
