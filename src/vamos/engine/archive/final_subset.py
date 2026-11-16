"""The final subset of an external archive that a result reports (``ExternalArchiveConfig.final_selection``)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def select_final_subset(F: NDArray[np.float64], selection: str, size: int, *, rng_seed: int = 0) -> NDArray[np.int_]:
    """Indices of the ``size`` rows of the archive's objectives ``F`` that ``selection`` keeps.

    ``selection`` is ``"farthest"``, ``"crowding"``, ``"hv_greedy"`` or ``"kmeans"``, the functions of
    :mod:`vamos.engine.algorithm.components.subset_selection` of the same name; ``kmeans`` is seeded with
    ``rng_seed``, so the same archive gives the same subset. An archive of at most ``size`` rows is kept
    whole, in its order.
    """
    from vamos.engine.algorithm.components.subset_selection import (
        select_top_k_crowding,
        select_top_k_farthest,
        select_top_k_hv_greedy,
        select_top_k_kmeans,
    )

    values = np.asarray(F, dtype=float)
    if values.shape[0] <= size:
        return np.arange(values.shape[0], dtype=int)
    if selection == "crowding":
        return np.asarray(select_top_k_crowding(values, size), dtype=int)
    if selection == "farthest":
        return np.asarray(select_top_k_farthest(values, size), dtype=int)
    if selection == "hv_greedy":
        return np.asarray(select_top_k_hv_greedy(values, size), dtype=int)
    if selection == "kmeans":
        return np.asarray(select_top_k_kmeans(values, size, rng=int(rng_seed)), dtype=int)
    raise ValueError(f"Unknown final_selection {selection!r}: expected farthest, crowding, hv_greedy or kmeans.")


__all__ = ["select_final_subset"]
