"""
Checkpointing utilities for saving and resuming optimization runs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.random import Generator


def save_checkpoint(
    path: str | Path,
    *,
    X: NDArray[np.generic],
    F: NDArray[np.generic],
    generation: int,
    n_eval: int,
    rng_state: dict[str, Any],
    G: NDArray[np.generic] | None = None,
    archive_X: NDArray[np.generic] | None = None,
    archive_F: NDArray[np.generic] | None = None,
    archive_G: NDArray[np.generic] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save algorithm state to a checkpoint file.

    Parameters
    ----------
    path : str | Path
        Target file path. ``.ckpt`` is appended when no suffix is present.
    X, F : np.ndarray
        Population decision and objective arrays.
    generation : int
        Current generation number.
    n_eval : int
        Total number of evaluations performed so far.
    rng_state : dict[str, Any]
        State from ``rng.bit_generator.state``.
    G, archive_X, archive_F, archive_G : np.ndarray | None, optional
        Optional constraint and archive arrays; ``archive_G`` holds the archived rows' constraint values.
    extra : dict[str, Any] | None, optional
        Algorithm-specific checkpoint payload.

    Returns
    -------
    Path
        Written checkpoint path.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".ckpt")

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "version": 1,
        "X": X,
        "F": F,
        "G": G,
        "generation": generation,
        "n_eval": n_eval,
        "rng_state": rng_state,
        "archive_X": archive_X,
        "archive_F": archive_F,
        "archive_G": archive_G,
        "extra": extra or {},
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_checkpoint(path: str | Path, *, trusted: bool = False) -> dict[str, Any]:
    """
    Load algorithm state from a checkpoint file.

    Parameters
    ----------
    path : str | Path
        Checkpoint file path.
    trusted : bool, default False
        Must be ``True`` to deserialize pickle checkpoint data. Only load
        checkpoint files created by trusted code.

    Returns
    -------
    dict[str, Any]
        Loaded checkpoint payload.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if not trusted:
        raise ValueError("Checkpoint loading uses pickle; pass trusted=True only for trusted checkpoint files.")

    with open(path, "rb") as f:
        checkpoint = cast(dict[str, Any], pickle.load(f))

    version = checkpoint.get("version", 0)
    if version != 1:
        raise ValueError(f"Unsupported checkpoint version: {version}")

    return checkpoint


def restore_rng(rng: Generator, state: dict[str, Any]) -> None:
    """
    Restore RNG state from checkpoint.

    Parameters
    ----------
    rng : Generator
        NumPy random generator to restore.
    state : dict[str, Any]
        State dictionary from ``checkpoint["rng_state"]``.
    """
    rng.bit_generator.state = state


__all__ = ["save_checkpoint", "load_checkpoint", "restore_rng"]
