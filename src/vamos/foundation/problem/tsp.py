# problem/tsp.py
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem

from .tsplib import load_tsplib_coords


def _default_coordinates() -> np.ndarray[Any, Any]:
    # Simple layout with a few asymmetric points to avoid symmetry degeneracy.
    return np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.3, 0.8),
            (0.5, 1.5),
            (-0.3, 1.0),
            (-0.6, 0.2),
        ],
        dtype=float,
    )


def _circle_coordinates(n_cities: int) -> np.ndarray[Any, Any]:
    if n_cities < 3:
        raise ValueError("TSP problems require at least 3 cities.")
    angles = np.linspace(0.0, 2.0 * math.pi, num=n_cities, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _validate_routes(routes: np.ndarray[Any, Any], n_cities: int) -> None:
    expected = np.arange(n_cities, dtype=int)
    if np.any(routes < 0) or np.any(routes >= n_cities):
        raise ValueError(f"TSP routes must contain city labels in [0, {n_cities - 1}].")
    for row in routes:
        if not np.array_equal(np.sort(row), expected):
            raise ValueError("TSP routes must be valid permutations of each city exactly once.")


def _coerce_routes(X: np.ndarray[Any, Any], n_cities: int) -> np.ndarray[Any, Any]:
    try:
        raw_routes = np.asarray(X)
    except ValueError as exc:
        raise ValueError(f"Expected routes of shape (N, {n_cities}).") from exc
    if raw_routes.ndim != 2 or raw_routes.shape[1] != n_cities:
        raise ValueError(f"Expected routes of shape (N, {n_cities}).")
    try:
        numeric_routes = raw_routes.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("TSP routes must contain integer city labels.") from exc
    if not np.all(np.isfinite(numeric_routes)) or not np.all(numeric_routes == np.floor(numeric_routes)):
        raise ValueError("TSP routes must contain integer city labels.")
    if np.any(numeric_routes < 0) or np.any(numeric_routes >= n_cities):
        raise ValueError(f"TSP routes must contain city labels in [0, {n_cities - 1}].")
    routes = numeric_routes.astype(int)
    _validate_routes(routes, n_cities)
    return routes


class TSPProblem(Problem):
    """
    Toy multi-objective travelling salesman problem.
    Objective 1: total tour length (closed tour).
    Objective 2: maximum edge length in the tour (encourages balanced legs).
    """

    def __init__(
        self,
        n_cities: int | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        dataset: str | None = None,
    ) -> None:
        if dataset is not None:
            coords = load_tsplib_coords(dataset)
        elif coordinates is not None:
            coords = np.asarray(coordinates, dtype=float)
        elif n_cities is not None:
            coords = _circle_coordinates(int(n_cities))
        else:
            coords = _default_coordinates()
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Coordinates must be an array-like of shape (n_cities, 2).")
        self.coordinates = coords
        self.n_var = coords.shape[0]
        self.n_obj = 2
        self.xl = 0
        self.xu = self.n_var - 1
        self.encoding = "permutation"
        self.labels = [f"City {i}" for i in range(self.n_var)]

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        routes = _coerce_routes(X, self.n_var)
        coords = self.coordinates
        paths = coords[routes]
        closed_paths = np.concatenate([paths, paths[:, :1, :]], axis=1)
        deltas = np.diff(closed_paths, axis=1)
        edges = np.linalg.norm(deltas, axis=2)
        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = edges.sum(axis=1)
        F[:, 1] = edges.max(axis=1)

    def describe(self) -> dict[str, Iterable[str] | int]:
        return {
            "n_var": self.n_var,
            "cities": self.labels,
        }
