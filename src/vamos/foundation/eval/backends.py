from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.exceptions import ConfigurationError, EvaluationError

from . import EvaluationBackend, EvaluationResult


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _eval_chunk(
    problem: Any,
    X_chunk: NDArray[np.generic],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Worker helper to evaluate a chunk; kept at module level for pickling."""
    F, G = evaluate_population_with_constraints(problem, X_chunk)
    return F, G


class SerialEvalBackend(EvaluationBackend):
    """Synchronous in-process evaluation (current default)."""

    def evaluate(self, X: NDArray[np.generic], problem: Any) -> EvaluationResult:
        F, G = evaluate_population_with_constraints(problem, X)
        return EvaluationResult(F=F, G=G)


class MultiprocessingEvalBackend(EvaluationBackend):
    """
    Parallel evaluation using multiprocessing.

    Notes:
        - Requires the problem instance to be picklable.
        - Best suited for expensive evaluations; overhead dominates for tiny problems.
    """

    def __init__(self, n_workers: int | None = None, chunk_size: int | None = None, timeout: float | None = None) -> None:
        self.n_workers = max(1, n_workers or os.cpu_count() or 1)
        self.chunk_size = chunk_size
        self.timeout = timeout

    def evaluate(self, X: NDArray[np.generic], problem: Any) -> EvaluationResult:
        if self.n_workers <= 1 or X.shape[0] <= 1:
            return SerialEvalBackend().evaluate(X, problem)

        n = X.shape[0]
        if self.chunk_size is not None and self.chunk_size > 0:
            chunk_size = self.chunk_size
        else:
            chunk_size = max(1, math.ceil(n / self.n_workers))
        slices = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

        F_parts: list[tuple[int, NDArray[np.float64]]] = []
        G_parts: list[tuple[int, NDArray[np.float64] | None]] = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            future_map = {ex.submit(_eval_chunk, problem, X[start:end]): (start, end) for start, end in slices}
            for fut in as_completed(future_map):
                start, end = future_map[fut]
                F_chunk, G_chunk = fut.result(timeout=self.timeout)
                F_parts.append((start, F_chunk))
                G_parts.append((start, G_chunk))

        # Restore original order
        if not F_parts:
            raise RuntimeError(
                "MultiprocessingEvalBackend: no results were collected from worker chunks. "
                "All futures may have failed or the pool was empty."
            )
        F = np.empty((n, F_parts[0][1].shape[1]), dtype=float)
        G_sample = G_parts[0][1]
        G_out: NDArray[np.float64] | None = None
        if G_sample is not None:
            G_out = np.empty((n, G_sample.shape[1]), dtype=float)
        for start, f_part in sorted(F_parts, key=lambda p: p[0]):
            F[start : start + f_part.shape[0]] = f_part
        missing_constraints = False
        if G_out is not None:
            for start, g_part in sorted(G_parts, key=lambda p: p[0]):
                if g_part is None:
                    missing_constraints = True
                    break
                G_out[start : start + g_part.shape[0]] = g_part
        if missing_constraints:
            raise ValueError(
                "MultiprocessingEvalBackend: one or more worker chunks returned no "
                "constraint data (G=None). Ensure all evaluations compute constraints."
            )

        return EvaluationResult(F=F, G=G_out)


class DaskEvalBackend(EvaluationBackend):
    """Experimental distributed evaluation using Dask.

    Dask is an optional third-party integration and is outside the VAMOS 1.0
    stable compatibility surface. When no explicit client or scheduler address
    is supplied, the backend attempts to discover the active
    ``dask.distributed.Client`` for the current process.

    Notes:
        - Requires ``dask.distributed``.
        - Raises on missing/unavailable Dask by default. Pass
          ``fallback_to_serial=True`` only to opt in explicitly to serial
          fallback.
    """

    def __init__(self, client: Any = None, address: str | None = None, *, fallback_to_serial: bool = False) -> None:
        """
        Initialize Dask backend.

        Parameters
        ----------
        client : Any, optional
            Existing ``dask.distributed.Client`` instance.
        address : str | None, optional
            Scheduler address used when ``client`` is not provided. When both
            ``client`` and ``address`` are omitted, an active client is
            discovered with ``dask.distributed.get_client`` when available.
        fallback_to_serial : bool, default False
            If True, evaluation falls back to ``SerialEvalBackend`` when Dask
            is unavailable or a scheduler call fails.
        """
        self.client = client
        self.address = address
        self.fallback_to_serial = bool(fallback_to_serial)
        self._connected = False
        self._logged_fallback = False
        self._owns_client = False

        if self.client is not None:
            self._connected = True
            return

        if not self.address:
            try:
                from dask.distributed import get_client
            except ImportError as exc:
                message = "'dask.distributed' is required for experimental Dask evaluation."
                if self.fallback_to_serial:
                    _logger().warning("%s Falling back to SerialEvalBackend.", message)
                    return
                raise ConfigurationError(
                    message,
                    suggestion='Install with: pip install "vamos-optimization[compute]".',
                ) from exc

            try:
                self.client = cast(Any, get_client)()
            except ValueError:
                _logger().debug("No active Dask client found; backend remains unconnected.")
                return
            self._connected = True
            self._owns_client = False
            return

        try:
            from dask.distributed import Client
        except ImportError as exc:
            message = "'dask.distributed' is required for DaskEvalBackend(address=...)."
            if self.fallback_to_serial:
                _logger().warning("%s Falling back to SerialEvalBackend.", message)
                return
            raise ConfigurationError(
                message,
                suggestion='Install with: pip install "vamos-optimization[compute]" or pass fallback_to_serial=True.',
            ) from exc

        self.client = cast(Any, Client)(self.address)
        self._owns_client = True
        self._connected = True

    def close(self) -> None:
        """Close the Dask client if this backend created it."""
        if self._owns_client and self.client is not None:
            try:
                self.client.close()
            except Exception:
                _logger().debug("Error closing Dask client.", exc_info=True)
            finally:
                self.client = None
                self._connected = False

    def evaluate(self, X: NDArray[np.generic], problem: Any) -> EvaluationResult:
        if not self._connected or (self.client is None and self.address is None):
            return self._fallback_or_raise(X, problem, "DaskEvalBackend is not connected to a Dask scheduler.")

        try:
            # Re-check client connection
            if self.client is None and self.address:
                try:
                    from dask.distributed import Client
                except ImportError as exc:
                    return self._fallback_or_raise(X, problem, "'dask.distributed' is required to connect by address.", exc)

                self.client = cast(Any, Client)(self.address)
                self._owns_client = True
                self._connected = True

            if self.client is None:
                return self._fallback_or_raise(X, problem, "DaskEvalBackend has no client after connection setup.")

            n = X.shape[0]

            # Determine worker count with fallback if scheduler is unreachable
            try:
                n_workers = len(self.client.scheduler_info()["workers"])
            except Exception as exc:
                return self._fallback_or_raise(X, problem, "DaskEvalBackend could not query scheduler workers.", exc)
            if n_workers <= 0:
                return self._fallback_or_raise(X, problem, "DaskEvalBackend scheduler reports zero workers.")
            chunk_size = max(1, math.ceil(n / n_workers))
            slices = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

            futures = []
            for start, end in slices:
                # Submit chunk
                fut = self.client.submit(_eval_chunk, problem, X[start:end])
                futures.append((start, fut))

            # Gather
            results = self.client.gather([f for _, f in futures])

            # Reassemble
            if not results:
                raise RuntimeError(
                    "DaskEvalBackend: no results returned from workers. All futures may have failed or the futures list was empty."
                )
            F_sample = results[0][0]
            G_sample = results[0][1]

            F = np.empty((n, F_sample.shape[1]), dtype=float)
            G = None
            if G_sample is not None:
                G = np.empty((n, G_sample.shape[1]), dtype=float)

            for i, (start, _) in enumerate(futures):
                f_chunk, g_chunk = results[i]
                end = start + f_chunk.shape[0]
                F[start:end] = f_chunk
                if G is not None:
                    if g_chunk is None:
                        raise ValueError(
                            "DaskEvalBackend: worker chunk returned no constraint data (G=None) "
                            "but earlier chunks did. Ensure all evaluations compute constraints."
                        )
                    G[start:end] = g_chunk

            return EvaluationResult(F=F, G=G)

        except Exception as exc:
            return self._fallback_or_raise(X, problem, "DaskEvalBackend evaluation failed.", exc)

    def _fallback_or_raise(
        self,
        X: NDArray[np.generic],
        problem: Any,
        message: str,
        exc: Exception | None = None,
    ) -> EvaluationResult:
        if self.fallback_to_serial:
            if not self._logged_fallback:
                _logger().warning("%s Falling back to SerialEvalBackend.", message, exc_info=exc is not None)
                self._logged_fallback = True
            return SerialEvalBackend().evaluate(X, problem)
        raise EvaluationError(f"{message} Pass fallback_to_serial=True to allow serial fallback.") from exc


def resolve_eval_strategy(
    name: str,
    *,
    n_workers: int | None = None,
    chunk_size: int | None = None,
    dask_address: str | None = None,
    dask_fallback_to_serial: bool = False,
) -> EvaluationBackend:
    _KNOWN = ("serial", "multiprocessing", "dask")
    key = (name or "serial").lower()
    if key == "multiprocessing":
        return MultiprocessingEvalBackend(n_workers=n_workers, chunk_size=chunk_size)
    if key == "dask":
        return DaskEvalBackend(address=dask_address, fallback_to_serial=dask_fallback_to_serial)
    if key != "serial":
        raise ConfigurationError(
            f"Unknown eval_strategy {name!r}.",
            suggestion=f"Valid strategies: {_KNOWN}.",
        )
    return SerialEvalBackend()


__all__ = [
    "EvaluationBackend",
    "SerialEvalBackend",
    "MultiprocessingEvalBackend",
    "DaskEvalBackend",
    "resolve_eval_strategy",
]
