from __future__ import annotations

import time
from typing import Any

import numpy as np

from ._problem_builder_security import (
    apply_process_limits,
    compile_constraint_function,
    compile_objective_function,
    normalize_resource_profile,
    require_trusted_local_code,
)

DEFAULT_RESOURCE_PROFILE = "basic"
PREVIEW_TIMEOUT_SECONDS = 10.0


def _run_preview_once(
    fn: Any,
    *,
    n_var: int,
    n_obj: int,
    bounds: list[tuple[float, float]],
    algorithm: str,
    budget: int,
    pop_size: int,
    seed: int,
    constraints: Any = None,
    n_constraints: int = 0,
    algorithm_config: Any = None,
    record_history: bool = False,
) -> dict[str, Any]:
    from vamos.foundation.problem.builder import make_problem
    from vamos.ux.studio.services import DynamicsCallback, _build_algorithm_config, _run_algorithm

    kw: dict[str, Any] = {}
    if constraints is not None and n_constraints > 0:
        kw["constraints"] = constraints
        kw["n_constraints"] = n_constraints
    problem = make_problem(fn, n_var=n_var, n_obj=n_obj, bounds=bounds, encoding="real", name="studio_preview", **kw)
    algo_cfg = (
        algorithm_config
        if algorithm_config is not None
        else _build_algorithm_config(algorithm, pop_size=pop_size, n_var=n_var, n_obj=n_obj)
    )
    callback = DynamicsCallback() if record_history else None
    t0 = time.perf_counter()
    result = _run_algorithm(
        problem,
        algorithm=algorithm,
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=seed,
        engine="numpy",
        live_viz=callback,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    F = result.get("F")
    X = result.get("X")
    if F is None:
        raise RuntimeError("Preview optimization returned no objectives.")
    return {
        "F": np.asarray(F),
        "X": np.asarray(X) if X is not None else None,
        "elapsed_ms": elapsed,
        "history": list(callback.history) if callback is not None else [],
    }


def _preview_worker(
    queue: Any,
    *,
    objective_code: str,
    constraint_code: str,
    n_var: int,
    n_obj: int,
    bounds: list[tuple[float, float]],
    algorithm: str,
    budget: int,
    pop_size: int,
    seed: int,
    n_constraints: int,
    timeout_seconds: float,
    resource_profile: str,
    algorithm_config: Any = None,
    record_history: bool = False,
) -> None:
    try:
        apply_process_limits(profile=resource_profile, timeout_seconds=timeout_seconds)
        fn = compile_objective_function(objective_code, trusted_local_code=True)
        constraints = None
        if constraint_code.strip() and n_constraints > 0:
            constraints = compile_constraint_function(constraint_code, trusted_local_code=True)
        payload = _run_preview_once(
            fn,
            n_var=n_var,
            n_obj=n_obj,
            bounds=bounds,
            algorithm=algorithm,
            budget=budget,
            pop_size=pop_size,
            seed=seed,
            constraints=constraints,
            n_constraints=n_constraints,
            algorithm_config=algorithm_config,
            record_history=record_history,
        )
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def run_preview_optimization(
    fn: Any,
    *,
    n_var: int,
    n_obj: int,
    bounds: list[tuple[float, float]],
    algorithm: str,
    budget: int,
    pop_size: int,
    seed: int,
    constraints: Any = None,
    n_constraints: int = 0,
    objective_code: str | None = None,
    constraint_code: str = "",
    timeout_seconds: float = PREVIEW_TIMEOUT_SECONDS,
    resource_profile: str = DEFAULT_RESOURCE_PROFILE,
    trusted_local_code: bool = False,
    algorithm_config: Any = None,
    record_history: bool = False,
    cancel: Any = None,
) -> dict[str, Any]:
    """Run reviewed local Python after an explicit trust acknowledgement.

    ``algorithm_config`` replaces the default configuration of ``algorithm``;
    ``record_history`` adds the population objectives of every generation to the
    payload under ``"history"``. Setting the ``threading.Event`` ``cancel``
    terminates the child process and raises ``InterruptedError``.
    """
    require_trusted_local_code(trusted_local_code)
    resource_profile = normalize_resource_profile(resource_profile)
    if objective_code is None or timeout_seconds <= 0:
        return _run_preview_once(
            fn,
            n_var=n_var,
            n_obj=n_obj,
            bounds=bounds,
            algorithm=algorithm,
            budget=budget,
            pop_size=pop_size,
            seed=seed,
            constraints=constraints,
            n_constraints=n_constraints,
            algorithm_config=algorithm_config,
            record_history=record_history,
        )

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_preview_worker,
        kwargs={
            "queue": result_queue,
            "objective_code": objective_code,
            "constraint_code": constraint_code,
            "n_var": n_var,
            "n_obj": n_obj,
            "bounds": bounds,
            "algorithm": algorithm,
            "budget": budget,
            "pop_size": pop_size,
            "seed": seed,
            "n_constraints": n_constraints,
            "timeout_seconds": timeout_seconds,
            "resource_profile": resource_profile,
            "algorithm_config": algorithm_config,
            "record_history": record_history,
        },
    )
    process.start()
    try:
        worker_result = _await_worker_result(result_queue, process, float(timeout_seconds), cancel=cancel)
    finally:
        result_queue.close()
        result_queue.join_thread()

    if not isinstance(worker_result, dict):
        raise RuntimeError("Preview worker returned an invalid result payload.")
    if not worker_result.get("ok", False):
        raise RuntimeError(str(worker_result.get("error", "Unknown preview worker error.")))
    payload = worker_result.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("Preview worker returned an invalid payload body.")
    payload["F"] = np.asarray(payload.get("F"))
    payload["X"] = np.asarray(payload["X"]) if payload.get("X") is not None else None
    payload["history"] = [np.asarray(F) for F in payload.get("history") or []]
    return payload


def _await_worker_result(result_queue: Any, process: Any, timeout_seconds: float, *, cancel: Any = None) -> object:
    """Read the worker's result before joining it.

    A child process does not exit until its queued payload has been read, so
    joining first would turn any large result into a timeout.
    """
    import queue as queue_mod

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            result = result_queue.get(timeout=0.2)
        except queue_mod.Empty:
            if cancel is not None and cancel.is_set():
                process.terminate()
                process.join(timeout=1.0)
                raise InterruptedError("Stopped from Studio.") from None
            if not process.is_alive():
                try:
                    result = result_queue.get(timeout=1.0)
                except queue_mod.Empty as exc:
                    if process.exitcode not in (0, None):
                        raise RuntimeError(f"Preview worker exited unexpectedly (exit code {process.exitcode}).") from exc
                    raise RuntimeError("Preview worker finished without returning a result.") from exc
                break
            if time.monotonic() >= deadline:
                process.terminate()
                process.join(timeout=1.0)
                raise TimeoutError(f"Preview timed out after {timeout_seconds:.1f}s. Try simpler code or a smaller budget.") from None
            continue
        break
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    return result


__all__ = ["run_preview_optimization"]
