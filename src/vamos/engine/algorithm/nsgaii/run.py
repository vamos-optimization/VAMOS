"""
Run loop and checkpoint helpers for NSGA-II.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.hooks import live_should_stop
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.eval.backends import EvaluationBackend

from .setup import initialize_run
from .state import build_result, finalize_genealogy, get_archive_contents

if TYPE_CHECKING:
    from .nsgaii import NSGAII


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


#: Default number of generations between checkpoint saves.
DEFAULT_CHECKPOINT_INTERVAL: int = 50


def notify_generation(
    algo: NSGAII,
    live_cb: LiveVisualization,
    generation: int,
    F: np.ndarray[Any, Any],
    problem: Any | None = None,
    evals: int | None = None,
) -> bool:
    """Notify live visualization of generation progress."""
    st = algo._st
    nd_mask = None
    ranks = None
    if st is not None and st.immigration_manager is not None and problem is not None:
        changed = bool(
            st.immigration_manager.apply_generation(
                generation=int(generation),
                state=st,
                problem=problem,
                kernel=algo.kernel,
            )
        )
        if changed:
            F = st.F
            if st.incremental_mode:
                st.ranks = None
                st.crowding = None
                st.fronts = None

    try:
        ranks, crowding = algo.kernel.nsga2_ranking(F)
        nd_mask = ranks == ranks.min(initial=0)
        if st is not None:
            st.publish_generation_ranking(F, ranks, crowding)
    except (ValueError, IndexError) as exc:
        _logger().debug("Failed to compute non-dominated front for viz: %s", exc)
        nd_mask = None

    stats: dict[str, Any] = {"evals": int(evals)} if evals is not None else {}
    if st is not None and st.immigration_manager is not None:
        ig = st.immigration_manager.stats_for_generation(int(generation))
        stats["immigration"] = {
            "events": int(ig.events),
            "inserted": int(ig.inserted),
            "replaced_indices": [int(i) for i in (ig.replaced_indices or [])],
            "replaced_pages": [int(p) for p in (ig.replaced_pages or [])],
            "mating_participation": int(ig.mating_participation),
        }

    live_mode = str(getattr(st, "live_callback_mode", "nd_only")).lower() if st is not None else "nd_only"
    if nd_mask is None:
        live_F = F
        live_X = st.X if st is not None else None
    elif live_mode in {"population", "population_archive"}:
        live_F = F
        live_X = st.X if st is not None else None
    else:
        live_F = F[nd_mask]
        live_X = st.X[nd_mask] if st is not None else None

    if st is not None and live_mode == "population_archive":
        archive_payload = get_archive_contents(st)
        if archive_payload is not None:
            stats["archive"] = {
                "X": np.asarray(archive_payload.get("X")),
                "F": np.asarray(archive_payload.get("F")),
            }

    if stats:
        live_cb.on_generation(generation, F=live_F, X=live_X, stats=stats)
    else:
        live_cb.on_generation(generation, F=live_F, X=live_X, stats=None)

    if st is not None and callable(st.generation_callback):
        try:
            copy_arrays = bool(st.generation_callback_copy)
            payload: dict[str, Any] = {
                "generation": int(generation),
                "evaluations": int(evals) if evals is not None else None,
                "population": {
                    "X": np.asarray(st.X).copy() if copy_arrays else np.asarray(st.X),
                    "F": np.asarray(st.F).copy() if copy_arrays else np.asarray(st.F),
                    "G": (
                        np.asarray(st.G).copy()
                        if copy_arrays and isinstance(st.G, np.ndarray)
                        else np.asarray(st.G)
                        if isinstance(st.G, np.ndarray)
                        else None
                    ),
                },
                "nondominated": None,
                "archive": None,
                "stats": stats or None,
            }
            if nd_mask is not None:
                payload["nondominated"] = {
                    "X": (np.asarray(st.X[nd_mask]).copy() if copy_arrays else np.asarray(st.X[nd_mask])),
                    "F": (np.asarray(st.F[nd_mask]).copy() if copy_arrays else np.asarray(st.F[nd_mask])),
                }
            archive_payload = get_archive_contents(st)
            if archive_payload is not None:
                x_arch = np.asarray(archive_payload.get("X"))
                f_arch = np.asarray(archive_payload.get("F"))
                payload["archive"] = {
                    "X": x_arch.copy() if copy_arrays else x_arch,
                    "F": f_arch.copy() if copy_arrays else f_arch,
                }
            callback_stop = bool(st.generation_callback(payload))
            if callback_stop:
                return True
        except Exception as exc:
            _logger().debug("generation_callback failed: %s", exc)

    return live_should_stop(live_cb)


def save_checkpoint(algo: NSGAII, checkpoint_dir: str, seed: int, generation: int, n_eval: int) -> None:
    """Save current state to checkpoint file."""
    from pathlib import Path

    from vamos.foundation.checkpoint import save_checkpoint

    st = algo._st
    if st is None:
        return

    path = Path(checkpoint_dir) / f"nsgaii_seed{seed}_gen{generation}.ckpt"
    save_checkpoint(
        path,
        X=st.X,
        F=st.F,
        generation=generation,
        n_eval=n_eval,
        rng_state=cast(dict[str, Any], st.rng.bit_generator.state),
        G=st.G,
        archive_X=st.archive_X,
        archive_F=st.archive_F,
        extra={
            "step": st.step,
            "replacements": st.replacements,
        },
    )
    _logger().info("Checkpoint saved: %s", path)


def run_nsgaii(
    algo: NSGAII,
    problem: Any,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None = None,
    live_viz: LiveVisualization | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the NSGA-II algorithm."""
    import signal

    live_cb, eval_strategy, max_eval, n_eval, hv_tracker = initialize_run(
        algo,
        problem,
        termination,
        seed,
        eval_strategy,
        live_viz,
        checkpoint=checkpoint,
    )
    st = algo._st
    assert st is not None, "State not initialized"

    interrupted = False
    main_thread_signals = threading.current_thread() is threading.main_thread()
    original_handler = signal.getsignal(signal.SIGINT) if main_thread_signals else None

    def _handle_interrupt(signum: int, frame: Any | None) -> None:
        nonlocal interrupted
        interrupted = True
        _logger().info("Interrupt received, finishing current generation...")

    if main_thread_signals:
        signal.signal(signal.SIGINT, _handle_interrupt)

    generation = st.generation
    step = st.step
    replacements = st.replacements
    stop_requested = notify_generation(
        algo,
        live_cb,
        generation,
        st.F,
        problem=problem,
        evals=n_eval,
    )
    hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn())

    try:
        while st.n_eval < max_eval and not hv_reached and not stop_requested and not interrupted:
            st.generation = generation
            st.step = step
            st.replacements = replacements
            X_off = algo.ask()
            eval_off = eval_strategy.evaluate(X_off, problem)
            hv_reached = algo.tell(eval_off)
            n_eval = st.n_eval
            replacements += X_off.shape[0]

            step += 1
            st.step = step
            st.replacements = replacements

            if st.incremental_mode:
                if not stop_requested:
                    stop_requested = live_should_stop(live_cb)
                new_generation = replacements // st.pop_size
                if new_generation != generation:
                    generation = new_generation
                    st.generation = generation
                    stop_requested = stop_requested or notify_generation(
                        algo,
                        live_cb,
                        generation,
                        st.F,
                        problem=problem,
                        evals=n_eval,
                    )
                    if hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn()):
                        hv_reached = True

                    if checkpoint_dir and generation % checkpoint_interval == 0:
                        save_checkpoint(algo, checkpoint_dir, seed, generation, n_eval)
            else:
                generation += 1
                st.generation = generation
                stop_requested = notify_generation(
                    algo,
                    live_cb,
                    generation,
                    st.F,
                    problem=problem,
                    evals=n_eval,
                )
                if hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn()):
                    hv_reached = True

                if checkpoint_dir and generation % checkpoint_interval == 0:
                    save_checkpoint(algo, checkpoint_dir, seed, generation, n_eval)
    finally:
        if main_thread_signals and original_handler is not None:
            signal.signal(signal.SIGINT, original_handler)

    result = build_result(st, st.n_eval, hv_reached, kernel=algo.kernel)
    result["interrupted"] = interrupted
    result["checkpoint"] = {
        "version": 1,
        "algorithm": "nsgaii",
        "X": st.X,
        "F": st.F,
        "G": st.G,
        "generation": st.generation,
        "n_eval": st.n_eval,
        "rng_state": cast(dict[str, Any], st.rng.bit_generator.state),
        "archive_X": st.archive_X,
        "archive_F": st.archive_F,
        "extra": {
            "step": st.step,
            "replacements": st.replacements,
        },
    }
    live_cb.on_end(final_F=st.F)
    finalize_genealogy(result, st, algo.kernel)
    return result


__all__ = ["run_nsgaii", "save_checkpoint", "notify_generation"]
