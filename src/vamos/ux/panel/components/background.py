"""Run slow reads off the Panel event loop so that a Studio page stays responsive."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, TypeVar

import panel as pn

T = TypeVar("T")


class BackgroundWork(Generic[T]):
    """One job at a time: run ``work`` in a thread, then ``apply`` its result on the session.

    ``apply`` receives the result, or the exception ``work`` raised. A request
    made while a job runs replaces any earlier waiting request and starts when
    the job ends, so the latest request always runs. Without a server session,
    as in tests, the work runs inline.
    """

    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor | None = None
        self._pending: tuple[Callable[[], T], Callable[[T | Exception], None]] | None = None
        self.busy = False

    def submit(self, work: Callable[[], T], apply: Callable[[T | Exception], None]) -> bool:
        """Start ``work``; ``False`` when it waits for the running job to end."""
        if self.busy:
            self._pending = (work, apply)
            return False
        document = pn.state.curdoc
        if document is None or document.session_context is None:
            apply(_run(work))
            return True
        self.busy = True
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vamos-studio")

        def finish(result: T | Exception) -> None:
            self.busy = False
            apply(result)
            if self._pending is not None:
                pending, self._pending = self._pending, None
                self.submit(*pending)

        def run() -> None:
            result = _run(work)
            document.add_next_tick_callback(lambda: finish(result))

        self._executor.submit(run)
        return True

    def shutdown(self) -> None:
        self._pending = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


class BackgroundRun(Generic[T]):
    """Run one long job in a thread, refresh the page while it runs, and report its end.

    ``on_tick`` runs on the session every ``period_ms`` while the job runs;
    ``on_done`` receives the result or the exception. Without a server
    session, as in tests, the job runs inline, then one tick and ``on_done`` run.
    """

    def __init__(self, *, on_tick: Callable[[], None], on_done: Callable[[T | Exception], None], period_ms: int = 1000) -> None:
        self._on_tick = on_tick
        self._on_done = on_done
        self._period_ms = period_ms
        self._ticker: Any = None
        self.running = False

    def start(self, work: Callable[[], T]) -> bool:
        """Start ``work``; ``False`` when a job is already running."""
        if self.running:
            return False
        document = pn.state.curdoc
        if document is None or document.session_context is None:
            result = _run(work)
            self._on_tick()
            self._on_done(result)
            return True
        self.running = True
        self._ticker = pn.state.add_periodic_callback(self._on_tick, period=self._period_ms)

        def run() -> None:
            result = _run(work)
            document.add_next_tick_callback(lambda: self._finish(result))

        threading.Thread(target=run, name="vamos-studio-run", daemon=True).start()
        return True

    def _finish(self, result: T | Exception) -> None:
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None
        self.running = False
        self._on_tick()
        self._on_done(result)


def _run(work: Callable[[], T]) -> T | Exception:
    try:
        return work()
    except Exception as exc:  # Reported to the page by ``apply``.
        return exc


__all__ = ["BackgroundRun", "BackgroundWork"]
