"""Run one durable-study operation for VAMOS Studio in a process of its own.

Studio starts this module with ``python -m vamos.ux.studio.study_worker``. It
owns the study for its whole lifetime, like ``vamos study run`` does, refreshes
a heartbeat that Studio reads, and records how the operation ended.
"""

from __future__ import annotations

import _thread
import argparse
import logging
import os
import signal
import threading
import time
import traceback
import uuid
from collections.abc import Sequence
from pathlib import Path

from vamos.study_artifacts import load_study
from vamos.ux.studio.study_jobs import (
    HEARTBEAT_FILE,
    HEARTBEAT_SECONDS,
    OUTCOME_FILE,
    STOP_FILE,
    STOP_POLL_SECONDS,
    STOPPED_BY_RUNNER,
    _read_json,
    write_json,
)

STOPPED_EXIT_CODE = 3


class _StopRequested(BaseException):
    """Unwind the owning thread without cancelling the durable study."""


def _interrupt_worker(_signum: int, _frame: object) -> None:
    raise _StopRequested


def _beat(job_dir: Path, stop: threading.Event, worker_id: str) -> None:
    next_beat = 0.0
    requested = False
    while not stop.is_set():
        try:
            request = _read_json(job_dir / STOP_FILE)
            if not requested and request is not None and request.get("worker_id") == worker_id:
                requested = True
                # Schedule an exception in this process's owning thread. Unlike
                # os._exit(), unwinding executes atomic writers' cleanup before
                # main() acknowledges the stop. No signal is sent to a PID.
                _thread.interrupt_main(signal.SIGTERM)
            if time.monotonic() >= next_beat:
                write_json(job_dir / HEARTBEAT_FILE, {"pid": os.getpid(), "worker_id": worker_id, "time": time.time()})
                next_beat = time.monotonic() + HEARTBEAT_SECONDS
        except OSError:  # A missed beat only ages the heartbeat; the next one retries.
            logging.getLogger(__name__).debug("Cannot write the Studio heartbeat.", exc_info=True)
        stop.wait(STOP_POLL_SECONDS)


def run_operation(study_root: Path, operation: str, *, retry_failed: bool = False) -> str:
    """Execute ``operation`` and return the resulting study state."""
    study = load_study(study_root)
    if operation == "run":
        study = study.run()
    elif operation == "resume":
        study = study.resume(retry_failed=retry_failed)
    elif operation == "retry":
        study = study.retry(failed_only=True)
    else:
        raise ValueError(f"Unknown study operation: {operation!r}.")
    return str(study.inspect().as_dict().get("state", "unknown"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vamos.ux.studio.study_worker")
    parser.add_argument("--study", required=True, help="Study directory.")
    parser.add_argument("--job-dir", required=True, help="Studio job directory for heartbeat and outcome.")
    parser.add_argument("--operation", choices=("run", "resume", "retry"), required=True)
    parser.add_argument("--retry-failed", action="store_true", help="With resume, also retry failed tasks.")
    args = parser.parse_args(argv)

    job_dir = Path(args.job_dir)
    stop = threading.Event()
    heartbeat = threading.Thread(target=_beat, args=(job_dir, stop, uuid.uuid4().hex), name="vamos-studio-heartbeat", daemon=True)
    previous_handler = signal.signal(signal.SIGTERM, _interrupt_worker)
    try:
        try:
            heartbeat.start()
            state = run_operation(Path(args.study), args.operation, retry_failed=args.retry_failed)
        finally:
            stop.set()
            if heartbeat.ident is not None:
                heartbeat.join()
    except _StopRequested:
        outcome: dict[str, object] = {"status": "stopped", "message": STOPPED_BY_RUNNER, "time": time.time()}
        code = STOPPED_EXIT_CODE
    except BaseException as exc:  # Report every ending, including an interrupt.
        traceback.print_exc()
        outcome = {"status": "failed", "message": f"{type(exc).__name__}: {exc}", "time": time.time()}
        code = 1
    else:
        outcome = {"status": "finished", "message": f"the study is {state}", "state": state, "time": time.time()}
        code = 0
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    write_json(job_dir / OUTCOME_FILE, outcome)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
