"""Run one durable-study operation for VAMOS Studio in a process of its own.

Studio starts this module with ``python -m vamos.ux.studio.study_worker``. It
owns the study for its whole lifetime, like ``vamos study run`` does, refreshes
a heartbeat that Studio reads, and records how the operation ended.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import traceback
import uuid
from collections.abc import Sequence
from pathlib import Path

from vamos.study_artifacts import load_study
from vamos.ux.studio.study_jobs import HEARTBEAT_FILE, HEARTBEAT_SECONDS, OUTCOME_FILE, STOP_FILE, _read_json, write_json


def _beat(job_dir: Path, stop: threading.Event, worker_id: str) -> None:
    while not stop.is_set():
        try:
            request = _read_json(job_dir / STOP_FILE)
            if request is not None and request.get("worker_id") == worker_id:
                write_json(job_dir / OUTCOME_FILE, {"status": "stopped", "message": "Stopped from Studio.", "time": time.time()})
                # End this worker, including its optimization thread. The study
                # remains resumable, just as after external process termination.
                os._exit(0)
            write_json(job_dir / HEARTBEAT_FILE, {"pid": os.getpid(), "worker_id": worker_id, "time": time.time()})
        except OSError:  # A missed beat only ages the heartbeat; the next one retries.
            logging.getLogger(__name__).debug("Cannot write the Studio heartbeat.", exc_info=True)
        stop.wait(HEARTBEAT_SECONDS)


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
    heartbeat.start()
    try:
        state = run_operation(Path(args.study), args.operation, retry_failed=args.retry_failed)
    except BaseException as exc:  # Report every ending, including an interrupt.
        traceback.print_exc()
        write_json(job_dir / OUTCOME_FILE, {"status": "failed", "message": f"{type(exc).__name__}: {exc}", "time": time.time()})
        return 1
    finally:
        stop.set()
        heartbeat.join()
    write_json(job_dir / OUTCOME_FILE, {"status": "finished", "message": f"the study is {state}", "state": state, "time": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
