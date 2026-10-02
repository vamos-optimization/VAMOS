"""Worker process for GUI runs: ``python -m vamos.experiment.gui.worker JOB_DIR``.

The worker reads ``request.json``, runs ``optimize`` with a
:class:`~vamos.experiment.gui.events.ProgressEventWriter` as live-visualization
hook, stores the canonical run under ``JOB_DIR/run``, and finishes with an
``end`` event. Failures are reported as a ``failed`` end event and a nonzero
exit status; the traceback goes to ``worker.log``.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from .events import EVENTS_FILENAME, ProgressEventWriter
from .jobs import REQUEST_FILENAME, RUN_DIRNAME, RunRequest


def run_job(job_dir: str | Path) -> int:
    """Execute the request stored in ``job_dir`` and return the exit status."""
    from vamos.experiment.artifacts import save_result
    from vamos.experiment.unified import optimize

    root = Path(job_dir)
    writer = ProgressEventWriter(root / EVENTS_FILENAME)
    try:
        request = RunRequest.from_json(json.loads((root / REQUEST_FILENAME).read_text(encoding="utf-8")))
        request.validate()
    except (OSError, ValueError, TypeError) as exc:
        writer.write_end("failed", error=f"Invalid run request: {exc}")
        sys.stderr.write(traceback.format_exc())
        return 2
    writer = ProgressEventWriter(root / EVENTS_FILENAME, request=request.to_json())
    writer.on_start()
    try:
        result = optimize(
            request.problem,
            algorithm=request.algorithm,
            max_evaluations=request.max_evaluations,
            pop_size=request.pop_size,
            engine=request.engine,
            seed=request.seed,
            live_viz=writer,
        )
        save_result(result, root / RUN_DIRNAME)
    except Exception as exc:  # noqa: BLE001 - every failure must reach the interface
        writer.write_end("failed", error=f"{type(exc).__name__}: {exc}")
        sys.stderr.write(traceback.format_exc())
        return 1
    evaluations = result.data.get("evaluations") if isinstance(result.data, dict) else None
    writer.write_end("succeeded", run_dir=RUN_DIRNAME, evaluations=evaluations if isinstance(evaluations, int) else None)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from vamos.foundation.logging import configure_vamos_logging

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: python -m vamos.experiment.gui.worker JOB_DIR\n")
        return 2
    configure_vamos_logging(level=logging.WARNING)
    return run_job(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
