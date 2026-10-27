"""irace backend of ``ModelBasedTuner``.

VAMOS writes an irace scenario and runs ``irace`` through ``Rscript``. irace's targetRunner is
``_irace_target_runner.py``, a standard-library client launched with this Python interpreter; it
forwards each experiment over a local TCP socket to an evaluation server inside this process,
which calls the tuner's evaluation function and returns the cost (the negated score when the task
maximizes, because irace minimizes).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socketserver
import subprocess
import sys
import tempfile
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from ._irace_space import configurations_file, decode_switches, parameter_file
from ._model_backend_fidelity import ScoreEvalFn
from .racing.random_search_tuner import TrialResult
from .racing.tuning_task import EvalContext, Instance, TuningTask

INSTANCE_SEED_MODES = ("task", "irace")
MANAGED_OPTIONS = {
    "parameterFile",
    "configurationsFile",
    "trainInstancesDir",
    "trainInstancesFile",
    "execDir",
    "logFile",
    "targetRunner",
    "targetRunnerLauncher",
    "maxExperiments",
    "parallel",
    "seed",
    "deterministic",
}
# irace 4.4.4 fixed psRace() failing when the post-selection race has nothing left to run (#97),
# which happens whenever a deterministic race has used every task pair.
MIN_IRACE_VERSION = (4, 4, 4)
_VERSION_SCRIPT = 'cat(as.character(utils::packageVersion("irace")))\n'
_RUN_SCRIPT = "irace::irace_cmdline(commandArgs(trailingOnly = TRUE))\n"
_ELITES_SCRIPT = (
    'log <- irace::read_logfile(commandArgs(trailingOnly = TRUE)[1])\ncat(log$allElites[[length(log$allElites)]], sep = "\\n")\n'
)


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class IraceTunerLike(Protocol):
    task: TuningTask
    max_trials: int
    seed: int
    n_jobs: int
    timeout_seconds: float | None
    budget_levels: list[int] | None
    irace_max_experiments: int | None
    irace_rscript: str
    irace_work_dir: str | None
    irace_options: dict[str, Any] | None
    irace_initial_configs: list[dict[str, Any]] | None
    irace_instance_seeds: str

    def _score_to_loss(self, score: float) -> float: ...


def _r_value(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"irace options take booleans, numbers or strings, not {type(value).__name__}.")


def _resolve_run_settings(tuner: IraceTunerLike) -> tuple[int, int]:
    """Validate irace-specific settings before R starts; return (budget per run, maxExperiments)."""
    if tuner.timeout_seconds is not None:
        raise ValueError("The irace backend has no wall-clock limit; bound the search with irace_max_experiments.")
    levels = sorted({int(level) for level in tuner.budget_levels or []})
    if len(levels) > 1:
        raise ValueError("The irace backend evaluates one fidelity level; give budget_levels at most one value.")
    budget = int(tuner.task.budget_per_run)
    if levels:
        budget = min(budget, max(1, levels[0]))
    if tuner.irace_instance_seeds not in INSTANCE_SEED_MODES:
        raise ValueError(f"irace_instance_seeds must be one of {INSTANCE_SEED_MODES}, not {tuner.irace_instance_seeds!r}.")
    clash = sorted(MANAGED_OPTIONS & set(tuner.irace_options or {}))
    if clash:
        raise ValueError(f"irace_options cannot set {clash}; ModelBasedTuner manages them.")
    pairs = len(tuner.task.instances) * len(tuner.task.seeds)
    if pairs < 1:
        raise ValueError("The irace backend needs at least one instance and one seed.")
    max_experiments = tuner.max_trials * pairs if tuner.irace_max_experiments is None else tuner.irace_max_experiments
    if int(max_experiments) < 1:
        raise ValueError("irace_max_experiments must be positive.")
    return budget, int(max_experiments)


@contextmanager
def _work_dir(path: str | None) -> Generator[Path, None, None]:
    if path is None:
        with tempfile.TemporaryDirectory(prefix="vamos-irace-") as tmp:
            yield Path(tmp)
        return
    work = Path(path).resolve()
    if (work / "irace.Rdata").exists():
        raise FileExistsError(f"{work / 'irace.Rdata'} exists; give irace_work_dir a new directory.")
    work.mkdir(parents=True, exist_ok=True)
    yield work


def _rscript(tuner: IraceTunerLike, work: Path, script: str, name: str, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    (work / name).write_text(script, encoding="utf-8")
    # R runs inside the work directory, so a path with a directory part is resolved against ours first;
    # a bare name keeps the PATH lookup.
    rscript = os.path.abspath(tuner.irace_rscript) if os.path.dirname(tuner.irace_rscript) else tuner.irace_rscript
    try:
        # R writes messages in the platform's native encoding.
        return subprocess.run([rscript, str(work / name), *args], cwd=work, text=True, errors="replace", check=False, **kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Backend 'irace' requires Rscript; {tuner.irace_rscript!r} was not found.") from exc


def _check_irace(tuner: IraceTunerLike, work: Path) -> str:
    probe = _rscript(tuner, work, _VERSION_SCRIPT, "irace_version.R", capture_output=True)
    version = probe.stdout.strip()
    parts = version.split(".")[:3]
    numbers = tuple(int(part) for part in parts) + (0,) * (3 - len(parts)) if all(p.isdigit() for p in parts) else ()
    if probe.returncode != 0 or not numbers or numbers < MIN_IRACE_VERSION:
        required = ".".join(map(str, MIN_IRACE_VERSION))
        raise RuntimeError(
            f"Backend 'irace' requires the R package `irace` >= {required} (install.packages('irace')); "
            f"Rscript reported {version or probe.stderr.strip()!r}."
        )
    return version


class _Evaluations:
    """Serve irace experiments by calling the evaluation function; thread-safe bookkeeping."""

    def __init__(self, tuner: IraceTunerLike, eval_fn: ScoreEvalFn, budget: int, lines: dict[str, tuple[Instance, int | None]]):
        self.tuner, self.eval_fn, self.budget, self.lines = tuner, eval_fn, budget, lines
        self.records: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self.lock = threading.Lock()

    def cost(self, argv: list[str]) -> float:
        config_id, _instance_id, irace_seed, line, *switches = argv
        assignment = decode_switches(self.tuner.task.param_space, switches)
        instance, seed = self.lines[line]
        run_seed = int(irace_seed) if seed is None else seed
        ctx = EvalContext(instance=instance, seed=run_seed, budget=self.budget, fidelity_level=0, previous_budget=None)
        result = self.eval_fn(dict(assignment), ctx)
        score = float(result[0] if isinstance(result, tuple) else result)
        with self.lock:
            self.records.append(dict(config_id=int(config_id), config=assignment, instance=instance.name, seed=run_seed, score=score))
        return self.tuner._score_to_loss(score)


def _serve(evaluations: _Evaluations, token: str) -> socketserver.ThreadingTCPServer:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            request = json.loads(self.rfile.readline())
            if not secrets.compare_digest(str(request.get("token", "")), token):
                reply: dict[str, Any] = dict(error="Rejected an experiment with a wrong token.")
            else:
                try:
                    reply = dict(cost=evaluations.cost([str(arg) for arg in request["argv"]]))
                except Exception as exc:  # irace stops the race; the cause is re-raised after it exits
                    with evaluations.lock:
                        evaluations.error = evaluations.error or exc
                    reply = dict(error=f"{type(exc).__name__}: {exc}")
            self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="vamos-irace-server", daemon=True).start()
    return server


def _write_inputs(tuner: IraceTunerLike, work: Path, max_experiments: int) -> dict[str, tuple[Instance, int | None]]:
    space = tuner.task.param_space
    (work / "parameters.txt").write_text(parameter_file(space), encoding="utf-8")
    if tuner.irace_instance_seeds == "task":
        lines: dict[str, tuple[Instance, int | None]] = {
            f"vamos_pair_{k}": (instance, int(seed))
            for k, (instance, seed) in enumerate((i, s) for i in tuner.task.instances for s in tuner.task.seeds)
        }
    else:
        lines = {f"vamos_instance_{k}": (instance, None) for k, instance in enumerate(tuner.task.instances)}
    (work / "instances.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    (work / "exec").mkdir(exist_ok=True)
    options: dict[str, Any] = {
        "parameterFile": (work / "parameters.txt").as_posix(),
        "trainInstancesDir": "",
        "trainInstancesFile": (work / "instances.txt").as_posix(),
        "execDir": (work / "exec").as_posix(),
        "logFile": (work / "irace.Rdata").as_posix(),
        "targetRunner": Path(__file__).with_name("_irace_target_runner.py").as_posix(),
        "targetRunnerLauncher": Path(sys.executable).as_posix(),
        "maxExperiments": max_experiments,
        "parallel": max(1, int(tuner.n_jobs)),
        "seed": int(tuner.seed),
        # Task pairs already carry their seed, so each pair is evaluated once per configuration.
        "deterministic": tuner.irace_instance_seeds == "task",
    }
    if tuner.irace_instance_seeds == "task":
        # A deterministic race cannot start on more instances than the task has pairs.
        options["firstTest"] = min(5, len(lines))
    if tuner.irace_initial_configs:
        (work / "configurations.txt").write_text(configurations_file(space, tuner.irace_initial_configs), encoding="utf-8")
        options["configurationsFile"] = (work / "configurations.txt").as_posix()
    options.update(tuner.irace_options or {})
    (work / "scenario.txt").write_text("".join(f"{k} = {_r_value(v)}\n" for k, v in options.items()), encoding="utf-8")
    return lines


def _history(tuner: IraceTunerLike, records: list[dict[str, Any]], elites: list[int]) -> tuple[dict[str, Any], list[TrialResult]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["config_id"], []).append(record)
    history = [
        TrialResult(
            trial_id=config_id,
            config=dict(rows[0]["config"]),
            score=float(tuner.task.aggregator([row["score"] for row in rows])),
            details={
                "state": "COMPLETE",
                "evaluations": len(rows),
                "irace_elite_rank": elites.index(config_id) + 1 if config_id in elites else None,
            },
        )
        for config_id, rows in grouped.items()
    ]
    if not elites or elites[0] not in grouped:
        raise RuntimeError("irace finished without an evaluated elite configuration.")
    return dict(grouped[elites[0]][0]["config"]), history


def run_irace(tuner: IraceTunerLike, eval_fn: ScoreEvalFn) -> tuple[dict[str, Any], list[TrialResult]]:
    """Run irace for the tuner's task; the best configuration is irace's first final elite."""
    budget, max_experiments = _resolve_run_settings(tuner)
    with _work_dir(tuner.irace_work_dir) as work:
        version = _check_irace(tuner, work)
        lines = _write_inputs(tuner, work, max_experiments)
        evaluations = _Evaluations(tuner, eval_fn, budget, lines)
        token = secrets.token_hex(16)
        server = _serve(evaluations, token)
        env = dict(os.environ, VAMOS_IRACE_PORT=str(server.server_address[1]), VAMOS_IRACE_TOKEN=token)
        _logger().info("Running irace %s in %s (maxExperiments=%d).", version, work, max_experiments)
        try:
            with open(work / "irace.log", "w", encoding="utf-8") as log:
                run = _rscript(
                    tuner,
                    work,
                    _RUN_SCRIPT,
                    "run_irace.R",
                    "--scenario",
                    str(work / "scenario.txt"),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        finally:
            server.shutdown()
            server.server_close()
        if evaluations.error is not None:
            raise RuntimeError(f"An irace experiment failed: {evaluations.error}") from evaluations.error
        if run.returncode != 0:
            tail = "".join((work / "irace.log").read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-20:])
            raise RuntimeError(f"irace exited with status {run.returncode}:\n{tail}")
        listing = _rscript(tuner, work, _ELITES_SCRIPT, "irace_elites.R", str(work / "irace.Rdata"), capture_output=True)
        if listing.returncode != 0:
            raise RuntimeError(f"Could not read the irace log: {listing.stderr.strip()}")
        elites = [int(line) for line in listing.stdout.split()]
        return _history(tuner, evaluations.records, elites)


__all__ = ["run_irace"]
