"""Generate the permanent VAMOS 1.0 stable API and command snapshots."""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "tests" / "compatibility" / "v1_0_0"

STABLE_API: dict[str, tuple[str, ...]] = {
    "vamos": (
        "__version__",
        "Problem",
        "make_problem",
        "make_problem_selection",
        "available_problem_names",
        "optimize",
        "OptimizationResult",
        "StudyResult",
        "save_result",
        "load_run",
        "load_result",
        "verify_run",
        "reproduce",
        "StoredRun",
        "RunManifest",
        "LoadLimits",
        "IncompleteRunMetadataError",
        "CompatibilityReport",
        "VerificationReport",
        "ReplayReport",
        "StudySpec",
        "StudyLoadLimits",
        "StudyPlanReport",
        "Study",
        "StudyReport",
        "StudySummary",
        "plan_study",
        "create_study",
        "load_study",
    ),
    "vamos.api": (
        "Problem",
        "make_problem",
        "make_problem_selection",
        "available_problem_names",
        "optimize",
        "OptimizationResult",
        "StudyResult",
        "save_result",
        "load_run",
        "load_result",
        "verify_run",
        "reproduce",
        "StoredRun",
        "RunManifest",
        "LoadLimits",
        "IncompleteRunMetadataError",
        "CompatibilityReport",
        "VerificationReport",
        "ReplayReport",
        "StudySpec",
        "StudyLoadLimits",
        "StudyPlanReport",
        "Study",
        "StudyReport",
        "StudySummary",
        "plan_study",
        "create_study",
        "load_study",
    ),
    "vamos.algorithms": (
        "NSGAIIConfig",
        "NSGAIIIConfig",
        "MOEADConfig",
        "SMSEMOAConfig",
        "SMPSOConfig",
        "SPEA2Config",
        "IBEAConfig",
        "AGEMOEAConfig",
        "RVEAConfig",
        "ProbabilityExpression",
        "ProbabilityValue",
        "available_algorithms",
        "available_crossover_methods",
        "available_mutation_methods",
    ),
    "vamos.problems": (
        "CEC2009CF1",
        "CEC2009UF1",
        "CEC2009UF2",
        "CEC2009UF3",
        "DTLZ1",
        "DTLZ2",
        "DTLZ3",
        "DTLZ4",
        "DTLZ7",
        "FeatureSelectionProblem",
        "HyperparameterTuningProblem",
        "TSP",
        "WFG1",
        "WFG2",
        "WFG3",
        "WFG4",
        "WFG5",
        "WFG6",
        "WFG7",
        "WFG8",
        "WFG9",
        "WeldedBeamDesignProblem",
        "ZDT1",
        "ZDT2",
        "ZDT3",
        "ZDT4",
        "ZDT6",
    ),
    "vamos.run_artifacts": (
        "CompatibilityReport",
        "IncompleteRunMetadataError",
        "LoadLimits",
        "ReplayReport",
        "RunManifest",
        "StoredRun",
        "VerificationReport",
        "load_result",
        "load_run",
        "reproduce",
        "save_result",
        "verify_run",
    ),
    "vamos.study_artifacts": (
        "Study",
        "StudyLoadLimits",
        "StudyPlanReport",
        "StudyReport",
        "StudySpec",
        "StudySummary",
        "create_study",
        "load_study",
        "plan_study",
    ),
}

STABLE_CLI: dict[str, tuple[str, ...]] = {
    "vamos": (
        "--config",
        "--validate-config",
        "--algorithm",
        "--engine",
        "--output-root",
        "--problem",
        "--max-evaluations",
        "--population-size",
        "--offspring-population-size",
        "--seed",
    ),
    "vamos results inspect": ("run_dir", "--json"),
    "vamos results verify": ("run_dir", "--json", "--require-level"),
    "vamos reproduce": ("run_dir", "--output", "--json"),
    "vamos study plan": ("config", "--output", "--json"),
    "vamos study create": ("config", "--output", "--json"),
    "vamos study run": ("study_dir", "--json"),
    "vamos study inspect": ("study_dir", "--json"),
    "vamos study resume": ("study_dir", "--retry-failed", "--json"),
    "vamos study retry": ("study_dir", "--failed", "--json"),
    "vamos study summarize": ("study_dir", "--format", "--output", "--json"),
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _signature(obj: Any) -> str | None:
    """Parameter names, kinds and defaults of ``obj``; annotations are left to the type checkers."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    parameters = [parameter.replace(annotation=inspect.Parameter.empty) for parameter in signature.parameters.values()]
    return str(signature.replace(parameters=parameters, return_annotation=inspect.Signature.empty))


def _public_methods(cls: type) -> dict[str, Any]:
    """Public methods that VAMOS defines on ``cls`` or its VAMOS bases (properties excluded)."""
    methods: dict[str, Any] = {}
    for name in sorted(dir(cls)):
        owner = next((base for base in cls.__mro__ if name in vars(base)), None)
        if name.startswith("_") or owner is None or not owner.__module__.startswith("vamos"):
            continue
        member = getattr(cls, name)
        if not isinstance(vars(owner)[name], property) and callable(member):
            methods[name] = member
    return methods


def _stable_signatures(modules: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Signatures of the stable callables and of the public methods of the stable classes.

    An object exported by several facades is recorded once, under the first facade name.
    """
    signatures: dict[str, str] = {}
    seen: set[int] = set()
    for module_name, names in sorted(modules.items()):
        module = importlib.import_module(module_name)
        for name in names:
            obj = getattr(module, name)
            # Functions and classes only: a type alias has no signature, and whether typing makes one
            # callable changes with the Python version (a union alias is callable before 3.14, not after).
            if id(obj) in seen or not (inspect.isroutine(obj) or inspect.isclass(obj)):
                continue
            seen.add(id(obj))
            members = {"": obj, **(_public_methods(obj) if inspect.isclass(obj) else {})}
            for member_name, member in members.items():
                signature = _signature(member)
                if signature is not None:
                    signatures[".".join(filter(None, (module_name, name, member_name)))] = signature
    return signatures


def _stable_api_snapshot() -> dict[str, Any]:
    modules: dict[str, list[str]] = {}
    for module_name, names in sorted(STABLE_API.items()):
        module = importlib.import_module(module_name)
        public = set(getattr(module, "__all__", ()))
        missing = [name for name in names if not hasattr(module, name) or name not in public]
        if missing:
            raise RuntimeError(f"{module_name} is missing stable exports: {', '.join(missing)}")
        modules[module_name] = sorted(names)
    return {
        "document_type": "vamos.stable-public-api",
        "schema_version": "1.0.0",
        "modules": modules,
        "signatures": _stable_signatures(modules),
    }


class _HelpRequested(BaseException):
    """Raised in place of printing help, to capture the parser that answers ``--help``."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__()
        self.parser = parser


def _parser_arguments(argv: Sequence[str]) -> list[str]:
    """Option strings and positional names of the parser that answers ``vamos <argv> --help``.

    Runs in the child process of ``_cli_arguments``. It replaces ``print_help`` and runs the real
    command line, so the names come from the parser's actions, not from its help text, where the
    description of another option can still mention a removed one.
    """
    from vamos.experiment.cli import main as cli_main

    def capture(parser: argparse.ArgumentParser, file: object = None) -> None:
        raise _HelpRequested(parser)

    argparse.ArgumentParser.print_help = capture
    sys.argv = ["vamos", *argv, "--help"]
    try:
        cli_main.main()
    except _HelpRequested as requested:
        names: set[str] = set()
        for action in requested.parser._actions:
            names.update(action.option_strings or [action.dest])
        return sorted(names)
    raise RuntimeError(f"vamos {' '.join(argv)} --help printed no parser help")


def _cli_arguments(command: str) -> frozenset[str]:
    """Arguments of a stable command, read from its argparse parser in a clean child process."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--cli-arguments", *command.split()[1:]],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Reading the arguments of `{command}` failed: {result.stderr}")
    return frozenset(json.loads(result.stdout))


def _stable_cli_snapshot() -> dict[str, Any]:
    commands: dict[str, Any] = {}
    for command, arguments in STABLE_CLI.items():
        available = _cli_arguments(command)
        missing = [argument for argument in arguments if argument not in available]
        if missing:
            raise RuntimeError(f"{command} is missing stable arguments: {', '.join(missing)}")
        commands[command] = {"arguments": sorted(arguments)}
    return {
        "document_type": "vamos.stable-cli-tree",
        "schema_version": "1.0.0",
        "commands": commands,
        "json_stdout_documents": 1,
        "diagnostics_stream": "stderr",
    }


def _algorithm_snapshot() -> dict[str, Any]:
    from vamos.engine.algorithm.registry import get_builtin_algorithm_names

    algorithms = importlib.import_module("vamos.algorithms")
    config_names = (
        "AGEMOEAConfig",
        "IBEAConfig",
        "MOEADConfig",
        "NSGAIIConfig",
        "NSGAIIIConfig",
        "RVEAConfig",
        "SMPSOConfig",
        "SMSEMOAConfig",
        "SPEA2Config",
    )
    classes = {name: getattr(algorithms, name) for name in config_names}
    configs = {name: [field.name for field in dataclasses.fields(cls)] for name, cls in classes.items()}
    return {
        "document_type": "vamos.stable-algorithm-configs",
        "schema_version": "1.0.0",
        # The built-in identifiers: available_algorithms() also lists the plugins installed here.
        "algorithm_ids": list(get_builtin_algorithm_names()),
        "config_fields": configs,
        "config_field_defaults": {name: _field_defaults(cls) for name, cls in classes.items()},
        "default_configs": {name: _json_value(cls.default().to_dict()) for name, cls in classes.items()},
    }


def _json_value(value: object) -> object:
    """``value`` as JSON data: tuples become lists, and other objects their ``repr``."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _field_defaults(cls: type) -> dict[str, object]:
    """Defaults of the optional fields of a configuration dataclass."""
    defaults: dict[str, object] = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = _json_value(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = _json_value(field.default_factory())
    return defaults


def _schema_snapshot() -> dict[str, Any]:
    from vamos.experiment.artifacts.manifest import DOCUMENT_TYPE as RUN_DOCUMENT_TYPE
    from vamos.experiment.artifacts.manifest import SCHEMA_VERSION as RUN_SCHEMA_VERSION
    from vamos.experiment.cli.study_command_result import COMMAND_DOCUMENT_TYPE, COMMAND_SCHEMA_VERSION
    from vamos.experiment.study.models import SCHEMA_VERSION as STUDY_SCHEMA_VERSION

    schemas = {
        RUN_DOCUMENT_TYPE: RUN_SCHEMA_VERSION,
        "vamos.environment": "1.0.0",
        "vamos.resolved-study-plan": STUDY_SCHEMA_VERSION,
        "vamos.study-attempt": STUDY_SCHEMA_VERSION,
        "vamos.study-event": STUDY_SCHEMA_VERSION,
        "vamos.study-manifest": STUDY_SCHEMA_VERSION,
        "vamos.study-report": STUDY_SCHEMA_VERSION,
        "vamos.study-spec": STUDY_SCHEMA_VERSION,
        "vamos.study-summary": STUDY_SCHEMA_VERSION,
        "vamos.study-task": STUDY_SCHEMA_VERSION,
        COMMAND_DOCUMENT_TYPE: COMMAND_SCHEMA_VERSION,
    }
    envelopes = {
        "vamos.replay-report": "1",
        "vamos.run-command-error": "1",
        "vamos.run-inspection": "1",
        "vamos.verification-report": "1",
    }
    return {
        "document_type": "vamos.stable-schema-inventory",
        "schema_version": "1.0.0",
        "artifact_schemas": schemas,
        "command_envelopes": envelopes,
    }


def _one_code(category: str, codes: Sequence[int]) -> int:
    distinct = sorted(set(codes))
    if len(distinct) != 1:
        raise RuntimeError(f"The outcomes of exit-code category {category!r} map to several codes: {distinct}")
    return distinct[0]


def _usage_exit_code(command: Callable[[list[str]], object]) -> int:
    """Exit code of a command line that the command's parser rejects."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            command(["--not-an-option"])
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 1
    raise RuntimeError("A command line with an unknown option did not exit")


def _exit_code_snapshot() -> dict[str, Any]:
    """Exit codes read from the code that maps outcomes to codes, one example outcome per kind."""
    from vamos.experiment.artifacts import errors as run_errors
    from vamos.experiment.cli import run_artifact_cli, study, study_command_result
    from vamos.experiment.study import errors as study_errors

    def example(cls: type[Any], reason: str = "example") -> Any:
        # The mappings only test the type and the reason, so the example skips each class's own constructor.
        error = cls.__new__(cls)
        error.reason = reason
        return error

    def run(*classes: type[Any]) -> list[int]:
        return [run_artifact_cli.run_artifact_exit_code(example(cls)) for cls in classes]

    def study_errors_codes(*classes: type[Any], reason: str = "example") -> list[int]:
        return [study_command_result.map_exit_code("run", example(cls, reason)) for cls in classes]

    def study_states(*states: str) -> list[int]:
        return [study_command_result.map_exit_code("run", state) for state in states]

    run_codes = {
        "success": 0,
        "usage": _one_code("usage", [_usage_exit_code(run_artifact_cli.run_results), _usage_exit_code(run_artifact_cli.run_reproduce)]),
        "integrity_or_resource_or_path": _one_code(
            "integrity_or_resource_or_path",
            run(
                run_errors.ArtifactIntegrityError,
                run_errors.ArtifactMissingError,
                run_errors.ArtifactResourceLimitError,
                run_errors.DuplicateJSONKeyError,
                run_errors.MalformedResultBundleError,
                run_errors.UnsafeArtifactPathError,
            ),
        ),
        "schema_or_layout": _one_code(
            "schema_or_layout",
            run(run_errors.ManifestValidationError, run_errors.UnsupportedSchemaError, run_errors.UnsupportedArtifactLayoutError),
        ),
        "verification_requirement": _one_code(
            "verification_requirement", run(run_errors.VerificationRequirementError, run_errors.EnvironmentIncompatibilityError)
        ),
        "replay_unavailable": _one_code(
            "replay_unavailable",
            run(
                run_errors.ReplayUnavailableError,
                run_errors.ComponentNotReconstructableError,
                run_errors.ResolvedSpecMismatchError,
                run_errors.UnsupportedReplayProviderError,
            ),
        ),
        "replay_execution_or_mismatch": _one_code(
            "replay_execution_or_mismatch", run(run_errors.ReplayExecutionError, run_errors.ReplayResultMismatchError)
        ),
        "output_collision": _one_code("output_collision", run(run_errors.OutputCollisionError)),
    }
    study_codes = {
        "success": _one_code("success", study_states("completed")),
        "usage_or_invalid_spec": _one_code(
            "usage_or_invalid_spec",
            [
                _usage_exit_code(study.study_main),
                *study_errors_codes(study_errors.InvalidStudySpecError, study_errors.UnresolvedStudyTaskError),
            ],
        ),
        "malformed_or_integrity": _one_code(
            "malformed_or_integrity",
            study_errors_codes(study_errors.MalformedStudyError, study_errors.StudyIntegrityError, study_errors.MissingStudyDocumentError),
        ),
        "schema_or_state_or_retry": _one_code(
            "schema_or_state_or_retry",
            study_errors_codes(
                study_errors.UnsupportedStudySchemaError,
                study_errors.PlanMismatchError,
                study_errors.InvalidStudyTransitionError,
                study_errors.UnsupportedStudyExecutionStateError,
                study_errors.RetryNotAllowedError,
            ),
        ),
        "collision_or_ownership": _one_code(
            "collision_or_ownership",
            [
                *study_errors_codes(study_errors.StudyOutputCollisionError),
                *study_errors_codes(study_errors.StudyError, reason="ACTIVE_LEASE"),
                *study_states("blocked"),
            ],
        ),
        "paused_or_completed_with_failures": _one_code(
            "paused_or_completed_with_failures", study_states("paused", "completed_with_failures")
        ),
        "infrastructure": _one_code(
            "infrastructure",
            study_errors_codes(study_errors.StudyInfrastructureError, study_errors.ResumeEnvironmentIncompatibilityError),
        ),
        "cancelled_or_interrupted": _one_code("cancelled_or_interrupted", study_states("cancelled", "interrupted")),
    }
    return {
        "document_type": "vamos.stable-exit-codes",
        "schema_version": "1.0.0",
        "run": run_codes,
        "study": study_codes,
    }


def snapshots() -> Mapping[str, object]:
    return {
        "stable_algorithm_configs.json": _algorithm_snapshot(),
        "stable_cli_tree.json": _stable_cli_snapshot(),
        "stable_exit_codes.json": _exit_code_snapshot(),
        "stable_public_api.json": _stable_api_snapshot(),
        "stable_schemas.json": _schema_snapshot(),
    }


CONFIGS_SNAPSHOT = "stable_algorithm_configs.json"
API_SNAPSHOT = "stable_public_api.json"
_CONFIG_SECTIONS = ("config_fields", "config_field_defaults", "default_configs")


def _added_config_fields(frozen: Mapping[str, Any], generated: Mapping[str, Any]) -> dict[str, list[str]] | None:
    """The fields that each stable configuration class adds after its frozen 1.0 fields.

    Stable configuration classes may gain optional fields during 1.x (see
    ``docs/project/stability-and-versioning.md``). A new field goes after the 1.0 ones, so that the
    constructor keeps the positions of its parameters, and has a default. None if a class loses,
    moves or reorders a frozen field, if a class appears or disappears, or if a new field has no default.
    """
    frozen_fields = frozen.get("config_fields", {})
    fields = generated.get("config_fields", {})
    defaults = generated.get("config_field_defaults", {})
    if frozen_fields.keys() != fields.keys():
        return None
    added: dict[str, list[str]] = {}
    for name, names in frozen_fields.items():
        current = list(fields[name])
        if current[: len(names)] != list(names) or any(field not in defaults.get(name, {}) for field in current[len(names) :]):
            return None
        added[name] = current[len(names) :]
    return added


def _without_added(section: Mapping[str, Mapping[str, Any]], added: Mapping[str, Sequence[str]]) -> dict[str, dict[str, Any]]:
    return {name: {key: value for key, value in values.items() if key not in added.get(name, ())} for name, values in section.items()}


def _keeps_frozen_configs(frozen: Mapping[str, Any], generated: Mapping[str, Any], added: Mapping[str, Sequence[str]]) -> bool:
    """Whether the configuration snapshot is the frozen one plus the added fields, their defaults among them."""
    rest = {key: value for key, value in frozen.items() if key not in _CONFIG_SECTIONS}
    if rest != {key: value for key, value in generated.items() if key not in _CONFIG_SECTIONS}:
        return False
    sections = _CONFIG_SECTIONS[1:]
    return all(_without_added(generated.get(section, {}), added) == frozen.get(section, {}) for section in sections)


def _extends_constructor(frozen: str, generated: str, added: Sequence[str]) -> bool:
    """Whether a constructor's signature is the frozen one followed by the added fields, each with a default."""
    if not added or not frozen.endswith(")") or not generated.startswith(frozen[:-1] + ", "):
        return False
    try:
        definition = ast.parse(f"def _constructor{generated}: pass").body[0]
    except SyntaxError:
        return False
    if not isinstance(definition, ast.FunctionDef):
        return False
    arguments = definition.args
    trailing = [argument.arg for argument in arguments.args[len(arguments.args) - len(added) :]]
    plain = arguments.vararg is None and arguments.kwarg is None and not arguments.kwonlyargs
    return plain and trailing == list(added) and len(arguments.defaults) >= len(added)


def _keeps_frozen_api(frozen: Mapping[str, Any], generated: Mapping[str, Any], added: Mapping[str, Sequence[str]]) -> bool:
    """Whether the API snapshot is the frozen one, but for the constructors of configuration classes with added fields."""
    frozen_signatures = frozen.get("signatures", {})
    signatures = generated.get("signatures", {})
    if {key: value for key, value in frozen.items() if key != "signatures"} != {
        key: value for key, value in generated.items() if key != "signatures"
    } or frozen_signatures.keys() != signatures.keys():
        return False
    return all(
        signatures[key] == signature or _extends_constructor(signature, signatures[key], added.get(key.rsplit(".", 1)[-1], ()))
        for key, signature in frozen_signatures.items()
    )


def _matches(name: str, committed_text: str, expected: bytes, added: Mapping[str, Sequence[str]] | None) -> bool:
    """Whether a committed snapshot holds the generated contract: exactly, or with optional configuration fields added."""
    if committed_text == expected.decode("utf-8"):
        return True
    if added is None:
        return False
    frozen = json.loads(committed_text)
    generated = json.loads(expected.decode("utf-8"))
    if name == CONFIGS_SNAPSHOT:
        return _keeps_frozen_configs(frozen, generated, added)
    if name == API_SNAPSHOT:
        return _keeps_frozen_api(frozen, generated, added)
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when committed snapshots differ from the generated contract.")
    parser.add_argument("--cli-arguments", nargs="*", metavar="WORD", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cli_arguments is not None:
        # Child process of _cli_arguments: print the arguments of `vamos <WORD...>`.
        print(json.dumps(_parser_arguments(args.cli_arguments)))
        return 0
    mismatches: list[str] = []
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    generated = snapshots()
    frozen_configs = OUTPUT_ROOT / CONFIGS_SNAPSHOT
    added = (
        _added_config_fields(json.loads(frozen_configs.read_text(encoding="utf-8")), json.loads(_json_bytes(generated[CONFIGS_SNAPSHOT])))
        if frozen_configs.is_file()
        else None
    )
    for name, value in generated.items():
        path = OUTPUT_ROOT / name
        expected = _json_bytes(value)
        # Git may materialize the canonical LF JSON with CRLF on Windows.
        # Text mode normalizes line endings while preserving every other
        # byte-level formatting decision in the snapshot.
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if args.check:
            if current is None or not _matches(name, current, expected, added):
                mismatches.append(path.relative_to(REPO_ROOT).as_posix())
        elif current is None or not _matches(name, current, expected, added):
            # A snapshot that only lacks optional configuration fields added after 1.0 stays the 1.0 baseline.
            path.write_bytes(expected)
    if mismatches:
        print("VAMOS 1.0 compatibility snapshots differ:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1
    print("VAMOS 1.0 compatibility snapshots: PASS" if args.check else "VAMOS 1.0 compatibility snapshots: UPDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
