"""Generate the permanent VAMOS 1.0 stable API and command snapshots."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
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


def _stable_api_snapshot() -> dict[str, Any]:
    modules: dict[str, list[str]] = {}
    for module_name, names in sorted(STABLE_API.items()):
        module = importlib.import_module(module_name)
        public = set(getattr(module, "__all__", ()))
        missing = [name for name in names if not hasattr(module, name) or name not in public]
        if missing:
            raise RuntimeError(f"{module_name} is missing stable exports: {', '.join(missing)}")
        modules[module_name] = sorted(names)
    return {"document_type": "vamos.stable-public-api", "schema_version": "1.0.0", "modules": modules}


def _help(command: str) -> str:
    args = command.split()[1:] + ["--help"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    environment["COLUMNS"] = "120"
    result = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", *args],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} --help failed: {result.stderr}")
    return result.stdout.replace("\r\n", "\n")


def _stable_cli_snapshot() -> dict[str, Any]:
    commands: dict[str, Any] = {}
    for command, arguments in STABLE_CLI.items():
        help_text = _help(command)
        missing = [argument for argument in arguments if argument not in help_text]
        if missing:
            raise RuntimeError(f"{command} help is missing stable arguments: {', '.join(missing)}")
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
    configs = {name: [field.name for field in dataclasses.fields(getattr(algorithms, name))] for name in config_names}
    return {
        "document_type": "vamos.stable-algorithm-configs",
        "schema_version": "1.0.0",
        # The built-in identifiers: available_algorithms() also lists the plugins installed here.
        "algorithm_ids": list(get_builtin_algorithm_names()),
        "config_fields": configs,
    }


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


def _exit_code_snapshot() -> dict[str, Any]:
    return {
        "document_type": "vamos.stable-exit-codes",
        "schema_version": "1.0.0",
        "run": {
            "success": 0,
            "usage": 2,
            "integrity_or_resource_or_path": 3,
            "schema_or_layout": 4,
            "verification_requirement": 5,
            "replay_unavailable": 6,
            "replay_execution_or_mismatch": 7,
            "output_collision": 8,
        },
        "study": {
            "success": 0,
            "usage_or_invalid_spec": 2,
            "malformed_or_integrity": 3,
            "schema_or_state_or_retry": 4,
            "collision_or_ownership": 5,
            "paused_or_completed_with_failures": 6,
            "infrastructure": 7,
            "cancelled_or_interrupted": 8,
        },
    }


def snapshots() -> Mapping[str, object]:
    return {
        "stable_algorithm_configs.json": _algorithm_snapshot(),
        "stable_cli_tree.json": _stable_cli_snapshot(),
        "stable_exit_codes.json": _exit_code_snapshot(),
        "stable_public_api.json": _stable_api_snapshot(),
        "stable_schemas.json": _schema_snapshot(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when committed snapshots differ from the generated contract.")
    args = parser.parse_args(argv)
    mismatches: list[str] = []
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, value in snapshots().items():
        path = OUTPUT_ROOT / name
        expected = _json_bytes(value)
        if args.check:
            # Git may materialize the canonical LF JSON with CRLF on Windows.
            # Text mode normalizes line endings while preserving every other
            # byte-level formatting decision in the snapshot.
            if not path.is_file() or path.read_text(encoding="utf-8") != expected.decode("utf-8"):
                mismatches.append(path.relative_to(REPO_ROOT).as_posix())
        else:
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
