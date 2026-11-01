"""Thin parser and renderer for the canonical durable-study CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .study_command import Operation, StudyCommandRequest, StudyCommandResult, execute_study_command
from .study_summary_output import SummaryFormat


def study_main(argv: Sequence[str] | None = None) -> None:
    """Parse, delegate, and render the ``vamos study`` command group."""
    request = _request(_parser().parse_args(argv))
    result = execute_study_command(request)
    if request.json_output:
        print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False))
        for warning in result.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
    else:
        _render_human(result)
    if result.exit_code:
        raise SystemExit(result.exit_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vamos study", description="Plan, create, run, inspect, resume, retry, and summarize durable studies."
    )
    commands = parser.add_subparsers(dest="study_command", required=True)

    plan = commands.add_parser("plan", help="Resolve a StudySpec without writing or executing.")
    plan.add_argument("config", help="Path to a StudySpec JSON object.")
    plan.add_argument("--output", help="Proposed study directory to inspect without reserving it.")
    _json_argument(plan)

    create = commands.add_parser("create", help="Atomically create a canonical study without executing it.")
    create.add_argument("config", help="Path to a StudySpec JSON object.")
    create.add_argument("--output", required=True, help="Absent destination for the canonical study.")
    _json_argument(create)

    for name, help_text in (
        ("run", "Run a newly created study sequentially."),
        ("inspect", "Inspect verified canonical state without writing."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("study_dir", help="Canonical StudyManifest root.")
        _json_argument(command)

    resume = commands.add_parser("resume", help="Reconcile and resume eligible unfinished work.")
    resume.add_argument("study_dir", help="Canonical StudyManifest root.")
    resume.add_argument("--retry-failed", action="store_true", help="Explicitly include eligible failed tasks.")
    _json_argument(resume)

    retry = commands.add_parser("retry", help="Explicitly retry eligible failed tasks.")
    retry.add_argument("study_dir", help="Canonical StudyManifest root.")
    retry.add_argument("--failed", action="store_true", required=True, help="Select retryable failed tasks only.")
    _json_argument(retry)

    summarize = commands.add_parser("summarize", help="Project one derived row per canonical task.")
    summarize.add_argument("study_dir", help="Canonical StudyManifest root.")
    summarize.add_argument(
        "--format", choices=("json", "csv"), default="json", dest="summary_format", help="Explicit output format (default: json)."
    )
    summarize.add_argument("--output", help="Absent explicit derived-output path; omitted means no write.")
    _json_argument(summarize)
    return parser


def _json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit one vamos.study-command-result/1 JSON document.")


def _request(args: argparse.Namespace) -> StudyCommandRequest:
    operation = cast(Operation, args.study_command)
    config = Path(args.config) if hasattr(args, "config") else None
    study_dir = Path(args.study_dir) if hasattr(args, "study_dir") else None
    output = Path(args.output) if getattr(args, "output", None) is not None else None
    return StudyCommandRequest(
        operation=operation,
        config=config,
        study_dir=study_dir,
        output=output,
        json_output=bool(args.json_output),
        retry_failed=bool(getattr(args, "retry_failed", False)),
        failed_only=bool(getattr(args, "failed", True)),
        summary_format=cast(SummaryFormat, getattr(args, "summary_format", "json")),
    )


def _render_human(result: StudyCommandResult) -> None:
    if result.errors:
        for error in result.errors:
            print(f"Error [{error.get('category', 'study')}]: {error.get('reason', 'UNKNOWN')}", file=sys.stderr)
            action = error.get("action")
            if action:
                print(f"Next: {action}", file=sys.stderr)
        return
    payload = result.payload
    print(f"Study {result.operation}: {result.status}")
    print(f"Study ID: {result.study_id or '-'}")
    print(f"Plan ID: {result.plan_id or '-'}")
    if "study_root" in payload:
        print(f"Study root: {payload['study_root']}")
    print(f"Canonical state changed: {'yes' if result.changed else 'no'}")
    if "execution_began" in payload:
        print(f"Execution began: {'yes' if payload['execution_began'] else 'no'}")
    _render_counts(payload)
    _render_summary(payload)
    for action in result.next_actions:
        print(f"Next: {action}")
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def _render_counts(payload: Mapping[str, Any]) -> None:
    report = payload.get("report")
    if not isinstance(report, Mapping):
        return
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        return
    policy = report.get("policy")
    if isinstance(policy, Mapping):
        print(f"Policy: on_error={policy.get('on_error', '-')}, max_attempts_per_task={policy.get('max_attempts_per_task', '-')}")
    ordered = ("tasks", "pending", "running", "succeeded", "failed", "interrupted", "cancelled", "skipped")
    print("Counts: " + ", ".join(f"{key}={counts.get(key, 0)}" for key in ordered))


def _render_summary(payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return
    rows = summary.get("rows")
    # The payload is frozen, so its lists arrive as tuples.
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return
    print("Tasks:")
    print("index  state       problem  algorithm  seed  evaluations")
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        values = (
            item.get("plan_index", "-"),
            item.get("state", "-"),
            item.get("problem_id", "-") or "-",
            item.get("algorithm_id", "-") or "-",
            item.get("seed", "-") if item.get("seed") is not None else "-",
            item.get("evaluations", "-") if item.get("evaluations") is not None else "-",
        )
        print(f"{values[0]!s:>5}  {values[1]!s:<10}  {values[2]!s:<7}  {values[3]!s:<9}  {values[4]!s:>4}  {values[5]}")


__all__ = ["study_main"]
