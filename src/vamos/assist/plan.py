from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TextIO

from vamos.engine.config.spec import (
    EXPERIMENT_SPEC_VERSION,
    validate_experiment_spec,
)
from vamos.engine.config.spec import (
    allowed_override_keys as spec_allowed_override_keys,
)
from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.experiment.cli.quickstart import available_templates, generate_quickstart_config
from vamos.experiment.cli.spec_args import parser_spec_keys
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.encoding import normalize_encoding

from .catalog import build_catalog
from .providers.protocol import PlanProvider, ProviderResponse

_PROBLEM_TYPES: tuple[str, ...] = ("real", "int", "binary")


def _canonical_problem_type(problem_type: str) -> str:
    normalized = normalize_encoding(problem_type)
    if normalized == "integer":
        return "int"
    if normalized in _PROBLEM_TYPES:
        return normalized
    raise ValueError("assist plan supports problem types: real, int, binary.")


def _resolve_plan_dir(out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path("results") / "assist_plans" / f"plan_{timestamp}"


@lru_cache(maxsize=1)
def _spec_allowed_overrides() -> tuple[str, ...]:
    default_config = ExperimentConfig()
    pre_parser = build_pre_parser()
    spec_defaults = load_spec_defaults(None)
    parser = build_parser(default_config=default_config, pre_parser=pre_parser, spec_defaults=spec_defaults)
    return tuple(sorted(parser_spec_keys(parser)))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _catalog_counts(catalog: Mapping[str, object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in ("algorithms", "kernels", "crossover_methods", "mutation_methods", "templates"):
        value = catalog.get(key)
        if isinstance(value, list):
            counts[f"{key}_count"] = len(value)
        else:
            counts[f"{key}_count"] = 0
    return counts


def _response_as_mapping(response: ProviderResponse) -> Mapping[str, object]:
    if not isinstance(response, Mapping):
        raise RuntimeError("Provider response must be a JSON object.")
    return response


def _response_kind(response: Mapping[str, object]) -> str:
    kind = response.get("kind")
    if isinstance(kind, str) and kind in {"plan", "questions"}:
        return kind
    raise RuntimeError("Provider response must include kind='plan' or kind='questions'.")


def _response_warnings(response: Mapping[str, object]) -> list[str]:
    raw = response.get("warnings", [])
    if raw is None:
        return []
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise RuntimeError("Provider response warnings must be a list of strings.")
    return list(raw)


def _response_questions(response: Mapping[str, object]) -> list[str]:
    raw = response.get("questions")
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) or not item.strip() for item in raw):
        raise RuntimeError("Provider questions response must include a non-empty questions list.")
    return [item.strip() for item in raw]


def _response_plan(
    response: Mapping[str, object],
    *,
    templates: list[str],
) -> tuple[str, str, dict[str, object], list[str]]:
    template_raw = response.get("template")
    if not isinstance(template_raw, str) or not template_raw.strip():
        raise RuntimeError("Provider plan response is missing a template.")
    selected_template = template_raw.strip().lower()
    if selected_template not in templates:
        available = ", ".join(templates)
        raise RuntimeError(f"Provider returned unknown template '{template_raw}'. Available templates: {available}")

    problem_type_raw = response.get("problem_type")
    if not isinstance(problem_type_raw, str):
        raise RuntimeError("Provider plan response is missing problem_type.")
    canonical_problem_type = _canonical_problem_type(problem_type_raw)

    overrides_raw = response.get("overrides", {})
    overrides: dict[str, object] = {}
    if overrides_raw is None:
        overrides_raw = {}
    if not isinstance(overrides_raw, Mapping):
        raise RuntimeError("Provider plan response overrides must be an object.")
    for key, value in overrides_raw.items():
        if not isinstance(key, str):
            raise RuntimeError("Provider plan response override keys must be strings.")
        overrides[key] = value

    warnings = _response_warnings(response)
    return selected_template, canonical_problem_type, overrides, warnings


def _collect_answers(questions: list[str], *, input_func: Callable[[str], str] = input) -> dict[str, str]:
    answers: dict[str, str] = {}
    for question in questions:
        answers[question] = input_func(f"{question} ").strip()
    return answers


def resolve_plan_template(
    template: str | None,
    *,
    is_tty: bool | None = None,
    input_func: Callable[[str], str] = input,
    output: TextIO | None = None,
) -> str:
    templates = available_templates()
    if template is not None:
        key = template.strip().lower()
        if key in templates:
            return key
        available = ", ".join(templates)
        raise ValueError(f"Unknown template '{template}'. Available templates: {available}")

    interactive = sys.stdout.isatty() if is_tty is None else is_tty
    if not interactive:
        available = ", ".join(templates)
        raise ValueError(
            "Missing --template in non-interactive mode. "
            f"Available templates: {available}. "
            'Example: vamos assist plan "<prompt>" --template demo'
        )

    out = output if output is not None else sys.stdout
    out.write("Select a template:\n")
    for idx, key in enumerate(templates, start=1):
        out.write(f"  {idx}. {key}\n")
    while True:
        raw = input_func(f"Choose template [1-{len(templates)}]: ").strip()
        if raw.isdigit():
            selected = int(raw)
            if 1 <= selected <= len(templates):
                return templates[selected - 1]
        normalized = raw.lower()
        if normalized in templates:
            return normalized
        out.write("Invalid selection. Enter a number or template key.\n")


def create_plan(
    prompt: str,
    template: str | None = None,
    problem_type: str = "real",
    out_dir: Path | None = None,
    mode: str = "template",
    provider: PlanProvider | None = None,
    provider_name: str | None = None,
) -> Path:
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    plan_dir = _resolve_plan_dir(out_dir)
    if plan_dir.exists():
        raise FileExistsError(f"Plan directory already exists: {plan_dir}")

    canonical_problem_type = _canonical_problem_type(problem_type)
    templates = available_templates()
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"template", "auto"}:
        raise ValueError("assist plan supports --mode template|auto.")

    catalog = build_catalog(problem_type=canonical_problem_type)
    selected_template: str
    selected_problem_type = canonical_problem_type
    overrides: dict[str, object] | None = None
    overrides_requested: dict[str, object] = {}
    overrides_applied: dict[str, object] = {}
    overrides_rejected: list[str] = []
    warnings: list[str] = []
    provider_metadata: dict[str, object] | None = None
    auto_metadata: dict[str, object] | None = None
    provider_request: dict[str, object] | None = None
    provider_response_for_trace: Mapping[str, object] | None = None
    provider_answers: dict[str, str] | None = None

    if normalized_mode == "template":
        selected_template = resolve_plan_template(template)
    else:
        if provider is None:
            hint = provider_name or "mock"
            raise RuntimeError(f"assist plan --mode auto requires a provider instance. Example: --provider {hint}")

        provider_response = _response_as_mapping(
            provider.propose(
                prompt,
                catalog,
                templates,
                problem_type_hint=canonical_problem_type,
            )
        )
        provider_response_for_trace = provider_response
        response_kind = _response_kind(provider_response)
        if response_kind == "questions":
            questions = _response_questions(provider_response)
            warnings.extend(_response_warnings(provider_response))
            if not sys.stdout.isatty():
                question_lines = "; ".join(questions)
                raise RuntimeError(
                    "Provider needs additional answers in non-interactive mode. "
                    f"Questions: {question_lines}. "
                    'Re-run interactively or use --mode template --template "<name>".'
                )
            answers = _collect_answers(questions)
            provider_answers = answers
            provider_response = _response_as_mapping(
                provider.propose(
                    prompt,
                    catalog,
                    templates,
                    problem_type_hint=canonical_problem_type,
                    answers=answers,
                )
            )
            provider_response_for_trace = provider_response
            if _response_kind(provider_response) != "plan":
                raise RuntimeError("Provider did not return a plan after receiving answers.")

        selected_template, selected_problem_type, overrides, response_warnings = _response_plan(
            provider_response,
            templates=templates,
        )
        requested = dict(overrides or {})
        allowed_keys = spec_allowed_override_keys(_spec_allowed_overrides())
        applied = {key: value for key, value in requested.items() if key in allowed_keys}
        rejected = sorted(set(requested) - set(applied))
        overrides_requested = requested
        overrides_applied = applied
        overrides_rejected = rejected
        overrides = overrides_applied

        warnings.extend(response_warnings)
        if selected_problem_type != canonical_problem_type:
            warnings.append(f"Provider selected problem_type '{selected_problem_type}' from hint '{canonical_problem_type}'.")
        if rejected:
            warnings.append(f"Rejected provider overrides not allowed by schema: {', '.join(rejected)}.")

        resolved_provider_name = provider_name or getattr(provider, "name", None)
        if not isinstance(resolved_provider_name, str) or not resolved_provider_name.strip():
            resolved_provider_name = "unknown"
        provider_metadata = {"name": resolved_provider_name}
        auto_metadata = {
            "template": selected_template,
            "problem_type": selected_problem_type,
            "overrides_requested": overrides_requested,
            "overrides_applied": overrides_applied,
            "overrides_rejected": overrides_rejected,
            "overrides": overrides_applied,
            "warnings": list(warnings),
        }
        provider_request = {
            "provider": resolved_provider_name,
            "problem_type_hint": canonical_problem_type,
            "templates_count": len(templates),
            "catalog_summary": _catalog_counts(catalog),
            "allowed_override_keys_count": len(allowed_keys),
        }
        if provider_answers:
            provider_request["answers"] = provider_answers

    config = generate_quickstart_config(template=selected_template, overrides=overrides)
    validate_experiment_spec(config, allowed_overrides=_spec_allowed_overrides())

    metadata: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "template": selected_template,
        "problem_type": selected_problem_type,
        "schema_version": EXPERIMENT_SPEC_VERSION,
        "warnings": warnings,
        "mode": normalized_mode,
    }
    if provider_metadata is not None:
        metadata["provider"] = provider_metadata
    if auto_metadata is not None:
        metadata["auto"] = auto_metadata

    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _write_json(plan_dir / "catalog.json", catalog)
    _write_json(plan_dir / "config.json", config)
    if normalized_mode == "auto":
        if provider_request is not None:
            _write_json(plan_dir / "provider_request.json", provider_request)
        if provider_response_for_trace is not None:
            response_payload: dict[str, object] = {str(key): value for key, value in provider_response_for_trace.items()}
            _write_json(plan_dir / "provider_response.json", response_payload)
    _write_json(plan_dir / "plan.json", metadata)
    return plan_dir


__all__ = ["create_plan", "resolve_plan_template"]
