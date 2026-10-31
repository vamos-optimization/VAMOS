from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping
from typing import Any, Literal, cast

from vamos.experiment.cli.quickstart import generate_quickstart_config

from .protocol import PlanResponse, ProviderResponse, QuestionsResponse

_OPENAI_SETUP_HINT = (
    "OpenAI provider setup required: install dependencies with "
    "`pip install vamos-optimization[openai]` (or `pip install openai`), set OPENAI_API_KEY, "
    "and run `vamos assist doctor` for diagnostics."
)
_DEFAULT_MODEL = "gpt-5.2"
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_OUTPUT_TOKENS = 900
_PROBLEM_TYPES: tuple[str, ...] = ("real", "int", "binary")
_LIST_LIMIT = 50
_MAX_QUESTIONS = 3
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)
_TRANSIENT_ERROR_HINTS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "rate limit",
    "too many requests",
    "temporary",
    "temporarily",
    "try again",
    "connection reset",
    "connection aborted",
    "service unavailable",
    "unavailable",
    "429",
    "503",
)


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _allowed_override_keys(templates: list[str]) -> list[str]:
    allowed: set[str] = set()
    for template in templates:
        try:
            config = generate_quickstart_config(template=template, overrides=None)
        except Exception:
            continue
        defaults = config.get("defaults")
        if isinstance(defaults, Mapping):
            for key in defaults:
                if isinstance(key, str):
                    allowed.add(key)
    return sorted(allowed)


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _truncate_list(items: list[str], *, limit: int = _LIST_LIMIT) -> tuple[list[str], bool]:
    if len(items) <= limit:
        return list(items), False
    return list(items[:limit]), True


def _build_compact_catalog_context(catalog: Mapping[str, object]) -> tuple[dict[str, object], list[str]]:
    context: dict[str, object] = {}
    notes: list[str] = []

    for key in ("algorithms", "kernels"):
        values = _as_string_list(catalog.get(key))
        truncated_values, was_truncated = _truncate_list(values)
        context[key] = truncated_values
        if was_truncated:
            notes.append(f"{key} truncated to first {_LIST_LIMIT} of {len(values)} entries.")

    for key in ("crossover_methods", "mutation_methods"):
        values = _as_string_list(catalog.get(key))
        if not values:
            continue
        if len(values) <= _LIST_LIMIT:
            context[key] = values
            continue
        preview, _ = _truncate_list(values)
        context[f"{key}_preview"] = preview
        notes.append(f"{key} preview truncated to first {_LIST_LIMIT} of {len(values)} entries.")

    return context, notes


def _nullable_enum(values: list[str]) -> dict[str, object]:
    return {"type": ["string", "null"], "enum": [*values, None]}


def _response_schema(templates: list[str], override_keys: list[str]) -> dict[str, object]:
    """Return the plan schema in the form OpenAI Structured Outputs accepts in strict mode.

    Strict mode needs a single object at the root, ``additionalProperties: false``
    on every object and every property listed in ``required``. A plan and a
    questions reply therefore share one object: ``kind`` selects the reply, and
    the fields of the other kind are ``null``. Overrides are a list of
    ``{key, value}`` pairs whose keys are the announced override keys, because a
    strict object cannot have open-ended properties.
    """
    overrides: dict[str, object]
    if override_keys:
        overrides = {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string", "enum": list(override_keys)},
                    "value": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]},
                },
                "required": ["key", "value"],
            },
        }
    else:
        overrides = {"type": "null"}
    properties: dict[str, object] = {
        "kind": {"type": "string", "enum": ["plan", "questions"]},
        "template": _nullable_enum(templates),
        "problem_type": _nullable_enum(list(_PROBLEM_TYPES)),
        "questions": {"type": ["array", "null"], "items": {"type": "string"}},
        "overrides": overrides,
        "warnings": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _response_strings(raw: object, field: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise RuntimeError(f"OpenAI response {field} must be a list of strings.")
    return list(raw)


def _overrides_from_pairs(raw: object) -> dict[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise RuntimeError("OpenAI plan response overrides must be a list of key/value pairs.")
    overrides: dict[str, object] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str) or "value" not in item:
            raise RuntimeError("OpenAI plan response overrides must be objects with a string 'key' and a 'value'.")
        overrides[item["key"]] = item["value"]
    return overrides


def _coerce_provider_response(raw: object) -> ProviderResponse:
    """Convert a reply in the strict schema into the provider protocol's response."""
    if not isinstance(raw, dict):
        raise RuntimeError("OpenAI response is not a JSON object.")
    warnings = _response_strings(raw.get("warnings"), "warnings")
    kind = raw.get("kind")
    if kind == "plan":
        template = raw.get("template")
        problem_type = raw.get("problem_type")
        if not isinstance(template, str) or problem_type not in _PROBLEM_TYPES:
            raise RuntimeError("OpenAI plan response is missing template/problem_type.")
        return PlanResponse(
            kind="plan",
            template=template,
            problem_type=cast(Literal["real", "int", "binary"], problem_type),
            overrides=_overrides_from_pairs(raw.get("overrides")),
            warnings=warnings,
        )
    if kind == "questions":
        questions = _response_strings(raw.get("questions"), "questions")
        if not questions:
            raise RuntimeError("OpenAI questions response is missing a valid questions list.")
        if len(questions) > _MAX_QUESTIONS:
            warnings.append(f"OpenAI asked {len(questions)} questions; only the first {_MAX_QUESTIONS} are kept.")
            questions = questions[:_MAX_QUESTIONS]
        return QuestionsResponse(kind="questions", questions=questions, warnings=warnings)
    raise RuntimeError("OpenAI response kind must be 'plan' or 'questions'.")


class OpenAIPlanProvider:
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model or os.getenv("VAMOS_ASSIST_OPENAI_MODEL", _DEFAULT_MODEL)
        self.temperature = (
            temperature if temperature is not None else _read_float_env("VAMOS_ASSIST_OPENAI_TEMPERATURE", _DEFAULT_TEMPERATURE)
        )
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else _read_int_env("VAMOS_ASSIST_OPENAI_MAX_OUTPUT_TOKENS", _DEFAULT_MAX_OUTPUT_TOKENS)
        )

    def _import_openai(self) -> Any:
        from openai import OpenAI  # type: ignore[import-not-found]

        return OpenAI

    def _get_client(self) -> Any:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(_OPENAI_SETUP_HINT)
        try:
            openai_cls = self._import_openai()
        except ImportError as exc:
            raise RuntimeError(_OPENAI_SETUP_HINT) from exc
        return openai_cls()

    def _responses_create(self, client: Any, **kwargs: object) -> Any:
        return client.responses.create(**kwargs)

    def _is_transient_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in _TRANSIENT_ERROR_HINTS)

    def _create_response_with_retry(self, client: Any, **kwargs: object) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._responses_create(client, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= _MAX_ATTEMPTS or not self._is_transient_error(exc):
                    break
                delay = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)] + random.uniform(0.0, 0.1)
                time.sleep(delay)

        if last_error is None:  # pragma: no cover - defensive
            raise RuntimeError("OpenAI provider failed unexpectedly. Run `vamos assist doctor` for diagnostics.")
        raise RuntimeError(
            f"OpenAI provider failed after {_MAX_ATTEMPTS} attempts: {last_error}. Run `vamos assist doctor` for diagnostics."
        ) from last_error

    def propose(
        self,
        prompt: str,
        catalog: Mapping[str, object],
        templates: list[str],
        problem_type_hint: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> ProviderResponse:
        if not templates:
            raise RuntimeError("OpenAI provider failed: no templates available.")
        allowed_keys = _allowed_override_keys(templates)
        schema = _response_schema(templates, allowed_keys)
        compact_catalog, catalog_notes = _build_compact_catalog_context(catalog)
        allowed_keys_limited, allowed_keys_truncated = _truncate_list(allowed_keys)
        if allowed_keys_truncated:
            catalog_notes.append(f"allowed_override_keys truncated to first {_LIST_LIMIT} of {len(allowed_keys)} entries.")

        system_prompt = (
            "You are the VAMOS assist planner.\n"
            "Return exactly one JSON object that matches the provided schema.\n"
            "Choose the best template from the provided list.\n"
            "Use problem_type in {real,int,binary}.\n"
            "If you have enough information, return kind='plan' with template and problem_type, "
            "and set questions to null.\n"
            "If clarifications are necessary, return kind='questions' with at most 3 short questions, "
            "and set template, problem_type and overrides to null.\n"
            "Give overrides as a list of {key, value} pairs, or null when none are needed. "
            "Keep them minimal and only use keys listed in allowed_override_keys."
        )
        user_payload: dict[str, object] = {
            "prompt": prompt,
            "problem_type_hint": problem_type_hint,
            "templates_count": len(templates),
            "catalog": compact_catalog,
            "allowed_override_keys": allowed_keys_limited,
            "notes": catalog_notes,
        }
        if answers:
            user_payload["answers"] = answers
        user_message = json.dumps(user_payload, indent=2, sort_keys=True)

        try:
            client = self._get_client()
            response = self._create_response_with_retry(
                client,
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "vamos_plan",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            raw = getattr(response, "output_text", None)
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("No output_text in OpenAI response.")
            parsed = json.loads(raw)
            return _coerce_provider_response(parsed)
        except RuntimeError as exc:
            if _OPENAI_SETUP_HINT in str(exc):
                raise
            if str(exc).startswith("OpenAI provider failed:"):
                raise
            raise RuntimeError(f"OpenAI provider failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI provider failed: {exc}") from exc


__all__ = ["OpenAIPlanProvider"]
