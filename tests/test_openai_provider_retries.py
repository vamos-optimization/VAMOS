from __future__ import annotations

import json
from typing import Any

import pytest

from vamos.assist.providers.openai_provider import OpenAIPlanProvider


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


def _plan_payload(**fields: object) -> dict[str, object]:
    """A plan reply in the shape of the strict response schema."""
    payload: dict[str, object] = {
        "kind": "plan",
        "template": "demo",
        "problem_type": "real",
        "questions": None,
        "overrides": None,
        "warnings": [],
    }
    payload.update(fields)
    return payload


def _propose_with_reply(monkeypatch: pytest.MonkeyPatch, reply: dict[str, object]) -> tuple[object, dict[str, Any]]:
    """Run ``propose`` against a fake Responses API; return the result and the request."""
    provider = OpenAIPlanProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: object())
    requests: list[dict[str, Any]] = []

    def _fake_responses_create(client: object, **kwargs: object) -> _FakeResponse:
        del client
        requests.append(kwargs)
        return _FakeResponse(json.dumps(reply))

    monkeypatch.setattr(provider, "_responses_create", _fake_responses_create)
    result = provider.propose(
        prompt="Plan a quick bi-objective run.",
        catalog={"algorithms": ["nsgaii"], "kernels": ["numpy"]},
        templates=["demo", "physics_design"],
        problem_type_hint="real",
    )
    return result, requests[0]


def _schema_nodes(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(node, dict):
        return []
    nodes = [(path, node)]
    for name, child in (node.get("properties") or {}).items():
        nodes.extend(_schema_nodes(child, f"{path}.{name}"))
    nodes.extend(_schema_nodes(node.get("items"), f"{path}[]"))
    for index, child in enumerate(node.get("anyOf") or []):
        nodes.extend(_schema_nodes(child, f"{path}|{index}"))
    return nodes


def _types(node: dict[str, Any]) -> list[str]:
    raw = node.get("type", [])
    return [raw] if isinstance(raw, str) else list(raw)


def test_plan_schema_follows_openai_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _, request = _propose_with_reply(monkeypatch, _plan_payload())
    text_format = request["text"]["format"]
    assert text_format["strict"] is True
    schema = text_format["schema"]

    # Structured Outputs: the root is one object, not a union.
    assert schema["type"] == "object"
    assert not {"anyOf", "oneOf", "allOf", "not", "enum", "$ref"} & set(schema)
    for path, node in _schema_nodes(schema):
        assert not {"oneOf", "allOf", "not", "const", "$ref"} & set(node), path
        if "object" in _types(node):
            # Every object closes its properties and requires all of them.
            assert node.get("additionalProperties") is False, path
            assert sorted(node["required"]) == sorted(node["properties"]), path
        if "enum" in node and "null" in _types(node):
            assert None in node["enum"], path

    override = schema["properties"]["overrides"]["items"]
    assert override["properties"]["key"]["enum"] == [
        "algorithm",
        "engine",
        "max_evaluations",
        "output_root",
        "plot",
        "population_size",
        "problem",
        "seed",
        "title",
    ]


def test_plan_overrides_round_trip_from_key_value_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = _plan_payload(
        overrides=[
            {"key": "max_evaluations", "value": 123},
            {"key": "algorithm", "value": "nsgaii"},
            {"key": "plot", "value": False},
        ],
        warnings=["small budget"],
    )

    result, _ = _propose_with_reply(monkeypatch, reply)

    assert result == {
        "kind": "plan",
        "template": "demo",
        "problem_type": "real",
        "overrides": {"max_evaluations": 123, "algorithm": "nsgaii", "plot": False},
        "warnings": ["small budget"],
    }


def test_questions_reply_is_coerced_and_incomplete_plans_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    questions = {
        "kind": "questions",
        "template": None,
        "problem_type": None,
        "questions": ["How many objectives?"],
        "overrides": None,
        "warnings": [],
    }
    result, _ = _propose_with_reply(monkeypatch, questions)
    assert result == {"kind": "questions", "questions": ["How many objectives?"], "warnings": []}

    with pytest.raises(RuntimeError, match="template"):
        _propose_with_reply(monkeypatch, _plan_payload(template=None))
    with pytest.raises(RuntimeError, match="questions"):
        _propose_with_reply(monkeypatch, dict(questions, questions=None))


def test_openai_provider_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIPlanProvider()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(provider, "_get_client", lambda: object())
    monkeypatch.setattr("vamos.assist.providers.openai_provider.time.sleep", lambda _: None)

    attempts: list[int] = []

    def _fake_responses_create(client: object, **kwargs: object) -> _FakeResponse:
        del client
        del kwargs
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("timeout while calling responses API")
        return _FakeResponse(json.dumps(_plan_payload()))

    monkeypatch.setattr(provider, "_responses_create", _fake_responses_create)

    result = provider.propose(
        prompt="Retry transient errors and return a plan.",
        catalog={"algorithms": ["nsgaii"], "kernels": ["numpy"]},
        templates=["demo"],
        problem_type_hint="real",
    )

    assert result["kind"] == "plan"
    assert result["template"] == "demo"
    assert len(attempts) == 3
