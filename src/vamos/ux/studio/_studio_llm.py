from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from typing import Any

_VAMOS_CODE_SYSTEM_PROMPT = """\
You are a code generator for the VAMOS multi-objective optimization framework.

The user will describe an optimization problem in natural language.  You must
return a JSON object with these fields:

  objective_code : str — Raw Python statements (NOT a function def).
      The variable `x` is a 1-D NumPy array of length n_var.
      Must end with `return [f0, f1, ...]` (one value per objective).
      Allowed imports: `math`, `numpy` (as np).

  constraint_code : str — Same format but for constraints (may be empty "").
      Convention: g(x) <= 0 is feasible.
      E.g. for stress <= 100: `return [stress - 100.0]`

  n_var : int — Number of decision variables.
  n_obj : int — Number of objectives (>= 2).
  bounds : str — Bounds as "lo, hi" per variable, one pair per line,
      or a single line if all variables share the same bounds.
      Example: "0.0, 5.0\\n1.0, 10.0"

Example — beam design (minimise cost and deflection, stress <= 100):

{
  "objective_code": "area = x[0] * x[1]\\ncost = 2.0 * x[0] + 3.0 * x[1]\\ndeflection = 1000.0 / (x[0] * x[1] ** 3 + 1e-6)\\nreturn [cost, deflection]",
  "constraint_code": "stress = 600.0 / (x[0] * x[1] ** 2 + 1e-6)\\nreturn [stress - 100.0]",
  "n_var": 2,
  "n_obj": 2,
  "bounds": "0.5, 5.0\\n1.0, 10.0"
}

Return ONLY the JSON object, nothing else.
"""

# The ranges of the Problem Builder form, so that a draft never leaves it half updated.
N_VAR_RANGE = (1, 100)
N_OBJ_RANGE = (2, 10)
DEFAULT_TIMEOUT_SECONDS = 120.0


def _gemini_text(description: str, *, api_key: str = "") -> str:
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("No Gemini API key provided. Paste it in the API Key field or set GEMINI_API_KEY.")
    try:
        # google-genai is intentionally absent from the canonical typing environment;
        # this runtime import remains guarded by the actionable error below.
        from google import genai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("google-genai package not installed. Run: pip install google-genai") from exc

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=os.getenv("VAMOS_ASSIST_GEMINI_MODEL", "gemini-2.0-flash"),
        contents=f"{_VAMOS_CODE_SYSTEM_PROMPT}\n\nUser request:\n{description}",
    )
    return _response_text(response.text, "Gemini")


def _openai_text(description: str, *, api_key: str = "") -> str:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("No OpenAI API key provided. Paste it in the API Key field or set OPENAI_API_KEY.")
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("openai package not installed. Run: pip install vamos-optimization[openai]") from exc

    client = OpenAI(api_key=key)
    response = client.responses.create(
        model=os.getenv("VAMOS_ASSIST_OPENAI_MODEL", "gpt-4o"),
        input=[
            {"role": "system", "content": _VAMOS_CODE_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        temperature=0.2,
        max_output_tokens=1500,
    )
    return _response_text(getattr(response, "output_text", None), "OpenAI")


def _anthropic_text(description: str, *, api_key: str = "") -> str:
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("anthropic package not installed. Run: pip install vamos-optimization[anthropic]") from exc

    # Without a key in the field, the SDK resolves ANTHROPIC_API_KEY or another configured credential.
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    message = client.messages.create(
        model=os.getenv("VAMOS_ASSIST_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1500,
        system=_VAMOS_CODE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": description}],
    )
    return _response_text(message.content[0].text, "Anthropic")


_PROVIDERS: dict[str, Callable[..., str]] = {
    "gemini": _gemini_text,
    "openai": _openai_text,
    "anthropic": _anthropic_text,
}


def _response_text(raw: object, provider: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(f"Empty response from {provider}.")
    return raw


def parse_problem_draft(text: str) -> dict[str, Any]:
    """Parse a provider's answer into a checked Problem Builder draft, or raise ``ValueError``.

    One parser for every provider: a surrounding Markdown code fence is removed, the
    answer must be a JSON object, and each field must have the type and range the
    Problem Builder form accepts.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rstrip()
        if body.endswith("```"):
            body = body[:-3]
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise ValueError(f"The LLM answer is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"The LLM answer must be a JSON object, not a {type(payload).__name__}.")
    for name in ("objective_code", "n_var", "n_obj", "bounds"):
        if name not in payload:
            raise ValueError(f"The LLM answer has no '{name}' field.")
    draft: dict[str, Any] = {
        "objective_code": _text_field(payload, "objective_code", required=True),
        "constraint_code": _text_field(payload, "constraint_code", required=False),
        "n_var": _int_field(payload, "n_var", N_VAR_RANGE),
        "n_obj": _int_field(payload, "n_obj", N_OBJ_RANGE),
        "bounds": _text_field(payload, "bounds", required=True),
    }
    return draft


def _text_field(payload: dict[str, Any], name: str, *, required: bool) -> str:
    value = payload.get(name, "")
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"The LLM answer's '{name}' must be a {'non-empty ' if required else ''}string; got {value!r}.")
    return value


def _int_field(payload: dict[str, Any], name: str, bounds: tuple[int, int]) -> int:
    value = payload.get(name)
    low, high = bounds
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"The LLM answer's '{name}' must be an integer from {low} to {high}; got {value!r}.")
    return value


def llm_generate_problem_code(
    description: str,
    provider: str = "openai",
    api_key: str = "",
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Generate a checked Problem Builder draft from a natural-language description.

    ``provider`` is ``"gemini"``, ``"openai"`` or ``"anthropic"``; another name raises
    ``ValueError`` before any request. A provider that does not answer within
    ``timeout_seconds`` raises ``TimeoutError``; its request is abandoned, not cancelled.
    """
    request = _PROVIDERS.get(provider)
    if request is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose one of: {', '.join(_PROVIDERS)}.")
    outcome: dict[str, object] = {}

    def call() -> None:
        try:
            outcome["text"] = request(description, api_key=api_key)
        except BaseException as exc:  # Re-raised in the caller's thread.
            outcome["error"] = exc

    worker = threading.Thread(target=call, name=f"vamos-studio-{provider}", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(f"The {provider} request did not answer within {timeout_seconds:g} s.")
    error = outcome.get("error")
    if isinstance(error, BaseException):
        raise error
    return parse_problem_draft(str(outcome.get("text", "")))


__all__ = ["llm_generate_problem_code"]
