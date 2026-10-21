from __future__ import annotations

import json
import os
from typing import Any, cast

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


def _llm_generate_gemini(description: str, *, api_key: str = "") -> dict[str, Any]:
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
    raw = response.text
    if not raw or not raw.strip():
        raise RuntimeError("Empty response from Gemini.")
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return cast(dict[str, Any], json.loads(text.strip()))


def _llm_generate_openai(description: str, *, api_key: str = "") -> dict[str, Any]:
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
    raw = getattr(response, "output_text", None)
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("Empty response from OpenAI.")
    return cast(dict[str, Any], json.loads(raw))


def _llm_generate_anthropic(description: str, *, api_key: str = "") -> dict[str, Any]:
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
    raw = message.content[0].text
    if not raw.strip():
        raise RuntimeError("Empty response from Anthropic.")
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return cast(dict[str, Any], json.loads(text))


def llm_generate_problem_code(
    description: str,
    provider: str = "openai",
    api_key: str = "",
) -> dict[str, Any]:
    """Generate VAMOS problem code from a natural-language description using an LLM."""
    if provider == "gemini":
        result = _llm_generate_gemini(description, api_key=api_key)
    elif provider == "anthropic":
        result = _llm_generate_anthropic(description, api_key=api_key)
    else:
        result = _llm_generate_openai(description, api_key=api_key)

    for key in ("objective_code", "n_var", "n_obj", "bounds"):
        if key not in result:
            raise RuntimeError(f"LLM response missing required field: {key}")
    result.setdefault("constraint_code", "")
    return result


__all__ = ["llm_generate_problem_code"]
