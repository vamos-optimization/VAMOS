"""Check the executable C-TAEA learning path."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import numpy as np

from vamos import OptimizationResult
from vamos.algorithms import CTAEAConfig

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "algorithms" / "ctaea.md"


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", PAGE.read_text(encoding="utf-8"), re.DOTALL)


def test_tutorial_example_prints_what_the_page_says() -> None:
    code = _python_blocks()[0]
    namespace: dict[str, object] = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, str(PAGE), "exec"), namespace)

    expected = [comment.strip() for comment in re.findall(r"print\(.*\)\s+# (.*)", code)]
    assert output.getvalue().split("\n")[:-1] == expected
    archive = namespace["archive"]
    assert isinstance(archive, dict)
    assert archive["X"].shape == (91, 7)
    result = namespace["result"]
    assert isinstance(result, OptimizationResult)
    assert np.all(result.data["G"] <= 0.0)


def test_tutorial_builder_is_the_default_configuration() -> None:
    namespace: dict[str, object] = {}
    exec(compile(_python_blocks()[1], str(PAGE), "exec"), namespace)

    config = namespace["config"]
    assert isinstance(config, CTAEAConfig)
    assert config.pop_size == 100
    assert config.crossover == ("sbx", {"prob": 1.0, "eta": 20.0})
    assert config.mutation == ("pm", {"prob": "1/n", "eta": 20.0})
