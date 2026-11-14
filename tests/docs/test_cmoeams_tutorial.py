"""Check the executable CMOEA-MS learning path."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import numpy as np

from vamos import OptimizationResult
from vamos.algorithms import CMOEAMSConfig

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "algorithms" / "cmoeams.md"


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
    population = namespace["population"]
    assert isinstance(population, dict)
    assert population["X"].shape == (100, 15)
    result = namespace["result"]
    assert isinstance(result, OptimizationResult)
    assert np.all(result.data["G"] <= 0.0)
    # 249 generations after the initial population, 222 of them in stage B.
    assert result.data["generation"] == 249


def test_tutorial_builder_is_platemos_de_variation() -> None:
    namespace: dict[str, object] = {}
    exec(compile(_python_blocks()[1], str(PAGE), "exec"), namespace)

    config = namespace["config"]
    assert isinstance(config, CMOEAMSConfig)
    assert config.pop_size == 100
    assert config.crossover == ("de", {"cr": 1.0, "f": 0.5})
    assert config.de_variant == "rand_1_bin"
    assert config.mutation == ("pm", {"prob": "1/n", "eta": 20.0})
    assert config.feasible_ratio_threshold == 0.5
