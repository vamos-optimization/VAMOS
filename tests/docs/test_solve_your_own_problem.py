from __future__ import annotations

import re
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / "docs" / "index.md"
GUIDE = ROOT / "docs" / "guide" / "custom-problem.md"
QUICKSTART = ROOT / "docs" / "guide" / "zero_to_hero.md"
EXAMPLES = ROOT / "docs" / "examples.md"
EXAMPLE = ROOT / "examples" / "journeys" / "solve_my_problem.py"


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", path.read_text(encoding="utf-8"), re.DOTALL)


def test_executable_journey_preserves_domain_mapping_and_result_alignment() -> None:
    example = runpy.run_path(str(EXAMPLE))

    objectives = example["objectives"]
    constraints = example["constraints"]
    require_feasible_returned_set = example["require_feasible_returned_set"]
    problem = example["build_problem"]()

    np.testing.assert_allclose(objectives([75.0, 6.0]), [0.290625, 0.9583333333333333])
    np.testing.assert_allclose(constraints([75.0, 6.0]), [0.0])
    assert constraints([60.0, 2.0])[0] > 0.0
    assert constraints([100.0, 10.0])[0] < 0.0

    assert problem.n_var == 2
    assert problem.n_obj == 2
    assert problem.n_constraints == 1
    np.testing.assert_allclose(problem.xl, [60.0, 2.0])
    np.testing.assert_allclose(problem.xu, [100.0, 10.0])

    result = example["solve"]()
    assert result.X is not None and result.F is not None
    assert result.X.shape == (40, 2)
    assert result.F.shape == (40, 2)
    assert result.data["evaluations"] == 400

    recomputed = np.asarray([objectives(x) for x in result.X], dtype=float)
    np.testing.assert_allclose(result.F, recomputed)

    G = result.data.get("G")
    assert G is not None
    assert G.shape == (40, 1)
    assert (G <= 0.0).all()
    require_feasible_returned_set(result)

    front_F, front_indices = result.front(return_indices=True)
    front_X = result.X[front_indices]
    assert len(front_F) == len(front_X) == len(front_indices)
    np.testing.assert_array_equal(result.F[front_indices], front_F)
    np.testing.assert_array_equal(result.X[front_indices], front_X)

    with pytest.raises(RuntimeError, match="infeasible rows"):
        require_feasible_returned_set(SimpleNamespace(data={"G": np.asarray([[1.0]])}))


def test_guide_scalar_walkthrough_is_executable_in_order() -> None:
    blocks = _python_blocks(GUIDE)
    namespace: dict = {}

    for block in blocks[:6]:
        exec(compile(block, str(GUIDE), "exec"), namespace)

    result = namespace["result"]
    assert result.X is not None and result.F is not None
    assert result.X.shape == (40, 2)
    assert result.F.shape == (40, 2)
    assert result.data["evaluations"] == 400
    assert namespace["G"] is not None
    assert (namespace["G"] <= 0.0).all()
    assert namespace["front_X"].shape[0] == namespace["front_F"].shape[0]


def test_guide_vectorized_contract_matches_scalar_model() -> None:
    blocks = _python_blocks(GUIDE)
    namespace: dict = {}
    exec(compile(blocks[-1], str(GUIDE), "exec"), namespace)

    problem = namespace["problem"]
    X = np.asarray([[75.0, 6.0], [100.0, 10.0]])
    out = {"F": np.empty((2, 2)), "G": np.empty((2, 1))}
    problem.evaluate(X, out)

    expected_F = np.asarray(
        [
            [0.290625, 0.9583333333333333],
            [1.25, 0.2],
        ]
    )
    expected_G = np.asarray([[0.0], [-550.0]])
    np.testing.assert_allclose(out["F"], expected_F)
    np.testing.assert_allclose(out["G"], expected_G)


def test_learning_path_explains_translation_not_only_api_calls() -> None:
    home = HOME.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    quickstart = QUICKSTART.read_text(encoding="utf-8")
    examples = EXAMPLES.read_text(encoding="utf-8")

    assert "Translate your model into VAMOS" in guide
    assert "`F[:, 0]`" in guide
    assert "`F[:, 1]`" in guide
    assert "`g(x) <= 0`" in guide
    assert "temperature * residence_time >= 450" in guide
    assert "teaching surrogates" in guide
    assert "vectorized=True" in guide
    assert "result.front(return_indices=True)" in guide
    assert "does **not** independently remove rows" in guide
    assert "filter `X`, `F`, and `G` with the same feasibility mask" in guide

    assert "Replace the benchmark with your model" in quickstart
    assert "custom-problem.md" in quickstart
    assert "maps two domain variables" in examples
    assert "Translate domain decisions and requirements" in home
