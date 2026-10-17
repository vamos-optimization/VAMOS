from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocSmokeCase:
    name: str
    source_path: str
    code: str


DOC_SMOKE_CASES = [
    *[
        DocSmokeCase(
            name=f"coverage_tutorial_{lesson}",
            source_path=f"docs/assets/coverage/{lesson}.py",
            code=f"""
import runpy
import tempfile
from pathlib import Path
import pytest
lesson = "{lesson}"
if lesson in {{"feature_selection", "model_tuning"}}:
    pytest.importorskip("sklearn")
if lesson in {{"ablation", "performance"}}:
    pytest.importorskip("moocore")
if lesson == "performance":
    pytest.importorskip("numba")
module = runpy.run_path(f"docs/assets/coverage/{{lesson}}.py")
with tempfile.TemporaryDirectory() as directory:
    if lesson in {{"tuning", "ablation", "performance"}}:
        module[f"run_{{lesson}}_example"](Path(directory) / "output")
    elif lesson in {{"binary", "integer", "permutation", "mixed"}}:
        module["run_lesson"](budget=80)
    else:
        module[f"run_{{lesson}}"](budget=80 if lesson != "engineering" else 400)
""",
        )
        for lesson in (
            "smsemoa",
            "spea2",
            "ibea",
            "smpso",
            "agemoea",
            "rvea",
            "binary",
            "integer",
            "permutation",
            "mixed",
            "engineering",
            "feature_selection",
            "model_tuning",
            "tuning",
            "ablation",
            "performance",
        )
    ],
    *[
        DocSmokeCase(
            name=f"study_tutorial_{lesson}",
            source_path="examples/tutorials/study_workflow.py",
            code=f"""
import runpy
import tempfile
from pathlib import Path

import pytest

lesson = "{lesson}"
if lesson in {{"t07", "t08"}}:
    pytest.importorskip("moocore")
if lesson == "t08":
    pytest.importorskip("matplotlib")
workflow = runpy.run_path("examples/tutorials/study_workflow.py")
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    if lesson == "t08":
        workflow["render_report"]("docs/assets/studies/recorded-measurements.json", root / "report")
    else:
        workflow["plan_or_execute_study"](root / "study", budget=80, execute=True)
        if lesson == "t07":
            workflow["measure_studies"]([root / "study"], root / "metrics")
""",
        )
        for lesson in ("t06", "t07", "t08")
    ],
    *[
        DocSmokeCase(
            name=f"fundamental_tutorial_{lesson}",
            source_path="examples/tutorials/fundamentals.py",
            code=f"""
import subprocess
import sys

subprocess.run([sys.executable, "examples/tutorials/fundamentals.py", "{lesson}"], check=True)
""",
        )
        for lesson in ("t01", "t02", "t03")
    ],
    DocSmokeCase(
        name="readme_quickstart",
        source_path="README.md",
        code="""
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=200,
    pop_size=40,
    engine="numpy",
    seed=42,
)
front = result.front()
assert front is not None
assert len(front) > 0
""",
    ),
    DocSmokeCase(
        name="readme_make_problem",
        source_path="README.md",
        code="""
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
    encoding="real",
)
result = optimize(problem, algorithm="nsgaii", max_evaluations=200, seed=42)
assert result.F is not None
assert result.F.shape[1] == 2
""",
    ),
    DocSmokeCase(
        name="getting_started_config_object",
        source_path="docs/guide/getting-started.md",
        code="""
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=10)
algo_cfg = NSGAIIConfig.default(pop_size=40, n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=algo_cfg,
    max_evaluations=200,
    seed=42,
    engine="numpy",
)
assert result.F is not None
assert result.F.shape[1] == problem.n_obj
""",
    ),
    DocSmokeCase(
        name="extending_custom_algorithm",
        source_path="docs/topics/extending.md",
        code="""
from typing import Any

import numpy as np

from vamos import make_problem
from vamos.engine.algorithm.registry import get_algorithms_registry, resolve_algorithm
from vamos.foundation.kernel.backend import KernelBackend


class MyAlgorithm:
    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.config = dict(config)
        self.kernel = kernel

    def run(self, problem, termination=("max_evaluations", 10), seed=0, eval_strategy=None, live_viz=None):
        return {
            "X": np.zeros((1, problem.n_var)),
            "F": np.zeros((1, problem.n_obj)),
            "evaluations": 0,
        }


def build_my_algorithm(cfg: dict[str, Any], kernel: KernelBackend | None = None) -> MyAlgorithm:
    return MyAlgorithm(cfg, kernel=kernel)


registry = get_algorithms_registry()
name = "_docs_smoke_algorithm"
if name not in registry:
    registry.register(name, build_my_algorithm)

problem = make_problem(lambda x: [x[0], x[1]], n_var=2, n_obj=2, bounds=[(0.0, 1.0), (0.0, 1.0)], encoding="real")
algo = resolve_algorithm(name)({"pop_size": 4}, None)
payload = algo.run(problem, termination=("max_evaluations", 10), seed=0)
assert payload["F"].shape == (1, 2)
""",
    ),
    DocSmokeCase(
        name="cookbook_resolved_backend_metadata",
        source_path="docs/guide/cookbook.md",
        code="""
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=200,
    pop_size=40,
    seed=42,
)
resolved = result.explain_defaults()["resolved_spec"]
assert resolved["backend"]["kernel"]["resolution"]["name"] == "numpy"
""",
    ),
    DocSmokeCase(
        name="analysis_guide_snippets",
        source_path="docs/topics/analysis.md",
        code="""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if importlib.util.find_spec("matplotlib") is not None:
    text = Path("docs/topics/analysis.md").read_text(encoding="utf-8")
    blocks = []
    for part in text.split("```python")[1:]:
        block, separator, _rest = part.partition("```")
        assert separator == "```"
        blocks.append(block.strip())
    assert len(blocks) == 4

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "analysis_snippets.py"
        script.write_text((chr(10) * 2).join(blocks), encoding="utf-8")
        env = os.environ.copy()
        env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
        subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            check=True,
            timeout=60,
            env=env,
        )
        assert (tmp_path / "cd_plot.png").is_file()
        assert (tmp_path / "front.png").is_file()
""",
    ),
    DocSmokeCase(
        name="tuning_model_hyperparameters",
        source_path="examples/tuning/hyperparam_tuning.py",
        code="""
import importlib.util
import os
import subprocess
import sys

# The canonical dev+docs quick check does not install optional examples
# dependencies. Full CI installs the examples extra, so this path executes
# there while remaining a valid no-op in the lean documentation environment.
if importlib.util.find_spec("sklearn") is not None:
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    subprocess.run(
        [sys.executable, "examples/tuning/hyperparam_tuning.py"],
        check=True,
        timeout=120,
        env=env,
    )
""",
    ),
    DocSmokeCase(
        name="journey_try_vamos",
        source_path="examples/journeys/try_vamos.py",
        code="""
import subprocess
import sys

subprocess.run([sys.executable, "examples/journeys/try_vamos.py"], check=True)
""",
    ),
    DocSmokeCase(
        name="journey_solve_my_problem",
        source_path="examples/journeys/solve_my_problem.py",
        code="""
import subprocess
import sys

subprocess.run([sys.executable, "examples/journeys/solve_my_problem.py"], check=True)
""",
    ),
    DocSmokeCase(
        name="journey_reproducible_study",
        source_path="examples/journeys/reproducible_study.py",
        code="""
import subprocess
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / "study"
    subprocess.run(
        [sys.executable, "examples/journeys/reproducible_study.py", "--output", str(output)],
        check=True,
    )
    assert (output / "study-manifest.json").is_file()
""",
    ),
    DocSmokeCase(
        name="journey_nsgaii_zdt1",
        source_path="examples/journeys/nsgaii_zdt1.py",
        code="""
import subprocess
import sys

subprocess.run(
    [sys.executable, "examples/journeys/nsgaii_zdt1.py", "--pop-size", "20", "--max-evaluations", "200"],
    check=True,
)
""",
    ),
    DocSmokeCase(
        name="journey_understand_results",
        source_path="examples/journeys/understand_results.py",
        code="""
import subprocess
import sys

subprocess.run([sys.executable, "examples/journeys/understand_results.py"], check=True)
""",
    ),
    DocSmokeCase(
        name="journey_moead_zdt1",
        source_path="examples/journeys/moead_zdt1.py",
        code="""
import subprocess
import sys

subprocess.run(
    [sys.executable, "examples/journeys/moead_zdt1.py", "--pop-size", "20", "--max-evaluations", "203"],
    check=True,
)
""",
    ),
    DocSmokeCase(
        name="journey_nsgaiii_dtlz2",
        source_path="examples/journeys/nsgaiii_dtlz2.py",
        code="""
import subprocess
import sys

subprocess.run(
    [sys.executable, "examples/journeys/nsgaiii_dtlz2.py", "--divisions", "3", "--max-evaluations", "203"],
    check=True,
)
""",
    ),
    DocSmokeCase(
        name="journey_prepare_studio_demo",
        source_path="examples/journeys/prepare_studio_demo.py",
        code="""
import subprocess
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / "study"
    subprocess.run(
        [sys.executable, "examples/journeys/prepare_studio_demo.py", "--output", str(output),
         "--pop-size", "20", "--max-evaluations", "80"],
        check=True,
    )
    assert (output / "study-manifest.json").is_file()
""",
    ),
    DocSmokeCase(
        name="constraints_public_example",
        source_path="docs/reference/constraints.md",
        code="""
import re
from pathlib import Path

text = Path("docs/reference/constraints.md").read_text(encoding="utf-8")
blocks = [part.partition("```")[0] for part in text.split("```python")[1:]]
assert blocks
namespace = {}
exec(compile(blocks[0], "docs/reference/constraints.md", "exec"), namespace)
assert namespace["result"].data["evaluations"] == 400
assert namespace["feasible"].any()
""",
    ),
    DocSmokeCase(
        name="cookbook_public_lifecycle",
        source_path="docs/guide/cookbook.md",
        code="""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

text = Path("docs/guide/cookbook.md").read_text(encoding="utf-8")
blocks = []
for number in (3, 13, 15):
    section = re.search(rf"^## {number}\\..*?(?=^## |\\Z)", text, flags=re.MULTILINE | re.DOTALL)
    assert section is not None
    blocks.extend(part.partition("```")[0] for part in section.group().split("```python")[1:])
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, "-c", (chr(10) * 2).join(blocks)], cwd=tmp, check=True)
""",
    ),
]
