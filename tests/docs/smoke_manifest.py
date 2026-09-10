from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocSmokeCase:
    name: str
    source_path: str
    code: str


DOC_SMOKE_CASES = [
    DocSmokeCase(
        name="paper_reviewer_data",
        source_path="paper/README.md",
        code="""
import subprocess
import sys

subprocess.run([sys.executable, "-S", "paper/review.py", "data"], check=True)
""",
    ),
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
]
