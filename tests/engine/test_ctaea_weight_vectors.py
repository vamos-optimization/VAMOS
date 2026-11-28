"""C-TAEA's weight_vectors (development after 1.5.2): files, directories and lattices, with the default left as it was.

Without the option, C-TAEA must give the results of VAMOS before it, bit for bit. Two kinds of checks:

- **Absolute, x86-64 only.** The digests below were computed with ``main`` at ``15813d91``, before the
  option, on x86-64 (Windows). Their runs avoid ``np.power`` (BLX-alpha, uniform mutation, and problems
  made of products and sums added in a fixed order), whose rounding depends on the platform's maths
  library; arm64's NumPy reductions round differently, so they are skipped there.
- **Relative, every platform.** A file or lattice that holds the default vectors runs the default run,
  bit for bit, and ``weight_vectors=None`` is the default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import CTAEAConfig
from vamos.engine.algorithm.catalog import initial_evaluations
from vamos.engine.algorithm.components.weight_vectors import read_weight_vectors, vector_digest
from vamos.engine.algorithm.ctaea import CTAEA, ctaea_weight_vectors
from vamos.engine.algorithm.ctaea.weights import MINIMUM_COMPONENT, configured_weight_vectors
from vamos.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.resources import weight_path

X86_64 = platform.machine().lower() in {"amd64", "x86_64"}
x86_64_only = pytest.mark.skipif(
    not X86_64,
    reason="the digests were computed on x86-64; arm64's NumPy reductions round differently, so its trajectories differ",
)


class _LinearC1DTLZ1:
    """C1-DTLZ1's linear front and constraint on ``n_obj`` objectives, with ``g = sum((x_i - 0.5)^2)``.

    Products, sums and squares only, added in a fixed order, so that the values are the same on
    every x86-64 platform. ``evaluations`` counts the evaluated rows.
    """

    n_constraints = 1
    encoding = "real"

    def __init__(self, n_obj: int, n_var: int) -> None:
        self.n_obj = n_obj
        self.n_var = n_var
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.evaluations = 0

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        self.evaluations += X.shape[0]
        m = self.n_obj
        g = np.zeros(X.shape[0])
        for j in range(m - 1, self.n_var):
            d = X[:, j] - 0.5
            g = g + d * d
        F = np.empty((X.shape[0], m))
        for i in range(m):
            f = 0.5 * (1.0 + g)
            for j in range(m - 1 - i):
                f = f * X[:, j]
            if i > 0:
                f = f * (1.0 - X[:, m - 1 - i])
            F[:, i] = f
        rest = np.zeros(X.shape[0])
        for i in range(m - 1):
            rest = rest + F[:, i] / 0.5
        out["F"] = F
        out["G"] = (F[:, m - 1] / 0.6 + rest - 1.0)[:, None]


class _ShiftedFront:
    """Two objectives whose unconstrained front (x1 = 0) is infeasible: the constraint asks for x2 >= 0.5."""

    n_var = 4
    n_obj = 2
    n_constraints = 1
    xl = np.zeros(4)
    xu = np.ones(4)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        g = 1.0 + X[:, 1] + X[:, 2] + X[:, 3]
        out["F"] = np.column_stack([X[:, 0], g * (1.0 - np.sqrt(X[:, 0] / g))])
        out["G"] = (0.5 - X[:, 1])[:, None]


def _problem(n_obj: int) -> Any:
    return _ShiftedFront() if n_obj == 2 else _LinearC1DTLZ1(n_obj, n_obj + 4)


# case -> (objectives, pop_size, max_evaluations): two objectives (20 vectors), three (91 of 100, one layer) and
# five (105 of 105 in two layers, 70 + 35); every budget ends with a partial generation.
CASES: dict[str, tuple[int, int, int]] = {
    "two-objectives": (2, 20, 20 * 15 + 7),
    "three-objectives": (3, 100, 91 * 6 + 5),
    "five-objectives-two-layers": (5, 105, 105 * 3 + 11),
}
# The digests of these runs with main at 15813d91, before the weight_vectors option, on x86-64.
GOLDEN_X86_64: dict[str, str] = {
    "two-objectives": "1daebd142acc9373476f48d344136dd2564b59566c82279a95ad90fcfc76526b",
    "three-objectives": "6517227c563c054d55d9cebf282f89c686a9efe1a4ba47864168c8a88dec46a2",
    "five-objectives-two-layers": "0ead699ef3e75a547d2838f8d2eef98080a7fc75e68af4051af799947a701cc4",
}


def _builder(pop_size: int) -> Any:
    # BLX-alpha and uniform mutation: no np.power, whose rounding depends on the platform's maths library.
    return CTAEAConfig.builder().pop_size(pop_size).crossover("blx_alpha", alpha=0.5, prob=0.9).mutation("uniform", prob="1/n", perturb=0.2)


def _run(case: str, config: dict[str, Any] | None = None, seed: int = 1) -> tuple[dict[str, Any], CTAEA]:
    n_obj, pop_size, budget = CASES[case]
    algorithm = CTAEA(config if config is not None else _builder(pop_size).build().to_dict(), kernel=NumPyKernel())
    result = algorithm.run(_problem(n_obj), ("max_evaluations", budget), seed=seed)
    return result, algorithm


def _digest(result: dict[str, Any], algorithm: CTAEA) -> str:
    """SHA-256 of both archives, the weight vectors, the counters and the state of the run's generator."""
    state = algorithm.state
    assert state is not None
    h = hashlib.sha256()
    for block in (result["population"], result["diversity_archive"]):
        for key in ("X", "F", "G"):
            h.update(np.ascontiguousarray(block[key], dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(result["weights"], dtype="<f8").tobytes())
    h.update(json.dumps([result["evaluations"], result["generation"], state.rng.bit_generator.state], sort_keys=True).encode())
    return h.hexdigest()


def _sbx_builder(pop_size: int) -> Any:
    return CTAEAConfig.builder().pop_size(pop_size).crossover("sbx", prob=1.0, eta=20.0).mutation("pm", prob="1/n", eta=20.0)


# ---------------------------------------------------------------------------
# The default is unchanged
# ---------------------------------------------------------------------------


@x86_64_only
@pytest.mark.parametrize("case", list(CASES))
def test_default_runs_give_the_results_of_main_before_the_option(case: str) -> None:
    assert _digest(*_run(case)) == GOLDEN_X86_64[case]


@pytest.mark.parametrize("case", list(CASES))
def test_weight_vectors_none_is_the_default(case: str) -> None:
    n_obj, pop_size, _ = CASES[case]
    default = _builder(pop_size).build()
    explicit = dataclasses.replace(default, weight_vectors=None)
    empty = _builder(pop_size).weight_vectors().build()

    assert "weight_vectors" not in default.to_dict()
    assert explicit == default and empty == default and empty.to_dict() == default.to_dict()
    reference = _digest(*_run(case))
    assert _digest(*_run(case, {**default.to_dict(), "weight_vectors": None})) == reference
    assert _digest(*_run(case, {**default.to_dict(), "weight_vectors": {"path": None, "divisions": None}})) == reference


@pytest.mark.parametrize(
    ("case", "pop_size", "lattice"),
    [
        ("two-objectives", 20, {"divisions": 19}),
        ("three-objectives", 91, {"divisions": 12}),
        ("five-objectives-two-layers", 105, {"divisions": 4, "inner_divisions": 3}),
    ],
)
def test_the_default_vectors_given_explicitly_run_the_default_run(
    case: str, pop_size: int, lattice: dict[str, int], tmp_path: Path
) -> None:
    """The lattice of the default's divisions, and a file of the default vectors, give the default run, bit for bit."""
    n_obj, _, budget = CASES[case]
    file = tmp_path / "default.dat"
    np.savetxt(file, ctaea_weight_vectors(pop_size, n_obj), fmt="%.17g")

    def run(config: dict[str, Any]) -> str:
        algorithm = CTAEA(config, kernel=NumPyKernel())
        return _digest(algorithm.run(_problem(n_obj), ("max_evaluations", budget), seed=2), algorithm)

    reference = run(_builder(pop_size).build().to_dict())
    assert run(_builder(pop_size).weight_vectors(**lattice).build().to_dict()) == reference
    assert run(_builder(pop_size).weight_vectors(path=str(file)).build().to_dict()) == reference


# ---------------------------------------------------------------------------
# Files, directories and lattices
# ---------------------------------------------------------------------------


def test_jmetals_w3d_100_runs_100_individuals_on_three_objectives_with_the_exact_budget() -> None:
    file = weight_path("W3D_100.dat")
    raw = read_weight_vectors(str(file), 3)
    assert raw.shape == (100, 3) and np.count_nonzero(raw == 0.0) > 0
    problem = _LinearC1DTLZ1(3, 7)
    config = _sbx_builder(100).weight_vectors(path=str(file)).build().to_dict()

    result = CTAEA(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 100 * 4 + 37), seed=1)

    assert result["evaluations"] == problem.evaluations == 437
    assert result["population"]["X"].shape == (100, 7) and result["diversity_archive"]["X"].shape == (100, 7)
    # The file's rows, in its order, with the zeros raised to C-TAEA's floor so that the Tchebycheff values are finite.
    np.testing.assert_array_equal(result["weights"], np.maximum(raw, MINIMUM_COMPONENT))
    assert result["weights"].min() == MINIMUM_COMPONENT
    assert np.all(np.isfinite(result["population"]["F"]))


def test_optimize_runs_rcm08_with_100_individuals_and_records_the_file() -> None:
    directory = str(weight_path("W3D_100.dat").parent)
    config = _sbx_builder(100).weight_vectors(path=directory).build()

    result = vamos.optimize("rcm08", algorithm="ctaea", algorithm_config=config, max_evaluations=350, seed=1, engine="numpy")

    spec = result.meta["run_artifact_resolved_spec"]
    file = str(weight_path("W3D_100.dat"))
    assert result.data["evaluations"] == 350
    assert result.data["weights"].shape == (100, 3) and result.data["population"]["X"].shape == (100, 7)
    # The directory is bound to the file the run read and the digest of its vectors.
    assert spec["algorithm"]["config"]["weight_vectors"] == {
        "path": file,
        "divisions": None,
        "sha256": vector_digest(read_weight_vectors(file, 3)),
    }
    assert spec["algorithm"]["resolution"]["weight_vectors"] == {"source": "file", "file": "W3D_100.dat", "vectors": 100}
    assert spec["population"]["initial_size"] == 100


def test_the_default_run_spec_records_no_weight_vectors() -> None:
    result = vamos.optimize("c1dtlz1", n_obj=3, algorithm="ctaea", pop_size=100, max_evaluations=182, seed=1, engine="numpy")
    algorithm = result.meta["run_artifact_resolved_spec"]["algorithm"]

    assert "weight_vectors" not in algorithm["config"]
    assert algorithm["resolution"] == {"contract_version": 1}
    assert result.data["weights"].shape == (91, 3)


def test_a_two_objective_file_is_read_and_bound(tmp_path: Path) -> None:
    """C-TAEA reads a weight file with two objectives too, where MOEA/D's weights are uniform."""
    file = tmp_path / "W2D_20.dat"
    t = np.linspace(0.0, 1.0, 20)[::-1]
    np.savetxt(file, np.column_stack([t, 1.0 - t]))
    config = _builder(20).weight_vectors(path=str(tmp_path)).build()

    result = vamos.optimize(_ShiftedFront(), algorithm="ctaea", algorithm_config=config, max_evaluations=60, seed=1, engine="numpy")

    np.testing.assert_array_equal(result.data["weights"], np.maximum(np.column_stack([t, 1.0 - t]), MINIMUM_COMPONENT))
    bound = result.meta["run_artifact_resolved_spec"]["algorithm"]["config"]["weight_vectors"]
    assert bound["path"] == str(file) and bound["sha256"] == vector_digest(read_weight_vectors(str(file), 2))


def test_a_lattice_needs_exactly_pop_size_vectors() -> None:
    weights = configured_weight_vectors(91, 3, {"divisions": 12})
    np.testing.assert_array_equal(weights, ctaea_weight_vectors(91, 3))
    two_layers = configured_weight_vectors(105, 5, {"divisions": 4, "inner_divisions": 3})
    np.testing.assert_array_equal(two_layers, ctaea_weight_vectors(105, 5))

    with pytest.raises(ConfigurationError, match=r"divisions=12 has 91 weight vectors .* pop_size=100"):
        configured_weight_vectors(100, 3, {"divisions": 12})
    # Layers that repeat directions are rejected, as for MOEA/D, NSGA-III and RVEA.
    with pytest.raises(ConfigurationError, match="repeats"):
        configured_weight_vectors(20, 3, {"divisions": 3, "inner_divisions": 3})


def test_a_file_with_another_number_of_vectors_raises_before_any_evaluation() -> None:
    problem = _LinearC1DTLZ1(3, 7)
    config = _sbx_builder(100).weight_vectors(path=str(weight_path("W3D_91.dat"))).build().to_dict()

    with pytest.raises(ConfigurationError, match=r"W3D_91\.dat' holds 91 weight vectors .* pop_size=100"):
        CTAEA(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 1000), seed=1)
    assert problem.evaluations == 0


def test_a_missing_path_or_a_directory_without_the_file_raises(tmp_path: Path) -> None:
    problem = _LinearC1DTLZ1(3, 7)

    for path, message in ((tmp_path / "missing.dat", "does not exist"), (tmp_path, r"has no W3D_100\.dat")):
        config = _sbx_builder(100).weight_vectors(path=str(path)).build().to_dict()
        with pytest.raises(ConfigurationError, match=message):
            CTAEA(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 1000), seed=1)
    assert problem.evaluations == 0
    assert not (tmp_path / "W3D_100.dat").exists()


def test_a_recorded_digest_is_checked(tmp_path: Path) -> None:
    file = tmp_path / "W3D_100.dat"
    file.write_text(weight_path("W3D_100.dat").read_text(encoding="utf-8"), encoding="utf-8")
    digest = vector_digest(read_weight_vectors(str(file), 3))
    config = dataclasses.replace(_sbx_builder(100).build(), weight_vectors={"path": str(file), "sha256": digest})

    assert configured_weight_vectors(100, 3, config.weight_vectors).shape == (100, 3)
    np.savetxt(file, read_weight_vectors(str(file), 3)[::-1])
    with pytest.raises(ConfigurationError, match="no longer holds"):
        configured_weight_vectors(100, 3, config.weight_vectors)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"path": None, "divisons": 3}, "divisons"),
        ({"divisions": 0}, "positive integer"),
        ({"inner_divisions": 2}, "needs divisions"),
        ({"path": "W3D_100.dat", "divisions": 3, "inner_divisions": 2}, "file or a lattice"),
        ({"divisions": 12, "sha256": "sha256:" + "0" * 64}, "needs path"),
        (12, "mapping"),
    ],
)
def test_invalid_options_are_rejected_when_the_configuration_is_built(value: Any, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        dataclasses.replace(_sbx_builder(100).build(), weight_vectors=value)


def test_configurations_round_trip() -> None:
    path = str(weight_path("W3D_100.dat"))
    by_path = dataclasses.replace(_sbx_builder(100).build(), weight_vectors=path)
    by_lattice = _sbx_builder(105).weight_vectors(divisions=4, inner_divisions=3).build()

    assert by_path.weight_vectors == {"path": path, "divisions": None}
    assert by_path.to_dict()["weight_vectors"] == {"path": path, "divisions": None}
    for config in (by_path, by_lattice, CTAEAConfig.default(n_var=7)):
        assert CTAEAConfig.from_dict(json.loads(config.to_json())) == config


def test_the_budget_checks_count_one_evaluation_per_vector() -> None:
    source = {"path": str(weight_path("W3D_100.dat")), "divisions": None}
    assert initial_evaluations("ctaea", 100, n_obj=3, weight_vectors=source) == 100
    assert initial_evaluations("ctaea", 100, n_obj=3, weight_vectors=None) == 91
    config = _sbx_builder(100).weight_vectors(**source).build()

    with pytest.raises(ConfigurationError, match="max_evaluations must be >= pop_size"):
        vamos.optimize("c1dtlz1", n_obj=3, algorithm="ctaea", algorithm_config=config, max_evaluations=95, seed=1)
    with pytest.raises(ValueError, match=r"max_evaluations >= 100, the number of its weight_vectors for pop_size=100"):
        CTAEA(config.to_dict(), kernel=NumPyKernel()).run(_LinearC1DTLZ1(3, 7), ("max_evaluations", 95), seed=1)


def test_a_study_binds_the_file_and_sizes_the_budget_by_it() -> None:
    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    directory = str(weight_path("W3D_100.dat").parent)

    def spec(max_evaluations: int) -> Any:
        return vamos.StudySpec(
            problems=["c1dtlz1"],
            algorithms=["ctaea"],
            seeds=[0],
            pop_size=100,
            n_obj=3,
            max_evaluations=max_evaluations,
            engine="numpy",
            algorithm_configs={"ctaea": {"weight_vectors": {"path": directory, "divisions": None}}},
        )

    resolved = vamos.plan_study(spec(200)).plan.tasks[0].resolved_run_spec["algorithm"]
    file = str(weight_path("W3D_100.dat"))
    assert resolved["config"]["weight_vectors"] == {"path": file, "divisions": None, "sha256": vector_digest(read_weight_vectors(file, 3))}
    assert resolved["resolution"]["weight_vectors"] == {"source": "file", "file": "W3D_100.dat", "vectors": 100}
    # 100 individuals: 95 evaluations, enough for the 91 default vectors, are not enough for them.
    with pytest.raises(UnresolvedStudyTaskError, match="INVALID_EVALUATION_BUDGET"):
        vamos.plan_study(spec(95))


def test_a_default_study_task_has_no_weight_vectors() -> None:
    spec = vamos.StudySpec(problems=["c1dtlz1"], algorithms=["ctaea"], seeds=[0], pop_size=100, max_evaluations=182, engine="numpy")

    resolved = vamos.plan_study(spec).plan.tasks[0].resolved_run_spec["algorithm"]

    assert "weight_vectors" not in resolved["config"]
    assert "weight_vectors" not in resolved["resolution"]


def _study(weight_vectors: Any = None, *, set_field: bool = True) -> Any:
    configs = {"ctaea": {"weight_vectors": weight_vectors}} if set_field else None
    return vamos.StudySpec(
        problems=["c1dtlz1"],
        algorithms=["ctaea"],
        seeds=[0],
        pop_size=100,
        n_obj=3,
        max_evaluations=200,
        engine="numpy",
        algorithm_configs=configs,
    )


@pytest.mark.parametrize("where", ["file", "directory"])
def test_a_study_binds_a_path_string_as_the_mapping_and_runs_it(where: str, tmp_path: Path) -> None:
    """Execution rebuilds the task's configuration with CTAEAConfig, which makes a path a mapping, so planning does too."""
    from vamos.experiment.artifacts.resolved_reconstruction import reconstruct_resolved_run

    file = str(weight_path("W3D_100.dat"))
    path = file if where == "file" else str(weight_path("W3D_100.dat").parent)
    digest = vector_digest(read_weight_vectors(file, 3))

    task = vamos.plan_study(_study(path)).plan.tasks[0]
    mapping = vamos.plan_study(_study({"path": path, "divisions": None})).plan.tasks[0]

    assert task.resolved_run_spec["algorithm"]["config"]["weight_vectors"] == {"path": file, "divisions": None, "sha256": digest}
    assert task.task_id == mapping.task_id
    assert reconstruct_resolved_run(task.resolved_run_spec, root=tmp_path).algorithm_config.to_dict()["weight_vectors"]["sha256"] == digest
    completed = vamos.create_study(_study(path), output=tmp_path / "study").run()
    assert completed.status == "completed" and [item.state for item in completed.tasks] == ["succeeded"]
    reference = completed.attempts[0].run_reference
    assert reference is not None
    stored = vamos.load_run(completed.root / "runs" / str(reference["run_id"]), verify="all")
    assert stored.manifest.resolved_spec["algorithm"]["config"]["weight_vectors"]["sha256"] == digest
    assert stored.manifest["outcome"]["evaluations"] == 200


@pytest.mark.parametrize("value", [None, {}, {"path": None, "divisions": None}], ids=["none", "empty", "no-path-no-divisions"])
def test_a_study_whose_weight_vectors_name_no_vectors_plans_the_default_task(value: Any, tmp_path: Path) -> None:
    """CTAEAConfig.to_dict() leaves such a source out, so the task is the default one and reconstructs."""
    from vamos.experiment.artifacts.resolved_reconstruction import reconstruct_resolved_run

    default = vamos.plan_study(_study(set_field=False)).plan.tasks[0]
    task = vamos.plan_study(_study(value)).plan.tasks[0]

    assert task.task_id == default.task_id
    assert task.resolved_run_spec == default.resolved_run_spec
    assert "weight_vectors" not in task.resolved_run_spec["algorithm"]["config"]
    reconstructed = reconstruct_resolved_run(task.resolved_run_spec, root=tmp_path).algorithm_config
    assert isinstance(reconstructed, CTAEAConfig) and reconstructed.weight_vectors is None


def test_a_study_rejects_invalid_weight_vectors_when_it_is_planned() -> None:
    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    with pytest.raises(UnresolvedStudyTaskError, match="UNRESOLVED_TASK"):
        vamos.plan_study(_study({"file": "W3D_100.dat"}))


def test_the_command_line_builder_reads_and_binds_the_option() -> None:
    from vamos.engine.algorithm.factory import build_algorithm
    from vamos.foundation.core.experiment_config import ExperimentConfig

    problem = _LinearC1DTLZ1(3, 7)
    file = str(weight_path("W3D_100.dat"))
    variations = {"ctaea": {"weight_vectors": {"path": str(weight_path("W3D_100.dat").parent)}}}

    algorithm, resolved = build_algorithm("ctaea", "numpy", problem, ExperimentConfig(population_size=100), variations=variations)
    result = algorithm.run(problem, termination=("max_evaluations", 200), seed=1)

    assert isinstance(resolved, CTAEAConfig)
    assert resolved.weight_vectors == {"path": file, "sha256": vector_digest(read_weight_vectors(file, 3))}
    assert result["evaluations"] == 200 and result["weights"].shape == (100, 3)


def test_the_archives_take_vectors_in_any_order(tmp_path: Path) -> None:
    """The archives index the subregions by row and assume no lattice, no layers and no order: W3D_100.dat reversed runs too."""
    raw = read_weight_vectors(str(weight_path("W3D_100.dat")), 3)
    file = tmp_path / "reversed.dat"
    np.savetxt(file, raw[::-1], fmt="%.17g")
    problem = _LinearC1DTLZ1(3, 7)
    config = _sbx_builder(100).weight_vectors(path=str(file)).build().to_dict()

    result = CTAEA(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 500), seed=1)

    np.testing.assert_array_equal(result["weights"], np.maximum(raw[::-1], MINIMUM_COMPONENT))
    assert result["evaluations"] == problem.evaluations == 500
    assert result["population"]["X"].shape == (100, 7) and result["diversity_archive"]["X"].shape == (100, 7)
