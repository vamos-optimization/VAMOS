"""Options of a registered problem, besides its dimensions, given by name through ``problem_kwargs``."""

from __future__ import annotations

import inspect
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.artifacts.errors import ComponentNotReconstructableError
from vamos.experiment.artifacts.jsonio import canonical_json_bytes, manifest_self_hash, sha256_bytes, stored_json_bytes
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection

# VAMOS 1.5.2: the sorted variant of RCM30's reformulation has 25 variables, one fewer than the default variant.
_CASES = [("bin_knapsack", {"capacity_ratio": 0.6}), ("zcat1", {"level": 2}), ("rcm30_reformulated", {"variant": "sorted"})]


def _optimize(name: str, **kwargs: Any) -> vamos.OptimizationResult:
    return vamos.optimize(name, algorithm="nsgaii", pop_size=8, max_evaluations=16, engine="numpy", seed=3, **kwargs)


@pytest.mark.parametrize(("name", "options"), _CASES, ids=[name for name, _ in _CASES])
def test_options_are_recorded_and_replay_exactly(name: str, options: dict[str, object], tmp_path: Path) -> None:
    result = _optimize(name, problem_kwargs=options)

    problem = result.meta["run_artifact_resolved_spec"]["problem"]
    assert problem["component_id"] == f"vamos.problem:{name}@1"
    assert problem["provider"]["type"] == "built_in"
    assert {key: problem["config"][key] for key in options} == options
    # The option changes the problem, so the objectives differ from the default instance's.
    assert not np.array_equal(result.F, _optimize(name).F)

    source = vamos.save_result(result, tmp_path / "source")
    verification = vamos.verify_run(source.root, require_level="exact")
    assert verification.component_reconstructability == "reconstructable"
    assert verification.effective_replayability == "exact"
    replay = vamos.reproduce(source.root, output=tmp_path / "replay")
    assert replay.exact is True
    assert vamos.load_run(replay.output_root).manifest.resolved_spec["problem"] == source.manifest.resolved_spec["problem"]


def test_options_build_the_same_instance_as_the_class() -> None:
    selection = make_problem_selection("bin_knapsack").with_options({"capacity_ratio": 0.6})
    knapsack = selection.instantiate()
    assert knapsack.capacity == pytest.approx(0.6 * knapsack.weights.sum())

    zcat = make_problem_selection("zcat1", n_obj=3).with_options({"level": 2, "bias": True}).instantiate()
    assert (zcat.n_obj, zcat.level, zcat.bias) == (3, 2, True)


def test_a_run_without_options_records_only_the_dimensions() -> None:
    config = _optimize("zcat1").meta["run_artifact_resolved_spec"]["problem"]["config"]
    assert set(config) == {"n_var", "n_obj", "encoding", "constraint_convention"}


def test_an_option_the_factory_does_not_declare_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match=r"n_var, n_obj and capacity_ratio; got capacity\b"):
        _optimize("bin_knapsack", problem_kwargs={"capacity": 0.6})
    with pytest.raises(ConfigurationError, match=r"accepts no options"):
        make_problem_selection("zdt1").with_options({"level": 2})
    with pytest.raises(ConfigurationError, match=r"dimensions, not options"):
        make_problem_selection("bin_knapsack").with_options({"n_var": 10})


@pytest.mark.parametrize("value", [object(), float("nan"), [2]], ids=["object", "nan", "list"])
def test_an_option_that_replay_cannot_rebuild_is_rejected(value: object) -> None:
    with pytest.raises(ConfigurationError, match="level"):
        _optimize("zcat1", problem_kwargs={"level": value})


def test_a_number_that_the_artifact_would_round_is_rejected() -> None:
    # The artifact records float(value), 2.0, but ZCAT builds level int(value), 1: replay would build another problem.
    value = Fraction(2**54 - 1, 2**53)
    with pytest.raises(ConfigurationError, match="level"):
        _optimize("zcat1", problem_kwargs={"level": value})
    with pytest.raises(ConfigurationError, match="level"):
        make_problem_selection("zcat1").with_options({"level": value})


def test_a_number_of_another_type_is_built_as_the_artifact_records_it(tmp_path: Path) -> None:
    selection = make_problem_selection("zcat1").with_options({"level": np.int64(2), "bias": True})
    assert dict(selection.options) == {"level": 2, "bias": True}
    assert type(selection.options["level"]) is int
    ratio = make_problem_selection("bin_knapsack").with_options({"capacity_ratio": np.float32(0.5)}).options["capacity_ratio"]
    assert type(ratio) is float and ratio == 0.5

    result = _optimize("zcat1", problem_kwargs={"level": Fraction(4, 2)})
    assert result.meta["run_artifact_resolved_spec"]["problem"]["config"]["level"] == 2.0
    replay = vamos.reproduce(vamos.save_result(result, tmp_path / "source").root, output=tmp_path / "replay")
    assert replay.exact is True


def test_replay_refuses_an_option_the_factory_does_not_declare(tmp_path: Path) -> None:
    source = vamos.save_result(_optimize("bin_knapsack", problem_kwargs={"capacity_ratio": 0.6}), tmp_path / "source")
    manifest = json.loads((source.root / "manifest.json").read_text(encoding="utf-8"))
    manifest["resolved_spec"]["problem"]["config"]["capacity"] = 0.6
    manifest["task_id"] = "sha256:" + sha256_bytes(canonical_json_bytes(manifest["resolved_spec"]))
    manifest["integrity"]["manifest_sha256"] = manifest_self_hash(manifest)
    (source.root / "manifest.json").write_bytes(stored_json_bytes(manifest))

    assert vamos.verify_run(source.root).effective_replayability != "exact"
    with pytest.raises(ComponentNotReconstructableError):
        vamos.reproduce(source.root, output=tmp_path / "replay")


def test_a_study_runs_the_problem_with_its_options(tmp_path: Path) -> None:
    spec = vamos.StudySpec(
        problems=["zcat1"],
        algorithms=["nsgaii"],
        seeds=[3],
        max_evaluations=16,
        pop_size=8,
        engine="numpy",
        problem_kwargs={"level": 2},
    )
    plan = vamos.plan_study(spec)
    assert plan.status == "ready"
    assert plan.plan.tasks[0].resolved_run_spec["problem"]["config"]["level"] == 2

    completed = vamos.create_study(spec, output=tmp_path / "study").run()
    reference = completed.attempts[0].run_reference
    assert reference is not None
    stored = vamos.load_result(tmp_path / "study" / "runs" / str(reference["run_id"]))
    assert np.array_equal(stored.F, _optimize("zcat1", problem_kwargs={"level": 2}).F)


def test_factory_options_default_to_the_constructor_defaults() -> None:
    # A registry factory declares a problem's options as keyword-only parameters; its defaults are the class's.
    with_options = 0
    for key, spec in get_problem_specs().items():
        defaults = spec.option_defaults()
        if not defaults:
            continue
        with_options += 1
        instance = make_problem_selection(key).instantiate()
        constructor = inspect.signature(type(instance)).parameters
        assert {name: constructor[name].default for name in defaults} == defaults, key
    assert with_options >= 21
