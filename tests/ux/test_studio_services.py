from pathlib import Path

import numpy as np

import vamos
from vamos.ux.studio.services import DynamicsCallback, build_demo_study_data, discover_study_directories, load_studio_data


def test_dynamics_final_front_replaces_the_last_recorded_generation() -> None:
    callback = DynamicsCallback()
    callback.on_generation(1, F=np.zeros((3, 2)))
    callback.on_generation(2, F=np.ones((3, 2)))
    final = np.array([[0.2, 0.8]])
    callback.on_end(final_F=final)
    assert len(callback.history) == 2
    np.testing.assert_array_equal(callback.history[-1], final)


def test_dynamics_records_an_uncaptured_final_generation() -> None:
    callback = DynamicsCallback()
    callback.on_generation(1, F=np.zeros((3, 2)))
    callback.on_generation(2)
    callback.on_end(final_F=np.ones((1, 2)))
    assert len(callback.history) == 2
    np.testing.assert_array_equal(callback.history[0], np.zeros((3, 2)))


def test_dynamics_records_the_final_front_without_generation_callbacks() -> None:
    callback = DynamicsCallback()
    callback.on_end(final_F=np.ones((1, 2)))
    assert len(callback.history) == 1


def test_discover_study_directories_finds_canonical_roots(tmp_path: Path) -> None:
    study_dir = tmp_path / "results" / "canonical"
    vamos.create_study(
        vamos.StudySpec(
            problems=["zdt1"],
            algorithms=["nsgaii"],
            seeds=[0],
            pop_size=4,
            max_evaluations=4,
            engine="numpy",
        ),
        output=study_dir,
    )

    study_dirs = discover_study_directories(tmp_path)

    assert study_dirs == [study_dir]


def test_load_studio_data_uses_summary_run_traceability(tmp_path: Path) -> None:
    study = vamos.create_study(
        vamos.StudySpec(
            problems=["zdt1"],
            algorithms=["nsgaii"],
            seeds=[0],
            pop_size=4,
            max_evaluations=4,
            engine="numpy",
        ),
        output=tmp_path / "study",
    ).run()

    runs, fronts = load_studio_data(study.root)

    assert len(runs) == len(fronts) == 1
    trace = runs[0].metadata["study_summary"]
    assert trace["study_id"] == study.study_id
    assert trace["selected_run_id"] == runs[0].experiment_id


def test_build_demo_study_data_marks_demo_fronts() -> None:
    runs, fronts = build_demo_study_data()

    assert len(runs) == 2
    assert len(fronts) == 2
    assert all(front.extra.get("demo") for front in fronts)
    assert fronts[0].points_F.shape[1] == 2


def test_studio_run_closes_only_the_backends_it_builds(monkeypatch) -> None:
    import vamos.ux.studio.services as services
    from vamos.engine.algorithm.config import NSGAIIConfig
    from vamos.foundation.eval.backends import SerialEvalBackend
    from vamos.foundation.problem.zdt1 import ZDT1Problem

    class ClosingSpy(SerialEvalBackend):
        closed = 0

        def close(self) -> None:
            self.closed += 1

    built: list[ClosingSpy] = []
    monkeypatch.setattr(services, "resolve_eval_strategy", lambda name, **kwargs: built.append(ClosingSpy()) or built[-1])
    run = {
        "algorithm": "nsgaii",
        "algorithm_config": NSGAIIConfig.default(pop_size=10, n_var=6),
        "termination": ("max_evaluations", 20),
        "seed": 1,
        "engine": "numpy",
    }

    services._run_algorithm(ZDT1Problem(n_var=6), eval_strategy="multiprocessing", **run)
    assert len(built) == 1 and built[0].closed == 1

    own = ClosingSpy()
    services._run_algorithm(ZDT1Problem(n_var=6), eval_strategy=own, **run)
    assert own.closed == 0


def test_studio_and_optimize_build_the_same_default_configs() -> None:
    """Studio and optimize share one default-configuration builder (VR-109).

    The only difference is the result mode of SMPSO: optimize returns its leaders archive, while
    Studio shows the final swarm, as for every other algorithm.
    """
    import dataclasses

    import pytest

    from vamos.exceptions import InvalidAlgorithmError
    from vamos.experiment.optimize import _build_algorithm_config as optimize_config
    from vamos.ux.studio.services import _build_algorithm_config as studio_config

    shapes = ((None, None, None), (40, 12, 2), (None, 30, 3), (24, None, None))
    for algorithm in ("agemoea", "ibea", "moead", "nsgaii", "nsgaiii", "rvea", "smpso", "smsemoa", "spea2", "NSGAII"):
        for encoding in (None, "real", "binary", "integer", "permutation", "mixed"):
            for pop_size, n_var, n_obj in shapes:
                arguments = {"pop_size": pop_size, "n_var": n_var, "n_obj": n_obj, "encoding": encoding}
                try:
                    expected = optimize_config(algorithm, **arguments)
                except ValueError as exc:
                    with pytest.raises(type(exc)):
                        studio_config(algorithm, **arguments)
                    continue
                if algorithm == "smpso":
                    assert expected.result_mode == "non_dominated"
                    expected = dataclasses.replace(expected, result_mode="population")
                assert studio_config(algorithm, **arguments) == expected, (algorithm, arguments)
    for build in (optimize_config, studio_config):
        with pytest.raises(InvalidAlgorithmError):
            build("not_an_algorithm", pop_size=None, n_var=None, n_obj=None, encoding=None)
