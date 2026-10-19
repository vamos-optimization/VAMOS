"""Tests for the Panel pages of VAMOS Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos

pytest.importorskip("panel")

from vamos.ux.panel import app as studio_app  # noqa: E402
from vamos.ux.panel.components.algorithm_config import CUSTOM_MODE, AlgorithmConfigurator  # noqa: E402
from vamos.ux.panel.launcher import _build_panel_command  # noqa: E402
from vamos.ux.panel.pages import batch, experiment, problem_builder, results, welcome  # noqa: E402
from vamos.ux.panel.shell import PAGES  # noqa: E402

pytestmark = pytest.mark.studio


def test_every_registered_page_has_a_renderer_and_opens() -> None:
    app = studio_app.StudioApp()

    assert set(PAGES) == set(studio_app.PAGE_RENDERERS)
    for slug in PAGES:
        view = app._load_page(slug)
        assert type(view).__name__ != "Alert", f"page {slug} failed to open: {getattr(view, 'object', '')}"


def test_welcome_lists_every_page() -> None:
    text = welcome.welcome_markdown()

    for slug, (label, _summary) in PAGES.items():
        if slug != "welcome":
            assert label in text


def test_launcher_serves_the_app_module_with_its_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    served: dict[str, Any] = {}

    def fake_serve(panels: dict[str, Any], **kwargs: Any) -> None:
        served.update(kwargs, template=panels["app"]())

    command = _build_panel_command("studies/x", ["--allow-websocket-origin=host:5010"], port=5010, show=False)
    monkeypatch.setattr(studio_app.pn, "serve", fake_serve)

    assert command[1:3] == ["-m", "vamos.ux.panel.app"]
    assert studio_app.serve(command[3:]) == 0
    assert served["port"] == 5010
    assert served["show"] is False
    assert "host:5010" in served["websocket_origin"]
    assert type(served["template"]).__name__ == "FastListTemplate"


def test_ipv6_loopback_is_served_through_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    served: dict[str, Any] = {}
    monkeypatch.setattr(studio_app.pn, "serve", lambda panels, **kwargs: served.update(kwargs))

    studio_app.serve(["--address=::1", "--port=5010"])

    assert served["websocket_origin"] == ["localhost:5010"]


def test_app_module_rejects_unknown_server_options() -> None:
    with pytest.raises(SystemExit):
        studio_app.parse_serve_args(["--autoreload-everything"])


def test_configurator_offers_only_algorithms_for_the_encoding() -> None:
    configurator = AlgorithmConfigurator()

    configurator.set_problem(encoding="binary", n_var=30, n_obj=2)

    assert "smpso" not in configurator.algorithm.options.values()
    assert configurator.selected_algorithm == "nsgaii"


def test_experiment_runs_with_the_customized_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(problem: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        kwargs["live_viz"].on_generation(0, F=np.array([[0.0, 1.0]]))
        return {"F": np.array([[0.0, 1.0]])}

    monkeypatch.setattr(experiment, "_run_algorithm", fake_run)
    state = experiment.ExperimentState()
    state.configurator.mode.value = CUSTOM_MODE
    state.configurator.set_values({"crossover": "blx_alpha", "selection": "random"})

    state.run_experiment()

    config = captured["algorithm_config"].to_dict()
    assert config["crossover"][0] == "blx_alpha"
    assert config["selection"][0] == "random"
    assert "customized configuration" in state.status
    assert len(state.generation_history) == 1


def test_experiment_problem_change_updates_the_algorithm_form() -> None:
    state = experiment.ExperimentState()

    state.problem_name = "dtlz2"

    assert state.configurator.form is not None
    assert state.configurator.form.n_obj == 3
    assert "3 objectives" in state.problem_info


def test_problem_builder_full_run_uses_the_defined_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_preview(fn: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"F": np.array([[0.0, 1.0]]), "X": None, "history": [np.array([[0.0, 1.0]]), np.array([[0.1, 0.9]])]}

    monkeypatch.setattr(problem_builder, "run_preview_optimization", fake_preview)
    state = problem_builder.ProblemBuilderState()
    state.objective_code = "return [x[0], 1.0 - x[0]]"
    state.trusted_local_code = True

    state.run_full_experiment()

    assert captured["objective_code"] == "return [x[0], 1.0 - x[0]]"
    assert captured["record_history"] is True
    assert captured["timeout_seconds"] == state.time_limit
    assert captured["algorithm_config"] is not None
    assert len(state.generation_history) == 2


def test_problem_builder_template_brings_its_bounds_and_constraints() -> None:
    state = problem_builder.ProblemBuilderState()

    state.template_name = "Engineering: beam design (cost vs deflection)"

    assert state.bounds_text == "1.0, 10.0"
    assert "stress" in state.constraint_code
    assert state.n_constraints == 1


def test_generated_script_rebuilds_the_customized_configuration() -> None:
    state = problem_builder.ProblemBuilderState()
    state.configurator.mode.value = CUSTOM_MODE
    state.configurator.set_values({"crossover": "blx_alpha"})

    state.do_generate_script()
    namespace: dict[str, Any] = {"__name__": "generated_script"}
    exec(compile(state.generated_script, "<generated>", "exec"), namespace)  # noqa: S102 - script built by Studio itself

    assert "from vamos.algorithms import NSGAIIConfig" in state.generated_script
    assert namespace["ALGORITHM_CONFIG"].to_dict()["crossover"][0] == "blx_alpha"


def test_generated_script_includes_constraints() -> None:
    state = problem_builder.ProblemBuilderState()
    state.template_name = "Engineering: beam design (cost vs deflection)"

    state.do_generate_script()

    assert "n_constraints=1" in state.generated_script


def _run_in_process(spec: Any, output: Path) -> Any:
    """Stand-in for the background runner: create the study and run it here."""
    vamos.create_study(spec, output=output).run()
    return studio_app.pn  # any value; the page only refreshes afterwards


def test_batch_plans_creates_and_monitors_a_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "create_and_start", _run_in_process)
    state = batch.BatchState(
        study_dir=str(tmp_path), algorithms=["nsgaii", "spea2"], problems=["zdt1"], n_runs=2, max_evaluations=400, batch_name="b1"
    )

    state.check()
    assert state.plan_valid and "4 runs" in state.plan_text
    state.launch()

    assert state.status_kind == "success"
    assert state.selected == str((tmp_path / "b1").resolve())
    assert state.progress.state == "completed"
    assert state.actions.open_results and not state.actions.stop
    assert {row["Algorithm"] for row in state.summary} == {"NSGA-II", "SPEA2"}
    assert state.summary_csv().getvalue().startswith("Problem,Algorithm,Runs")
    assert r"\begin{tabular}" in state.summary_tex().getvalue()
    assert "b1" in batch.progress_markdown(state.progress)


def test_batch_refuses_an_impossible_plan(tmp_path: Path) -> None:
    state = batch.BatchState(study_dir=str(tmp_path), algorithms=["nsgaiii"], problems=["zdt1"], default_population=False, pop_size=100)

    state.launch()

    assert not state.plan_valid
    assert state.status_kind == "warning"
    assert not any(tmp_path.iterdir())


def test_batch_lists_existing_studies_and_their_failures(tmp_path: Path) -> None:
    spec = vamos.StudySpec(
        problems=["dtlz2"], algorithms=["moead"], seeds=[0], pop_size=40, max_evaluations=200, engine="numpy", on_error="continue"
    )
    vamos.create_study(spec, output=tmp_path / "failing").run()

    state = batch.BatchState(study_dir=str(tmp_path))
    state.refresh_studies()

    assert list(state.study_options().values()) == [str((tmp_path / "failing").resolve())]
    assert state.progress.state == "completed_with_failures"
    assert state.actions.retry_failed  # The study classifies this execution failure as retryable.
    assert state.progress.failures[0].label() == "dtlz2 / MOEA/D, seed 0"


def test_app_hands_a_batch_to_explore_results_and_stops_page_work(tmp_path: Path) -> None:
    study = vamos.create_study(
        vamos.StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[0], pop_size=8, max_evaluations=40, engine="numpy"),
        output=tmp_path / "done",
    ).run()
    app = studio_app.StudioApp(study_dir=str(tmp_path))
    stopped: list[str] = []
    app.on_page_exit(lambda: stopped.append("batch"))

    app.open_results(str(study.root))

    assert stopped == ["batch"]
    assert app.page == "results"
    assert app.focus_study == ""


def test_results_reports_a_missing_study_instead_of_showing_demo_data(tmp_path: Path) -> None:
    state = results.ResultsState(study_dir=str(tmp_path / "missing"))

    state.load_data()

    assert state.status_kind == "danger"
    assert state.fronts == []


def test_results_finds_loads_and_exports_a_study(tmp_path: Path) -> None:
    spec = vamos.StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[0], pop_size=8, max_evaluations=40, engine="numpy")
    vamos.create_study(spec, output=tmp_path / "study").run()
    state = results.ResultsState(study_dir=str(tmp_path))

    state.find_studies()
    state.load_data()

    assert state.selected_study == str((tmp_path / "study").resolve())
    assert state.problem == "zdt1"
    exported = state.ranking_csv().getvalue().splitlines()
    assert exported[0].startswith("Problem,Algorithm,Rank,Score")
    assert len(exported) - 1 == len(state.ranking().rows)


def test_results_labels_demo_data() -> None:
    state = results.ResultsState()

    state.load_demo()

    assert state.source == "demo"
    assert state.status_kind == "warning"


def test_results_explains_a_study_without_completed_runs(tmp_path: Path) -> None:
    # MOEA/D needs 91 weight vectors for three objectives, so this single task fails and the study pauses.
    spec = vamos.StudySpec(problems=["dtlz2"], algorithms=["moead"], seeds=[0], pop_size=40, max_evaluations=200, engine="numpy")
    vamos.create_study(spec, output=tmp_path / "paused").run()
    state = results.ResultsState(study_dir=str(tmp_path / "paused"))

    state.load_data()

    assert state.status_kind == "warning"
    assert "1 failed" in state.status
    assert "vamos study resume" in state.status


def test_compare_page_analyzes_a_study(tmp_path: Path) -> None:
    from vamos.ux.panel.pages import compare

    spec = vamos.StudySpec(
        problems=["zdt1"], algorithms=["nsgaii", "spea2"], seeds=[0, 1, 2], pop_size=8, max_evaluations=80, engine="numpy"
    )
    study = vamos.create_study(spec, output=tmp_path / "pair").run()
    state = compare.CompareState(study_dir=str(tmp_path))

    state.refresh_studies()

    assert state.selected == str(study.root.resolve())
    assert state.pivot == "NSGA-II" and state.status_kind == "success"
    assert "smallest possible p-value" in state.power_warning()
    for view in (compare.verdict_view, compare.comparisons_view, compare.ranking_view, compare.cost_view):
        assert type(view(state)).__name__ == "Column"
    for view in (compare.effect_view, compare.distribution_view, compare.fronts_view, compare.attainment_view):
        assert type(view(state)).__name__ == "Plotly"
    assert r"\begin{tabular}" in state.comparison_tex().getvalue()
    assert state.comparisons_csv().getvalue().startswith("Problem,Algorithm,Pivot")
    assert compare.smallest_p_value(5, 5) == pytest.approx(2 / 252)


def test_app_hands_a_batch_to_the_comparison(tmp_path: Path) -> None:
    app = studio_app.StudioApp(study_dir=str(tmp_path))

    app.open_compare(str(tmp_path / "some-study"))

    assert app.page == "compare"
    assert app.focus_study == ""
