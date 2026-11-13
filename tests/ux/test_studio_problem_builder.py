"""Tests for the VAMOS Studio Problem Builder backend logic."""

from __future__ import annotations

import multiprocessing as mp
import signal
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vamos.ux.studio import _problem_builder_preview as preview
from vamos.ux.studio.problem_builder_backend import (
    compile_objective_function,
    example_objectives,
    generate_script,
    parse_bounds_text,
    run_preview_optimization,
)

# ======================================================================
# compile_objective_function
# ======================================================================


def _send_large_preview_history(connection: Any) -> None:
    history = [np.full((512, 2), generation, dtype=float) for generation in range(256)]
    connection.send({"ok": True, "payload": {"history": history}})
    connection.close()


def _preview_worker_without_threads(**kwargs: Any) -> None:
    with patch("threading.Thread.start", side_effect=RuntimeError("can't start new thread")):
        preview._preview_worker(**kwargs)


def _exceed_preview_cpu_limit(_connection: Any) -> None:
    import math
    import resource

    # Spawn imports consume CPU too. Budget from the current usage so this
    # test reaches SIGXCPU rather than installing an already-exceeded hard limit.
    usage = resource.getrusage(resource.RUSAGE_SELF)
    soft_limit = math.ceil(usage.ru_utime + usage.ru_stime) + 1
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (soft_limit, soft_limit + 1))
    while True:
        pass


@pytest.mark.skipif(not hasattr(signal, "SIGXCPU"), reason="CPU resource-limit signals require POSIX")
def test_preview_cpu_limit_is_reported_before_the_wall_clock_deadline() -> None:
    ctx = mp.get_context("spawn")
    connection, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_exceed_preview_cpu_limit, args=(sender,))
    process.start()
    sender.close()
    try:
        with pytest.raises(TimeoutError, match="CPU time limit"):
            preview._await_worker_result(connection, process, 30.0)
        assert process.exitcode == -signal.SIGXCPU
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        connection.close()


@pytest.mark.parametrize("pipe_ready", [True, False])
def test_preview_cpu_limit_exit_is_a_timeout_on_both_receive_paths(pipe_ready: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Mock()
    connection.poll.return_value = pipe_ready
    connection.recv.side_effect = EOFError
    process = Mock(exitcode=-24)
    process.is_alive.return_value = False
    monkeypatch.setattr(preview.signal, "SIGXCPU", 24, raising=False)
    monkeypatch.setattr(preview.time, "monotonic", lambda: 0.0)

    with pytest.raises(TimeoutError, match="CPU time limit"):
        preview._await_worker_result(connection, process, 30.0)


def test_preview_receives_history_larger_than_the_process_pipe() -> None:
    ctx = mp.get_context("spawn")
    connection, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_send_large_preview_history, args=(sender,))
    process.start()
    sender.close()
    try:
        result = preview._await_worker_result(connection, process, 60.0)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        connection.close()

    assert process.exitcode == 0
    assert isinstance(result, dict) and result["ok"] is True
    history = result["payload"]["history"]
    assert len(history) == 256
    assert sum(front.nbytes for front in history) > 1024 * 1024
    for generation, front in enumerate(history):
        np.testing.assert_array_equal(front, np.full((512, 2), generation))


@pytest.mark.parametrize("alive", [True, False])
def test_preview_deadline_is_reported_even_if_the_worker_just_exited(alive: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Mock()
    connection.poll.return_value = False
    process = Mock(exitcode=-9)
    process.is_alive.return_value = alive
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(preview.time, "monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="timed out"):
        preview._await_worker_result(connection, process, 0.5)
    assert process.terminate.call_count == int(alive)
    process.join.assert_called_once()


def test_preview_crash_before_the_deadline_is_not_reported_as_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Mock()
    connection.poll.return_value = False
    process = Mock(exitcode=-9)
    process.is_alive.return_value = False
    monkeypatch.setattr(preview.time, "monotonic", lambda: 0.0)

    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        preview._await_worker_result(connection, process, 10.0)


@pytest.mark.parametrize("exitcode", [0, 1])
def test_preview_closed_pipe_is_reported_without_waiting_for_the_deadline(exitcode: int) -> None:
    connection = Mock()
    connection.poll.return_value = True
    connection.recv.side_effect = EOFError
    process = Mock(exitcode=exitcode)

    message = "exited unexpectedly" if exitcode else "without returning a result"
    with pytest.raises(RuntimeError, match=message):
        preview._await_worker_result(connection, process, 10.0)


def test_preview_closed_pipe_at_the_deadline_is_reported_as_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Mock()
    connection.poll.return_value = True
    connection.recv.side_effect = EOFError
    process = Mock(exitcode=-9)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(preview.time, "monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="timed out"):
        preview._await_worker_result(connection, process, 0.5)


class TestCompileObjectiveFunction:
    """Test the user-code compilation helper."""

    def test_simple_function(self) -> None:
        fn = compile_objective_function("return [x[0], 1 - x[0]]", trusted_local_code=True)
        result = fn(np.array([0.3, 0.5]))
        assert len(result) == 2
        assert abs(result[0] - 0.3) < 1e-9
        assert abs(result[1] - 0.7) < 1e-9

    def test_multiline_with_math(self) -> None:
        code = "import math\nf0 = x[0] ** 2\nf1 = math.sqrt(x[0])\nreturn [f0, f1]"
        fn = compile_objective_function(code, trusted_local_code=True)
        result = fn(np.array([4.0]))
        assert abs(result[0] - 16.0) < 1e-9
        assert abs(result[1] - 2.0) < 1e-9

    def test_numpy_available(self) -> None:
        code = "return [float(np.sum(x)), float(np.prod(x))]"
        fn = compile_objective_function(code, trusted_local_code=True)
        result = fn(np.array([2.0, 3.0]))
        assert abs(result[0] - 5.0) < 1e-9
        assert abs(result[1] - 6.0) < 1e-9

    def test_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            compile_objective_function("return [x[0] +]", trusted_local_code=True)

    def test_disallow_unsafe_import(self) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            compile_objective_function("import os\nreturn [x[0], x[1]]", trusted_local_code=True)

    def test_disallow_unsafe_builtin(self) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            compile_objective_function("open('tmp.txt', 'w')\nreturn [x[0], x[1]]", trusted_local_code=True)

    def test_all_templates_compile(self) -> None:
        """Every built-in template must compile without error."""
        rng = np.random.default_rng(0)
        for name, template in example_objectives().items():
            fn = compile_objective_function(template["code"], trusted_local_code=True)
            n_var = int(template["n_var"])
            x = rng.random(n_var)
            result = fn(x)
            assert isinstance(result, list), f"Template '{name}' must return a list"
            assert len(result) == int(template["n_obj"]), f"Template '{name}' objective count mismatch"


# ======================================================================
# _parse_bounds_text
# ======================================================================


class TestParseBoundsText:
    """Test the bounds textarea parser."""

    def test_single_line(self) -> None:
        result = parse_bounds_text("0.0, 1.0", 3)
        assert result == [(0.0, 1.0)] * 3

    def test_multi_line(self) -> None:
        text = "0.0, 1.0\n-5.0, 5.0"
        result = parse_bounds_text(text, 2)
        assert result == [(0.0, 1.0), (-5.0, 5.0)]

    def test_with_parens(self) -> None:
        result = parse_bounds_text("(0, 10)", 2)
        assert result == [(0.0, 10.0)] * 2

    def test_error_wrong_count(self) -> None:
        result = parse_bounds_text("0,1\n0,1\n0,1", 2)
        assert isinstance(result, str)
        assert "Expected" in result

    def test_error_inverted(self) -> None:
        result = parse_bounds_text("5.0, 0.0", 1)
        assert isinstance(result, str)
        assert "lower" in result.lower() or "Lower" in result

    def test_error_not_numbers(self) -> None:
        result = parse_bounds_text("a, b", 1)
        assert isinstance(result, str)


# ======================================================================
# _generate_script
# ======================================================================


class TestGenerateScript:
    """Test the export script generator."""

    def test_valid_python(self) -> None:
        script = generate_script(
            "return [x[0], 1 - x[0]]",
            name="test_problem",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="nsgaii",
            budget=5000,
        )
        # Must be valid Python
        compile(script, "<test>", "exec")

    def test_contains_make_problem(self) -> None:
        script = generate_script(
            "return [x[0]]",
            name="test",
            n_var=1,
            n_obj=1,
            bounds=[(0.0, 1.0)],
            algorithm="nsgaii",
            budget=3000,
        )
        assert "make_problem" in script
        assert "optimize" in script
        assert "max_evaluations=3000" in script

    def test_contains_problem_name(self) -> None:
        script = generate_script(
            "return [x[0]]",
            name="my cool problem",
            n_var=1,
            n_obj=1,
            bounds=[(0.0, 1.0)],
            algorithm="moead",
            budget=5000,
        )
        assert "my cool problem" in script
        assert 'algorithm="moead"' in script


# ======================================================================
# run_preview_optimization (integration)
# ======================================================================


class TestRunPreviewOptimization:
    """Integration test: compile + optimize end-to-end."""

    def test_basic_preview(self) -> None:
        fn = compile_objective_function("return [x[0], 1 - x[0]]", trusted_local_code=True)
        result = run_preview_optimization(
            fn,
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="nsgaii",
            budget=500,
            pop_size=20,
            seed=42,
            trusted_local_code=True,
        )
        assert "F" in result
        assert result["F"].shape[1] == 2
        assert result["F"].shape[0] > 0
        assert result["elapsed_ms"] > 0

    def test_three_objective(self) -> None:
        code = "return [x[0], x[1], 1 - x[0] - x[1]]"
        fn = compile_objective_function(code, trusted_local_code=True)
        result = run_preview_optimization(
            fn,
            n_var=3,
            n_obj=3,
            bounds=[(0.0, 1.0)] * 3,
            algorithm="nsgaii",
            budget=500,
            pop_size=20,
            seed=0,
            trusted_local_code=True,
        )
        assert result["F"].shape[1] == 3

    def test_preview_wall_clock_timeout_for_non_terminating_code(self) -> None:
        fn = compile_objective_function("return [x[0], x[1]]", trusted_local_code=True)
        with pytest.raises(TimeoutError, match="timed out"):
            run_preview_optimization(
                fn,
                n_var=2,
                n_obj=2,
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                algorithm="nsgaii",
                budget=200,
                pop_size=20,
                seed=0,
                objective_code="while True:\n    pass\nreturn [x[0], x[1]]",
                timeout_seconds=0.5,
                resource_profile="none",  # Isolate the wall-clock watchdog from CPU-limit signals.
                trusted_local_code=True,
            )

    def test_invalid_resource_profile(self) -> None:
        fn = compile_objective_function("return [x[0], x[1]]", trusted_local_code=True)
        with pytest.raises(ValueError, match="Unknown resource-limit profile"):
            run_preview_optimization(
                fn,
                n_var=2,
                n_obj=2,
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                algorithm="nsgaii",
                budget=200,
                pop_size=20,
                seed=0,
                objective_code="return [x[0], x[1]]",
                resource_profile="invalid_profile",
                trusted_local_code=True,
            )

    @pytest.mark.parametrize("allow_threads", [True, False])
    def test_child_process_records_generation_history(self, allow_threads: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        from vamos.ux.studio.algorithm_form import default_algorithm_config

        code = "return [x[0], 1.0 - x[0] + sum(x[1:])]"
        fn = compile_objective_function(code, trusted_local_code=True)
        config = default_algorithm_config("nsgaii", encoding="real", n_var=10, n_obj=2, pop_size=20)
        if not allow_threads:
            monkeypatch.setattr(preview, "_preview_worker", _preview_worker_without_threads)
        result = run_preview_optimization(
            fn,
            n_var=10,
            n_obj=2,
            bounds=[(0.0, 1.0)] * 10,
            algorithm="nsgaii",
            budget=400,
            pop_size=20,
            seed=0,
            objective_code=code,
            timeout_seconds=120.0,
            trusted_local_code=True,
            algorithm_config=config,
            record_history=True,
        )

        assert len(result["history"]) >= 15
        assert all(F.shape[1] == 2 and 0 < F.shape[0] <= 20 for F in result["history"])

    def test_child_process_reports_errors_without_starting_threads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(preview, "_preview_worker", _preview_worker_without_threads)
        code = "raise ValueError('intentional preview failure')"
        fn = compile_objective_function(code, trusted_local_code=True)

        with pytest.raises(RuntimeError, match="intentional preview failure"):
            run_preview_optimization(
                fn,
                n_var=2,
                n_obj=2,
                bounds=[(0.0, 1.0)] * 2,
                algorithm="nsgaii",
                budget=20,
                pop_size=20,
                seed=0,
                objective_code=code,
                timeout_seconds=20.0,
                trusted_local_code=True,
            )


class TestGenerateScriptConfiguration:
    """The generated script runs the configuration Studio ran."""

    def test_default_configuration_passes_the_population_size(self) -> None:
        script = generate_script(
            "return [x[0], 1 - x[0]]", name="t", n_var=1, n_obj=2, bounds=[(0, 1)], algorithm="nsgaii", budget=1000, pop_size=40
        )

        assert "pop_size=40," in script
        assert "ALGORITHM_CONFIG" not in script

    def test_customized_configuration_is_rebuilt_from_its_dictionary(self) -> None:
        from vamos.algorithms import NSGAIIConfig

        config = NSGAIIConfig.default(pop_size=40, n_var=1).to_dict()
        script = generate_script(
            "return [x[0], 1 - x[0]]",
            name="t",
            n_var=1,
            n_obj=2,
            bounds=[(0, 1)],
            algorithm="nsgaii",
            budget=1000,
            algorithm_config=config,
            config_class="NSGAIIConfig",
        )
        namespace: dict[str, object] = {"__name__": "generated"}
        exec(compile(script, "<generated>", "exec"), namespace)  # noqa: S102 - script built from fixed test inputs

        assert "algorithm_config=ALGORITHM_CONFIG," in script
        rebuilt = namespace["ALGORITHM_CONFIG"]
        assert isinstance(rebuilt, NSGAIIConfig)
        assert rebuilt.to_dict() == config

    def test_unscriptable_configuration_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot reproduce"):
            generate_script(
                "return [x[0], 1 - x[0]]",
                name="t",
                n_var=1,
                n_obj=2,
                bounds=[(0, 1)],
                algorithm="nsgaii",
                budget=1000,
                algorithm_config={"generation_callback": object()},
                config_class="NSGAIIConfig",
            )


# ======================================================================
# Generated script and LLM drafts run what Studio runs
# ======================================================================


def test_generated_script_runs_a_numpy_objective(tmp_path: Any) -> None:
    import subprocess
    import sys

    from vamos.algorithms import NSGAIIConfig

    config = NSGAIIConfig.default(pop_size=8, n_var=3)
    # Studio evaluates this code with math and numpy (as np) in scope; the script must too.
    code = "f0 = float(np.sum(x ** 2))\nf1 = math.fsum((v - 1.0) ** 2 for v in x)\nreturn [f0, f1]"
    script = generate_script(
        code,
        name="numpy objective",
        n_var=3,
        n_obj=2,
        bounds=[(0.0, 1.0)] * 3,
        algorithm="nsgaii",
        budget=24,
        seed=7,
        algorithm_config=dict(config.to_dict()),
        config_class="NSGAIIConfig",
    )
    path = tmp_path / "problem.py"
    path.write_text(script, encoding="utf-8")

    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=240)

    assert proc.returncode == 0, proc.stderr
    assert "seed=7," in script
    assert "seed=42" not in script
    assert "non-dominated" not in script
    assert "solutions" in proc.stdout


def _install_fake_provider(monkeypatch: pytest.MonkeyPatch, provider: str, text: str) -> list[str]:
    import sys
    from types import ModuleType, SimpleNamespace

    calls: list[str] = []

    class FakeAnthropic:
        def __init__(self, **_kwargs: object) -> None:
            self.messages = SimpleNamespace(
                create=lambda **_kw: calls.append("anthropic") or SimpleNamespace(content=[SimpleNamespace(text=text)])
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.responses = SimpleNamespace(create=lambda **_kw: calls.append("openai") or SimpleNamespace(output_text=text))

    module = ModuleType(provider)
    module.Anthropic = FakeAnthropic  # type: ignore[attr-defined]
    module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, provider, module)
    return calls


_DRAFT = '{"objective_code": "return [x[0], 1 - x[0]]", "constraint_code": "", "n_var": 3, "n_obj": 2, "bounds": "0.0, 1.0"}'


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_llm_draft_in_a_code_fence_is_accepted_from_every_provider(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    from vamos.ux.studio import _studio_llm

    _install_fake_provider(monkeypatch, provider, "```json\n" + _DRAFT + "\n```")

    draft = _studio_llm.llm_generate_problem_code("a problem", provider=provider, api_key="test-key")

    assert draft["n_var"] == 3
    assert draft["objective_code"] == "return [x[0], 1 - x[0]]"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('"return [x[0], x[1]]"', "JSON object"),
        ('{"objective_code": "return [x[0]]", "n_var": "three", "n_obj": 2, "bounds": "0, 1"}', "n_var"),
        ('{"objective_code": "return [x[0]]", "n_var": 2, "n_obj": 1, "bounds": "0, 1"}', "n_obj"),
        ('{"objective_code": 5, "n_var": 2, "n_obj": 2, "bounds": "0, 1"}', "objective_code"),
        ('{"objective_code": "return [x[0]]", "n_var": 2, "n_obj": 2, "bounds": [0, 1]}', "bounds"),
    ],
)
def test_llm_draft_parsing_rejects_strings_and_wrong_types_without_touching_the_form(
    monkeypatch: pytest.MonkeyPatch, text: str, message: str
) -> None:
    pytest.importorskip("panel")
    from vamos.ux.panel.pages import problem_builder

    _install_fake_provider(monkeypatch, "openai", text)
    state = problem_builder.ProblemBuilderState()
    before = (state.objective_code, state.constraint_code, state.n_var, state.n_obj, state.bounds_text)
    state.ai_provider = "openai"
    state.ai_api_key = "test-key"
    state.ai_description = "a problem"

    state.ai_generate()

    assert (state.objective_code, state.constraint_code, state.n_var, state.n_obj, state.bounds_text) == before
    assert state.ai_status.startswith("Error:")
    assert message in state.ai_status


def test_llm_unknown_provider_is_rejected_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.ux.studio import _studio_llm

    calls = _install_fake_provider(monkeypatch, "openai", _DRAFT)

    with pytest.raises(ValueError, match="gemni"):
        _studio_llm.llm_generate_problem_code("a problem", provider="gemni", api_key="test-key")
    assert calls == []


def test_llm_request_that_does_not_answer_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from vamos.ux.studio import _studio_llm

    release = threading.Event()
    monkeypatch.setitem(_studio_llm._PROVIDERS, "openai", lambda _description, *, api_key: release.wait(10) and _DRAFT)
    try:
        with pytest.raises(TimeoutError, match="openai"):
            _studio_llm.llm_generate_problem_code("a problem", provider="openai", api_key="test-key", timeout_seconds=0.2)
    finally:
        release.set()


# The distribution each Studio provider imports, and the module that import loads.
_PROVIDER_PACKAGES = {"gemini": ("google-genai", "google.genai"), "openai": ("openai", "openai"), "anthropic": ("anthropic", "anthropic")}


def test_every_studio_llm_provider_has_a_package_entry() -> None:
    from vamos.ux.studio import _studio_llm

    assert set(_studio_llm._PROVIDERS) == set(_PROVIDER_PACKAGES)


@pytest.mark.parametrize("provider", sorted(_PROVIDER_PACKAGES))
def test_each_llm_provider_installs_from_the_extra_its_error_names(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    import re
    import sys
    from pathlib import Path

    from vamos.ux.studio import _studio_llm

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    distribution, module = _PROVIDER_PACKAGES[provider]
    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    extra = pyproject["project"]["optional-dependencies"].get(provider, [])
    assert distribution in [re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0] for requirement in extra]

    # A None entry in sys.modules makes the provider's import fail as if the package were absent.
    for name in (module, module.split(".")[0]):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(RuntimeError, match=re.escape(f'pip install "vamos-optimization[{provider}]"')):
        _studio_llm._PROVIDERS[provider]("a problem", api_key="test-key")
