from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vamos.ux.panel import launcher
from vamos.ux.studio import _studio_llm
from vamos.ux.studio._problem_builder_security import TRUSTED_LOCAL_CODE_WARNING
from vamos.ux.studio.problem_builder_backend import compile_objective_function, run_preview_optimization

ROOT = Path(__file__).resolve().parents[2]


def test_raw_python_is_disabled_before_explicit_trust() -> None:
    with pytest.raises(PermissionError, match="current operating-system user"):
        compile_objective_function("return [x[0], x[1]]")


def test_explicit_trust_allows_reviewed_local_code() -> None:
    fn = compile_objective_function("return [x[0], 1.0 - x[0]]", trusted_local_code=True)

    assert fn(np.array([0.25])) == [0.25, 0.75]
    assert "permissions of your current operating-system user" in TRUSTED_LOCAL_CODE_WARNING


def test_preview_rejects_an_unacknowledged_callable_without_invoking_it() -> None:
    invoked = False

    def untrusted(_value: object) -> list[float]:
        nonlocal invoked
        invoked = True
        return [0.0, 1.0]

    with pytest.raises(PermissionError, match="trusted_local_code=True"):
        run_preview_optimization(
            untrusted,
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="nsgaii",
            budget=20,
            pop_size=10,
            seed=0,
        )

    assert invoked is False


def test_launcher_command_binds_to_loopback_by_default() -> None:
    command = launcher._build_panel_command("results", (), show=False)

    assert "--address=127.0.0.1" in command
    assert "--port=5006" in command
    assert "--show" not in command


def test_remote_binding_is_rejected_without_dangerous_use_acknowledgement() -> None:
    with pytest.raises(SystemExit) as raised:
        launcher.main(["--address", "0.0.0.0", "--no-browser"])

    assert raised.value.code == 2


def test_remote_binding_requires_and_accepts_separate_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setitem(sys.modules, "panel", ModuleType("panel"))

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert launcher.main(["--address", "0.0.0.0", "--allow-remote-binding", "--no-browser"]) == 0
    assert "--address=0.0.0.0" in commands[0]
    assert "WARNING" in capsys.readouterr().err


def test_llm_and_ui_require_review_before_any_generated_code_execution() -> None:
    llm_source = (ROOT / "src/vamos/ux/studio/_studio_llm.py").read_text(encoding="utf-8")
    ui_source = (ROOT / "src/vamos/ux/panel/pages/problem_builder.py").read_text(encoding="utf-8")
    controls_source = (ROOT / "src/vamos/ux/panel/pages/problem_builder_ai.py").read_text(encoding="utf-8")

    assert "exec(" not in llm_source
    assert "eval(" not in llm_source
    assert "never executed automatically" in controls_source
    assert "self.trusted_local_code = False" in ui_source
    assert ui_source.index("if not self.trusted_local_code:") < ui_source.index("compile_objective_function(self.objective_code")


def test_ui_invalidates_trust_after_edit_or_llm_generation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("panel")
    from vamos.ux.panel.pages import problem_builder

    state = problem_builder.ProblemBuilderState()
    assert state.trusted_local_code is False

    state.trusted_local_code = True
    state.objective_code = "return [x[0], 1.0 - x[0]]"
    assert state.trusted_local_code is False

    generated = {
        "objective_code": "return [x[0], x[1]]",
        "constraint_code": "",
        "n_var": 2,
        "n_obj": 2,
        "bounds": "0.0, 1.0",
    }
    secret = "studio-test-secret"
    monkeypatch.setattr(problem_builder, "llm_generate_problem_code", lambda *args, **kwargs: generated)
    state.ai_description = "two objectives"
    state.ai_api_key = secret
    state.trusted_local_code = True
    state.ai_generate()

    assert state.objective_code == generated["objective_code"]
    assert state.trusted_local_code is False
    assert "not executed" in state.ai_status
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def test_active_studio_guidance_does_not_claim_code_is_isolated() -> None:
    paths = (
        ROOT / "docs/guide/studio.md",
        ROOT / "src/vamos/ux/panel/pages/problem_builder.py",
        ROOT / "src/vamos/ux/panel/pages/problem_builder_ai.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)

    assert "sandbox" not in text
    assert "secure execution" not in text
    assert "safe execution of untrusted code" not in text
    assert "not a security boundary" in text


def test_studio_inspection_module_has_no_execution_primitives() -> None:
    source = (ROOT / "src/vamos/ux/studio/data.py").read_text(encoding="utf-8")

    for forbidden in ("optimize(", "reproduce(", "subprocess", "exec(", "eval("):
        assert forbidden not in source


def test_studio_provider_code_does_not_log_secrets_or_generated_code() -> None:
    source = (ROOT / "src/vamos/ux/studio/_studio_llm.py").read_text(encoding="utf-8")

    assert "logging" not in source
    assert "print(" not in source
    assert _studio_llm.__all__ == ["llm_generate_problem_code"]


@pytest.mark.parametrize(
    ("provider", "module_name", "env_var"), [("anthropic", "anthropic", "ANTHROPIC_API_KEY"), ("openai", "openai", "OPENAI_API_KEY")]
)
def test_provider_clients_receive_the_api_key_field(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    module_name: str,
    env_var: str,
) -> None:
    payload = '{"objective_code": "return [x[0], x[1]]", "n_var": 2, "n_obj": 2, "bounds": "0, 1"}'
    clients: list[dict[str, object]] = []

    class FakeAnthropic:
        def __init__(self, **kwargs: object) -> None:
            clients.append(kwargs)
            self.messages = SimpleNamespace(create=lambda **_kw: SimpleNamespace(content=[SimpleNamespace(text=payload)]))

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            clients.append(kwargs)
            self.responses = SimpleNamespace(create=lambda **_kw: SimpleNamespace(output_text=payload))

    fake = ModuleType(module_name)
    fake.Anthropic = FakeAnthropic  # type: ignore[attr-defined]
    fake.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake)
    monkeypatch.delenv(env_var, raising=False)

    result = _studio_llm.llm_generate_problem_code("two objectives", provider=provider, api_key="field-key")

    assert clients == [{"api_key": "field-key"}]
    assert result["n_obj"] == 2
