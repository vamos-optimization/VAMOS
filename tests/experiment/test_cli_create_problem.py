"""Tests for the vamos create-problem CLI wizard."""

from __future__ import annotations

from vamos.experiment.cli.create_problem import (
    _generate_class,
    _generate_functional,
    _to_class_name,
    _to_identifier,
    run_create_problem,
)

# ======================================================================
# Helpers
# ======================================================================


class TestHelpers:
    """Test identifier and class-name conversion."""

    def test_to_identifier_simple(self) -> None:
        assert _to_identifier("my problem") == "my_problem"

    def test_to_identifier_special_chars(self) -> None:
        assert _to_identifier("My Cool Problem!") == "my_cool_problem"

    def test_to_identifier_empty(self) -> None:
        assert _to_identifier("") == "custom_problem"

    def test_to_class_name_simple(self) -> None:
        assert _to_class_name("my problem") == "MyProblem"

    def test_to_class_name_underscores(self) -> None:
        assert _to_class_name("my_cool_problem") == "MyCoolProblem"

    def test_to_class_name_empty(self) -> None:
        assert _to_class_name("") == "CustomProblem"


# ======================================================================
# Template generation
# ======================================================================


class TestFunctionalTemplate:
    """Test functional (make_problem) template generation."""

    def test_contains_make_problem_import(self) -> None:
        code = _generate_functional(
            name="test_fn",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            budget=5000,
        )
        assert "from vamos import make_problem, optimize" in code

    def test_contains_function_definition(self) -> None:
        code = _generate_functional(
            name="test_fn",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            budget=5000,
        )
        assert "def test_fn(x):" in code

    def test_contains_bounds(self) -> None:
        code = _generate_functional(
            name="test_fn",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (-5.0, 5.0)],
            budget=5000,
        )
        assert "(-5.0, 5.0)" in code

    def test_contains_optimize_call(self) -> None:
        code = _generate_functional(
            name="test_fn",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            budget=3000,
        )
        assert "max_evaluations=3000" in code
        assert 'algorithm="nsgaii"' in code

    def test_valid_python_syntax(self) -> None:
        code = _generate_functional(
            name="test_fn",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            budget=5000,
        )
        # Must not raise
        compile(code, "<test>", "exec")

    def test_three_objectives(self) -> None:
        code = _generate_functional(
            name="tri_obj",
            n_var=3,
            n_obj=3,
            bounds=[(0.0, 1.0)] * 3,
            budget=5000,
        )
        assert "n_obj=3" in code
        assert "f0" in code
        assert "f1" in code
        assert "f2" in code


class TestClassTemplate:
    """Test class-based template generation."""

    def test_contains_class_definition(self) -> None:
        code = _generate_class(
            name="my problem",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            budget=5000,
        )
        assert "class MyProblem:" in code

    def test_contains_numpy_import(self) -> None:
        code = _generate_class(
            name="my problem",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            budget=5000,
        )
        assert "import numpy as np" in code

    def test_contains_evaluate_method(self) -> None:
        code = _generate_class(
            name="my problem",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            budget=5000,
        )
        assert "def evaluate(self, X" in code

    def test_valid_python_syntax(self) -> None:
        code = _generate_class(
            name="my problem",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            budget=5000,
        )
        compile(code, "<test>", "exec")

    def test_bounds_in_init(self) -> None:
        code = _generate_class(
            name="my problem",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (-5.0, 5.0)],
            budget=5000,
        )
        assert "[0.0, -5.0]" in code  # xl
        assert "[1.0, 5.0]" in code  # xu


# ======================================================================
# CLI end-to-end (non-interactive --yes)
# ======================================================================


class TestCLI:
    """End-to-end CLI tests using --yes for non-interactive mode."""

    def test_default_functional_output(self, tmp_path) -> None:
        out_file = str(tmp_path / "my_problem.py")
        run_create_problem(["--yes", "--output", out_file])
        content = (tmp_path / "my_problem.py").read_text(encoding="utf-8")
        assert "from vamos import make_problem" in content
        assert "def my_problem(x):" in content
        compile(content, out_file, "exec")

    def test_class_style(self, tmp_path) -> None:
        out_file = str(tmp_path / "my_cls.py")
        run_create_problem(["--yes", "--style", "class", "--output", out_file])
        content = (tmp_path / "my_cls.py").read_text(encoding="utf-8")
        assert "class MyProblem:" in content
        assert "def evaluate(self, X" in content
        compile(content, out_file, "exec")

    def test_custom_metadata(self, tmp_path) -> None:
        out_file = str(tmp_path / "custom.py")
        run_create_problem(
            [
                "--yes",
                "--name",
                "portfolio optimizer",
                "--n-var",
                "5",
                "--n-obj",
                "3",
                "--budget",
                "8000",
                "--output",
                out_file,
            ]
        )
        content = (tmp_path / "custom.py").read_text(encoding="utf-8")
        assert "n_var=5" in content
        assert "n_obj=3" in content
        assert "max_evaluations=8000" in content
        assert "portfolio_optimizer" in content
        compile(content, out_file, "exec")

    def test_output_file_created(self, tmp_path) -> None:
        out_file = tmp_path / "subdir" / "problem.py"
        run_create_problem(["--yes", "--output", str(out_file)])
        assert out_file.exists()


# ======================================================================
# Overwrites, inputs and names
# ======================================================================


def test_create_problem_refuses_to_overwrite(tmp_path, capsys) -> None:
    import pytest

    target = tmp_path / "mine.py"
    target.write_text("# my edits\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        run_create_problem(["--yes", "--output", str(target)])

    assert excinfo.value.code != 0
    assert "--force" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "# my edits\n"

    run_create_problem(["--yes", "--output", str(target), "--force"])
    assert "make_problem" in target.read_text(encoding="utf-8")


def test_create_problem_rejects_non_positive_dimensions(tmp_path, capsys) -> None:
    import pytest

    for flag, value in (("--n-var", "0"), ("--n-obj", "-1"), ("--budget", "0")):
        target = tmp_path / f"p{flag.strip('-')}.py"
        with pytest.raises(SystemExit) as excinfo:
            run_create_problem(["--yes", "--output", str(target), flag, value])
        assert excinfo.value.code == 2
        assert flag in capsys.readouterr().err
        assert not target.exists()


def test_create_problem_escapes_name(tmp_path) -> None:
    import ast

    for style in ("functional", "class"):
        for name in ('a"b', 'tri"""ple', "back\\slash", "end\\", "3d problem", "class"):
            target = tmp_path / f"{style}-{abs(hash(name))}.py"
            run_create_problem(["--yes", "--style", style, "--name", name, "--output", str(target)])
            source = target.read_text(encoding="utf-8")
            ast.parse(source)
            if style == "functional":
                assert f"name={name!r}," in source


def test_create_problem_default_file_does_not_shadow_a_module(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_create_problem(["--yes", "--name", "numpy"])

    assert not (tmp_path / "numpy.py").exists()
    assert (tmp_path / "numpy_problem.py").exists()


def test_generated_script_reports_the_front_not_the_population() -> None:
    code = _generate_functional(name="p", n_var=2, n_obj=2, bounds=[(0.0, 1.0)] * 2, budget=100)

    assert "result.front()" in code
