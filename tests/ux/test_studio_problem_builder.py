"""Tests for the VAMOS Studio Problem Builder backend logic."""

from __future__ import annotations

import numpy as np
import pytest

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

    def test_preview_timeout_for_non_terminating_code(self) -> None:
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

    def test_child_process_returns_a_large_generation_history(self) -> None:
        from vamos.ux.studio.algorithm_form import default_algorithm_config

        code = "return [x[0], 1.0 - x[0] + sum(x[1:])]"
        fn = compile_objective_function(code, trusted_local_code=True)
        config = default_algorithm_config("nsgaii", encoding="real", n_var=10, n_obj=2, pop_size=100)
        result = run_preview_optimization(
            fn,
            n_var=10,
            n_obj=2,
            bounds=[(0.0, 1.0)] * 10,
            algorithm="nsgaii",
            budget=20000,
            pop_size=100,
            seed=0,
            objective_code=code,
            timeout_seconds=120.0,
            trusted_local_code=True,
            algorithm_config=config,
            record_history=True,
        )

        # Tens of generations of 100 points exceed a pipe buffer; the parent must read before joining.
        assert len(result["history"]) >= 150
        assert all(F.shape[1] == 2 for F in result["history"])


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
