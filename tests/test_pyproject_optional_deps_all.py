from __future__ import annotations

from pathlib import Path

from tests.architecture._dependencies import OPTIMIZATION_FRAMEWORKS, distribution_name, optional_requirements


def test_pyproject_all_extra_matches_union_of_other_optional_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
        import tomli as tomllib  # type: ignore[import-not-found]

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]
    optional_deps = project["optional-dependencies"]

    all_extra = optional_deps["all"]
    expected_union: list[str] = []
    for group_name, deps in optional_deps.items():
        if group_name == "all":
            continue
        for dep in deps:
            if dep not in expected_union:
                expected_union.append(dep)

    assert all_extra == expected_union


def test_no_extra_installs_another_optimization_framework() -> None:
    frameworks = {
        group: sorted({distribution_name(requirement) for requirement in requirements} & set(OPTIMIZATION_FRAMEWORKS))
        for group, requirements in optional_requirements().items()
    }

    declared = {group: names for group, names in frameworks.items() if names}
    assert declared == {}, f"extras that install another optimization framework: {declared}"
