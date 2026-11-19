"""The extras of pyproject.toml install what users of VAMOS need, and ``all`` gathers only those."""

from __future__ import annotations

from tests.architecture._dependencies import LLM_SDKS, OPTIMIZATION_FRAMEWORKS, distribution_name, optional_requirements

# The extras for users of VAMOS, in the order ``all`` gathers them, and those for working on VAMOS itself.
USER_EXTRAS = ("compute", "analysis", "tuning", "studio", "examples")
TOOLING_EXTRAS = ("dev", "docs")
# Development after 1.5.0: user extras that ``all`` leaves out, because their package has a copyleft license
# (modact, which evaluates the MODAct problems, is GPL-3.0); a user installs them by name.
COPYLEFT_EXTRAS = ("modact",)


def _distributions(requirements: list[str]) -> set[str]:
    return {distribution_name(requirement) for requirement in requirements}


def test_every_extra_is_a_user_or_a_tooling_extra() -> None:
    known = {*USER_EXTRAS, *TOOLING_EXTRAS, *COPYLEFT_EXTRAS, "all"}
    unclassified = sorted(set(optional_requirements()) - known)
    missing = sorted(known - set(optional_requirements()))
    assert (unclassified, missing) == ([], []), f"extras that are neither user nor tooling extras: {unclassified}; missing: {missing}"


def test_all_extra_is_the_union_of_the_user_extras() -> None:
    extras = optional_requirements()
    expected = list(dict.fromkeys(requirement for group in USER_EXTRAS for requirement in extras[group]))

    assert extras["all"] == expected, (
        f"all has {sorted(set(extras['all']) - set(expected))} beyond the user extras "
        f"and lacks {sorted(set(expected) - set(extras['all']))}"
    )


def test_all_extra_installs_no_copyleft_extra() -> None:
    extras = optional_requirements()
    copyleft = set().union(*(_distributions(extras[group]) for group in COPYLEFT_EXTRAS))

    assert copyleft == {"modact"}
    assert sorted(_distributions(extras["all"]) & copyleft) == []


def test_all_extra_installs_no_development_tooling() -> None:
    extras = optional_requirements()
    user = set().union(*(_distributions(extras[group]) for group in USER_EXTRAS))
    tooling = set().union(*(_distributions(extras[group]) for group in TOOLING_EXTRAS)) - user

    installed = sorted(_distributions(extras["all"]) & tooling)
    assert installed == [], f"all installs development tooling: {installed}"


def test_no_extra_installs_another_optimization_framework() -> None:
    frameworks = {
        group: sorted(_distributions(requirements) & set(OPTIMIZATION_FRAMEWORKS))
        for group, requirements in optional_requirements().items()
    }

    declared = {group: names for group, names in frameworks.items() if names}
    assert declared == {}, f"extras that install another optimization framework: {declared}"


def test_no_extra_installs_an_llm_sdk() -> None:
    sdks = {group: sorted(_distributions(requirements) & set(LLM_SDKS)) for group, requirements in optional_requirements().items()}

    declared = {group: names for group, names in sdks.items() if names}
    assert declared == {}, f"extras that install an LLM client: {declared}"
