"""Assignments saved from the tuning spaces of VAMOS 1.3.0 keep building the configurations they built.

``data/tuning_assignments_1_3_0.json`` holds seeded draws of every space of VAMOS 1.3.0, before the
spaces became YAML grammars, with ``pop_size`` among their parameters and the external archive as
``use_external_archive`` and ``archive_unbounded``, and the configuration that each one built then.
The same draws with the names of VAMOS 1.2.1 (``crossover_eta``, ``selection_pressure``, ...) build
the same configurations too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.racing.parameter_names import bridge_assignment

FIXTURE = json.loads((Path(__file__).parent / "data" / "tuning_assignments_1_3_0.json").read_text(encoding="utf-8"))["spaces"]


def _plain(value: object) -> Any:
    return json.loads(json.dumps(value, default=str))


def _same_fields(built: dict[str, Any], stored: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """The stored fields that the new configuration gives another value (fields added since 1.3.0 are not compared)."""
    return {key: (value, built.get(key)) for key, value in stored.items() if built.get(key) != value}


@pytest.mark.parametrize("name", sorted(FIXTURE))
def test_an_assignment_of_1_3_0_builds_the_same_configuration(name: str) -> None:
    for entry in FIXTURE[name]:
        built = _plain(config_from_assignment(name, dict(entry["assignment"])).to_dict())
        assert _same_fields(built, entry["config"]) == {}, entry["assignment"]


@pytest.mark.parametrize("name", sorted(FIXTURE))
def test_the_same_assignment_with_the_names_of_1_2_1_builds_it_too(name: str) -> None:
    for entry in FIXTURE[name]:
        earlier = bridge_assignment(entry["assignment"])
        built = _plain(config_from_assignment(name, earlier).to_dict())
        assert _same_fields(built, entry["config"]) == {}, earlier


# Spaces added after VAMOS 1.3.0, of which no assignment was saved then.
ADDED_AFTER_1_3_0 = {"pps"}


def test_the_fixture_covers_every_space_with_its_population() -> None:
    from vamos.engine.tuning.racing import bridge_spaces

    spaces = {name[len("build_") : -len("_config_space")] for name in bridge_spaces.__all__}
    assert ADDED_AFTER_1_3_0 <= spaces
    assert sorted(FIXTURE) == sorted(spaces - ADDED_AFTER_1_3_0)
    assert all("pop_size" in entry["assignment"] for entries in FIXTURE.values() for entry in entries)
