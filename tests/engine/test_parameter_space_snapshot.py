"""The committed snapshot of what the built-in grammars are and what the bridges build from them.

``data/parameter_space_snapshot.json`` records, for every built-in space, the SHA-256 of its grammar
and the configurations that ``config_from_assignment`` builds from its default and from three seeded
draws. A change to a grammar, or to a bridge, that changes them fails here until the snapshot is
written again with ``python tools/update_parameter_spaces.py``; such a change says in the changelog
which seeded tuning runs draw other trials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.update_parameter_spaces import snapshot_entry
from vamos.engine.tuning.grammar import builtin_parameter_space_names

SNAPSHOT = json.loads((Path(__file__).parent / "data" / "parameter_space_snapshot.json").read_text(encoding="utf-8"))
UPDATE = "python tools/update_parameter_spaces.py"


def test_the_snapshot_covers_every_built_in_space() -> None:
    assert sorted(SNAPSHOT["spaces"]) == sorted(builtin_parameter_space_names()), f"Run {UPDATE}."


@pytest.mark.parametrize("name", sorted(builtin_parameter_space_names()))
def test_the_grammar_and_what_its_bridge_builds_match_the_snapshot(name: str) -> None:
    current = snapshot_entry(name)
    stored = SNAPSHOT["spaces"][name]

    assert current["sha256"] == stored["sha256"], f"The grammar of {name!r} changed. Run {UPDATE} and say which seeded tuning runs change."
    assert current["configurations"] == stored["configurations"], f"The bridge of {name!r} builds other configurations. Run {UPDATE}."
