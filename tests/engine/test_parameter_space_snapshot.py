"""The committed snapshot of what the built-in grammars are and what the bridges build from them.

``data/parameter_space_snapshot.json`` records, for every built-in space, the SHA-256 of its grammar
and the configurations that ``config_from_assignment`` builds from its default and from three seeded
draws. A change to a grammar, or to a bridge, that changes them fails here until the snapshot is
written again with ``python tools/update_parameter_spaces.py``; such a change says in the changelog
which seeded tuning runs draw other trials.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from tools.update_parameter_spaces import snapshot_entry
from vamos.engine.tuning.grammar import builtin_parameter_space_names

SNAPSHOT = json.loads((Path(__file__).parent / "data" / "parameter_space_snapshot.json").read_text(encoding="utf-8"))
UPDATE = "python tools/update_parameter_spaces.py"
# Seeded real draws can differ by one unit in the last place between platforms: macOS arm64 gives ibea_binary's
# third draw kappa 0.17290681255163817 where x86-64 gives 0.1729068125516382. Real numbers are compared to 12
# significant digits; everything else, including which branch each draw took, exactly.
REAL_DIGITS = 1e-12


def _differences(current: Any, stored: Any, path: str = "") -> list[str]:
    """Where ``current`` differs from ``stored``: real numbers within ``REAL_DIGITS``, anything else exactly."""
    if isinstance(current, float) or isinstance(stored, float):
        same = (
            not isinstance(current, bool)
            and not isinstance(stored, bool)
            and isinstance(current, (int, float))
            and isinstance(stored, (int, float))
            and math.isclose(current, stored, rel_tol=REAL_DIGITS, abs_tol=0.0)
        )
        return [] if same else [f"{path or '.'}: {current!r} != {stored!r}"]
    if isinstance(current, dict) and isinstance(stored, dict):
        if list(current) != list(stored):
            return [f"{path or '.'}: keys {list(current)} != {list(stored)}"]
        return [d for key in current for d in _differences(current[key], stored[key], f"{path}.{key}")]
    if isinstance(current, list) and isinstance(stored, list):
        if len(current) != len(stored):
            return [f"{path or '.'}: {len(current)} items != {len(stored)}"]
        return [d for i, (a, b) in enumerate(zip(current, stored)) for d in _differences(a, b, f"{path}[{i}]")]
    return [] if type(current) is type(stored) and current == stored else [f"{path or '.'}: {current!r} != {stored!r}"]


@pytest.mark.parametrize(
    ("current", "stored", "same"),
    [
        ({"kappa": 0.17290681255163817}, {"kappa": 0.1729068125516382}, True),  # one unit in the last place
        ({"kappa": 0.1729}, {"kappa": 0.1730}, False),
        ({"crossover": ["hux", {"prob": 0.9}]}, {"crossover": ["uniform", {"prob": 0.9}]}, False),
        ({"size": 8}, {"size": 9}, False),
        ({"flag": True}, {"flag": 1}, False),
        ({"a": 1.0, "b": 2.0}, {"b": 2.0, "a": 1.0}, False),
        ([1.0, 2.0], [1.0], False),
        ({"prob": 0.9}, {"prob": 0.9}, True),
    ],
)
def test_the_snapshot_compares_real_numbers_to_twelve_digits_and_the_rest_exactly(current: Any, stored: Any, same: bool) -> None:
    assert (_differences(current, stored) == []) is same


def test_the_snapshot_covers_every_built_in_space() -> None:
    assert sorted(SNAPSHOT["spaces"]) == sorted(builtin_parameter_space_names()), f"Run {UPDATE}."


@pytest.mark.parametrize("name", sorted(builtin_parameter_space_names()))
def test_the_grammar_and_what_its_bridge_builds_match_the_snapshot(name: str) -> None:
    current = snapshot_entry(name)
    stored = SNAPSHOT["spaces"][name]

    assert current["sha256"] == stored["sha256"], f"The grammar of {name!r} changed. Run {UPDATE} and say which seeded tuning runs change."
    differences = _differences(current["configurations"], stored["configurations"])
    assert not differences, f"The bridge of {name!r} builds other configurations: {differences[:3]}. Run {UPDATE}."
