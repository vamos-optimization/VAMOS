"""Check the built-in parameter-space grammars and write the snapshot of what their bridges build.

The YAML files in ``src/vamos/resources/parameter_spaces`` are the source of truth, and VAMOS reads
them at run time. This tool reads every one of them, fails unless each passes rules G1 to G6 and is
named after its space, and writes ``tests/engine/data/parameter_space_snapshot.json``: the SHA-256 of
every grammar and the configurations that ``config_from_assignment`` builds from its default and from
three seeded draws.

``--check`` writes nothing and exits with status 1 when a grammar is invalid or the snapshot is out of
date.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAMMARS = REPO_ROOT / "src" / "vamos" / "resources" / "parameter_spaces"
SNAPSHOT = REPO_ROOT / "tests" / "engine" / "data" / "parameter_space_snapshot.json"
SNAPSHOT_POPULATION = 100
SNAPSHOT_OBJECTIVES = 2
SNAPSHOT_DRAWS = 3


def grammar_errors() -> list[str]:
    """Why each YAML file in the grammar folder is not a valid built-in space, if it is not."""
    from vamos.engine.tuning.grammar import ParameterSpaceError, parameter_space_from_yaml

    errors = []
    for path in sorted(GRAMMARS.iterdir()):
        if path.suffix != ".yaml":
            errors.append(f"{path.name}: the folder holds only the grammars, as <space>.yaml files")
            continue
        try:
            space = parameter_space_from_yaml(path.read_text(encoding="utf-8"))
        except ParameterSpaceError as error:
            errors.append(f"{path.name}: {error}")
            continue
        if space.name != path.stem:
            errors.append(f"{path.name} holds the space {space.name!r}: name the file after the space")
    return errors


def _plain(value: object) -> Any:
    return json.loads(json.dumps(value, default=str))


def snapshot_entry(name: str) -> dict[str, Any]:
    """The SHA-256 of the built-in grammar ``name`` and what its bridge builds from its default and three draws."""
    import numpy as np

    from vamos.engine.tuning import config_from_assignment
    from vamos.engine.tuning.grammar import builtin_parameter_space

    space = builtin_parameter_space(name)
    rng = np.random.default_rng(0)
    points = [space.default_values(), *(space.sample(rng) for _ in range(SNAPSHOT_DRAWS))]
    settings = {"pop_size": SNAPSHOT_POPULATION, "n_obj": SNAPSHOT_OBJECTIVES}
    configurations = [_plain(config_from_assignment(name, {**point, **settings}).to_dict()) for point in points]
    return {"sha256": space.identity.sha256, "configurations": configurations}


def snapshot_text() -> str:
    from vamos.engine.tuning.grammar import builtin_parameter_space_names

    names = sorted(builtin_parameter_space_names())
    settings = json.dumps({"pop_size": SNAPSHOT_POPULATION, "n_obj": SNAPSHOT_OBJECTIVES, "draws": SNAPSHOT_DRAWS})
    lines = ["{", f' "settings": {settings},', ' "spaces": {']
    for index, name in enumerate(names):
        comma = "," if index < len(names) - 1 else ""
        lines.append(f"  {json.dumps(name)}: {json.dumps(snapshot_entry(name), separators=(',', ':'))}{comma}")
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the YAML parameter-space grammars and write the snapshot of what the bridges build."
    )
    parser.add_argument("--check", action="store_true", help="write nothing; exit with status 1 on an invalid grammar or a stale snapshot")
    args = parser.parse_args(argv)
    errors = grammar_errors()
    for error in errors:
        print(f"Invalid grammar: {error}")
    if errors:
        return 1
    snapshot = snapshot_text()
    if SNAPSHOT.exists() and SNAPSHOT.read_text(encoding="utf-8") == snapshot:
        print("The grammars are valid and the snapshot is current.")
        return 0
    if args.check:
        print(f"Out of date: {SNAPSHOT.relative_to(REPO_ROOT)}. Run python tools/update_parameter_spaces.py.")
        return 1
    SNAPSHOT.write_text(snapshot, encoding="utf-8", newline="\n")
    print(f"Wrote: {SNAPSHOT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
