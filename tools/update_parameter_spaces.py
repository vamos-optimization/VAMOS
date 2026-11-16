"""Write what the built-in parameter-space grammars generate: the packaged bundle and the test snapshot.

The YAML files in ``src/vamos/resources/parameter_spaces`` are the source of truth. This tool, which
needs PyYAML, writes from them:

- ``parameter_spaces.json`` next to them, the canonical form that VAMOS reads at run time (VAMOS does
  not depend on PyYAML);
- ``tests/engine/data/parameter_space_snapshot.json``, the SHA-256 of every grammar and the
  configurations that ``config_from_assignment`` builds from its default and from three seeded draws.

``--check`` writes nothing and exits with status 1 when either file is out of date.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAMMARS = REPO_ROOT / "src" / "vamos" / "resources" / "parameter_spaces"
BUNDLE = GRAMMARS / "parameter_spaces.json"
SNAPSHOT = REPO_ROOT / "tests" / "engine" / "data" / "parameter_space_snapshot.json"
SNAPSHOT_POPULATION = 100
SNAPSHOT_OBJECTIVES = 2
SNAPSHOT_DRAWS = 3


def bundle_text() -> str:
    """``parameter_spaces.json`` from the YAML grammars: one space per line, in name order."""
    from vamos.engine.tuning.grammar import SCHEMA, SCHEMA_VERSION, canonical_json, parameter_space_document, parameter_space_from_yaml

    lines = ["{", f' "schema": "{SCHEMA}",', f' "schema_version": "{SCHEMA_VERSION}",', ' "spaces": {']
    paths = sorted(GRAMMARS.glob("*.yaml"))
    for index, path in enumerate(paths):
        space = parameter_space_from_yaml(path.read_text(encoding="utf-8"))
        if space.name != path.stem:
            raise SystemExit(f"{path.name} holds the space {space.name!r}: name the file after the space.")
        comma = "," if index < len(paths) - 1 else ""
        lines.append(f"  {json.dumps(space.name)}: {canonical_json(parameter_space_document(space))}{comma}")
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


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
    lines = [
        "{",
        ' "settings": ' + json.dumps({"pop_size": SNAPSHOT_POPULATION, "n_obj": SNAPSHOT_OBJECTIVES, "draws": SNAPSHOT_DRAWS}) + ",",
        ' "spaces": {',
    ]
    for index, name in enumerate(names):
        comma = "," if index < len(names) - 1 else ""
        lines.append(f"  {json.dumps(name)}: {json.dumps(snapshot_entry(name), separators=(',', ':'))}{comma}")
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def _stale(path: Path, text: str) -> bool:
    return not path.exists() or path.read_text(encoding="utf-8") != text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the parameter-space bundle and the test snapshot from the YAML grammars.")
    parser.add_argument("--check", action="store_true", help="exit with status 1 when a file is out of date, writing nothing")
    args = parser.parse_args(argv)
    stale: list[Path] = []
    bundle = bundle_text()
    if _stale(BUNDLE, bundle):
        stale.append(BUNDLE)
        if not args.check:
            BUNDLE.write_text(bundle, encoding="utf-8", newline="\n")
    # The snapshot reads the bundle that VAMOS packages, so it is computed after the bundle is written.
    if not (args.check and stale):
        snapshot = snapshot_text()
        if _stale(SNAPSHOT, snapshot):
            stale.append(SNAPSHOT)
            if not args.check:
                SNAPSHOT.write_text(snapshot, encoding="utf-8", newline="\n")
    for path in stale:
        print(f"{'Out of date' if args.check else 'Wrote'}: {path.relative_to(REPO_ROOT)}")
    if args.check and stale:
        print("Run python tools/update_parameter_spaces.py.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
