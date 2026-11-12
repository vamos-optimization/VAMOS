"""Print pip constraints that pin each core dependency of pyproject.toml at its declared minimum.

CI's tests-minimum job installs VAMOS with them on the oldest supported Python, so the
``>=`` bounds users rely on are tested. They are generated, not stored, so a dependency
update cannot move them away from pyproject.toml.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, the version the minimum profile runs on.
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

ROOT = Path(__file__).resolve().parents[1]
_MINIMUM = re.compile(r"([A-Za-z0-9._-]+)\s*>=\s*([0-9][0-9.]*)")


def minimum_constraints(pyproject: Path = ROOT / "pyproject.toml") -> list[str]:
    dependencies = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    pins = []
    for requirement in dependencies:
        match = _MINIMUM.fullmatch(requirement.strip())
        if match is None:
            raise SystemExit(f"Core dependency without a single >= bound: {requirement}")
        pins.append(f"{match[1]}=={match[2]}")
    return pins


def main() -> int:
    print("\n".join(minimum_constraints()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
