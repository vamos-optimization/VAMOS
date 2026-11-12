from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Iterable, Sequence

from vamos.foundation.exceptions import _suggest_names


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


from importlib.resources import as_file
from pathlib import Path

from vamos.foundation.problem.registry import (
    ProblemSelection,
    available_problem_names,
    get_problem_specs,
    make_problem_selection,
)
from vamos.resources import reference_front_path

# Preset names map to problem keys defined in the registry.
PROBLEM_SET_PRESETS: dict[str, Sequence[str]] = {
    "families": ("zdt1", "dtlz2", "wfg4", "tsp6"),
    "zdt": ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6"),
    "dtlz": ("dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"),
    "wfg": ("wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"),
    "zcat": (
        "zcat1",
        "zcat2",
        "zcat3",
        "zcat4",
        "zcat5",
        "zcat6",
        "zcat7",
        "zcat8",
        "zcat9",
        "zcat10",
        "zcat11",
        "zcat12",
        "zcat13",
        "zcat14",
        "zcat15",
        "zcat16",
        "zcat17",
        "zcat18",
        "zcat19",
        "zcat20",
    ),
    "lz": ("lz09_f1", "lz09_f2", "lz09_f3", "lz09_f4", "lz09_f5", "lz09_f6", "lz09_f7", "lz09_f8", "lz09_f9"),
    "cec": (
        "cec2009_uf1",
        "cec2009_uf2",
        "cec2009_uf3",
        "cec2009_uf4",
        "cec2009_uf5",
        "cec2009_uf6",
        "cec2009_uf7",
        "cec2009_uf8",
        "cec2009_uf9",
        "cec2009_uf10",
        "cec2009_cf1",
    ),
    "lsmop": ("lsmop1", "lsmop2", "lsmop3", "lsmop4", "lsmop5", "lsmop6", "lsmop7", "lsmop8", "lsmop9"),
    "constrained_many": (
        "c1dtlz1",
        "c1dtlz3",
        "c2dtlz2",
        "c3dtlz1",
        "dc1dtlz1",
        "dc1dtlz3",
        "dc2dtlz1",
        "dc2dtlz3",
        "dc3dtlz1",
        "dc3dtlz3",
        "mw1",
        "mw2",
        "mw3",
        "mw4",
        "mw5",
        "mw6",
        "mw7",
        "mw8",
        "mw9",
        "mw10",
        "mw11",
        "mw12",
        "mw13",
        "mw14",
    ),
    "tsp": ("tsp6",),
    "tsplib": ("kroa100", "krob100", "kroc100", "krod100", "kroe100"),
    "real_world": ("ml_tuning", "welded_beam", "fs_real"),
}
_PROBLEM_DOCS = "docs/reference/problems.md"
_TROUBLESHOOTING_DOCS = "docs/guide/troubleshooting.md"


def _format_unknown_problem(name: str, options: list[str]) -> str:
    parts = [f"Unknown problem '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_PROBLEM_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


def _format_unknown_problem_set(name: str, options: list[str]) -> str:
    parts = [f"Unknown problem set '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_PROBLEM_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


def resolve_reference_front_path(problem_key: str, explicit_path: str | None, n_obj: int | None = None) -> str | None:
    """
    Return a path to a reference front, using an explicit override when provided.

    The packaged front of a problem with ``n_obj`` objectives is ``<key>.<n_obj>d.csv`` or, failing
    that, ``<key>.csv``. A file with another number of objectives is never returned. Without
    ``n_obj``, the problem's registered default is used; a key that is not registered takes
    ``<key>.csv`` as it is.
    """
    if explicit_path:
        return explicit_path

    key = str(problem_key).strip().lower()
    if n_obj is None:
        n_obj = _default_n_obj(key)
    candidates = [key] if n_obj is None or "." in key else [f"{key}.{int(n_obj)}d", key]

    for candidate in candidates:
        try:
            with as_file(reference_front_path(candidate)) as p:
                if n_obj is not None and _front_columns(p) != int(n_obj):
                    _logger().debug("Reference front %r has %d objectives, not %d", candidate, _front_columns(p), n_obj)
                    continue
                return str(p)
        except Exception:
            _logger().debug("No reference front found for candidate %r", candidate, exc_info=True)

    _logger().debug("No reference front found for %r", problem_key)
    return None


def _default_n_obj(key: str) -> int | None:
    spec = get_problem_specs().get(key)
    return None if spec is None else int(spec.default_n_obj)


def _front_columns(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return len(line.split(","))
    return 0


def _validate_problem_name(name: str) -> None:
    available = sorted(available_problem_names())
    if name not in available:
        raise ValueError(_format_unknown_problem(name, available))


def _make_selection(name: str, n_var: int | None, n_obj: int | None) -> ProblemSelection:
    _validate_problem_name(name)
    return make_problem_selection(name, n_var=n_var, n_obj=n_obj)


def resolve_problem_selection(args: Namespace) -> Iterable[ProblemSelection]:
    """
    Decide which problems to run based on CLI args, returning registry-backed selections.
    """
    if getattr(args, "problem_set", None):
        preset_name = args.problem_set
        if preset_name not in PROBLEM_SET_PRESETS:
            raise ValueError(_format_unknown_problem_set(preset_name, sorted(PROBLEM_SET_PRESETS)))
        return tuple(_make_selection(name, args.n_var, args.n_obj) for name in PROBLEM_SET_PRESETS[preset_name])

    if getattr(args, "problem", None):
        return (_make_selection(args.problem, args.n_var, args.n_obj),)

    # Fallback or default
    return (_make_selection("zdt1", args.n_var, args.n_obj),)


def resolve_problem_selections(args: Namespace) -> Iterable[ProblemSelection]:
    """
    Return a list of ProblemSelection objects to run.
    Handles 'all', specific sets, or single problems.
    """
    if getattr(args, "problem_set", None):
        if args.problem_set == "all":
            all_names: list[str] = []
            for key, problems in PROBLEM_SET_PRESETS.items():
                if key == "families":
                    continue
                all_names.extend(problems)
            return tuple(_make_selection(name, args.n_var, args.n_obj) for name in all_names)
        return resolve_problem_selection(args)

    if getattr(args, "problem", None):
        return (_make_selection(args.problem, args.n_var, args.n_obj),)

    return (_make_selection("zdt1", args.n_var, args.n_obj),)
