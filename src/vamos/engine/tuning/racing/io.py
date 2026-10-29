from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, cast

from .param_space import ParamSpace, _active_assignment
from .random_search_tuner import TrialResult


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Best-effort conversion for scalar types not handled by stdlib json."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            _logger().debug("Failed to convert scalar-like value with item().", exc_info=True)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def filter_active_config(config: dict[str, Any], param_space: ParamSpace) -> dict[str, Any]:
    """
    Return a copy of `config` containing only parameters that are active.

    Parameters declared in the ParamSpace are kept only when their conditions hold
    for the active parents, so descendants of an inactive parameter are dropped, as
    in ``ParamSpace.sample``. Keys not declared in the ParamSpace (e.g.,
    population_size, offspring_size, repair) are always preserved.
    """
    active = _active_assignment(param_space, config)
    return {name: value for name, value in config.items() if name in active or name not in param_space.params}


def history_to_dict(
    history: list[TrialResult],
    param_space: ParamSpace,
    *,
    include_raw: bool = False,
) -> list[dict[str, Any]]:
    """
    Convert TrialResult list into JSON-serializable dicts, filtering out
    inactive parameters using param_space.
    """
    data: list[dict[str, Any]] = []
    for trial in history:
        clean_config = filter_active_config(trial.config, param_space)
        record: dict[str, Any] = {
            "trial_id": trial.trial_id,
            "score": trial.score,
            "config": clean_config,
            "details": trial.details,
        }
        if include_raw:
            record["raw_config"] = trial.config
        data.append(record)
    return data


def save_history_json(
    history: list[TrialResult],
    param_space: ParamSpace,
    path: str | Path,
    *,
    include_raw: bool = False,
) -> None:
    """Persist filtered tuning history to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = history_to_dict(history, param_space, include_raw=include_raw)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def save_history_csv(
    history: list[TrialResult],
    param_space: ParamSpace,
    path: str | Path,
    *,
    include_raw: bool = False,
) -> None:
    """Persist filtered tuning history to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        headers = ["trial_id", "score", "config_json"]
        if include_raw:
            headers.append("raw_config_json")
        writer.writerow(headers)
        for trial in history:
            clean_config = filter_active_config(trial.config, param_space)
            row = [trial.trial_id, trial.score, json.dumps(clean_config, default=_json_default)]
            if include_raw:
                row.append(json.dumps(trial.config, default=_json_default))
            writer.writerow(row)


def save_checkpoint(
    best_configs: list[dict[str, Any]],
    elite_archive: list[dict[str, Any]],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a tuning checkpoint with elite configurations for warm restart.

    Parameters
    ----------
    best_configs
        Best configuration dictionaries found so far.
    elite_archive
        Elite archive entries containing configuration and score data.
    path
        Destination path for the checkpoint JSON file.
    metadata
        Optional extra metadata such as stage index or experiment counts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "best_configs": best_configs,
        "elite_archive": elite_archive,
        "metadata": metadata or {},
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, indent=2, default=_json_default)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a tuning checkpoint for warm restart.

    Parameters
    ----------
    path
        Path to a checkpoint file produced by :func:`save_checkpoint`.

    Returns
    -------
    dict[str, Any]
        Mapping with ``best_configs``, ``elite_archive``, and ``metadata`` keys.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return cast(dict[str, Any], payload)


__all__ = [
    "filter_active_config",
    "history_to_dict",
    "save_history_json",
    "save_history_csv",
    "save_checkpoint",
    "load_checkpoint",
]
