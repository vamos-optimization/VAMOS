from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal, cast

CanonicalPrunePolicy = Literal["crowding", "hv", "mc_hv", "knn", "maxmin", "ref_dirs"]
PrunePolicy = CanonicalPrunePolicy
DeduplicateIn = Literal["objective", "decision", "both"]

_VALID_PRUNE_POLICIES = {"crowding", "hv", "mc_hv", "knn", "maxmin", "ref_dirs"}
_VALID_DEDUPLICATE_IN = ("objective", "decision", "both")
_HV_POLICIES = ("hv", "mc_hv")


def normalize_prune_policy(policy: str) -> PrunePolicy:
    if policy not in _VALID_PRUNE_POLICIES:
        valid = ", ".join(sorted(_VALID_PRUNE_POLICIES))
        raise ValueError(f"Unsupported pruning '{policy}'. Expected one of: {valid}.")
    return cast(PrunePolicy, policy)


def _is_integer(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def check_external_archive_field(name: str, value: object) -> None:
    """Raise when ``value`` is not valid for the :class:`ExternalArchiveConfig` field ``name``.

    Only the field itself is checked. Rules that relate two fields, such as
    ``truncate_size <= capacity``, are checked when the configuration is built.
    Wrong types raise ``TypeError`` and out-of-range values ``ValueError``; both
    messages name the field.
    """
    if name in ("capacity", "truncate_size"):
        if value is None:
            return
        if not _is_integer(value):
            raise TypeError(f"{name} must be a positive integer or null, got {value!r}.")
        if int(cast(numbers.Integral, value)) <= 0:
            raise ValueError(f"{name} must be > 0 when provided, got {value!r}.")
    elif name == "pruning":
        if not isinstance(value, str):
            raise TypeError(f"pruning must be a string, got {value!r}.")
        normalize_prune_policy(value)
    elif name == "hv_ref_point":
        if value is None:
            return
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"hv_ref_point must be a list of numbers or null, got {value!r}.")
        if len(value) == 0 or not all(_is_real(item) and math.isfinite(float(cast(float, item))) for item in value):
            raise ValueError(f"hv_ref_point must be a non-empty list of finite numbers, got {list(value)!r}.")
    elif name == "rng_seed":
        if not _is_integer(value):
            raise TypeError(f"rng_seed must be an integer, got {value!r}.")
    elif name in ("objective_tolerance", "decision_tolerance"):
        if not _is_real(value):
            raise TypeError(f"{name} must be a number, got {value!r}.")
        number = float(cast(float, value))
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be a finite number >= 0, got {value!r}.")
    elif name == "deduplicate_in":
        if value not in _VALID_DEDUPLICATE_IN:
            raise ValueError(f"deduplicate_in must be one of {', '.join(_VALID_DEDUPLICATE_IN)}, got {value!r}.")
    else:
        raise TypeError(f"Unknown external archive field {name!r}.")


@dataclass(frozen=True)
class ExternalArchiveConfig:
    """Configuration for the single external result archive path.

    ``objective_tolerance`` is the Chebyshev distance under which two objective
    vectors count as duplicates. It is absolute, so the default is 0: only identical
    vectors are merged, whatever the scale of the objectives.

    Every field is validated when the configuration is built. A field that the
    chosen pruning does not use (``hv_ref_point`` without HV pruning,
    ``rng_seed`` without ``ref_dirs``, or a pruning other than ``crowding``
    without a ``capacity``) gives a ``UserWarning``.
    """

    capacity: int | None = None
    pruning: PrunePolicy = "crowding"
    hv_ref_point: list[float] | None = None
    rng_seed: int = 0
    objective_tolerance: float = 0.0
    truncate_size: int | None = None
    deduplicate_in: DeduplicateIn = "objective"
    decision_tolerance: float = 1e-32

    def __post_init__(self) -> None:
        for field in fields(self):
            check_external_archive_field(field.name, getattr(self, field.name))
        object.__setattr__(self, "pruning", normalize_prune_policy(self.pruning))
        if self.capacity is not None:
            object.__setattr__(self, "capacity", int(self.capacity))
        if self.truncate_size is not None:
            object.__setattr__(self, "truncate_size", int(self.truncate_size))
            if self.capacity is None:
                raise ValueError("truncate_size requires a finite capacity.")
            if self.truncate_size > self.capacity:
                raise ValueError("truncate_size must be <= capacity.")
        if self.hv_ref_point is not None:
            object.__setattr__(self, "hv_ref_point", [float(value) for value in self.hv_ref_point])
        object.__setattr__(self, "rng_seed", int(self.rng_seed))
        object.__setattr__(self, "objective_tolerance", float(self.objective_tolerance))
        object.__setattr__(self, "decision_tolerance", float(self.decision_tolerance))
        self._warn_about_unused_fields()

    def _warn_about_unused_fields(self) -> None:
        unused: list[str] = []
        if self.capacity is None and self.pruning != "crowding":
            unused.append(f"pruning={self.pruning!r} has no effect without a capacity: the archive is unbounded")
        if self.hv_ref_point is not None and (self.capacity is None or self.pruning not in _HV_POLICIES):
            unused.append(f"hv_ref_point is used only by HV pruning of a bounded archive (pruning={self.pruning!r})")
        if self.rng_seed != 0 and (self.capacity is None or self.pruning != "ref_dirs"):
            unused.append(f"rng_seed is used only by ref_dirs pruning of a bounded archive (pruning={self.pruning!r})")
        for message in unused:
            warnings.warn(f"External archive: {message}.", UserWarning, stacklevel=3)


__all__ = [
    "CanonicalPrunePolicy",
    "DeduplicateIn",
    "ExternalArchiveConfig",
    "PrunePolicy",
    "check_external_archive_field",
    "normalize_prune_policy",
]
