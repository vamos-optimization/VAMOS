"""Exact replay planning built on shared resolved-spec reconstruction."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol, EngineName
from vamos.experiment.types import TerminationSpec

from .component_support import component_reconstructability
from .errors import (
    ComponentNotReconstructableError,
    OutputCollisionError,
    ReplayUnavailableError,
    UnsupportedReplayProviderError,
)
from .jsonio import canonical_json_bytes, sha256_bytes
from .lineage import MAX_REPLAY_LINEAGE_DEPTH
from .models import RunManifest, StoredRun, deep_freeze, deep_thaw
from .reports import VerificationReport
from .resolved_reconstruction import (
    ReconstructedRun,
    instantiate_reconstructed_problem,
    reconstruct_resolved_run,
)


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """Internal immutable, pre-validated execution plan."""

    source_root: Path
    output_root: Path
    source_run_id: str
    root_run_id: str
    lineage_depth: int
    new_run_id: str
    source_manifest_sha256: str
    requested_spec: Mapping[str, Any]
    resolved_spec: Mapping[str, Any]
    problem: str
    n_var: int
    n_obj: int
    encoding: str
    algorithm: str
    algorithm_config: AlgorithmConfigProtocol
    termination: TerminationSpec
    engine: EngineName
    eval_strategy: str | None
    seed: int
    expected_arrays: tuple[str, ...]
    replay_plan_sha256: str


def build_replay_plan(stored: StoredRun, verification: VerificationReport, output: str | Path | None) -> ReplayPlan:
    """Construct and semantically prove a plan before any optimization begins."""
    _ensure_components(stored)
    if any(reason.code == "component_not_reconstructable" for reason in verification.reasons):
        # Raise the reconstruction's own error (for example a resolved-spec mismatch), not the generic level.
        reconstruct_resolved_run(stored.manifest.resolved_spec, root=stored.root)
    if verification.effective_replayability != "exact":
        raise ReplayUnavailableError(
            operation="reproduce run",
            field="$.effective_replayability",
            path=stored.root,
            reason="is not exact",
            expected="exact",
            actual=verification.effective_replayability,
            action="Run verify_run(path, require_level='exact') and correct every blocking finding.",
        )
    reconstructed = reconstruct_resolved_run(stored.manifest.resolved_spec, root=stored.root)
    resolved = cast(dict[str, Any], deep_thaw(reconstructed.resolved_spec))
    new_run_id = str(uuid.uuid4())
    output_root = _output_root(stored.root, new_run_id, output)
    _reject_collision(output_root)
    root_run_id, depth = _lineage(stored.manifest, stored.root)
    source_hash = str(stored.manifest["integrity"]["manifest_sha256"])
    descriptor = stored.manifest.artifact("result_bundle")
    expected_arrays = tuple(sorted(descriptor.array_contract)) if descriptor is not None and descriptor.array_contract else ()
    payload = {
        "source_run_id": stored.manifest.run_id,
        "source_manifest_sha256": source_hash,
        "resolved_spec": resolved,
        "expected_arrays": list(expected_arrays),
        "compatibility_level": verification.effective_replayability,
    }
    plan_hash = sha256_bytes(canonical_json_bytes(payload))
    return ReplayPlan(
        source_root=stored.root,
        output_root=output_root,
        source_run_id=stored.manifest.run_id,
        root_run_id=root_run_id,
        lineage_depth=depth,
        new_run_id=new_run_id,
        source_manifest_sha256=source_hash,
        requested_spec=cast(Mapping[str, Any], deep_freeze(stored.manifest.requested_spec)),
        resolved_spec=cast(Mapping[str, Any], deep_freeze(resolved)),
        problem=reconstructed.problem,
        n_var=reconstructed.n_var,
        n_obj=reconstructed.n_obj,
        encoding=reconstructed.encoding,
        algorithm=reconstructed.algorithm,
        algorithm_config=reconstructed.algorithm_config,
        termination=reconstructed.termination,
        engine=reconstructed.engine,
        eval_strategy=reconstructed.eval_strategy,
        seed=reconstructed.seed,
        expected_arrays=expected_arrays,
        replay_plan_sha256=plan_hash,
    )


def _ensure_components(stored: StoredRun) -> None:
    status, reasons = component_reconstructability(stored.manifest)
    if status == "reconstructable":
        return
    reason = reasons[0] if reasons else None
    error_type = (
        UnsupportedReplayProviderError
        if reason is not None and reason.code in {"custom_provider", "unsupported_provider"}
        else ComponentNotReconstructableError
    )
    raise error_type(
        operation="reproduce run",
        field=reason.field if reason is not None else "$.resolved_spec",
        path=stored.root,
        reason="contains a component outside exact built-in replay support",
        expected="reconstructable built-in components",
        actual=status,
        action=reason.action if reason is not None else "Regenerate with supported built-ins.",
    )


def _lineage(manifest: RunManifest, root_path: Path) -> tuple[str, int]:
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        return manifest.run_id, 1
    root = lineage.get("root_run_id")
    depth = lineage.get("depth")
    if isinstance(root, str) and isinstance(depth, int):
        if depth >= MAX_REPLAY_LINEAGE_DEPTH:
            raise ReplayUnavailableError(
                operation="reproduce run",
                field="$.lineage.depth",
                path=root_path,
                reason="has reached the bounded replay-lineage limit",
                expected=f"depth below {MAX_REPLAY_LINEAGE_DEPTH}",
                actual=depth,
                action="Use the root or an earlier replay as the source instead of extending this lineage.",
                optimization_executed=False,
            )
        return root, depth + 1
    return manifest.run_id, 1


def _output_root(source: Path, new_run_id: str, output: str | Path | None) -> Path:
    candidate = Path(output) if output is not None else source.parent / "replays" / new_run_id
    return candidate.absolute()


def _reject_collision(path: Path) -> None:
    if os.path.lexists(path):
        raise OutputCollisionError(
            operation="reproduce run",
            path=path,
            reason="already exists",
            expected="an output path that does not exist",
            actual="occupied output path",
            action="Choose another --output directory; replay never overwrites or merges runs.",
        )


__all__ = [
    "ReconstructedRun",
    "ReplayPlan",
    "build_replay_plan",
    "instantiate_reconstructed_problem",
    "reconstruct_resolved_run",
]
