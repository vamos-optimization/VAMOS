"""
NSGA-II variants: generational, steady-state, bounded archive, and unbounded archive.

This advanced example runs DTLZ2 (3 objectives) with four configurations:
1) Standard generational NSGA-II
2) Steady-state NSGA-II
3) NSGA-II with bounded external archive (crowding, capacity=100)
4) NSGA-II with unbounded external archive

For the unbounded archive run, the script reports two top-100 selections:
- **crowding**: the most spread solutions selected by iterative crowding distance
- **balanced_sum**: compromise solutions ranked by normalized objective sum (lower is better)

Usage:
    python examples/advanced/nsgaii_variants_archives.py

Optional environment variables:
    VAMOS_ADV_ENGINE (default: numba)
    VAMOS_ADV_EVALS  (default: 40000)
    VAMOS_ADV_SEED   (default: 11)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import DTLZ2

ENGINE = os.environ.get("VAMOS_ADV_ENGINE", "numba")
MAX_EVALUATIONS = int(os.environ.get("VAMOS_ADV_EVALS", "40000"))
SEED = int(os.environ.get("VAMOS_ADV_SEED", "11"))
POP_SIZE = 100


def _build_generational_config() -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        # Canonical generational NSGA-II: produce a full offspring batch each step.
        .offspring_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )


def _build_steady_state_config() -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        # Generate one offspring per step (classic steady-state behavior).
        .offspring_size(1)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )


def _build_bounded_archive_config() -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .offspring_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        # For NSGA-II here, capacity + pruning are the effective controls.
        .external_archive(capacity=100, pruning="crowding")
        .build()
    )


def _build_unbounded_archive_config() -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .offspring_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        # Keep all non-dominated archive insertions (no capacity cap).
        .external_archive()
        .build()
    )


def _run_variant(name: str, problem: DTLZ2, config: NSGAIIConfig) -> dict[str, Any]:
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=config,
        max_evaluations=MAX_EVALUATIONS,
        seed=SEED,
        engine=ENGINE,
    )
    return {
        "name": name,
        "result": result,
    }


def _print_summary(entry: dict[str, Any]) -> None:
    name = str(entry["name"])
    result = entry["result"]

    print(f"\n[{name}]")
    print(f"  Returned solutions: {len(result)}")
    archive = result.data.get("archive")
    if isinstance(archive, dict) and archive.get("F") is not None:
        archive_f = np.asarray(archive.get("F"), dtype=float)
        print(f"  Archive size: {archive_f.shape[0]}")
    else:
        print("  Archive size: n/a")

    if result.F is not None and len(result.F) > 0:
        top1 = result.top_k(k=1, source="result", method="balanced_sum", nondominated_only=False)
        best_idx = int(top1["indices"][0])
        print(f"  Best compromise in result (index={best_idx}): {result.F[best_idx]}")


def _report_top_100_unbounded(entry: dict[str, Any]) -> None:
    result = entry["result"]

    # --- Top-100 by crowding distance (most spread solutions) ---
    try:
        top_cd = result.top_k(k=100, source="archive", method="crowding", nondominated_only=True)
    except ValueError:
        print("\n[unbounded_archive] No archive solutions available.")
        return

    top_cd_f = np.asarray(top_cd["F"], dtype=float)
    print(f"\n[unbounded_archive] Top {top_cd_f.shape[0]} most spread solutions (crowding)")
    print("  rank | crowding | objectives")
    for i in range(min(10, top_cd_f.shape[0])):
        print(f"  {i + 1:4d} | {top_cd['scores'][i]:.6f} | {np.array2string(top_cd_f[i], precision=6, suppress_small=True)}")
    if top_cd_f.shape[0] > 10:
        print(f"  ... ({top_cd_f.shape[0] - 10} more rows)")

    # --- Top-100 by balanced_sum (best compromise solutions) ---
    try:
        top_balanced = result.top_k(k=100, source="archive", method="balanced_sum", nondominated_only=True)
    except ValueError:
        return

    top_f = np.asarray(top_balanced["F"], dtype=float)
    top_scores = np.asarray(top_balanced["scores"], dtype=float)
    top_x = top_balanced["X"]

    print(f"\n[unbounded_archive] Top {top_f.shape[0]} best compromise solutions (balanced_sum)")
    print("  rank | score | objectives")
    for i in range(min(10, top_f.shape[0])):
        print(f"  {i + 1:4d} | {top_scores[i]:.6f} | {np.array2string(top_f[i], precision=6, suppress_small=True)}")
    if top_f.shape[0] > 10:
        print(f"  ... ({top_f.shape[0] - 10} more rows)")

    if top_x is not None and top_f.shape[0] > 0:
        print("\n  Best decision vector (balanced_sum, rank=1):")
        print(f"  {np.array2string(np.asarray(top_x)[0], precision=6, suppress_small=True)}")


def main() -> None:
    print("Running NSGA-II advanced variants on DTLZ2")
    print(f"  engine={ENGINE}, evaluations={MAX_EVALUATIONS}, seed={SEED}")

    problem = DTLZ2(n_obj=3)
    runs = [
        _run_variant("generational", problem, _build_generational_config()),
        _run_variant("steady_state", problem, _build_steady_state_config()),
        _run_variant("bounded_archive_crowding_100", problem, _build_bounded_archive_config()),
        _run_variant("unbounded_archive", problem, _build_unbounded_archive_config()),
    ]

    for entry in runs:
        _print_summary(entry)

    _report_top_100_unbounded(runs[-1])


if __name__ == "__main__":
    main()
