from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .state import ConfigState, EliteEntry
from .stats import build_score_matrix, select_configs_by_frace

if TYPE_CHECKING:
    from .scenario import Scenario
    from .tuning_task import TuningTask


def compute_aggregated_scores(configs: list[ConfigState], task: TuningTask) -> tuple[list[int], npt.NDArray[np.float64]]:
    """
    Compute aggregated scores for all alive configs that have at least one score.
    """
    alive_indices: list[int] = []
    agg_values: list[float] = []
    for idx, state in enumerate(configs):
        if not state.alive or not state.scores:
            continue
        alive_indices.append(idx)
        agg_values.append(float(task.aggregator(state.scores)))
    if not agg_values:
        return [], np.array([], dtype=float)
    return alive_indices, np.asarray(agg_values, dtype=float)


def rank_based_elimination(
    configs: list[ConfigState],
    alive_indices: list[int],
    agg_scores: npt.NDArray[np.float64],
    *,
    task: TuningTask,
    scenario: Scenario,
) -> bool:
    n_alive = len(alive_indices)
    if n_alive <= 1:
        return False

    order = np.argsort(-agg_scores if task.maximize else agg_scores)

    target_keep = max(
        scenario.min_survivors,
        int(math.ceil(n_alive * (1.0 - scenario.elimination_fraction))),
    )
    target_keep = max(1, min(target_keep, n_alive))

    keep_rows = {int(idx) for idx in order[:target_keep]}
    eliminated_any = False
    for row_idx, cfg_idx in enumerate(alive_indices):
        if row_idx not in keep_rows:
            if configs[cfg_idx].alive:
                configs[cfg_idx].alive = False
                eliminated_any = True
    return eliminated_any


def fill_survivors(
    keep: npt.NDArray[np.bool_],
    agg_scores: npt.NDArray[np.float64],
    *,
    maximize: bool,
    k: int,
) -> npt.NDArray[np.bool_]:
    """Return ``keep`` with at least ``k`` rows kept, or all of them when there are fewer.

    Every row that ``keep`` already keeps stays kept; the best of the other rows by aggregate
    score fill the remaining places, the first row winning a tie.
    """
    filled = np.array(keep, dtype=bool)
    missing = min(k, filled.size) - int(filled.sum())
    if missing > 0:
        order = np.argsort(-agg_scores if maximize else agg_scores, kind="stable")
        filled[order[~filled[order]][:missing]] = True
    return filled


def eliminate_configs(configs: list[ConfigState], *, task: TuningTask, scenario: Scenario) -> bool:
    """
    Eliminate configurations based on the current scores.
    """
    alive_indices, agg_scores = compute_aggregated_scores(configs, task)
    if agg_scores.size == 0 or len(alive_indices) <= 1:
        return False

    lengths = [len(configs[idx].scores) for idx in alive_indices]
    min_len = min(lengths)
    max_len = max(lengths)

    if not scenario.use_statistical_tests:
        # Even in rank-only mode, wait for a minimum number of blocks to reduce
        # noisy eliminations of recently spawned/restarted configurations.
        if min_len < scenario.min_blocks_before_elimination:
            return False
        return rank_based_elimination(
            configs,
            alive_indices,
            agg_scores,
            task=task,
            scenario=scenario,
        )

    # In statistical mode, the first test waits for first_test blocks, as irace's firstTest.
    if min_len < max(scenario.first_test, scenario.min_blocks_before_elimination) or min_len <= 1:
        return False

    # Keep waiting until all alive configs are aligned on the same number of blocks.
    if min_len != max_len:
        return False

    scores, aligned_indices = build_score_matrix(configs)
    # F-race: Friedman on the ranks of each block and Conover's post-hoc, or Wilcoxon with two alive.
    keep_mask = select_configs_by_frace(scores, maximize=task.maximize, alpha=scenario.alpha)
    # Fewer survivors than the minimum: keep those of the test and add the best of the others by aggregate score.
    keep_mask = fill_survivors(keep_mask, agg_scores, maximize=task.maximize, k=max(1, scenario.min_survivors))

    eliminated_any = False
    for row_idx, cfg_idx in enumerate(aligned_indices):
        if not keep_mask[row_idx]:
            if configs[cfg_idx].alive:
                configs[cfg_idx].alive = False
                eliminated_any = True
    return eliminated_any


def update_elite_archive(
    configs: list[ConfigState],
    *,
    task: TuningTask,
    scenario: Scenario,
    elite_archive: list[EliteEntry],
) -> list[EliteEntry]:
    """
    Update the elite archive based on the current alive configurations.
    """
    if not scenario.use_elitist_restarts:
        return elite_archive

    indices, agg_scores = compute_aggregated_scores(configs, task)
    if not indices:
        return elite_archive

    n_alive = len(indices)
    if n_alive == 0:
        return elite_archive

    k = max(1, int(math.ceil(scenario.elite_fraction * n_alive)))

    if task.maximize:
        order = np.argsort(-agg_scores)
    else:
        order = np.argsort(agg_scores)

    elite_entries: list[EliteEntry] = []
    for rank in range(min(k, n_alive)):
        row_idx = int(order[rank])
        cfg_idx = indices[row_idx]
        state = configs[cfg_idx]
        score = float(agg_scores[row_idx])
        elite_entries.append(EliteEntry(config=dict(state.config), score=score))

    all_elites = elite_archive + elite_entries
    if not all_elites:
        return []

    if task.maximize:
        all_elites.sort(key=lambda e: e.score, reverse=True)
    else:
        all_elites.sort(key=lambda e: e.score)

    return all_elites[: scenario.max_elite_archive_size]


__all__ = [
    "compute_aggregated_scores",
    "eliminate_configs",
    "fill_survivors",
    "rank_based_elimination",
    "update_elite_archive",
]
