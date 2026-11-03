from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vamos.engine.tuning import (
    EvalContext,
    Instance,
    ParamSpace,
    TrialResult,
    TuningTask,
    available_model_based_backends,
    filter_active_config,
    save_history_csv,
    save_history_json,
)
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.stats import select_configs_by_paired_test


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def rank_history_topk(history: list[TrialResult], param_space: ParamSpace, k: int) -> list[dict[str, Any]]:
    if not history:
        return []
    best_by_cfg: dict[str, dict[str, Any]] = {}
    for tr in history:
        active_cfg = filter_active_config(dict(tr.config), param_space)
        cfg_json = json.dumps(active_cfg, sort_keys=True)
        row = best_by_cfg.get(cfg_json)
        score = float(tr.score)
        if row is None or score > float(row["score"]):
            best_by_cfg[cfg_json] = {"score": score, "config": active_cfg}
    ranked = sorted(best_by_cfg.values(), key=lambda d: float(d["score"]), reverse=True)
    return ranked[: max(1, int(k))]


def evaluate_config_split(
    *,
    config: dict[str, Any],
    eval_fn: EvalFn,
    instances: list[Instance],
    seeds: list[int],
    budget: int,
    aggregator: Callable[[list[float]], float],
) -> dict[str, Any]:
    scores: list[float] = []
    rows_total = 0
    rows_ok = 0
    for inst in instances:
        for seed in seeds:
            rows_total += 1
            ctx = EvalContext(instance=inst, seed=int(seed), budget=int(budget))
            try:
                result = eval_fn(config, ctx)
                score = float(result[0]) if isinstance(result, tuple) else float(result)
            except Exception:
                score = float("nan")
            # A failed run scores NaN (see make_evaluator), so only finite scores are successful rows.
            rows_ok += int(np.isfinite(score))
            scores.append(score)
    valid = [s for s in scores if np.isfinite(s)]
    agg = float(aggregator(valid)) if valid else float("nan")
    return {
        "score_agg": agg,
        "score_mean": float(np.mean(valid)) if valid else float("nan"),
        "score_median": float(np.median(valid)) if valid else float("nan"),
        "score_p25": float(np.percentile(valid, 25)) if valid else float("nan"),
        "score_p10": float(np.percentile(valid, 10)) if valid else float("nan"),
        "rows_total": int(rows_total),
        "rows_ok": int(rows_ok),
        "fail_rate": float(1.0 - (rows_ok / rows_total if rows_total else 0.0)),
    }


def append_summary(
    output_dir: Path,
    *,
    summary_updates: dict[str, Any] | None = None,
    artifact_updates: dict[str, str] | None = None,
) -> None:
    summary_path = output_dir / "tuning_summary.json"
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary_updates:
        payload.update(summary_updates)
    if artifact_updates:
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        artifacts.update(artifact_updates)
        payload["artifacts"] = artifacts
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_statistical_finisher(
    *,
    candidates: list[dict[str, Any]],
    eval_fn: EvalFn,
    instances: list[Instance],
    seeds: list[int],
    budget: int,
    aggregator: Callable[[list[float]], float],
    alpha: float,
    min_blocks: int,
    failure_score: float,
    use_friedman: bool,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    blocks = [(inst, int(seed)) for inst in instances for seed in seeds]
    if not blocks:
        return None

    n_cfg = len(candidates)
    n_blocks = len(blocks)
    scores = np.full((n_cfg, n_blocks), float(failure_score), dtype=float)
    block_rows: list[dict[str, Any]] = []

    for cfg_idx, row in enumerate(candidates):
        cfg = dict(row["config"])
        for block_idx, (inst, seed) in enumerate(blocks):
            ctx = EvalContext(instance=inst, seed=int(seed), budget=int(budget))
            try:
                result = eval_fn(cfg, ctx)
                score = float(result[0]) if isinstance(result, tuple) else float(result)
            except Exception:
                score = float(failure_score)
            if not np.isfinite(score):
                score = float(failure_score)
            scores[cfg_idx, block_idx] = score
            block_rows.append(
                {
                    "candidate_idx": int(cfg_idx),
                    "instance": str(inst.name),
                    "seed": int(seed),
                    "block_idx": int(block_idx),
                    "score": float(score),
                }
            )

    agg_scores = np.asarray([float(aggregator(scores[i, :].tolist())) for i in range(n_cfg)], dtype=float)
    winner_idx = int(np.argmax(agg_scores))
    keep_mask = np.ones(n_cfg, dtype=bool)
    method = "aggregate_only"
    friedman_pvalue: float | None = None

    if n_cfg >= 2 and n_blocks >= int(min_blocks):
        should_run_paired = True
        if bool(use_friedman) and n_cfg >= 3 and n_blocks >= 3:
            try:
                from scipy.stats import friedmanchisquare  # type: ignore[import-untyped]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    _, p_val = friedmanchisquare(*[scores[i, :] for i in range(n_cfg)])
                if np.isfinite(p_val):
                    friedman_pvalue = float(p_val)
                    if float(p_val) > float(alpha):
                        should_run_paired = False
                        method = "friedman_no_difference"
            except Exception:
                _logger().debug("Friedman pre-check failed; continuing with paired tests.", exc_info=True)

        if should_run_paired:
            keep_mask = select_configs_by_paired_test(
                scores=scores,
                maximize=True,
                alpha=float(alpha),
                aggregator=aggregator,
            )
            if not bool(keep_mask.any()):
                keep_mask[winner_idx] = True
            alive_idx = np.flatnonzero(keep_mask)
            if alive_idx.size > 0:
                best_local = int(np.argmax(agg_scores[alive_idx]))
                winner_idx = int(alive_idx[best_local])
            method = "paired_holm"

    candidate_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        row_scores = scores[idx, :]
        candidate_rows.append(
            {
                "candidate_idx": int(idx),
                "tune_score": float(row["score"]),
                "score_agg": float(agg_scores[idx]),
                "score_mean": float(np.mean(row_scores)),
                "score_median": float(np.median(row_scores)),
                "score_p25": float(np.percentile(row_scores, 25)),
                "score_p10": float(np.percentile(row_scores, 10)),
                "kept_by_test": bool(keep_mask[idx]),
                "selected": bool(idx == winner_idx),
                "config_json": json.dumps(row["config"], sort_keys=True),
            }
        )

    return {
        "winner_idx": int(winner_idx),
        "winner_config": dict(candidates[winner_idx]["config"]),
        "method": str(method),
        "alpha": float(alpha),
        "num_candidates": int(n_cfg),
        "num_blocks": int(n_blocks),
        "friedman_pvalue": (None if friedman_pvalue is None else float(friedman_pvalue)),
        "candidate_rows": candidate_rows,
        "block_rows": block_rows,
    }


def persist_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    task: TuningTask,
    best_config: dict[str, Any],
    history: list[TrialResult],
    elapsed_seconds: float,
    resolved_jobs: int,
    *,
    available_backends_fn: Callable[[], dict[str, bool]] | None = None,
) -> None:
    best_active = filter_active_config(best_config, task.param_space)
    best_score = max((float(h.score) for h in history), default=float("nan"))

    best_raw_path = output_dir / "best_config_raw.json"
    best_active_path = output_dir / "best_config_active.json"
    summary_path = output_dir / "tuning_summary.json"
    history_json_path = output_dir / "tuning_history.json"
    history_csv_path = output_dir / "tuning_history.csv"

    best_raw_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    best_active_path.write_text(json.dumps(best_active, indent=2), encoding="utf-8")
    save_history_json(history, task.param_space, history_json_path, include_raw=True)
    save_history_csv(history, task.param_space, history_csv_path, include_raw=True)

    available_backends = available_model_based_backends() if available_backends_fn is None else dict(available_backends_fn())
    arg_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "schema_version": "vamos_tuning_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": str(args.backend),
        "problem": str(args.problem),
        "algorithm": str(args.algorithm),
        "n_var": int(args.n_var),
        "n_obj": int(args.n_obj),
        "budget_per_run": int(task.budget_per_run),
        "seed": int(args.seed),
        "n_jobs_resolved": int(resolved_jobs),
        "aggregate_mode": str(args.aggregate_mode),
        "runtime_penalty": float(args.runtime_penalty),
        "failure_score": float(args.failure_score),
        "trials_observed": int(len(history)),
        "best_score": float(best_score),
        "elapsed_seconds": float(elapsed_seconds),
        "available_model_backends": available_backends,
        "args": arg_payload,
        "artifacts": {
            "best_config_raw": best_raw_path.name,
            "best_config_active": best_active_path.name,
            "tuning_summary": summary_path.name,
            "history_json": history_json_path.name,
            "history_csv": history_csv_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _logger().info("Saved: %s", best_raw_path)
    _logger().info("Saved: %s", best_active_path)
    _logger().info("Saved: %s", summary_path)
    _logger().info("Saved: %s", history_json_path)
    _logger().info("Saved: %s", history_csv_path)
