from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from vamos.engine.tuning import Instance, TrialResult, TuningTask, filter_active_config
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.tuning_task import TuningFailedError

from ._tune_post import (
    append_summary,
    evaluate_config_split,
    rank_history_topk,
    run_statistical_finisher,
)
from ._tune_scoring import evaluation_ledger


def write_split_artifacts(
    out_dir: Path,
    split_manifest: list[dict[str, Any]],
    *,
    train_seeds: list[int],
    validation_seeds: list[int],
    test_seeds: list[int],
) -> tuple[Path, Path]:
    split_csv_path = out_dir / "split_instances.csv"
    with split_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["instance", "suite", "split", "shared_instance"])
        writer.writeheader()
        writer.writerows(split_manifest)
    split_seed_path = out_dir / "split_seeds.json"
    split_seed_path.write_text(
        json.dumps(
            {
                "train_seeds": train_seeds,
                "validation_seeds": validation_seeds,
                "test_seeds": test_seeds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return split_csv_path, split_seed_path


def run_tuning_with_finisher(
    *,
    args: argparse.Namespace,
    task: TuningTask,
    eval_fn: EvalFn,
    resolved_jobs: int,
    train_instances: list[Instance],
    train_seeds: list[int],
    out_dir: Path,
    run_backend_fn: Any,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[TrialResult], float, dict[str, Any] | None, dict[str, str] | None]:
    """Run the backend and the optional statistical finisher.

    The summary updates count the tuning evaluations and how many failed. When every one of
    them failed, the command stops with a nonzero exit code instead of reporting a configuration.
    """
    ledger = evaluation_ledger(eval_fn)
    before = ledger.counts() if ledger is not None else (0, 0)
    t0 = time.perf_counter()
    try:
        best_config, history = run_backend_fn(args, task, eval_fn, resolved_jobs=resolved_jobs)
    except TuningFailedError as exc:
        raise SystemExit(_every_evaluation_failed_message(str(exc), ledger)) from exc
    elapsed = time.perf_counter() - t0

    finisher_summary_updates: dict[str, Any] | None = None
    if ledger is not None:
        # Racing with n_jobs > 1 evaluates in worker processes, which count in their own copy of the
        # evaluator; its trials record their evaluations, so those records are counted instead.
        recorded = _recorded_evaluations(history)
        evaluations, failures = recorded if recorded is not None else (ledger.evaluations - before[0], ledger.failures - before[1])
        if evaluations and failures == evaluations:
            raise SystemExit(_every_evaluation_failed_message(f"every one of the {evaluations} tuning evaluations failed", ledger))
        finisher_summary_updates = {"evaluations": evaluations, "failed_evaluations": failures}
    finisher_artifacts: dict[str, str] | None = None
    if bool(args.run_statistical_finisher):
        finisher_budget = int(args.finisher_budget) if int(args.finisher_budget) > 0 else int(args.budget)
        finisher_candidates = rank_history_topk(history, task.param_space, int(args.finisher_topk))
        finisher_result = run_statistical_finisher(
            candidates=finisher_candidates,
            eval_fn=eval_fn,
            instances=train_instances,
            seeds=train_seeds,
            budget=finisher_budget,
            aggregator=task.aggregator,
            alpha=float(args.finisher_alpha),
            min_blocks=int(args.finisher_min_blocks),
            failure_score=float(args.failure_score),
            use_friedman=bool(args.finisher_use_friedman),
        )
        if finisher_result is not None:
            best_config = dict(finisher_result["winner_config"])
            fin_summary_path = out_dir / "statistical_finisher_summary.json"
            fin_candidates_path = out_dir / "statistical_finisher_candidates.csv"
            fin_blocks_path = out_dir / "statistical_finisher_blocks.csv"
            fin_summary_path.write_text(json.dumps(finisher_result, indent=2), encoding="utf-8")
            candidate_rows = list(finisher_result["candidate_rows"])
            block_rows = list(finisher_result["block_rows"])
            with fin_candidates_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(candidate_rows[0].keys()))
                writer.writeheader()
                writer.writerows(candidate_rows)
            with fin_blocks_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(block_rows[0].keys()))
                writer.writeheader()
                writer.writerows(block_rows)
            finisher_summary_updates = {
                **(finisher_summary_updates or {}),
                "statistical_finisher_ran": True,
                "statistical_finisher_method": str(finisher_result["method"]),
                "statistical_finisher_budget": int(finisher_budget),
            }
            finisher_artifacts = {
                "statistical_finisher_summary": fin_summary_path.name,
                "statistical_finisher_candidates": fin_candidates_path.name,
                "statistical_finisher_blocks": fin_blocks_path.name,
            }
            logger.info(
                "Statistical finisher selected candidate %s using method '%s'.",
                finisher_result["winner_idx"],
                finisher_result["method"],
            )
    return best_config, history, elapsed, finisher_summary_updates, finisher_artifacts


def _recorded_evaluations(history: list[TrialResult]) -> tuple[int, int] | None:
    """Evaluations and failures that every trial of ``history`` records, or ``None`` if some does not.

    Racing records ``num_evals``, and random search and irace ``evaluations``, each with
    ``failed_evaluations``; the model-based backends record neither and evaluate in this process.
    """
    evaluations = failures = 0
    for trial in history:
        details = getattr(trial, "details", None) or {}
        count = details.get("num_evals", details.get("evaluations"))
        failed = details.get("failed_evaluations")
        if not isinstance(count, int) or not isinstance(failed, int):
            return None
        evaluations += count
        failures += failed
    return (evaluations, failures) if history else None


def _every_evaluation_failed_message(reason: str, ledger: Any) -> str:
    cause = f" First failure: {ledger.first_failure}." if ledger is not None and ledger.first_failure else ""
    return f"vamos tune: {reason}; no configuration was selected.{cause} The log shows each failure."


def run_validation_stage(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    history: list[TrialResult],
    task: TuningTask,
    eval_fn: EvalFn,
    validation_instances: list[Instance],
    validation_seeds: list[int],
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    champions: dict[str, dict[str, Any]] = {}
    if not bool(args.run_validation):
        return champions

    logger.info("Running validation split evaluation...")
    val_budget = int(args.validation_budget) if int(args.validation_budget) > 0 else int(args.budget)
    ranked = rank_history_topk(history, task.param_space, int(args.validation_topk))
    validation_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        metrics = evaluate_config_split(
            config=dict(row["config"]),
            eval_fn=eval_fn,
            instances=validation_instances,
            seeds=validation_seeds,
            budget=val_budget,
            aggregator=task.aggregator,
        )
        validation_rows.append(
            {
                "rank": int(rank),
                "tune_score": float(row["score"]),
                "config_json": json.dumps(row["config"], sort_keys=True),
                **metrics,
            }
        )
    validation_rows.sort(key=lambda d: float(d["score_agg"]), reverse=True)
    val_csv_path = out_dir / "validation_metrics.csv"
    with val_csv_path.open("w", encoding="utf-8", newline="") as fh:
        if validation_rows:
            dict_writer = csv.DictWriter(fh, fieldnames=list(validation_rows[0].keys()))
            dict_writer.writeheader()
            dict_writer.writerows(validation_rows)
        else:
            plain_writer = csv.writer(fh)
            plain_writer.writerow(
                [
                    "rank",
                    "tune_score",
                    "config_json",
                    "score_agg",
                    "score_mean",
                    "score_median",
                    "score_p25",
                    "score_p10",
                    "rows_total",
                    "rows_ok",
                    "fail_rate",
                ]
            )

    if validation_rows:
        best_global = validation_rows[0]
        best_robust = sorted(validation_rows, key=lambda d: float(d["score_p25"]), reverse=True)[0]
        best_fast = sorted(validation_rows, key=lambda d: float(d["score_mean"]), reverse=True)[0]
        champions = {
            "champion_global": json.loads(str(best_global["config_json"])),
            "champion_robust": json.loads(str(best_robust["config_json"])),
            "champion_fast": json.loads(str(best_fast["config_json"])),
        }
        champions_path = out_dir / "selected_configs_validation.json"
        champions_path.write_text(json.dumps(champions, indent=2), encoding="utf-8")
        append_summary(
            out_dir,
            summary_updates={"validation_ran": True, "validation_budget": int(val_budget)},
            artifact_updates={
                "validation_metrics": val_csv_path.name,
                "selected_configs_validation": champions_path.name,
            },
        )
    else:
        append_summary(
            out_dir,
            summary_updates={"validation_ran": True, "validation_budget": int(val_budget)},
            artifact_updates={"validation_metrics": val_csv_path.name},
        )
    return champions


def run_test_stage(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    champions: dict[str, dict[str, Any]],
    best_config: dict[str, Any],
    task: TuningTask,
    eval_fn: EvalFn,
    test_instances: list[Instance],
    test_seeds: list[int],
    logger: logging.Logger,
) -> None:
    if not bool(args.run_test):
        return

    logger.info("Running test split evaluation...")
    test_budget = int(args.test_budget) if int(args.test_budget) > 0 else int(args.budget)
    candidates = dict(champions)
    if not candidates:
        candidates = {"champion_global": filter_active_config(dict(best_config), task.param_space)}
    unique_candidates: dict[str, dict[str, Any]] = {}
    for name, cfg in candidates.items():
        unique_candidates[name] = dict(cfg)
    test_rows: list[dict[str, Any]] = []
    for label, cfg in unique_candidates.items():
        metrics = evaluate_config_split(
            config=dict(cfg),
            eval_fn=eval_fn,
            instances=test_instances,
            seeds=test_seeds,
            budget=test_budget,
            aggregator=task.aggregator,
        )
        test_rows.append(
            {
                "candidate": str(label),
                "config_json": json.dumps(cfg, sort_keys=True),
                **metrics,
            }
        )
    test_rows.sort(key=lambda d: float(d["score_agg"]), reverse=True)
    test_csv_path = out_dir / "test_metrics.csv"
    with test_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)
    append_summary(
        out_dir,
        summary_updates={"test_ran": True, "test_budget": int(test_budget)},
        artifact_updates={"test_metrics": test_csv_path.name},
    )
