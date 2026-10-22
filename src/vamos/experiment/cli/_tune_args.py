from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ._tune_irace import add_irace_arguments, validate_irace_arguments


def build_parser(
    *,
    builders: Mapping[str, Any],
    all_backends: Sequence[str],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VAMOS tuning CLI (unified backends)")
    parser.add_argument("--problem", type=str, default="zdt1", help="Problem ID (e.g., zdt1).")
    parser.add_argument("--instances", type=str, default="", help="Optional comma-separated problem IDs. Overrides --problem.")
    parser.add_argument("--algorithm", type=str, default="nsgaii", choices=sorted(builders), help="Algorithm family to tune.")
    parser.add_argument("--backend", type=str, default="optuna", choices=tuple(all_backends), help="Tuning backend.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Apply tiny smoke-test defaults for quick CLI verification. "
            "Keeps the requested algorithm/backend, clamps budgets and workers, "
            "and disables validation/test/finisher stages."
        ),
    )
    parser.add_argument(
        "--backend-fallback",
        type=str,
        default="error",
        choices=["error", "racing", "random"],
        help="Fallback backend if requested model backend is unavailable.",
    )
    parser.add_argument("--list-backends", action="store_true", help="Print backend availability and exit.")

    parser.add_argument("--n-var", type=int, default=30, help="Number of variables.")
    parser.add_argument("--n-obj", type=int, default=2, help="Number of objectives.")
    parser.add_argument("--budget", type=int, default=5000, help="Max evaluations per algorithm run.")
    parser.add_argument("--tune-budget", type=int, default=200, help="Racing experiments or model trials.")
    parser.add_argument("--seed", type=int, default=1, help="Global seed.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Seeds per configuration.")
    parser.add_argument("--validation-seeds", type=str, default="", help="Validation seed list or ranges (e.g., 1001:1011).")
    parser.add_argument("--test-seeds", type=str, default="", help="Test seed list or ranges (e.g., 2001:2011).")
    parser.add_argument("--pop-size", type=int, default=100, help="Fallback fixed population size.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1 means CPU cores - 1).")
    parser.add_argument("--ref-point", type=str, default=None, help="HV reference point: comma list, e.g. 10,10,10.")
    parser.add_argument(
        "--aggregate-mode",
        type=str,
        default="mean",
        choices=["mean", "median", "p25", "p10"],
        help="Aggregation across instance/seed scores.",
    )
    parser.add_argument(
        "--runtime-penalty",
        type=float,
        default=0.0,
        help="Lambda in score = HV - lambda*log1p(runtime_seconds).",
    )
    parser.add_argument(
        "--failure-score",
        type=float,
        default=0.0,
        help="Score used when an evaluation fails.",
    )

    parser.add_argument("--multi-fidelity", action=argparse.BooleanOptionalAction, default=True, help="Enable multi-fidelity schedule.")
    parser.add_argument("--fidelity-levels", type=str, default=None, help="Comma budgets, e.g. 500,1000,1500.")
    parser.add_argument("--fidelity-promotion-ratio", type=float, default=0.3, help="Promotion ratio for racing multi-fidelity.")
    parser.add_argument("--fidelity-min-configs", type=int, default=3, help="Minimum promoted configs per fidelity level.")
    parser.add_argument("--fidelity-warm-start", dest="fidelity_warm_start", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--initial-configs", type=int, default=20, help="Initial sampled configs for racing.")
    parser.add_argument("--elimination-fraction", type=float, default=0.25, help="Racing elimination fraction.")
    parser.add_argument("--min-blocks-before-elimination", type=int, default=3, help="Racing grace blocks before pruning.")
    parser.add_argument("--use-statistical-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance for racing statistical elimination.")

    parser.add_argument("--bohb-reduction-factor", type=int, default=3, help="Reduction factor for BOHB/Hyperband-style backends.")
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="Optional wallclock timeout. 0 disables.")
    parser.add_argument("--show-progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fidelity-min-instance-frac",
        type=float,
        default=1.0,
        help="Minimum instance fraction at lowest budget for model backends (1.0 disables instance subsampling).",
    )
    parser.add_argument(
        "--fidelity-min-seed-count",
        type=int,
        default=0,
        help="Minimum seed count at lowest budget for model backends (0 uses all seeds).",
    )
    parser.add_argument(
        "--fidelity-max-seed-count",
        type=int,
        default=0,
        help="Maximum seed count at highest budget for model backends (0 uses all seeds).",
    )
    parser.add_argument(
        "--fidelity-selection-seed",
        type=int,
        default=-1,
        help="Seed for deterministic fidelity subsampling (-1 uses --seed).",
    )
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default="",
        help="Optional Optuna storage URL for persistent/restartable studies (e.g. sqlite:///results/tune.db).",
    )
    parser.add_argument(
        "--optuna-study-name",
        type=str,
        default="",
        help="Optional Optuna study name. Used with --optuna-storage.",
    )
    parser.add_argument(
        "--optuna-load-if-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --optuna-storage, resume existing study if present.",
    )
    add_irace_arguments(parser)

    parser.add_argument("--split-seed", type=int, default=42, help="Seed used to split instances into train/validation/test.")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="suite_stratified",
        choices=["suite_stratified", "random"],
        help="Instance split strategy.",
    )
    parser.add_argument("--train-frac", type=float, default=0.6, help="Train fraction for instance split.")
    parser.add_argument("--validation-frac", type=float, default=0.2, help="Validation fraction for instance split.")
    parser.add_argument("--run-validation", action=argparse.BooleanOptionalAction, default=True, help="Evaluate top-k on validation split.")
    parser.add_argument("--run-test", action=argparse.BooleanOptionalAction, default=False, help="Evaluate selected configs on test split.")
    parser.add_argument("--validation-budget", type=int, default=0, help="Validation evaluation budget (0 uses --budget).")
    parser.add_argument("--test-budget", type=int, default=0, help="Test evaluation budget (0 uses --budget).")
    parser.add_argument("--validation-topk", type=int, default=5, help="Top-k configs from tuning history to validate.")
    parser.add_argument(
        "--run-statistical-finisher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run statistical finisher on train split top-k candidates before validation/test.",
    )
    parser.add_argument("--finisher-topk", type=int, default=5, help="Top-k candidates for statistical finisher.")
    parser.add_argument("--finisher-min-blocks", type=int, default=3, help="Minimum blocks required for finisher statistical tests.")
    parser.add_argument("--finisher-budget", type=int, default=0, help="Finisher evaluation budget (0 uses --budget).")
    parser.add_argument("--finisher-alpha", type=float, default=0.05, help="Significance level for finisher statistical tests.")
    parser.add_argument(
        "--finisher-use-friedman",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Friedman pre-check before finisher paired tests.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("results") / "tuning")
    parser.add_argument("--name", type=str, default="", help="Optional run name (otherwise auto-generated).")
    return parser


def parse_args(
    argv: Sequence[str] | None,
    *,
    builders: Mapping[str, Any],
    all_backends: Sequence[str],
    parse_csv_ints: Callable[..., tuple[int, ...] | None],
) -> argparse.Namespace:
    parser = build_parser(builders=builders, all_backends=all_backends)
    args = parser.parse_args(argv)
    args.fidelity_levels = parse_csv_ints(args.fidelity_levels, parser, "--fidelity-levels", min_len=2)
    validate_irace_arguments(args, parser)
    if args.budget <= 0:
        parser.error("--budget must be > 0.")
    if args.tune_budget <= 0:
        parser.error("--tune-budget must be > 0.")
    if args.n_var <= 0:
        parser.error("--n-var must be > 0.")
    if args.n_obj <= 0:
        parser.error("--n-obj must be > 0.")
    if args.pop_size <= 0:
        parser.error("--pop-size must be > 0.")
    if args.n_seeds <= 0:
        parser.error("--n-seeds must be > 0.")
    if args.validation_topk <= 0:
        parser.error("--validation-topk must be > 0.")
    if args.finisher_topk <= 0:
        parser.error("--finisher-topk must be > 0.")
    if args.finisher_min_blocks <= 0:
        parser.error("--finisher-min-blocks must be > 0.")
    if args.validation_budget < 0 or args.test_budget < 0 or args.finisher_budget < 0:
        parser.error("--validation-budget, --test-budget and --finisher-budget must be >= 0.")
    if not (0.0 < float(args.finisher_alpha) < 1.0):
        parser.error("--finisher-alpha must be in (0, 1).")
    if float(args.runtime_penalty) < 0.0:
        parser.error("--runtime-penalty must be >= 0.")
    if not (0.0 < float(args.fidelity_min_instance_frac) <= 1.0):
        parser.error("--fidelity-min-instance-frac must be in (0, 1].")
    if int(args.fidelity_min_seed_count) < 0:
        parser.error("--fidelity-min-seed-count must be >= 0.")
    if int(args.fidelity_max_seed_count) < 0:
        parser.error("--fidelity-max-seed-count must be >= 0.")
    if int(args.fidelity_min_seed_count) > 0 and int(args.fidelity_max_seed_count) > 0:
        if int(args.fidelity_min_seed_count) > int(args.fidelity_max_seed_count):
            parser.error("--fidelity-min-seed-count cannot exceed --fidelity-max-seed-count.")
    if not (0.0 < float(args.train_frac) < 1.0):
        parser.error("--train-frac must be in (0, 1).")
    if not (0.0 < float(args.validation_frac) < 1.0):
        parser.error("--validation-frac must be in (0, 1).")
    if float(args.train_frac) + float(args.validation_frac) >= 1.0:
        parser.error("--train-frac + --validation-frac must be < 1.")
    return args
