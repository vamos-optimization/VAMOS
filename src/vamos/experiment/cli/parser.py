"""
CLI argument parsing and validation helpers.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

from vamos.engine.config.spec import allowed_override_keys, validate_experiment_spec
from vamos.foundation.core.experiment_config import ExperimentConfig

from .args import build_parser, build_pre_parser
from .args_algorithms import _as_block
from .common import _collect_generic_variation, collect_nsgaii_variation_args
from .loaders import load_spec_defaults
from .spec_args import parser_spec_keys
from .validation import finalize_args

# Algorithms whose ``defaults.<name>`` block sets command-line options, and their operator sub-blocks.
_OPTION_BLOCKS = ("nsgaii", "moead", "smsemoa", "nsgaiii", "agemoea", "rvea")
_OPERATOR_BLOCKS = ("crossover", "mutation")


def _config_key_path(dest: str, spec_keys: set[str]) -> tuple[str, ...] | None:
    """Path under ``defaults`` of the config key that sets option ``dest``, if a config key can set it.

    ``max_evaluations`` -> ``(max_evaluations,)``; ``nsgaii_crossover_prob`` -> ``(nsgaii, crossover, prob)``;
    ``nsgaii_crossover`` -> ``(nsgaii, crossover, method)``; ``moead_aggregation`` -> ``(moead, aggregation)``.
    """
    if dest in spec_keys:
        return (dest,)
    block, _, rest = dest.partition("_")
    if block not in _OPTION_BLOCKS or not rest:
        return None
    operator, _, field = rest.partition("_")
    if operator in _OPERATOR_BLOCKS:
        return (block, operator, field or "method")
    return (block, rest)


def _sets_key(defaults: Mapping[str, object], path: tuple[str, ...]) -> bool:
    node: object = defaults
    for part in path:
        block = _as_block(node)
        if part not in block:
            return False
        node = block[part]
    return True


def _config_keys(parser: argparse.ArgumentParser, args: argparse.Namespace, defaults: Mapping[str, object]) -> dict[str, str]:
    """Map each option whose value comes from the config file to its config key.

    An option keeps the config value as its parser default, so a value the command line changed is left out.
    """
    spec_keys = parser_spec_keys(parser)
    keys: dict[str, str] = {}
    for action in parser._actions:
        path = _config_key_path(action.dest, spec_keys) if action.option_strings else None
        if path is None or not _sets_key(defaults, path):
            continue
        if getattr(args, action.dest, None) == parser.get_default(action.dest):
            keys[action.option_strings[0]] = "defaults." + ".".join(path)
    return keys


# Config-file aware parser.
def parse_args(default_config: ExperimentConfig, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and finalize the arguments of a standard ``vamos`` run.

    With ``--validate-config`` the returned namespace has ``validate_config`` set; the caller then
    runs the pre-run checks instead of the experiment.
    """
    pre_parser = build_pre_parser()
    pre_args, remaining = pre_parser.parse_known_args(argv)

    if getattr(pre_args, "validate_config", False) and not getattr(pre_args, "config", None):
        pre_parser.error("--validate-config requires --config.")

    try:
        spec_defaults = load_spec_defaults(pre_args.config)
    except Exception as exc:
        pre_parser.error(str(exc))
    parser = build_parser(
        default_config=default_config,
        pre_parser=pre_parser,
        spec_defaults=spec_defaults,
    )
    if pre_args.config:
        try:
            spec_keys = parser_spec_keys(parser)
            allowed_keys = allowed_override_keys(spec_keys)
            validate_experiment_spec(spec_defaults.spec, allowed_overrides=allowed_keys)
        except Exception as exc:
            pre_parser.error(str(exc))
    args = parser.parse_args(remaining)
    args.validate_config = bool(getattr(pre_args, "validate_config", False))
    if pre_args.config:
        parser.config_keys = _config_keys(parser, args, spec_defaults.experiment_defaults)
    return finalize_args(
        parser,
        args,
        spec_defaults=spec_defaults,
        config_path=getattr(pre_args, "config", None),
    )


__all__ = ["parse_args", "collect_nsgaii_variation_args", "_collect_generic_variation"]
