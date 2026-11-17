"""The CLI's ``--moead-aggregation`` names every aggregation MOEA/D accepts, and accepts each of them.

The option takes no ``choices``: the name goes to ``MOEADConfig``, which validates it (aliases such as
``tschebyscheff`` included), so its help is the place that lists the aggregations.
"""

from __future__ import annotations

import argparse
from typing import get_args

import pytest

from vamos.engine.algorithm._builder_decomposition import build_moead_algorithm
from vamos.engine.algorithm.config.types import AggregationName
from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.common import _collect_moead_variation
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _parser() -> argparse.ArgumentParser:
    return build_parser(default_config=ExperimentConfig(), pre_parser=build_pre_parser(), spec_defaults=load_spec_defaults(None))


def _aggregation_action(parser: argparse.ArgumentParser) -> argparse.Action:
    return next(action for action in parser._actions if "--moead-aggregation" in action.option_strings)


def test_the_help_lists_every_aggregation() -> None:
    action = _aggregation_action(_parser())
    assert action.choices is None
    names = get_args(AggregationName)
    assert set(names) >= {"inverse_weight_tchebycheff", "augmented_tchebycheff", "inverted_pbi", "modified_tchebycheff"}
    missing = [name for name in names if name not in (action.help or "")]
    assert missing == []


@pytest.mark.parametrize("name", get_args(AggregationName))
def test_the_cli_accepts_every_aggregation(name: str) -> None:
    args = _parser().parse_args(["--algorithm", "moead", "--moead-aggregation", name])
    variation = _collect_moead_variation(args)
    assert variation is not None and variation["aggregation"] == name
    _, config = build_moead_algorithm(kernel=NumPyKernel(), problem=ZDT1Problem(n_var=6), pop_size=20, moead_variation=variation)
    assert config.to_dict()["aggregation"][0] == name  # type: ignore[index]


def test_an_unknown_aggregation_fails_with_the_list() -> None:
    args = _parser().parse_args(["--algorithm", "moead", "--moead-aggregation", "chebyshev"])
    with pytest.raises(ConfigurationError, match="aggregation must be one of"):
        build_moead_algorithm(
            kernel=NumPyKernel(), problem=ZDT1Problem(n_var=6), pop_size=20, moead_variation=_collect_moead_variation(args)
        )
