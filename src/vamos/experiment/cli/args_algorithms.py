from __future__ import annotations

import argparse

from vamos.engine.config.spec import SpecBlock

from .types import SpecDefaults


def _as_block(value: object) -> SpecBlock:
    return value if isinstance(value, dict) else {}


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _add_nsga_family_arguments(parser: argparse.ArgumentParser, *, spec_defaults: SpecDefaults) -> None:
    nsgaii_defaults = spec_defaults.nsgaii_defaults
    moead_defaults = spec_defaults.moead_defaults
    smsemoa_defaults = spec_defaults.smsemoa_defaults
    nsgaiii_defaults = spec_defaults.nsgaiii_defaults

    nsgaii_crossover = _as_block(nsgaii_defaults.get("crossover"))
    nsgaii_mutation = _as_block(nsgaii_defaults.get("mutation"))
    moead_crossover = _as_block(moead_defaults.get("crossover"))
    moead_mutation = _as_block(moead_defaults.get("mutation"))
    smsemoa_crossover = _as_block(smsemoa_defaults.get("crossover"))
    smsemoa_mutation = _as_block(smsemoa_defaults.get("mutation"))
    nsgaiii_crossover = _as_block(nsgaiii_defaults.get("crossover"))
    nsgaiii_mutation = _as_block(nsgaiii_defaults.get("mutation"))

    parser.add_argument(
        "--nsgaii-crossover",
        choices=(
            "sbx",
            "blx_alpha",
            "ox",
            "order",
            "pmx",
            "cycle",
            "cx",
            "position",
            "position_based",
            "pos",
            "edge",
            "edge_recombination",
            "erx",
            "oxd",
        ),
        default=_as_str(nsgaii_crossover.get("method")),
        help="Crossover operator for NSGA-II. Continuous problems use sbx/blx_alpha; permutations use ox/pmx/cycle/position/edge.",
    )
    parser.add_argument(
        "--nsgaii-crossover-prob",
        default=_as_float(nsgaii_crossover.get("prob")),
        help="Crossover probability for NSGA-II real-coded operators (default: 1.0).",
    )
    parser.add_argument(
        "--nsgaii-crossover-eta",
        type=float,
        default=_as_float(nsgaii_crossover.get("eta")),
        help="Distribution index eta for SBX crossover (default: 20.0).",
    )
    parser.add_argument(
        "--nsgaii-crossover-alpha",
        type=float,
        default=_as_float(nsgaii_crossover.get("alpha")),
        help="Alpha for BLX-alpha crossover (default: 0.5).",
    )
    parser.add_argument(
        "--nsgaii-mutation",
        choices=("polynomial", "non_uniform", "swap", "insert", "scramble", "inversion", "displacement"),
        default=_as_str(nsgaii_mutation.get("method")),
        help="Mutation operator for NSGA-II. Continuous problems use polynomial/non_uniform; permutations use swap/insert/scramble/inversion/displacement.",
    )
    parser.add_argument(
        "--nsgaii-mutation-prob",
        type=str,
        default=_as_str(nsgaii_mutation.get("prob")),
        help="Mutation probability for NSGA-II operators (allow expressions like 1/n; uses n_var).",
    )
    parser.add_argument(
        "--nsgaii-mutation-eta",
        type=float,
        default=_as_float(nsgaii_mutation.get("eta")),
        help="Distribution index eta for polynomial mutation (default: 20.0).",
    )
    parser.add_argument(
        "--nsgaii-mutation-perturbation",
        type=float,
        default=_as_float(nsgaii_mutation.get("perturbation")),
        help="Perturbation magnitude for non-uniform mutation (default: 0.5).",
    )
    parser.add_argument(
        "--nsgaii-repair",
        choices=("clip", "reflect", "random", "resample", "round", "none"),
        default=_as_str(nsgaii_defaults.get("repair")),
        help="Repair strategy for NSGA-II (continuous encoding; use 'none' to disable).",
    )

    parser.add_argument(
        "--moead-crossover",
        choices=("sbx", "uniform"),
        default=_as_str(moead_crossover.get("method")),
        help="Crossover method for MOEA/D (sbx for real, uniform for binary/integer).",
    )
    parser.add_argument("--moead-crossover-prob", default=_as_float(moead_crossover.get("prob")), help="Crossover probability for MOEA/D.")
    parser.add_argument(
        "--moead-crossover-eta", type=float, default=_as_float(moead_crossover.get("eta")), help="SBX eta for MOEA/D (real encoding)."
    )
    parser.add_argument(
        "--moead-mutation",
        choices=("polynomial", "bitflip", "reset"),
        default=_as_str(moead_mutation.get("method")),
        help="Mutation method for MOEA/D (pm for real, bitflip for binary, reset for integer).",
    )
    parser.add_argument(
        "--moead-mutation-prob",
        default=_as_str(moead_mutation.get("prob")),
        help="Mutation probability for MOEA/D (allow expressions like 1/n; uses n_var).",
    )
    parser.add_argument(
        "--moead-mutation-eta",
        type=float,
        default=_as_float(moead_mutation.get("eta")),
        help="Polynomial mutation eta for MOEA/D (real encoding).",
    )
    parser.add_argument(
        "--moead-mutation-step",
        type=int,
        default=_as_int(moead_mutation.get("step")),
        help="Integer creep step for MOEA/D (integer encoding).",
    )
    parser.add_argument(
        "--moead-aggregation",
        default=_as_str(moead_defaults.get("aggregation")),
        help="Aggregation method for MOEA/D (tchebycheff, weighted_sum, pbi).",
    )

    parser.add_argument(
        "--smsemoa-crossover",
        choices=("sbx", "uniform"),
        default=_as_str(smsemoa_crossover.get("method")),
        help="Crossover method for SMS-EMOA (sbx for real, uniform for binary/integer).",
    )
    parser.add_argument(
        "--smsemoa-crossover-prob", default=_as_float(smsemoa_crossover.get("prob")), help="Crossover probability for SMS-EMOA."
    )
    parser.add_argument(
        "--smsemoa-crossover-eta", type=float, default=_as_float(smsemoa_crossover.get("eta")), help="SBX eta for SMS-EMOA (real encoding)."
    )
    parser.add_argument(
        "--smsemoa-mutation",
        choices=("polynomial", "bitflip", "reset"),
        default=_as_str(smsemoa_mutation.get("method")),
        help="Mutation method for SMS-EMOA (pm for real, bitflip for binary, reset for integer).",
    )
    parser.add_argument(
        "--smsemoa-mutation-prob",
        default=_as_str(smsemoa_mutation.get("prob")),
        help="Mutation probability for SMS-EMOA (allow expressions like 1/n; uses n_var).",
    )
    parser.add_argument(
        "--smsemoa-mutation-eta",
        type=float,
        default=_as_float(smsemoa_mutation.get("eta")),
        help="Polynomial mutation eta for SMS-EMOA (real encoding).",
    )
    parser.add_argument(
        "--smsemoa-mutation-step",
        type=int,
        default=_as_int(smsemoa_mutation.get("step")),
        help="Integer creep step for SMS-EMOA (integer encoding).",
    )

    parser.add_argument(
        "--nsga3-crossover",
        dest="nsgaiii_crossover",
        choices=("sbx", "uniform"),
        default=_as_str(nsgaiii_crossover.get("method")),
        help="Crossover method for NSGA-III (sbx for real, uniform for binary/integer).",
    )
    parser.add_argument(
        "--nsga3-crossover-prob",
        dest="nsgaiii_crossover_prob",
        default=_as_float(nsgaiii_crossover.get("prob")),
        help="Crossover probability for NSGA-III.",
    )
    parser.add_argument(
        "--nsga3-crossover-eta",
        dest="nsgaiii_crossover_eta",
        type=float,
        default=_as_float(nsgaiii_crossover.get("eta")),
        help="SBX eta for NSGA-III (real encoding).",
    )
    parser.add_argument(
        "--nsga3-mutation",
        dest="nsgaiii_mutation",
        choices=("polynomial", "bitflip", "reset"),
        default=_as_str(nsgaiii_mutation.get("method")),
        help="Mutation method for NSGA-III (pm for real, bitflip for binary, reset for integer).",
    )
    parser.add_argument(
        "--nsga3-mutation-prob",
        dest="nsgaiii_mutation_prob",
        default=_as_str(nsgaiii_mutation.get("prob")),
        help="Mutation probability for NSGA-III (allow expressions like 1/n; uses n_var).",
    )
    parser.add_argument(
        "--nsga3-mutation-eta",
        dest="nsgaiii_mutation_eta",
        type=float,
        default=_as_float(nsgaiii_mutation.get("eta")),
        help="Polynomial mutation eta for NSGA-III (real encoding).",
    )
    parser.add_argument(
        "--nsga3-mutation-step",
        dest="nsgaiii_mutation_step",
        type=int,
        default=_as_int(nsgaiii_mutation.get("step")),
        help="Integer creep step for NSGA-III (real encoding).",
    )


def _add_agemoea_arguments(parser: argparse.ArgumentParser, *, spec_defaults: SpecDefaults) -> None:
    agemoea_defaults = spec_defaults.agemoea_defaults
    agemoea_crossover = _as_block(agemoea_defaults.get("crossover"))
    agemoea_mutation = _as_block(agemoea_defaults.get("mutation"))

    parser.add_argument(
        "--agemoea-crossover",
        choices=("sbx", "blx_alpha"),
        default=_as_str(agemoea_crossover.get("method")),
        help="Crossover method for AGE-MOEA (default: sbx).",
    )
    parser.add_argument(
        "--agemoea-crossover-prob", default=_as_float(agemoea_crossover.get("prob")), help="Crossover probability for AGE-MOEA."
    )
    parser.add_argument(
        "--agemoea-crossover-eta", type=float, default=_as_float(agemoea_crossover.get("eta")), help="SBX eta for AGE-MOEA (real encoding)."
    )
    parser.add_argument(
        "--agemoea-mutation",
        choices=("polynomial", "non_uniform"),
        default=_as_str(agemoea_mutation.get("method")),
        help="Mutation method for AGE-MOEA (default: polynomial).",
    )
    parser.add_argument(
        "--agemoea-mutation-prob",
        type=str,
        default=_as_str(agemoea_mutation.get("prob")),
        help="Mutation probability for AGE-MOEA (allow expressions like 1/n).",
    )
    parser.add_argument(
        "--agemoea-mutation-eta", type=float, default=_as_float(agemoea_mutation.get("eta")), help="Polynomial mutation eta for AGE-MOEA."
    )


def _add_rvea_arguments(parser: argparse.ArgumentParser, *, spec_defaults: SpecDefaults) -> None:
    rvea_defaults = spec_defaults.rvea_defaults
    rvea_crossover = _as_block(rvea_defaults.get("crossover"))
    rvea_mutation = _as_block(rvea_defaults.get("mutation"))

    parser.add_argument(
        "--rvea-crossover",
        choices=("sbx", "blx_alpha"),
        default=_as_str(rvea_crossover.get("method")),
        help="Crossover method for RVEA (default: sbx).",
    )
    parser.add_argument("--rvea-crossover-prob", default=_as_float(rvea_crossover.get("prob")), help="Crossover probability for RVEA.")
    parser.add_argument(
        "--rvea-crossover-eta", type=float, default=_as_float(rvea_crossover.get("eta")), help="SBX eta for RVEA (real encoding)."
    )
    parser.add_argument(
        "--rvea-mutation",
        choices=("polynomial", "non_uniform"),
        default=_as_str(rvea_mutation.get("method")),
        help="Mutation method for RVEA (default: polynomial).",
    )
    parser.add_argument(
        "--rvea-mutation-prob",
        type=str,
        default=_as_str(rvea_mutation.get("prob")),
        help="Mutation probability for RVEA (allow expressions like 1/n).",
    )
    parser.add_argument(
        "--rvea-mutation-eta", type=float, default=_as_float(rvea_mutation.get("eta")), help="Polynomial mutation eta for RVEA."
    )


def add_algorithm_arguments(
    parser: argparse.ArgumentParser,
    *,
    spec_defaults: SpecDefaults,
) -> None:
    """Register algorithm-specific arguments on the parser."""
    _add_nsga_family_arguments(parser, spec_defaults=spec_defaults)
    _add_agemoea_arguments(parser, spec_defaults=spec_defaults)
    _add_rvea_arguments(parser, spec_defaults=spec_defaults)
