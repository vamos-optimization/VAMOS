"""
The built-in tuning spaces, as the flat views that the tuners sample.

Each space is a YAML grammar packaged with VAMOS (``vamos/resources/parameter_spaces``, schema
``vamos.parameter-space`` 1.0.0), which is its source of truth; ``build_<space>_config_space()``
returns its view (:func:`vamos.engine.tuning.grammar.view.to_algorithm_config_space`). The spaces
do not tune ``pop_size``, a run setting that the caller adds to every assignment.
"""

from __future__ import annotations

from ..grammar.builtin import builtin_parameter_space
from ..grammar.view import to_algorithm_config_space
from .config_space import AlgorithmConfigSpace


def _view(name: str) -> AlgorithmConfigSpace:
    """The view of the built-in grammar ``name``."""
    return to_algorithm_config_space(builtin_parameter_space(name))


def build_nsgaii_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaii")


def build_nsgaii_constrained_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaii_constrained")


def build_nsgaii_permutation_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaii_permutation")


def build_nsgaii_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaii_mixed")


def build_nsgaii_binary_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaii_binary")


def build_nsgaii_integer_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaii_integer")


def build_moead_config_space() -> AlgorithmConfigSpace:
    return _view("moead")


def build_moead_constrained_config_space() -> AlgorithmConfigSpace:
    return _view("moead_constrained")


def build_moead_permutation_config_space() -> AlgorithmConfigSpace:
    return _view("moead_permutation")


def build_moead_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("moead_mixed")


def build_moead_binary_config_space() -> AlgorithmConfigSpace:
    return _view("moead_binary")


def build_moead_integer_config_space() -> AlgorithmConfigSpace:
    return _view("moead_integer")


def build_nsgaiii_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaiii")


def build_nsgaiii_constrained_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaiii_constrained")


def build_nsgaiii_permutation_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaiii_permutation")


def build_nsgaiii_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaiii_mixed")


def build_nsgaiii_binary_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaiii_binary")


def build_nsgaiii_integer_config_space() -> AlgorithmConfigSpace:
    return _view("nsgaiii_integer")


def build_smsemoa_config_space() -> AlgorithmConfigSpace:
    return _view("smsemoa")


def build_smsemoa_constrained_config_space() -> AlgorithmConfigSpace:
    return _view("smsemoa_constrained")


def build_smsemoa_permutation_config_space() -> AlgorithmConfigSpace:
    return _view("smsemoa_permutation")


def build_smsemoa_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("smsemoa_mixed")


def build_smsemoa_binary_config_space() -> AlgorithmConfigSpace:
    return _view("smsemoa_binary")


def build_smsemoa_integer_config_space() -> AlgorithmConfigSpace:
    return _view("smsemoa_integer")


def build_spea2_config_space() -> AlgorithmConfigSpace:
    return _view("spea2")


def build_spea2_permutation_config_space() -> AlgorithmConfigSpace:
    return _view("spea2_permutation")


def build_spea2_binary_config_space() -> AlgorithmConfigSpace:
    return _view("spea2_binary")


def build_spea2_integer_config_space() -> AlgorithmConfigSpace:
    return _view("spea2_integer")


def build_spea2_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("spea2_mixed")


def build_ibea_config_space() -> AlgorithmConfigSpace:
    return _view("ibea")


def build_ibea_permutation_config_space() -> AlgorithmConfigSpace:
    return _view("ibea_permutation")


def build_ibea_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("ibea_mixed")


def build_ibea_binary_config_space() -> AlgorithmConfigSpace:
    return _view("ibea_binary")


def build_ibea_integer_config_space() -> AlgorithmConfigSpace:
    return _view("ibea_integer")


def build_smpso_config_space() -> AlgorithmConfigSpace:
    return _view("smpso")


def build_smpso_constrained_config_space() -> AlgorithmConfigSpace:
    return _view("smpso_constrained")


def build_smpso_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("smpso_mixed")


def build_agemoea_config_space() -> AlgorithmConfigSpace:
    return _view("agemoea")


def build_agemoea_constrained_config_space() -> AlgorithmConfigSpace:
    return _view("agemoea_constrained")


def build_agemoea_permutation_config_space() -> AlgorithmConfigSpace:
    return _view("agemoea_permutation")


def build_agemoea_binary_config_space() -> AlgorithmConfigSpace:
    return _view("agemoea_binary")


def build_agemoea_integer_config_space() -> AlgorithmConfigSpace:
    return _view("agemoea_integer")


def build_agemoea_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("agemoea_mixed")


def build_rvea_config_space() -> AlgorithmConfigSpace:
    return _view("rvea")


def build_rvea_constrained_config_space() -> AlgorithmConfigSpace:
    return _view("rvea_constrained")


def build_rvea_permutation_config_space() -> AlgorithmConfigSpace:
    return _view("rvea_permutation")


def build_rvea_binary_config_space() -> AlgorithmConfigSpace:
    return _view("rvea_binary")


def build_rvea_integer_config_space() -> AlgorithmConfigSpace:
    return _view("rvea_integer")


def build_rvea_mixed_config_space() -> AlgorithmConfigSpace:
    return _view("rvea_mixed")


def build_ccmo_config_space() -> AlgorithmConfigSpace:
    return _view("ccmo")


def build_ctaea_config_space() -> AlgorithmConfigSpace:
    return _view("ctaea")


def build_top_config_space() -> AlgorithmConfigSpace:
    return _view("top")


def build_cmoeams_config_space() -> AlgorithmConfigSpace:
    return _view("cmoeams")


__all__ = [
    # NSGA-II
    "build_nsgaii_config_space",
    "build_nsgaii_constrained_config_space",
    "build_nsgaii_permutation_config_space",
    "build_nsgaii_mixed_config_space",
    "build_nsgaii_binary_config_space",
    "build_nsgaii_integer_config_space",
    # MOEA/D
    "build_moead_config_space",
    "build_moead_constrained_config_space",
    "build_moead_permutation_config_space",
    "build_moead_mixed_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
    # NSGA-III
    "build_nsgaiii_config_space",
    "build_nsgaiii_constrained_config_space",
    "build_nsgaiii_permutation_config_space",
    "build_nsgaiii_mixed_config_space",
    "build_nsgaiii_binary_config_space",
    "build_nsgaiii_integer_config_space",
    # SMS-EMOA
    "build_smsemoa_config_space",
    "build_smsemoa_constrained_config_space",
    "build_smsemoa_permutation_config_space",
    "build_smsemoa_mixed_config_space",
    "build_smsemoa_binary_config_space",
    "build_smsemoa_integer_config_space",
    # SPEA2
    "build_spea2_config_space",
    "build_spea2_permutation_config_space",
    "build_spea2_binary_config_space",
    "build_spea2_integer_config_space",
    "build_spea2_mixed_config_space",
    # IBEA
    "build_ibea_config_space",
    "build_ibea_permutation_config_space",
    "build_ibea_mixed_config_space",
    "build_ibea_binary_config_space",
    "build_ibea_integer_config_space",
    # SMPSO
    "build_smpso_config_space",
    "build_smpso_constrained_config_space",
    "build_smpso_mixed_config_space",
    # AGE-MOEA
    "build_agemoea_config_space",
    "build_agemoea_constrained_config_space",
    "build_agemoea_permutation_config_space",
    "build_agemoea_binary_config_space",
    "build_agemoea_integer_config_space",
    "build_agemoea_mixed_config_space",
    # RVEA
    "build_rvea_config_space",
    "build_rvea_constrained_config_space",
    "build_rvea_permutation_config_space",
    "build_rvea_binary_config_space",
    "build_rvea_integer_config_space",
    "build_rvea_mixed_config_space",
    # CCMO
    "build_ccmo_config_space",
    # C-TAEA
    "build_ctaea_config_space",
    # ToP
    "build_top_config_space",
    # CMOEA-MS
    "build_cmoeams_config_space",
]
