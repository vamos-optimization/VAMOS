"""
Bridge from sampled hyperparameters to concrete algorithm configs.
"""

from __future__ import annotations

from .bridge_assignments import config_from_assignment
from .bridge_spaces import (
    # AGE-MOEA
    build_agemoea_binary_config_space,
    build_agemoea_config_space,
    build_agemoea_integer_config_space,
    build_agemoea_mixed_config_space,
    build_agemoea_permutation_config_space,
    # IBEA
    build_ibea_binary_config_space,
    build_ibea_config_space,
    build_ibea_integer_config_space,
    build_ibea_mixed_config_space,
    build_ibea_permutation_config_space,
    # MOEA/D
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_integer_config_space,
    build_moead_mixed_config_space,
    build_moead_permutation_config_space,
    # NSGA-II
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_constrained_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    # NSGA-III
    build_nsgaiii_binary_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_integer_config_space,
    build_nsgaiii_mixed_config_space,
    build_nsgaiii_permutation_config_space,
    # RVEA
    build_rvea_binary_config_space,
    build_rvea_config_space,
    build_rvea_integer_config_space,
    build_rvea_mixed_config_space,
    build_rvea_permutation_config_space,
    # SMPSO
    build_smpso_config_space,
    build_smpso_mixed_config_space,
    # SMS-EMOA
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_mixed_config_space,
    build_smsemoa_permutation_config_space,
    # SPEA2
    build_spea2_binary_config_space,
    build_spea2_config_space,
    build_spea2_integer_config_space,
    build_spea2_mixed_config_space,
    build_spea2_permutation_config_space,
)

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
    "build_moead_permutation_config_space",
    "build_moead_mixed_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
    # NSGA-III
    "build_nsgaiii_config_space",
    "build_nsgaiii_permutation_config_space",
    "build_nsgaiii_mixed_config_space",
    "build_nsgaiii_binary_config_space",
    "build_nsgaiii_integer_config_space",
    # SMS-EMOA
    "build_smsemoa_config_space",
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
    "build_smpso_mixed_config_space",
    # AGE-MOEA
    "build_agemoea_config_space",
    "build_agemoea_permutation_config_space",
    "build_agemoea_binary_config_space",
    "build_agemoea_integer_config_space",
    "build_agemoea_mixed_config_space",
    # RVEA
    "build_rvea_config_space",
    "build_rvea_permutation_config_space",
    "build_rvea_binary_config_space",
    "build_rvea_integer_config_space",
    "build_rvea_mixed_config_space",
    # Bridge
    "config_from_assignment",
]
