"""
Configuration space builders for tuning.

Implementations live in algorithm-focused submodules to satisfy the monolith
guard; this module is a stable import surface.
"""

from __future__ import annotations

from .bridge_spaces_agemoea import (
    build_agemoea_binary_config_space,
    build_agemoea_config_space,
    build_agemoea_constrained_config_space,
    build_agemoea_integer_config_space,
    build_agemoea_mixed_config_space,
    build_agemoea_permutation_config_space,
)
from .bridge_spaces_ccmo import build_ccmo_config_space
from .bridge_spaces_ibea import (
    build_ibea_binary_config_space,
    build_ibea_config_space,
    build_ibea_integer_config_space,
    build_ibea_mixed_config_space,
    build_ibea_permutation_config_space,
)
from .bridge_spaces_moead import (
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_constrained_config_space,
    build_moead_integer_config_space,
    build_moead_mixed_config_space,
    build_moead_permutation_config_space,
)
from .bridge_spaces_nsgaii import (
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_constrained_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
)
from .bridge_spaces_nsgaiii import (
    build_nsgaiii_binary_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_constrained_config_space,
    build_nsgaiii_integer_config_space,
    build_nsgaiii_mixed_config_space,
    build_nsgaiii_permutation_config_space,
)
from .bridge_spaces_rvea import (
    build_rvea_binary_config_space,
    build_rvea_config_space,
    build_rvea_constrained_config_space,
    build_rvea_integer_config_space,
    build_rvea_mixed_config_space,
    build_rvea_permutation_config_space,
)
from .bridge_spaces_smpso import (
    build_smpso_config_space,
    build_smpso_constrained_config_space,
    build_smpso_mixed_config_space,
)
from .bridge_spaces_smsemoa import (
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_constrained_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_mixed_config_space,
    build_smsemoa_permutation_config_space,
)
from .bridge_spaces_spea2 import (
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
]
