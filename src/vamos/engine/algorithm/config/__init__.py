"""Algorithm configuration module.

This package provides configuration dataclasses and fluent builders for all
multi-objective optimization algorithms.

Examples:
    from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig

    # Fluent builder
    cfg = NSGAIIConfig.builder().pop_size(100).crossover("sbx", prob=1.0).build()

    # Quick defaults
    cfg = NSGAIIConfig.default(pop_size=100, n_var=30)
"""

from .agemoea import AGEMOEAConfig
from .ccmo import CCMOConfig
from .ctaea import CTAEAConfig
from .generic import GenericAlgorithmConfig
from .ibea import IBEAConfig
from .moead import MOEADConfig
from .nsgaii import NSGAIIConfig
from .nsgaiii import NSGAIIIConfig
from .rvea import RVEAConfig
from .smpso import SMPSOConfig
from .smsemoa import SMSEMOAConfig
from .spea2 import SPEA2Config
from .top import ToPConfig
from .types import (
    AlgorithmConfigMapping,
    AlgorithmConfigProtocol,
    ProbabilityExpression,
    ProbabilityValue,
)

__all__ = [
    # NSGA-II
    "NSGAIIConfig",
    # MOEA/D
    "MOEADConfig",
    # SPEA2
    "SPEA2Config",
    # IBEA
    "IBEAConfig",
    # SMS-EMOA
    "SMSEMOAConfig",
    # SMPSO
    "SMPSOConfig",
    # NSGA-III
    "NSGAIIIConfig",
    # AGE-MOEA
    "AGEMOEAConfig",
    # RVEA
    "RVEAConfig",
    # CCMO
    "CCMOConfig",
    # C-TAEA
    "CTAEAConfig",
    # ToP
    "ToPConfig",
    # Plugin/custom
    "GenericAlgorithmConfig",
    # Typing helpers
    "AlgorithmConfigMapping",
    "AlgorithmConfigProtocol",
    "ProbabilityExpression",
    "ProbabilityValue",
]
