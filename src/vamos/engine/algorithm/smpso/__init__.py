"""
SMPSO algorithm module.

This package provides the SMPSO (Speed-constrained Multiobjective Particle Swarm Optimization)
implementation with modular components:
- `core.py`: main SMPSO class (run/ask/tell loop)
- `setup.py`: initialization/config helpers
- `state.py`: SMPSOState + result building
- `operators/policies/smpso.py`: mutation operator building
- `helpers.py`: personal best updates, evaluation helpers

References:
    Nebro, A.J., Durillo, J.J., Garcia-Nieto, J., Coello Coello, C.A.,
    Luna, F. and Alba, E. (2009). SMPSO: A new PSO-based metaheuristic
    for multi-objective optimization. IEEE MCDM'09, pp. 66-73.
"""

from vamos.engine.operators.policies.smpso import build_mutation_operator

from .helpers import update_personal_bests
from .initialization import initialize_smpso_run
from .smpso import SMPSO
from .state import SMPSOState, build_smpso_result

__all__ = [
    "SMPSO",
    # Helpers
    "update_personal_bests",
    # Operators
    "build_mutation_operator",
    # Setup
    "initialize_smpso_run",
    # State
    "SMPSOState",
    "build_smpso_result",
]
