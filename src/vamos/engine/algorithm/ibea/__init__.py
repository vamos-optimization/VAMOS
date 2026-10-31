"""
IBEA algorithm module.

This package provides the IBEA (Indicator-Based Evolutionary Algorithm)
implementation with modular components:
- `core.py`: main IBEA class (run/ask/tell loop)
- `setup.py`: initialization/config helpers
- `state.py`: IBEAState + result building
    - `operators/policies/ibea.py`: variation pipeline building
- `helpers.py`: indicator computation, fitness, environmental selection

References:
    E. Zitzler and S. Künzli, "Indicator-Based Selection in Multiobjective
    Search," in Proc. PPSN VIII, 2004, pp. 832-842.
"""

from vamos.engine.operators.policies.ibea import build_variation_pipeline

from .helpers import (
    apply_constraint_penalty,
    compute_indicator_matrix,
    environmental_selection,
    epsilon_indicator,
    hypervolume_indicator,
    ibea_fitness,
)
from .ibea import IBEA
from .initialization import initialize_ibea_run
from .state import IBEAState, build_ibea_result

__all__ = [
    "IBEA",
    # Helpers
    "apply_constraint_penalty",
    "compute_indicator_matrix",
    "environmental_selection",
    "epsilon_indicator",
    "hypervolume_indicator",
    "ibea_fitness",
    # Operators
    "build_variation_pipeline",
    # Setup
    "initialize_ibea_run",
    # State
    "IBEAState",
    "build_ibea_result",
]
