"""
Registry names for the CEC 2009 benchmark problems (UF1-UF10 and CF1).

The ``CEC2009UF*Problem`` and ``CEC2009CF1Problem`` names are the built-in NumPy
implementations of :mod:`vamos.foundation.problem.cec2009`; no optional dependency
is needed. The public facade exposes the same classes as ``vamos.problems.CEC2009UF1``
and so on.
"""

from __future__ import annotations

from .cec2009 import CEC2009_CF1 as CEC2009CF1Problem
from .cec2009 import CEC2009_UF1 as CEC2009UF1Problem
from .cec2009 import CEC2009_UF2 as CEC2009UF2Problem
from .cec2009 import CEC2009_UF3 as CEC2009UF3Problem
from .cec2009 import CEC2009_UF4 as CEC2009UF4Problem
from .cec2009 import CEC2009_UF5 as CEC2009UF5Problem
from .cec2009 import CEC2009_UF6 as CEC2009UF6Problem
from .cec2009 import CEC2009_UF7 as CEC2009UF7Problem
from .cec2009 import CEC2009_UF8 as CEC2009UF8Problem
from .cec2009 import CEC2009_UF9 as CEC2009UF9Problem
from .cec2009 import CEC2009_UF10 as CEC2009UF10Problem


def has_cec2009() -> bool:
    """Return True: the CEC 2009 problems are built in."""
    return True


__all__ = [
    "CEC2009UF1Problem",
    "CEC2009UF2Problem",
    "CEC2009UF3Problem",
    "CEC2009UF4Problem",
    "CEC2009UF5Problem",
    "CEC2009UF6Problem",
    "CEC2009UF7Problem",
    "CEC2009UF8Problem",
    "CEC2009UF9Problem",
    "CEC2009UF10Problem",
    "CEC2009CF1Problem",
    "has_cec2009",
]
