"""
Problem facade: curated benchmark and real-world problem classes.

Use `make_problem_selection()` when you need registry-based instantiation by name.
"""

from __future__ import annotations

from vamos.foundation.problem.cec2009 import CEC2009_CF1, CEC2009_UF1, CEC2009_UF2, CEC2009_UF3
from vamos.foundation.problem.dascmop import DASCMOPProblem
from vamos.foundation.problem.dtlz import DTLZ1Problem, DTLZ2Problem, DTLZ3Problem, DTLZ4Problem, DTLZ7Problem
from vamos.foundation.problem.lircmop import LIRCMOPProblem
from vamos.foundation.problem.real_world.engineering import WeldedBeamDesignProblem
from vamos.foundation.problem.real_world.feature_selection import FeatureSelectionProblem
from vamos.foundation.problem.real_world.hyperparam import HyperparameterTuningProblem
from vamos.foundation.problem.tsp import TSPProblem
from vamos.foundation.problem.wfg import (
    WFG1Problem,
    WFG2Problem,
    WFG3Problem,
    WFG4Problem,
    WFG5Problem,
    WFG6Problem,
    WFG7Problem,
    WFG8Problem,
    WFG9Problem,
)
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.foundation.problem.zdt2 import ZDT2Problem
from vamos.foundation.problem.zdt3 import ZDT3Problem
from vamos.foundation.problem.zdt4 import ZDT4Problem
from vamos.foundation.problem.zdt6 import ZDT6Problem

CEC2009UF1 = CEC2009_UF1
CEC2009UF2 = CEC2009_UF2
CEC2009UF3 = CEC2009_UF3
CEC2009CF1 = CEC2009_CF1

DASCMOP = DASCMOPProblem
LIRCMOP = LIRCMOPProblem

DTLZ1 = DTLZ1Problem
DTLZ2 = DTLZ2Problem
DTLZ3 = DTLZ3Problem
DTLZ4 = DTLZ4Problem
DTLZ7 = DTLZ7Problem

TSP = TSPProblem

WFG1 = WFG1Problem
WFG2 = WFG2Problem
WFG3 = WFG3Problem
WFG4 = WFG4Problem
WFG5 = WFG5Problem
WFG6 = WFG6Problem
WFG7 = WFG7Problem
WFG8 = WFG8Problem
WFG9 = WFG9Problem

ZDT1 = ZDT1Problem
ZDT2 = ZDT2Problem
ZDT3 = ZDT3Problem
ZDT4 = ZDT4Problem
ZDT6 = ZDT6Problem

__all__ = [
    "CEC2009UF1",
    "CEC2009UF2",
    "CEC2009UF3",
    "CEC2009CF1",
    "DASCMOP",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ7",
    "LIRCMOP",
    "TSP",
    "WFG1",
    "WFG2",
    "WFG3",
    "WFG4",
    "WFG5",
    "WFG6",
    "WFG7",
    "WFG8",
    "WFG9",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT6",
    "FeatureSelectionProblem",
    "HyperparameterTuningProblem",
    "WeldedBeamDesignProblem",
]
