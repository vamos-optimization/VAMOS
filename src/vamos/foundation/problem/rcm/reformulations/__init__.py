"""Reformulations of 23 CEC 2021 RCM problems whose variables decode into designs that meet the hard constraints.

Standard evolutionary algorithms find no feasible design of many RCM problems because of their equality
constraints (``|h| <= 1e-4``) or ordering constraints. Each reformulation here searches other variables, a
genotype, which :meth:`~._base.RCMReformulation.decode` maps to a complete decision vector of the original
problem (the instance's ``original``) that meets those constraints by construction; the objectives and the
remaining constraints are the original's at that vector (see :class:`~._base.RCMReformulation`).

- RCM22, RCM24 and RCM28: closed-form decoders (:mod:`.process`).
- RCM30-RCM35: increasing switching angles, in three variants (:mod:`.sopwm`): Wang et al.'s cumulative
  variables (the default), sorted angles, and sorted angles with the fundamental set by one more gene.
- RCM36-RCM39, RCM40-RCM46 and RCM47-RCM49: a few controls, with the other variables solved by a Newton
  power flow from a flat start (:mod:`.distributed_generation`, :mod:`.optimal_power_flow`,
  :mod:`.microgrid`).

The decoders other than the cumulative one are those of the scripts that computed the reference fronts of
the CRE and RCM suites with VAMOS, without their search methods.
"""

from __future__ import annotations

from vamos.foundation.problem.rcm.reformulations._base import RCMReformulation
from vamos.foundation.problem.rcm.reformulations.distributed_generation import DistributedGenerationReformulation
from vamos.foundation.problem.rcm.reformulations.microgrid import MicrogridReformulation
from vamos.foundation.problem.rcm.reformulations.optimal_power_flow import OptimalPowerFlowReformulation
from vamos.foundation.problem.rcm.reformulations.process import ProcessReformulation
from vamos.foundation.problem.rcm.reformulations.sopwm import VARIANTS, SOPWMReformulation

# The RCM problems that have a reformulation.
REFORMULATED = (22, 24, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49)


def make_reformulation(number: int) -> RCMReformulation:
    """The reformulation of RCM``number`` with its default variant (``"cumulative"`` for RCM30-RCM35)."""
    if number in (22, 24, 28):
        return ProcessReformulation(number)
    if 30 <= number <= 35:
        return SOPWMReformulation(number)
    if 36 <= number <= 39:
        return DistributedGenerationReformulation(number)
    if 40 <= number <= 46:
        return OptimalPowerFlowReformulation(number)
    if 47 <= number <= 49:
        return MicrogridReformulation(number)
    raise ValueError(f"RCM{number} has no reformulation; the reformulated problems are RCM22, RCM24, RCM28 and RCM30 to RCM49.")


__all__ = [
    "REFORMULATED",
    "VARIANTS",
    "DistributedGenerationReformulation",
    "MicrogridReformulation",
    "OptimalPowerFlowReformulation",
    "ProcessReformulation",
    "RCMReformulation",
    "SOPWMReformulation",
    "make_reformulation",
]
