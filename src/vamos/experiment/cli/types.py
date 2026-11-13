from __future__ import annotations

from dataclasses import dataclass

from vamos.engine.config.spec import ExperimentSpec, ProblemOverrides, SpecBlock


@dataclass(frozen=True)
class SpecDefaults:
    spec: ExperimentSpec
    problem_overrides: ProblemOverrides
    experiment_defaults: SpecBlock
    nsgaii_defaults: SpecBlock
    moead_defaults: SpecBlock
    smsemoa_defaults: SpecBlock
    nsgaiii_defaults: SpecBlock
    spea2_defaults: SpecBlock
    ibea_defaults: SpecBlock
    smpso_defaults: SpecBlock
    agemoea_defaults: SpecBlock
    rvea_defaults: SpecBlock
    ccmo_defaults: SpecBlock
