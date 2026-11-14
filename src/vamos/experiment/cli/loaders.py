from __future__ import annotations

from typing import cast

from vamos.engine.config.loader import load_experiment_spec
from vamos.engine.config.spec import ExperimentSpec, ProblemOverrides, SpecBlock, normalize_problem_overrides

from .types import SpecDefaults


def _as_spec_block(value: object) -> SpecBlock:
    if isinstance(value, dict):
        return cast(SpecBlock, value)
    return {}


def load_spec_defaults(config_path: str | None) -> SpecDefaults:
    """Load YAML/JSON spec if provided and return defaults per algorithm and problems."""
    spec: ExperimentSpec = {}
    problem_overrides: ProblemOverrides = {}
    experiment_defaults: SpecBlock = {}
    nsgaii_defaults: SpecBlock = {}
    moead_defaults: SpecBlock = {}
    smsemoa_defaults: SpecBlock = {}
    nsgaiii_defaults: SpecBlock = {}
    spea2_defaults: SpecBlock = {}
    ibea_defaults: SpecBlock = {}
    smpso_defaults: SpecBlock = {}
    agemoea_defaults: SpecBlock = {}
    rvea_defaults: SpecBlock = {}
    ccmo_defaults: SpecBlock = {}
    if config_path:
        raw = load_experiment_spec(config_path)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError("Experiment spec must be a mapping (YAML/JSON object).")
        spec = cast(ExperimentSpec, raw)

        problems = spec.get("problems", {})
        if problems is None:
            problems = {}
        if not isinstance(problems, dict):
            raise ValueError("Experiment spec 'problems' must be a mapping of problem_key -> overrides.")
        # Problem keys are case-insensitive; the runs look their overrides up by registry key.
        problem_overrides = normalize_problem_overrides(problems)
        if "problems" in spec:
            spec["problems"] = problem_overrides

        defaults = spec.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("Experiment spec 'defaults' must be a mapping when provided.")
        experiment_defaults = _as_spec_block(defaults)

        nsgaii_raw = experiment_defaults.get("nsgaii", {})
        if nsgaii_raw is None:
            nsgaii_raw = {}
        if not isinstance(nsgaii_raw, dict):
            raise ValueError("Experiment spec 'defaults.nsgaii' must be a mapping when provided.")
        nsgaii_defaults = cast(SpecBlock, nsgaii_raw)
        moead_raw = experiment_defaults.get("moead", {})
        if moead_raw is None:
            moead_raw = {}
        if not isinstance(moead_raw, dict):
            raise ValueError("Experiment spec 'defaults.moead' must be a mapping when provided.")
        moead_defaults = cast(SpecBlock, moead_raw)
        smsemoa_raw = experiment_defaults.get("smsemoa", {})
        if smsemoa_raw is None:
            smsemoa_raw = {}
        if not isinstance(smsemoa_raw, dict):
            raise ValueError("Experiment spec 'defaults.smsemoa' must be a mapping when provided.")
        smsemoa_defaults = cast(SpecBlock, smsemoa_raw)
        nsgaiii_raw = experiment_defaults.get("nsgaiii", {})
        if nsgaiii_raw is None:
            nsgaiii_raw = {}
        if not isinstance(nsgaiii_raw, dict):
            raise ValueError("Experiment spec 'defaults.nsgaiii' must be a mapping when provided.")
        nsgaiii_defaults = cast(SpecBlock, nsgaiii_raw)

        def _algo_defaults(key: str) -> SpecBlock:
            raw = experiment_defaults.get(key, {})
            if raw is None:
                raw = {}
            if not isinstance(raw, dict):
                raise ValueError(f"Experiment spec 'defaults.{key}' must be a mapping when provided.")
            return cast(SpecBlock, raw)

        spea2_defaults = _algo_defaults("spea2")
        ibea_defaults = _algo_defaults("ibea")
        smpso_defaults = _algo_defaults("smpso")
        agemoea_defaults = _algo_defaults("agemoea")
        rvea_defaults = _algo_defaults("rvea")
        ccmo_defaults = _algo_defaults("ccmo")
    return SpecDefaults(
        spec=spec,
        problem_overrides=problem_overrides,
        experiment_defaults=experiment_defaults,
        nsgaii_defaults=nsgaii_defaults,
        moead_defaults=moead_defaults,
        smsemoa_defaults=smsemoa_defaults,
        nsgaiii_defaults=nsgaiii_defaults,
        spea2_defaults=spea2_defaults,
        ibea_defaults=ibea_defaults,
        smpso_defaults=smpso_defaults,
        agemoea_defaults=agemoea_defaults,
        rvea_defaults=rvea_defaults,
        ccmo_defaults=ccmo_defaults,
    )
