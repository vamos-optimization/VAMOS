from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import TypedDict

DEFAULT_QUICKSTART_BUDGET = 5000


@dataclass(frozen=True)
class TemplateDefaults:
    problem: str
    algorithm: str
    engine: str
    budget: int
    pop_size: int
    seed: int
    plot: bool = True


@dataclass(frozen=True)
class QuickstartTemplate:
    key: str
    label: str
    description: str
    defaults: TemplateDefaults
    requires: tuple[str, ...] = ()


class _TemplateDefaultsSpec(TypedDict):
    problem: str
    algorithm: str
    engine: str
    budget: int
    pop_size: int
    seed: int
    plot: bool


class _TemplateSpec(TypedDict):
    label: str
    description: str
    defaults: _TemplateDefaultsSpec
    requires: tuple[str, ...]


_EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "examples": ("sklearn",),
}
_EXTRA_HINTS: dict[str, str] = {
    "analysis": 'pip install "vamos-optimization[analysis]"',
    "compute": 'pip install "vamos-optimization[compute]"',
    "examples": 'pip install "vamos-optimization[examples]"',
}

_TEMPLATE_DATA: dict[str, _TemplateSpec] = {
    "demo": {
        "label": "Quick demo (ZDT1)",
        "description": "Small, fast benchmark run with default settings.",
        "defaults": {
            "problem": "zdt1",
            "algorithm": "nsgaii",
            "engine": "numpy",
            "budget": DEFAULT_QUICKSTART_BUDGET,
            "pop_size": 100,
            "seed": 42,
            "plot": True,
        },
        "requires": (),
    },
    "physics_design": {
        "label": "Physics/engineering trade-offs (welded beam)",
        "description": "Mixed-variable structural design with constraints.",
        "defaults": {
            "problem": "welded_beam",
            "algorithm": "nsgaii",
            "engine": "numpy",
            "budget": 4000,
            "pop_size": 80,
            "seed": 42,
            "plot": True,
        },
        "requires": (),
    },
    "bio_feature_selection": {
        "label": "Biology feature selection (breast cancer dataset)",
        "description": "Binary feature selection on real data (requires scikit-learn).",
        "defaults": {
            "problem": "fs_real",
            "algorithm": "nsgaii",
            "engine": "numpy",
            "budget": 2000,
            "pop_size": 60,
            "seed": 7,
            "plot": True,
        },
        "requires": ("examples",),
    },
    "chem_hyperparam_tuning": {
        "label": "Chemistry model tuning (SVM hyperparameters)",
        "description": "Mixed-variable hyperparameter tuning (requires scikit-learn).",
        "defaults": {
            "problem": "ml_tuning",
            "algorithm": "nsgaii",
            "engine": "numpy",
            "budget": 2000,
            "pop_size": 50,
            "seed": 7,
            "plot": True,
        },
        "requires": ("examples",),
    },
}

_TEMPLATE_CACHE: dict[str, QuickstartTemplate] | None = None


def _build_template(key: str, data: _TemplateSpec) -> QuickstartTemplate:
    defaults_data = data["defaults"]
    defaults = TemplateDefaults(
        problem=defaults_data["problem"],
        algorithm=defaults_data["algorithm"],
        engine=defaults_data["engine"],
        budget=defaults_data["budget"],
        pop_size=defaults_data["pop_size"],
        seed=defaults_data["seed"],
        plot=defaults_data["plot"],
    )
    return QuickstartTemplate(
        key=key,
        label=data["label"],
        description=data["description"],
        defaults=defaults,
        requires=data["requires"],
    )


def _get_template_cache() -> dict[str, QuickstartTemplate]:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = {key: _build_template(key, data) for key, data in _TEMPLATE_DATA.items()}
    return _TEMPLATE_CACHE


def template_keys() -> set[str]:
    return set(_TEMPLATE_DATA)


def list_templates() -> list[QuickstartTemplate]:
    templates = _get_template_cache()
    return [value for _, value in sorted(templates.items(), key=lambda item: item[0])]


def get_template(key: str) -> QuickstartTemplate:
    templates = _get_template_cache()
    template = templates.get(key.lower())
    if template is None:
        available = ", ".join(sorted(templates))
        raise ValueError(f"Unknown template '{key}'. Available: {available}.")
    return template


def missing_extras(template: QuickstartTemplate) -> list[str]:
    missing = []
    for extra in template.requires:
        modules = _EXTRA_MODULES.get(extra, ())
        if not modules:
            continue
        if any(find_spec(module) is None for module in modules):
            missing.append(extra)
    return missing


def extra_hint(extra: str) -> str | None:
    return _EXTRA_HINTS.get(extra)


__all__ = [
    "QuickstartTemplate",
    "TemplateDefaults",
    "extra_hint",
    "get_template",
    "list_templates",
    "missing_extras",
    "template_keys",
]
