from __future__ import annotations

from ...real_world.engineering import WeldedBeamDesignProblem
from ...real_world.feature_selection import FeatureSelectionProblem
from ...real_world.hyperparam import HyperparameterTuningProblem
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update(
        {
            "ml_tuning": ProblemSpec(
                key="ml_tuning",
                label="ML Hyperparameter Tuning",
                default_n_var=4,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="mixed",
                description="SVM hyperparameter tuning on a small dataset (requires scikit-learn).",
                factory=lambda _n_var, _n_obj: HyperparameterTuningProblem(),
            ),
            "welded_beam": ProblemSpec(
                key="welded_beam",
                label="Welded Beam Design",
                default_n_var=6,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="mixed",
                description="Classic welded-beam design with mixed variables and constraints.",
                factory=lambda _n_var, _n_obj: WeldedBeamDesignProblem(),
            ),
            "fs_real": ProblemSpec(
                key="fs_real",
                label="Feature Selection (real data)",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="binary",
                description="Binary feature selection on a real dataset (requires scikit-learn).",
                factory=lambda _n_var, _n_obj: FeatureSelectionProblem(),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
