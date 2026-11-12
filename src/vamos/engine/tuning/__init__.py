"""Auto-tuning utilities for VAMOS.

Parameter types use signature: ParamType(name, ...)
- Real(name, low, high, log=False)
- Int(name, low, high, log=False)
- Categorical(name, choices)
- Boolean(name)

Also provides RandomSearchTuner, RacingTuner, AlgorithmConfigSpace,
and config space builders for each algorithm.
"""

from .ablation import AblationPlan, AblationTask, AblationVariant, build_ablation_plan
from .backends import ModelBasedTuner, available_model_based_backends
from .convergence import ConvergenceDiagnostic, ConvergenceProfile, TuningCallback, diagnose, score_with_profile
from .racing import (
    # Config space
    AlgorithmConfigSpace,
    BaseParam,
    Boolean,
    Categorical,
    Condition,
    ConditionalBlock,
    ConfigState,
    EliteEntry,
    EvalContext,
    Instance,
    Int,
    ModelBasedSampler,
    # Parameter types
    ParamSpace,
    ParamType,
    RacingTuner,
    # Tuners
    RandomSearchTuner,
    Real,
    # Samplers
    Sampler,
    # Scenario
    Scenario,
    TrialResult,
    # Tuning task
    TuningTask,
    TwoPhaseScenario,
    # Two-phase tuner
    TwoPhaseTuner,
    UniformSampler,
    build_agemoea_binary_config_space,
    build_agemoea_config_space,
    build_agemoea_integer_config_space,
    build_agemoea_mixed_config_space,
    build_agemoea_permutation_config_space,
    build_ibea_binary_config_space,
    build_ibea_config_space,
    build_ibea_integer_config_space,
    build_ibea_mixed_config_space,
    build_ibea_permutation_config_space,
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_integer_config_space,
    build_moead_mixed_config_space,
    build_moead_permutation_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_constrained_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_integer_config_space,
    build_nsgaiii_mixed_config_space,
    build_nsgaiii_permutation_config_space,
    build_rvea_binary_config_space,
    build_rvea_config_space,
    build_rvea_integer_config_space,
    build_rvea_mixed_config_space,
    build_rvea_permutation_config_space,
    build_smpso_config_space,
    build_smpso_mixed_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_mixed_config_space,
    build_smsemoa_permutation_config_space,
    build_spea2_binary_config_space,
    # Config space builders
    build_spea2_config_space,
    build_spea2_integer_config_space,
    build_spea2_mixed_config_space,
    build_spea2_permutation_config_space,
    config_from_assignment,
    # I/O
    filter_active_config,
    history_to_dict,
    load_checkpoint,
    save_checkpoint,
    save_history_csv,
    save_history_json,
)
from .racing.stats import a12_magnitude, vargha_delaney_a12

__all__ = [
    # Ablation planning
    "AblationPlan",
    "AblationTask",
    "AblationVariant",
    "build_ablation_plan",
    # External backends
    "ModelBasedTuner",
    "available_model_based_backends",
    # Parameter types
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Boolean",
    "Condition",
    "ConditionalBlock",
    "ParamType",
    "BaseParam",
    # Samplers
    "Sampler",
    "UniformSampler",
    "ModelBasedSampler",
    # Tuning task
    "TuningTask",
    "EvalContext",
    "Instance",
    # Scenario
    "Scenario",
    # Config space
    "AlgorithmConfigSpace",
    # Tuners
    "RandomSearchTuner",
    "RacingTuner",
    "TrialResult",
    "ConfigState",
    "EliteEntry",
    "TwoPhaseTuner",
    "TwoPhaseScenario",
    # I/O
    "filter_active_config",
    "history_to_dict",
    "save_history_json",
    "save_history_csv",
    "save_checkpoint",
    "load_checkpoint",
    # Config space builders
    "build_spea2_config_space",
    "build_spea2_permutation_config_space",
    "build_spea2_binary_config_space",
    "build_spea2_integer_config_space",
    "build_spea2_mixed_config_space",
    "build_ibea_config_space",
    "build_ibea_permutation_config_space",
    "build_ibea_mixed_config_space",
    "build_ibea_binary_config_space",
    "build_ibea_integer_config_space",
    "build_smpso_config_space",
    "build_smpso_mixed_config_space",
    "build_agemoea_config_space",
    "build_agemoea_permutation_config_space",
    "build_agemoea_binary_config_space",
    "build_agemoea_integer_config_space",
    "build_agemoea_mixed_config_space",
    "build_rvea_config_space",
    "build_rvea_permutation_config_space",
    "build_rvea_binary_config_space",
    "build_rvea_integer_config_space",
    "build_rvea_mixed_config_space",
    "build_moead_config_space",
    "build_moead_mixed_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
    "build_moead_permutation_config_space",
    "build_nsgaiii_config_space",
    "build_nsgaiii_permutation_config_space",
    "build_nsgaiii_mixed_config_space",
    "build_nsgaiii_binary_config_space",
    "build_nsgaiii_integer_config_space",
    "build_smsemoa_config_space",
    "build_smsemoa_permutation_config_space",
    "build_smsemoa_mixed_config_space",
    "build_smsemoa_binary_config_space",
    "build_smsemoa_integer_config_space",
    "build_nsgaii_config_space",
    "build_nsgaii_constrained_config_space",
    "build_nsgaii_permutation_config_space",
    "build_nsgaii_mixed_config_space",
    "build_nsgaii_binary_config_space",
    "build_nsgaii_integer_config_space",
    "config_from_assignment",
    # Convergence-aware evaluation
    "ConvergenceProfile",
    "ConvergenceDiagnostic",
    "TuningCallback",
    "diagnose",
    "score_with_profile",
    # Statistical utilities
    "vargha_delaney_a12",
    "a12_magnitude",
]
