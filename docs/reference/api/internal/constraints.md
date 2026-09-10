# Constraint implementation

Low-level constraint strategies and violation helpers. For user-defined constrained problems, use the public problem-definition workflow instead.

!!! warning "Internal implementation API"
    These deep modules are not stable public imports. Documentation does not create a compatibility guarantee. See the [stability policy](../../../project/stability-and-versioning.md).

[Public problem definition](../problem-definition.md) · [Constraints guide](../../constraints.md)

::: vamos.foundation.constraints
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false
      members:
        - ConstraintInfo
        - compute_constraint_info
        - FeasibilityFirstStrategy
        - PenaltyCVStrategy
        - CVAsObjectiveStrategy
        - EpsilonConstraintStrategy
        - get_constraint_strategy

::: vamos.foundation.constraints.utils
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false
