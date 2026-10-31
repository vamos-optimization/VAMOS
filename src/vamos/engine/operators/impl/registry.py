"""
Registry for variation operators.
"""

from __future__ import annotations

from vamos.foundation.registry import Registry

# Global registry for variation operators (classes or factories).
# Key: operator name (e.g. "sbx", "polynomial").
# Value: Class type or factory function.
_operator_registry: Registry[object] | None = None


def _get_registry() -> Registry[object]:
    global _operator_registry
    if _operator_registry is not None:
        return _operator_registry

    reg: Registry[object] = Registry("VariationOperators")

    # Register common operators
    from vamos.engine.operators.impl.real import (
        ArithmeticCrossover,
        BLXAlphaBetaCrossover,
        BLXAlphaCrossover,
        CauchyMutation,
        ClampRepair,
        DEMatingCrossover,
        FuzzyCrossover,
        GaussianMutation,
        LaplaceCrossover,
        LevyFlightMutation,
        LinkedPolynomialMutation,
        MidpointBaseRepair,
        NonUniformMutation,
        PCXCrossover,
        PolynomialMutation,
        PowerLawMutation,
        ReflectRepair,
        ResampleRepair,
        RoundRepair,
        SBXCrossover,
        SPXCrossover,
        UNDXCrossover,
        UniformMutation,
        UniformResetMutation,
        WholeArithmeticCrossover,
        WrappingRepair,
    )
    from vamos.engine.operators.impl.repair import GradientRepair

    # Crossover
    reg.register("sbx", SBXCrossover)
    reg.register("blx_alpha", BLXAlphaCrossover)
    reg.register("blx_alpha_beta", BLXAlphaBetaCrossover)
    reg.register("arithmetic", ArithmeticCrossover)
    reg.register("whole_arithmetic", WholeArithmeticCrossover)
    reg.register("laplace", LaplaceCrossover)
    reg.register("fuzzy", FuzzyCrossover)
    reg.register("de", DEMatingCrossover)
    reg.register("pcx", PCXCrossover)
    reg.register("undx", UNDXCrossover)
    reg.register("simplex", SPXCrossover)

    # Mutation
    reg.register("polynomial", PolynomialMutation)
    reg.register("pm", PolynomialMutation)
    reg.register("non_uniform", NonUniformMutation)
    reg.register("gaussian", GaussianMutation)
    reg.register("uniform_reset", UniformResetMutation)
    reg.register("cauchy", CauchyMutation)
    reg.register("uniform", UniformMutation)
    reg.register("linked_polynomial", LinkedPolynomialMutation)
    reg.register("levy_flight", LevyFlightMutation)
    reg.register("power_law", PowerLawMutation)

    # Repair
    reg.register("clip", ClampRepair)
    reg.register("clamp", ClampRepair)
    reg.register("reflect", ReflectRepair)
    reg.register("random", ResampleRepair)
    reg.register("resample", ResampleRepair)
    reg.register("round", RoundRepair)
    reg.register("wrap", WrappingRepair)
    reg.register("wrapping", WrappingRepair)
    reg.register("midpoint", MidpointBaseRepair)
    reg.register("midpoint_base", MidpointBaseRepair)
    reg.register("gradient", GradientRepair)

    # Binary operators
    from vamos.engine.operators.impl.binary import (
        BitFlipMutation,
        HuxCrossover,
        OnePointCrossover,
        SegmentInversionMutation,
        TwoPointCrossover,
    )
    from vamos.engine.operators.impl.binary import (
        UniformCrossover as BinaryUniformCrossover,
    )

    reg.register("bitflip", BitFlipMutation)
    reg.register("one_point", OnePointCrossover)
    reg.register("two_point", TwoPointCrossover)
    reg.register("binary_uniform", BinaryUniformCrossover)
    reg.register("hux", HuxCrossover)
    reg.register("segment_inversion", SegmentInversionMutation)

    # Permutation operators
    from vamos.engine.operators.impl.permutation import (
        AlternatingEdgesCrossover,
        CycleCrossover,
        DisplacementMutation,
        EdgeRecombinationCrossover,
        InsertMutation,
        InversionMutation,
        OrderCrossover,
        PMXCrossover,
        PositionBasedCrossover,
        ScrambleMutation,
        SwapMutation,
        TwoOptMutation,
    )

    reg.register("swap", SwapMutation)
    reg.register("pmx", PMXCrossover)
    reg.register("cx", CycleCrossover)
    reg.register("cycle", CycleCrossover)
    reg.register("position_based", PositionBasedCrossover)
    reg.register("erx", EdgeRecombinationCrossover)
    reg.register("aex", AlternatingEdgesCrossover)
    reg.register("ox", OrderCrossover)
    reg.register("order", OrderCrossover)
    reg.register("insert", InsertMutation)
    reg.register("scramble", ScrambleMutation)
    reg.register("inversion", InversionMutation)
    reg.register("displacement", DisplacementMutation)
    reg.register("two_opt", TwoOptMutation)

    # Integer operators
    from vamos.engine.operators.impl.integer import (
        ArithmeticIntegerCrossover,
        BoundaryIntegerMutation,
        CreepMutation,
        GaussianIntegerMutation,
        IntegerPolynomialMutation,
        IntegerSBXCrossover,
        RandomResetMutation,
        UniformIntegerCrossover,
    )

    reg.register("int_uniform", UniformIntegerCrossover)
    reg.register("int_arithmetic", ArithmeticIntegerCrossover)
    reg.register("int_sbx", IntegerSBXCrossover)
    reg.register("reset", RandomResetMutation)
    reg.register("int_pm", IntegerPolynomialMutation)
    reg.register("pm_int", IntegerPolynomialMutation)
    reg.register("creep", CreepMutation)
    reg.register("boundary", BoundaryIntegerMutation)
    reg.register("int_gaussian", GaussianIntegerMutation)

    # Mixed operators
    from vamos.engine.operators.impl.mixed import MixedCrossover, MixedMutation

    reg.register("mixed", MixedCrossover)
    reg.register("mixed_mutation", MixedMutation)

    _operator_registry = reg
    return reg


def get_operator_registry() -> Registry[object]:
    """Return the global variation operator registry (lazily initialized)."""
    return _get_registry()


def __getattr__(name: str) -> object:
    if name == "operator_registry":
        return _get_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
