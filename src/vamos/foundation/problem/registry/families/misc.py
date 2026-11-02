from __future__ import annotations

from ...binary import BinaryFeatureSelectionProblem, BinaryKnapsackProblem, BinaryQUBOProblem
from ...integer import IntegerJobAssignmentProblem, IntegerResourceAllocationProblem
from ...mixed import MixedDesignProblem
from ...tsp import TSPProblem
from ..common import ProblemFactory, ProblemSpec


def _tsp_factory(n_var: int, _ignored: int | None = None) -> TSPProblem:
    return TSPProblem(n_cities=n_var)


def _tsplib_tsp_factory(dataset: str) -> ProblemFactory:
    def _factory(_n_var: int, _ignored: int | None = None) -> TSPProblem:
        return TSPProblem(dataset=dataset)

    return _factory


SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update(
        {
            "tsp6": ProblemSpec(
                key="tsp6",
                label="TSP (6 cities)",
                default_n_var=6,
                default_n_obj=2,
                allow_n_obj_override=False,
                encoding="permutation",
                description="Toy traveling salesman instance with 6 cities (permutation encoding).",
                factory=_tsp_factory,
            ),
            "kroa100": ProblemSpec(
                key="kroa100",
                label="TSPLIB KroA100",
                default_n_var=100,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="permutation",
                description="TSPLIB 100-city KroA instance (permutation encoding).",
                factory=_tsplib_tsp_factory("kroA100"),
            ),
            "krob100": ProblemSpec(
                key="krob100",
                label="TSPLIB KroB100",
                default_n_var=100,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="permutation",
                description="TSPLIB 100-city KroB instance (permutation encoding).",
                factory=_tsplib_tsp_factory("kroB100"),
            ),
            "kroc100": ProblemSpec(
                key="kroc100",
                label="TSPLIB KroC100",
                default_n_var=100,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="permutation",
                description="TSPLIB 100-city KroC instance (permutation encoding).",
                factory=_tsplib_tsp_factory("kroC100"),
            ),
            "krod100": ProblemSpec(
                key="krod100",
                label="TSPLIB KroD100",
                default_n_var=100,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="permutation",
                description="TSPLIB 100-city KroD instance (permutation encoding).",
                factory=_tsplib_tsp_factory("kroD100"),
            ),
            "kroe100": ProblemSpec(
                key="kroe100",
                label="TSPLIB KroE100",
                default_n_var=100,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="permutation",
                description="TSPLIB 100-city KroE instance (permutation encoding).",
                factory=_tsplib_tsp_factory("kroE100"),
            ),
            "bin_feat": ProblemSpec(
                key="bin_feat",
                label="Binary Feature Selection",
                default_n_var=50,
                default_n_obj=2,
                allow_n_obj_override=False,
                encoding="binary",
                description="Synthetic feature-selection style binary benchmark.",
                factory=lambda n_var, _n_obj: BinaryFeatureSelectionProblem(n_var=n_var),
            ),
            "bin_knapsack": ProblemSpec(
                key="bin_knapsack",
                label="Binary Knapsack",
                default_n_var=50,
                default_n_obj=2,
                allow_n_obj_override=False,
                encoding="binary",
                description="Knapsack-like binary benchmark trading value vs capacity deviation.",
                factory=lambda n_var, _n_obj: BinaryKnapsackProblem(n_var=n_var),
            ),
            "bin_qubo": ProblemSpec(
                key="bin_qubo",
                label="Binary QUBO",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                encoding="binary",
                description="Quadratic unconstrained binary optimization surrogate.",
                factory=lambda n_var, _n_obj: BinaryQUBOProblem(n_var=n_var),
            ),
            "int_alloc": ProblemSpec(
                key="int_alloc",
                label="Integer Resource Allocation",
                default_n_var=20,
                default_n_obj=2,
                allow_n_obj_override=False,
                encoding="integer",
                description="Resource allocation with integer budgets and diminishing returns.",
                factory=lambda n_var, _n_obj: IntegerResourceAllocationProblem(n_var=n_var),
            ),
            "int_jobs": ProblemSpec(
                key="int_jobs",
                label="Integer Job Assignment",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                encoding="integer",
                description="Assign job types to positions with mismatch and diversity objectives.",
                factory=lambda n_var, _n_obj: IntegerJobAssignmentProblem(n_positions=n_var),
            ),
            "mixed_design": ProblemSpec(
                key="mixed_design",
                label="Mixed Design",
                default_n_var=9,
                default_n_obj=2,
                allow_n_obj_override=False,
                encoding="mixed",
                description="Mixed real/integer/categorical benchmark.",
                factory=lambda n_var, _n_obj: MixedDesignProblem(n_var=n_var),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
