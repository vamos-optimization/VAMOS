from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

from vamos.engine.operators.impl.binary import (
    bit_flip_mutation,
    hux_crossover,
    one_point_crossover,
    segment_inversion_mutation,
    two_point_crossover,
    uniform_crossover,
)
from vamos.engine.operators.impl.integer import (
    arithmetic_integer_crossover,
    boundary_integer_mutation,
    creep_mutation,
    gaussian_integer_mutation,
    integer_polynomial_mutation,
    integer_sbx_crossover,
    random_reset_mutation,
    uniform_integer_crossover,
)
from vamos.engine.operators.impl.permutation import (
    alternating_edges_crossover,
    cycle_crossover,
    displacement_mutation,
    edge_recombination_crossover,
    insert_mutation,
    inversion_mutation,
    order_crossover,
    pmx_crossover,
    position_based_crossover,
    scramble_mutation,
    swap_mutation,
    two_opt_mutation,
)

BinaryCrossoverOp: TypeAlias = Callable[[np.ndarray[Any, Any], float, np.random.Generator], np.ndarray[Any, Any]]
BinaryMutationOp: TypeAlias = Callable[[np.ndarray[Any, Any], float, np.random.Generator], None]
IntCrossoverOp: TypeAlias = Callable[..., np.ndarray[Any, Any]]
IntMutationOp: TypeAlias = Callable[..., None]
PermCrossoverOp: TypeAlias = Callable[[np.ndarray[Any, Any], float, np.random.Generator], np.ndarray[Any, Any]]
PermMutationOp: TypeAlias = Callable[[np.ndarray[Any, Any], float, np.random.Generator], None]


BINARY_CROSSOVER_COMMON: dict[str, BinaryCrossoverOp] = {
    "one_point": one_point_crossover,
    "single_point": one_point_crossover,
    "1point": one_point_crossover,
    "two_point": two_point_crossover,
    "2point": two_point_crossover,
    "uniform": uniform_crossover,
    "hux": hux_crossover,
}

BINARY_MUTATION_COMMON: dict[str, BinaryMutationOp] = {
    "bitflip": bit_flip_mutation,
    "bit_flip": bit_flip_mutation,
    "segment_inversion": segment_inversion_mutation,
}

INT_CROSSOVER_COMMON: dict[str, IntCrossoverOp] = {
    "uniform": uniform_integer_crossover,
    "blend": arithmetic_integer_crossover,
    "arithmetic": arithmetic_integer_crossover,
    "sbx": integer_sbx_crossover,
}

INT_MUTATION_COMMON: dict[str, IntMutationOp] = {
    "reset": random_reset_mutation,
    "random_reset": random_reset_mutation,
    "creep": creep_mutation,
    "polynomial": integer_polynomial_mutation,
    # The operator registry names polynomial mutation "pm", the name that resolved specs carry.
    "pm": integer_polynomial_mutation,
    "gaussian": gaussian_integer_mutation,
    "boundary": boundary_integer_mutation,
}

PERM_CROSSOVER_COMMON: dict[str, PermCrossoverOp] = {
    "ox": order_crossover,
    "order": order_crossover,
    "pmx": pmx_crossover,
    "cycle": cycle_crossover,
    "cx": cycle_crossover,
    "position": position_based_crossover,
    "position_based": position_based_crossover,
    "pos": position_based_crossover,
    "edge": edge_recombination_crossover,
    "edge_recombination": edge_recombination_crossover,
    "erx": edge_recombination_crossover,
    "aex": alternating_edges_crossover,
    "alternating_edges": alternating_edges_crossover,
}

PERM_MUTATION_COMMON: dict[str, PermMutationOp] = {
    "swap": swap_mutation,
    "insert": insert_mutation,
    "scramble": scramble_mutation,
    "inversion": inversion_mutation,
    "displacement": displacement_mutation,
    "two_opt": two_opt_mutation,
}


__all__ = [
    "BINARY_CROSSOVER_COMMON",
    "BINARY_MUTATION_COMMON",
    "INT_CROSSOVER_COMMON",
    "INT_MUTATION_COMMON",
    "PERM_CROSSOVER_COMMON",
    "PERM_MUTATION_COMMON",
    "BinaryCrossoverOp",
    "BinaryMutationOp",
    "IntCrossoverOp",
    "IntMutationOp",
    "PermCrossoverOp",
    "PermMutationOp",
]
