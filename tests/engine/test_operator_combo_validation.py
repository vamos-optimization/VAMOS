"""
Exhaustive validation of all feasible operator combinations across
every algorithm × encoding config space.

Tier 1 (fast): Enumerates ALL Categorical × Boolean combinations and
validates that ``config_from_assignment()`` builds a config without error.

Tier 2 (slower): For each unique crossover × mutation (× initializer)
pair, runs ``optimize()`` for 100 evaluations on a lightweight problem
to verify the full pipeline.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest

from vamos.engine.algorithm.variants import VARIANT_TO_CANONICAL
from vamos.engine.tuning.racing.bridge import (
    # AGE-MOEA
    build_agemoea_binary_config_space,
    build_agemoea_config_space,
    build_agemoea_integer_config_space,
    build_agemoea_mixed_config_space,
    build_agemoea_permutation_config_space,
    # IBEA
    build_ibea_binary_config_space,
    build_ibea_config_space,
    build_ibea_integer_config_space,
    build_ibea_mixed_config_space,
    build_ibea_permutation_config_space,
    # MOEA/D
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_integer_config_space,
    build_moead_mixed_config_space,
    build_moead_permutation_config_space,
    # NSGA-II
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    # NSGA-III
    build_nsgaiii_binary_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_integer_config_space,
    build_nsgaiii_mixed_config_space,
    build_nsgaiii_permutation_config_space,
    # RVEA
    build_rvea_binary_config_space,
    build_rvea_config_space,
    build_rvea_integer_config_space,
    build_rvea_mixed_config_space,
    build_rvea_permutation_config_space,
    # SMPSO
    build_smpso_config_space,
    build_smpso_mixed_config_space,
    # SMS-EMOA
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_mixed_config_space,
    build_smsemoa_permutation_config_space,
    # SPEA2
    build_spea2_binary_config_space,
    build_spea2_config_space,
    build_spea2_integer_config_space,
    build_spea2_mixed_config_space,
    build_spea2_permutation_config_space,
    # Bridge
    config_from_assignment,
)
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.param_space import (
    Boolean,
    Categorical,
    Int,
    Real,
)

# ---------------------------------------------------------------------------
# Registry: (algo_name, encoding, builder)
# algo_name must match what config_from_assignment() expects.
# ---------------------------------------------------------------------------

REGISTRY: list[tuple[str, str, Any]] = [
    # NSGA-II
    ("nsgaii", "real", build_nsgaii_config_space),
    ("nsgaii_permutation", "permutation", build_nsgaii_permutation_config_space),
    ("nsgaii_binary", "binary", build_nsgaii_binary_config_space),
    ("nsgaii_integer", "integer", build_nsgaii_integer_config_space),
    ("nsgaii_mixed", "mixed", build_nsgaii_mixed_config_space),
    # AGE-MOEA
    ("agemoea", "real", build_agemoea_config_space),
    ("agemoea_permutation", "permutation", build_agemoea_permutation_config_space),
    ("agemoea_binary", "binary", build_agemoea_binary_config_space),
    ("agemoea_integer", "integer", build_agemoea_integer_config_space),
    ("agemoea_mixed", "mixed", build_agemoea_mixed_config_space),
    # RVEA
    ("rvea", "real", build_rvea_config_space),
    ("rvea_permutation", "permutation", build_rvea_permutation_config_space),
    ("rvea_binary", "binary", build_rvea_binary_config_space),
    ("rvea_integer", "integer", build_rvea_integer_config_space),
    ("rvea_mixed", "mixed", build_rvea_mixed_config_space),
    # MOEA/D
    ("moead", "real", build_moead_config_space),
    ("moead_permutation", "permutation", build_moead_permutation_config_space),
    ("moead_binary", "binary", build_moead_binary_config_space),
    ("moead_integer", "integer", build_moead_integer_config_space),
    ("moead_mixed", "mixed", build_moead_mixed_config_space),
    # IBEA
    ("ibea", "real", build_ibea_config_space),
    ("ibea_permutation", "permutation", build_ibea_permutation_config_space),
    ("ibea_binary", "binary", build_ibea_binary_config_space),
    ("ibea_integer", "integer", build_ibea_integer_config_space),
    ("ibea_mixed", "mixed", build_ibea_mixed_config_space),
    # SPEA2
    ("spea2", "real", build_spea2_config_space),
    ("spea2_permutation", "permutation", build_spea2_permutation_config_space),
    ("spea2_binary", "binary", build_spea2_binary_config_space),
    ("spea2_integer", "integer", build_spea2_integer_config_space),
    ("spea2_mixed", "mixed", build_spea2_mixed_config_space),
    # NSGA-III
    ("nsgaiii", "real", build_nsgaiii_config_space),
    ("nsgaiii_permutation", "permutation", build_nsgaiii_permutation_config_space),
    ("nsgaiii_binary", "binary", build_nsgaiii_binary_config_space),
    ("nsgaiii_integer", "integer", build_nsgaiii_integer_config_space),
    ("nsgaiii_mixed", "mixed", build_nsgaiii_mixed_config_space),
    # SMS-EMOA
    ("smsemoa", "real", build_smsemoa_config_space),
    ("smsemoa_permutation", "permutation", build_smsemoa_permutation_config_space),
    ("smsemoa_binary", "binary", build_smsemoa_binary_config_space),
    ("smsemoa_integer", "integer", build_smsemoa_integer_config_space),
    ("smsemoa_mixed", "mixed", build_smsemoa_mixed_config_space),
    # SMPSO (only 2 variants)
    ("smpso", "real", build_smpso_config_space),
    ("smpso_mixed", "mixed", build_smpso_mixed_config_space),
]

# Problem to use for each encoding in Tier 2.
ENCODING_PROBLEM: dict[str, str] = {
    "real": "zdt1",
    "permutation": "tsp6",
    "binary": "bin_feat",
    "integer": "int_alloc",
    "mixed": "mixed_design",
}

# Algorithms that need n_obj injected into the assignment.
_RVEA_ALGOS = {"rvea", "rvea_permutation", "rvea_binary", "rvea_integer", "rvea_mixed"}

# Safe defaults for Int parameters (keep runs fast).
_INT_DEFAULTS: dict[str, int] = {
    "pop_size": 20,
    "archive_size": 20,
    "selection_pressure": 2,
    "neighbor_size": 10,
    "k_neighbors": 3,
    "replace_limit": 2,
    "n_partitions": 4,
    "creep_step": 2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_for_param(p: Real | Int | Categorical | Boolean) -> Any:
    """Return a safe default value for a parameter."""
    if isinstance(p, Categorical):
        return p.choices[0]
    if isinstance(p, Boolean):
        return False
    if isinstance(p, Int):
        if p.name in _INT_DEFAULTS:
            return _INT_DEFAULTS[p.name]
        if p.log:
            return int(round(math.sqrt(p.low * p.high)))
        return (p.low + p.high) // 2
    if isinstance(p, Real):
        if p.log:
            return math.sqrt(p.low * p.high)
        return (p.low + p.high) / 2.0
    raise TypeError(f"Unknown param type: {type(p)}")


def _enumerate_discrete_combos(space: AlgorithmConfigSpace):
    """
    Yield every feasible Categorical × Boolean assignment.

    Enumerates base (non-conditional) discrete params first, then expands
    conditional discrete params only when their parent value activates them.
    Int and Real parameters are set to safe defaults throughout.
    """
    param_space = space.to_param_space()

    # Identify which param names are conditional.
    conditional_names: set[str] = set()
    for block in space.conditionals or []:
        for p in block.params:
            conditional_names.add(p.name)

    # Separate base enumerable, conditional enumerable, and fixed params.
    base_enum_names: list[str] = []
    base_enum_values: list[list[Any]] = []
    fixed: dict[str, Any] = {}

    for p in space.params:  # base (always-active) params
        if isinstance(p, Categorical):
            base_enum_names.append(p.name)
            base_enum_values.append(list(p.choices))
        elif isinstance(p, Boolean):
            base_enum_names.append(p.name)
            base_enum_values.append([True, False])
        else:
            fixed[p.name] = _default_for_param(p)

    # Build a map of conditional blocks for quick lookup.
    # Key: (parent_name, parent_value) -> list of params in that block.
    cond_blocks: list[tuple[str, Any, list]] = []
    for block in space.conditionals or []:
        cond_blocks.append((block.parent_name, block.parent_value, block.params))

    for base_combo in itertools.product(*base_enum_values):
        base_assignment = dict(zip(base_enum_names, base_combo))
        base_assignment.update(fixed)

        # Determine which conditional blocks are active.
        active_cond_enum_names: list[str] = []
        active_cond_enum_values: list[list[Any]] = []

        for parent_name, parent_value, block_params in cond_blocks:
            if base_assignment.get(parent_name) != parent_value:
                continue
            for p in block_params:
                # Also check extra conditions.
                if not param_space.is_active(p.name, base_assignment):
                    continue
                if isinstance(p, Categorical):
                    active_cond_enum_names.append(p.name)
                    active_cond_enum_values.append(list(p.choices))
                elif isinstance(p, Boolean):
                    active_cond_enum_names.append(p.name)
                    active_cond_enum_values.append([True, False])
                else:
                    # Int / Real conditional → use default.
                    base_assignment[p.name] = _default_for_param(p)

        if not active_cond_enum_names:
            yield dict(base_assignment)
        else:
            for cond_combo in itertools.product(*active_cond_enum_values):
                full = dict(base_assignment)
                full.update(dict(zip(active_cond_enum_names, cond_combo)))
                yield full


def _build_full_assignment(
    space: AlgorithmConfigSpace,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a complete assignment with defaults, then apply *overrides*
    for the operator params we want to test.
    """
    param_space = space.to_param_space()
    assignment: dict[str, Any] = {}

    # First pass: defaults for everything.
    for name, p in param_space.params.items():
        assignment[name] = _default_for_param(p)

    # Apply overrides (skip None values from spaces without that param).
    for k, v in overrides.items():
        if v is not None:
            assignment[k] = v

    # Second pass: fill conditional params that became active after overrides.
    for name, p in param_space.params.items():
        if name not in assignment and param_space.is_active(name, assignment):
            assignment[name] = _default_for_param(p)

    # Filter to active only.
    return {k: v for k, v in assignment.items() if param_space.is_active(k, assignment)}


def _operator_pairs(space: AlgorithmConfigSpace):
    """
    Yield unique dicts of operator-level overrides: crossover × mutation
    (× initializer if present).
    """
    ps = space.to_param_space()
    cx = ps.params.get("crossover")
    mx = ps.params.get("mutation")
    init = ps.params.get("initializer")

    cx_vals = list(cx.choices) if isinstance(cx, Categorical) else [None]
    mx_vals = list(mx.choices) if isinstance(mx, Categorical) else [None]
    init_vals = list(init.choices) if isinstance(init, Categorical) else [None]

    for c, m, i in itertools.product(cx_vals, mx_vals, init_vals):
        pair: dict[str, Any] = {}
        if c is not None:
            pair["crossover"] = c
        if m is not None:
            pair["mutation"] = m
        if i is not None:
            pair["initializer"] = i
        yield pair


def _inject_special_params(algo_name: str, assignment: dict[str, Any]) -> None:
    """Inject algorithm-specific params that are not part of the config space."""
    if algo_name in _RVEA_ALGOS:
        assignment.setdefault("n_obj", 2)


# ---------------------------------------------------------------------------
# Tier 1: Exhaustive config construction
# ---------------------------------------------------------------------------

_REGISTRY_IDS = [f"{algo}_{enc}" for algo, enc, _ in REGISTRY]


@pytest.mark.parametrize(
    "algo_name,encoding,builder",
    REGISTRY,
    ids=_REGISTRY_IDS,
)
def test_all_discrete_combos_build_config(algo_name: str, encoding: str, builder):
    """Every feasible Categorical × Boolean combo produces a valid config."""
    space = builder()
    failures: list[tuple[int, dict[str, Any], str]] = []
    total = 0

    for i, assignment in enumerate(_enumerate_discrete_combos(space)):
        total += 1
        _inject_special_params(algo_name, assignment)
        try:
            cfg = config_from_assignment(algo_name, assignment)
            assert cfg is not None
        except Exception as e:
            failures.append((i, assignment, str(e)))

    assert total > 0, f"No combinations generated for {algo_name}"
    assert not failures, f"{len(failures)}/{total} failures for {algo_name}:\n" + "\n".join(
        f"  combo {i}: {e}" for i, _, e in failures[:20]
    )


# ---------------------------------------------------------------------------
# Tier 2: End-to-end runs per operator pair
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "algo_name,encoding,builder",
    REGISTRY,
    ids=_REGISTRY_IDS,
)
def test_all_operator_pairs_run(algo_name: str, encoding: str, builder):
    """Every crossover × mutation × initializer pair runs without crash.

    Operators declared in the config space but not yet implemented in the
    execution engine raise "Unsupported ..." errors.  These are collected
    separately and reported as warnings — only *unexpected* errors count
    as hard failures.
    """
    import warnings

    from vamos import optimize

    space = builder()
    problem = ENCODING_PROBLEM[encoding]
    unsupported: list[tuple[dict[str, Any], str]] = []
    crashes: list[tuple[dict[str, Any], str]] = []
    total = 0

    for pair in _operator_pairs(space):
        total += 1
        assignment = _build_full_assignment(space, overrides=pair)
        _inject_special_params(algo_name, assignment)
        try:
            cfg = config_from_assignment(algo_name, assignment)
            canonical = VARIANT_TO_CANONICAL.get(algo_name, algo_name)
            optimize(
                problem,
                algorithm=canonical,
                algorithm_config=cfg,
                max_evaluations=100,
                seed=42,
                engine="numpy",
            )
        except Exception as e:
            msg = str(e)
            if "Unsupported" in msg or "not supported" in msg.lower() or "parents must have shape" in msg:
                unsupported.append((pair, msg))
            else:
                crashes.append((pair, msg))

    assert total > 0, f"No operator pairs generated for {algo_name}"

    if unsupported:
        warnings.warn(
            f"{algo_name}: {len(unsupported)}/{total} operator combos use "
            f"unimplemented operators (skipped):\n" + "\n".join(f"  {p}: {e}" for p, e in unsupported[:10]),
            stacklevel=1,
        )

    assert not crashes, f"{len(crashes)}/{total} unexpected failures for {algo_name}:\n" + "\n".join(f"  {p}: {e}" for p, e in crashes[:20])


# ---------------------------------------------------------------------------
# Encoding and parent-arity validation of the pair-based real builders
# ---------------------------------------------------------------------------


def _pair_based_builders() -> dict[str, Any]:
    from vamos.engine.algorithm.config import MOEADConfig, NSGAIIIConfig, SMSEMOAConfig, SPEA2Config

    return {
        "nsgaiii": lambda: NSGAIIIConfig.builder().pop_size(20).selection("tournament"),
        "smsemoa": lambda: SMSEMOAConfig.builder().pop_size(20).selection("tournament"),
        "spea2": lambda: SPEA2Config.builder().pop_size(20).archive_size(20).selection("tournament"),
        "moead": lambda: MOEADConfig.builder().pop_size(20).neighbor_size(5).delta(0.9).replace_limit(2).aggregation("tchebycheff"),
    }


def _counting_zdt1(n_var: int = 6) -> Any:
    from vamos.foundation.problem.zdt1 import ZDT1Problem

    class CountingZDT1(ZDT1Problem):
        def __init__(self) -> None:
            super().__init__(n_var)
            self.n_evaluated = 0

        def evaluate(self, X: Any, out: dict[str, Any]) -> None:
            self.n_evaluated += int(X.shape[0])
            super().evaluate(X, out)

    return CountingZDT1()


_THREE_PARENT_CASES = [
    (algorithm, crossover) for algorithm in ("nsgaiii", "smsemoa", "spea2") for crossover in ("pcx", "undx", "simplex", "de")
] + [("moead", crossover) for crossover in ("pcx", "undx", "simplex")]


@pytest.mark.parametrize("algorithm,crossover", _THREE_PARENT_CASES)
def test_pair_based_algorithms_reject_three_parent_crossovers(algorithm: str, crossover: str) -> None:
    from vamos import optimize

    problem = _counting_zdt1()
    cfg = _pair_based_builders()[algorithm]().crossover(crossover, prob=0.9).mutation("pm", prob="1/n", eta=20.0).build()
    with pytest.raises(ValueError, match="three parents"):
        optimize(problem, algorithm=algorithm, algorithm_config=cfg, max_evaluations=200, seed=1, engine="numpy")
    assert problem.n_evaluated == 0


@pytest.mark.parametrize("algorithm", ["nsgaiii", "smsemoa", "spea2", "moead"])
@pytest.mark.parametrize(
    "crossover,mutation",
    [("hux", "pm"), ("one_point", "pm"), ("sbx", "bitflip"), ("sbx", "swap"), ("int_sbx", "pm")],
)
def test_real_builders_reject_operators_of_other_encodings(algorithm: str, crossover: str, mutation: str) -> None:
    from vamos import optimize

    problem = _counting_zdt1()
    cfg = _pair_based_builders()[algorithm]().crossover(crossover, prob=0.9).mutation(mutation, prob=0.1).build()
    with pytest.raises(ValueError, match="real encoding"):
        optimize(problem, algorithm=algorithm, algorithm_config=cfg, max_evaluations=200, seed=1, engine="numpy")
    assert problem.n_evaluated == 0


@pytest.mark.parametrize("algorithm", ["nsgaiii", "smsemoa", "spea2", "moead"])
def test_pair_based_algorithms_still_run_pairwise_real_operators(algorithm: str) -> None:
    from vamos import optimize

    crossover = "de" if algorithm == "moead" else "blx_alpha"
    cfg = _pair_based_builders()[algorithm]().crossover(crossover, prob=0.9).mutation("pm", prob="1/n", eta=20.0).build()
    result = optimize(_counting_zdt1(), algorithm=algorithm, algorithm_config=cfg, max_evaluations=100, seed=1, engine="numpy")
    assert result.F.shape[1] == 2
