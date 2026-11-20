"""The reformulated RCM problems: decoded designs are feasible on the original problem and keep its values.

``data/rcm_reformulation_reference_values.json`` holds the designs that the authors' decoder scripts (Evolver
``scripts/realworld_repair``) produce for fixed genotypes; ``tools/rcm_reformulation_reference_values.py``
regenerates it. The cumulative variant of RCM30-RCM35 is checked against Wang et al.'s Eqs. 18-21, written out
below one angle at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos import available_problem_names, optimize
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.rcm import EQUALITY_TOLERANCE, RCMProblem
from vamos.foundation.problem.rcm.power_electronics import ORDER_THRESHOLD
from vamos.foundation.problem.rcm.reformulations import REFORMULATED, VARIANTS, RCMReformulation, SOPWMReformulation, make_reformulation
from vamos.foundation.problem.rcm.reformulations._base import NOT_CONVERGED
from vamos.foundation.problem.rcm.reformulations.sopwm import GAP
from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection
from vamos.foundation.problem.resolver import resolve_reference_front_path

_FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "data" / "rcm_reformulation_reference_values.json").read_text(encoding="utf-8")
)
Array = np.ndarray[Any, Any]

# (genotype size, number of inequalities of the original) of the default reformulation of each problem.
_DIMENSIONS = {
    22: (5, 2),
    24: (1, 0),
    28: (2, 4),
    **dict.fromkeys((30, 31, 32), (26, 24)),
    **dict.fromkeys((33, 34, 35), (31, 29)),
    **dict.fromkeys(range(36, 40), (4, 0)),
    **dict.fromkeys(range(40, 47), (8, 0)),
    **dict.fromkeys(range(47, 50), (7, 0)),
}
# Every problem with each of its variants (None: the problem has no variants).
_CASES = [(k, variant) for k in REFORMULATED for variant in (VARIANTS if 30 <= k <= 35 else (None,))]
# The least share of uniform genotypes that decodes successfully, by problem; measured: 1.0 for the closed-form
# decoders, 0.981 for RCM36-RCM39, 0.0026 for RCM40-RCM46 and 0.400 for RCM47-RCM49 (20 000 genotypes).
_LEAST_SHARE = {**dict.fromkeys(range(36, 40), 0.95), **dict.fromkeys(range(40, 47), 0.0005), **dict.fromkeys(range(47, 50), 0.3)}
_SAMPLES = 2000


def _problem(k: int, variant: str | None) -> RCMReformulation:
    return make_reformulation(k) if variant is None else SOPWMReformulation(k, variant=variant)


def _genotypes(problem: RCMReformulation, n: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    return np.asarray(problem.xl + rng.random((n, problem.n_var)) * (problem.xu - problem.xl))


def _original(problem: RCMReformulation, X: Array) -> tuple[Array, Array]:
    F, G = evaluate_population_with_constraints(problem.original, X)
    assert G is not None
    return F, G


def _id(case: tuple[int, str | None]) -> str:
    return f"rcm{case[0]}" + ("" if case[1] is None else f"-{case[1]}")


def test_every_reformulation_is_registered_by_name() -> None:
    names = {name for name in available_problem_names() if name.endswith("_reformulated")}
    assert names == {f"rcm{k:02d}_reformulated" for k in REFORMULATED} and len(names) == 23


@pytest.mark.parametrize("k", REFORMULATED)
def test_registered_dimensions_and_constraints(k: int) -> None:
    n_var, n_inequalities = _DIMENSIONS[k]
    selection = make_problem_selection(f"rcm{k:02d}_reformulated")
    problem = selection.instantiate()
    assert isinstance(problem, RCMReformulation) and isinstance(problem.original, RCMProblem)
    assert problem.original.key == f"rcm{k:02d}" and problem.key == f"rcm{k:02d}_reformulated"
    assert (selection.n_var, selection.n_obj) == (problem.n_var, problem.n_obj) == (n_var, problem.original.n_obj)
    assert (problem.n_inequalities, problem.n_constraints) == (n_inequalities, n_inequalities + 1)
    assert selection.spec.encoding == "real" and np.all(problem.xl < problem.xu)
    with pytest.raises(ValueError, match="fixed number of objectives"):
        make_problem_selection(f"rcm{k:02d}_reformulated", n_obj=problem.n_obj + 1)


@pytest.mark.parametrize("k", [k for k in REFORMULATED if not 30 <= k <= 35])
def test_a_reformulation_without_variants_has_a_fixed_number_of_variables(k: int) -> None:
    spec = get_problem_specs()[f"rcm{k:02d}_reformulated"]
    assert spec.option_defaults() == {}
    with pytest.raises(ValueError, match="fixed number of variables"):
        make_problem_selection(spec.key, n_var=spec.default_n_var + 1)


@pytest.mark.parametrize("k", range(30, 36))
def test_sopwm_variants_are_options_of_the_registered_problem(k: int) -> None:
    d = len(RCMProblem(k).xl)
    key = f"rcm{k}_reformulated"
    assert get_problem_specs()[key].option_defaults() == {"variant": "cumulative"}
    for variant, n_var in (("cumulative", d + 1), ("sorted", d), ("level", d + 1)):
        problem = make_problem_selection(key).with_options({"variant": variant}).instantiate()
        assert isinstance(problem, SOPWMReformulation) and (problem.variant, problem.n_var) == (variant, n_var)
        # The variant's own number of variables is accepted too.
        assert make_problem_selection(key, n_var=n_var).with_options({"variant": variant}).instantiate().n_var == n_var
    with pytest.raises(ValueError, match="has 25 variables|has 30 variables"):
        make_problem_selection(key, n_var=d + 5).with_options({"variant": "sorted"}).instantiate()
    with pytest.raises(ValueError, match="Unknown variant"):
        SOPWMReformulation(k, variant="sort")


def test_reformulations_reject_other_problems() -> None:
    with pytest.raises(ValueError, match="no reformulation"):
        make_reformulation(23)
    with pytest.raises(ValueError, match="RCM30 to RCM35"):
        SOPWMReformulation(36)


@pytest.mark.parametrize("case", _CASES, ids=_id)
def test_decoded_designs_are_feasible_on_the_original_and_keep_its_values(case: tuple[int, str | None]) -> None:
    k, variant = case
    problem = _problem(k, variant)
    U = _genotypes(problem, _SAMPLES, seed=k)
    X = problem.decode(U)
    F, G = evaluate_population_with_constraints(problem, U)
    assert G is not None
    assert X.shape == (_SAMPLES, problem.original.n_var) and F.shape == (_SAMPLES, problem.n_obj)
    assert G.shape == (_SAMPLES, problem.n_constraints) and np.isfinite(F).all() and np.isfinite(G).all()

    decoded = np.isfinite(X).all(axis=1)
    assert np.isnan(X[~decoded]).all()
    assert decoded.mean() >= _LEAST_SHARE.get(k, 1.0)
    # Every successfully decoded design lies within the original bounds and is feasible there.
    Xd = X[decoded]
    assert np.all((Xd >= problem.original.xl) & (Xd <= problem.original.xu))
    F_original, G_original = _original(problem, Xd)
    assert np.all(G_original <= 0.0)
    # F and the inequalities are the original's at the decoded design, bit for bit.
    np.testing.assert_array_equal(F[decoded], F_original)
    np.testing.assert_array_equal(G[decoded, : problem.n_inequalities], G_original[:, : problem.n_inequalities])
    # The last constraint: the largest equality |h| - 1e-4 of the original (0 without equalities), positive on a failure.
    equalities = G_original[:, problem.n_inequalities :]
    largest = equalities.max(axis=1) if equalities.shape[1] else np.zeros(Xd.shape[0])
    np.testing.assert_array_equal(G[decoded, -1], largest)
    assert np.all(G[decoded, -1] >= -EQUALITY_TOLERANCE) and np.all(G[~decoded, -1] > 0.0)
    # Feasible on the reformulation exactly when the decoded design is feasible on the original.
    np.testing.assert_array_equal(np.all(G <= 0.0, axis=1), decoded)


# One problem per decoder: the sorted and level decoders work row by row up to four rows, and the power flows
# do not depend on the other rows of the batch.
_ROW_CASES = [(30, "sorted"), (35, "sorted"), (30, "level"), (35, "level"), (36, None), (40, None), (47, None)]


@pytest.mark.parametrize("case", _ROW_CASES, ids=_id)
def test_a_row_decodes_alone_as_in_a_batch(case: tuple[int, str | None]) -> None:
    problem = _problem(*case)
    U = _genotypes(problem, 40, seed=case[0] + 2)
    if case[1] is not None:
        U[:8, :-1] = np.round(U[:8, :-1] / 10.0) * 10.0  # angles that tie, and crowd at 0 or 90
    batch = problem.decode(U)
    alone = np.vstack([problem.decode(U[i : i + 1]) for i in range(U.shape[0])])
    pairs = np.vstack([problem.decode(U[i : i + 2]) for i in range(0, U.shape[0], 2)])
    np.testing.assert_array_equal(alone, batch)
    np.testing.assert_array_equal(pairs, batch)


@pytest.mark.parametrize("k", [36, 40, 47])
def test_power_flows_converge_on_most_genotypes(k: int) -> None:
    problem = make_reformulation(k)
    _, _, measure = problem._decoded(_genotypes(problem, _SAMPLES, seed=1))
    converged = measure < NOT_CONVERGED
    assert converged.mean() >= {36: 0.99, 40: 0.99, 47: 0.8}[k]
    # A converged design outside the bounds has the bound excess as its measure, below that of a solve that failed.
    assert np.all(measure[~converged] >= NOT_CONVERGED)


def test_a_failed_decoding_is_evaluated_at_the_clipped_design() -> None:
    problem = make_reformulation(40)
    U = _genotypes(problem, 200, seed=3)
    decoded, failed, measure = problem._decoded(U)
    assert failed.sum() > 150
    F, G = evaluate_population_with_constraints(problem, U)
    assert G is not None
    F_clipped, _ = _original(problem, np.clip(decoded[failed], problem.original.xl, problem.original.xu))
    np.testing.assert_array_equal(F[failed], F_clipped)
    np.testing.assert_array_equal(G[failed, -1], measure[failed])
    # RCM40-RCM46: the measure of a converged design is the sum of the excesses of its voltages over 1.
    excess = np.maximum(np.abs(decoded[failed, :26]) - 1.0, 0.0).sum(axis=1)
    converged = measure[failed] < NOT_CONVERGED
    np.testing.assert_array_equal(G[failed, -1][converged], excess[converged])


def test_an_unsolvable_microgrid_control_fails_with_the_largest_measure() -> None:
    # Without a direction of the active droop gains (a = 0), the flat-start solve is not attempted.
    problem = make_reformulation(47)
    U = np.array([[0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
    assert np.isnan(problem.decode(U)).all()
    F, G = evaluate_population_with_constraints(problem, U)
    assert G is not None and np.isfinite(F).all() and G[0, -1] >= NOT_CONVERGED


def _reference_problem(key: str, variant: str | None, n_var: int) -> RCMReformulation:
    selection = make_problem_selection(f"{key}_reformulated", n_var=n_var if variant else None)
    problem = (selection.with_options({"variant": variant}) if variant else selection).instantiate()
    assert isinstance(problem, RCMReformulation)
    return problem


_REFERENCE = [(case, key) for case in _FIXTURE["cases"] for key in case["problems"]]


def test_reference_values_come_from_the_authors_scripts() -> None:
    assert _FIXTURE["source"] == "Evolver scripts/realworld_repair, commit 4a2612bb2"
    decoded = {f"{key}-{case['variant']}" for case, key in _REFERENCE}
    assert decoded == {f"rcm{k}-{v}" for k, v in _CASES if v != "cumulative"}


@pytest.mark.parametrize(("case", "key"), _REFERENCE, ids=[f"{key}-{case['variant']}" for case, key in _REFERENCE])
def test_decoders_reproduce_the_scripts(case: dict[str, Any], key: str) -> None:
    U = np.asarray(case["U"], dtype=float)
    problem = _reference_problem(key, case["variant"], U.shape[1])
    X = problem.decode(U)
    expected = np.asarray([row if row is not None else [np.nan] * X.shape[1] for row in case["X"]], dtype=float)
    script_decoded = np.asarray(case["decoded"])
    if case["decoder"].startswith("process_and_sopwm."):
        # Closed form: the scripts' arithmetic, so the same designs where both run on one platform; exp and cos
        # can differ by an ulp between platforms' math libraries. The scripts' sorted angles can leave the first
        # one about 1e-21 below 0 (a design outside the bounds, which the scripts reject), and VAMOS clips it to 0.
        assert np.isfinite(X).all()
        np.testing.assert_allclose(X, np.clip(expected, problem.original.xl, problem.original.xu), rtol=1e-13, atol=1e-13)
        assert script_decoded.sum() >= 19
    else:
        # Newton power flows: the same rows decode, to the same designs up to the rounding of the solve.
        np.testing.assert_array_equal(np.isfinite(X).all(axis=1), script_decoded)
        np.testing.assert_allclose(X[script_decoded], expected[script_decoded], rtol=1e-12, atol=1e-12)
        assert script_decoded.sum() >= 8


def _paper_angles(t: list[float], total: float) -> list[float]:
    """Wang et al.'s Eqs. 18-21, one angle at a time, in degrees with VAMOS's gap."""
    shares = 0.0
    for t_i in t:
        shares += t_i
    alpha = total / shares  # Eq. 18
    scaled = [alpha * t_i for t_i in t]  # Eq. 19
    scaled = [scaled[0]] + [value + GAP for value in scaled[1:]]  # Eq. 20
    angles, y = [], 0.0
    for value in scaled:  # Eq. 21
        y += value
        angles.append(y)
    return angles


@pytest.mark.parametrize("k", range(30, 36))
def test_cumulative_variant_follows_wang_et_als_equations(k: int) -> None:
    problem = SOPWMReformulation(k)
    d = problem.n_var - 1
    assert problem.xl.tolist() == [0.0] * (d + 1) and problem.xu.tolist() == [1.0] * d + [90.0 - (d - 1) * GAP]
    U = _genotypes(problem, 300, seed=k)
    U[:20, -1] = problem.xu[-1]  # the largest total
    U[20:30, : d // 2] = 0.0  # some shares at 0
    U[30:40, -1] = 0.0  # the smallest total
    X = problem.decode(U)
    expected = np.array([_paper_angles(row[:-1].tolist(), float(row[-1])) for row in U])
    # The last angle can exceed 90 by rounding at the largest total; the decoder caps it.
    assert np.all(expected <= 90.0 + 1e-12)
    np.testing.assert_array_equal(X, np.minimum(expected, 90.0))
    np.testing.assert_allclose(X[:20, -1], 90.0, rtol=0.0, atol=1e-12)
    # The angles lie in [0, 90] and increase by more than the ordering threshold of the original problem.
    assert np.all((X >= 0.0) & (X <= 90.0)) and np.all(np.diff(X, axis=1) > ORDER_THRESHOLD)
    assert GAP > ORDER_THRESHOLD


def test_cumulative_variant_with_every_share_at_zero_spreads_the_total_evenly() -> None:
    problem = SOPWMReformulation(30)
    U = np.append(np.zeros(25), 45.0)[None, :]
    X = problem.decode(U)
    np.testing.assert_allclose(X[0], np.cumsum(np.full(25, 45.0 / 25) + np.r_[0.0, np.full(24, GAP)]), rtol=1e-14)


@pytest.mark.parametrize("variant", ["sorted", "level"])
def test_sorted_and_level_angles_increase_and_stay_within_bounds(variant: str) -> None:
    problem = SOPWMReformulation(35, variant=variant)
    U = _genotypes(problem, 500, seed=7)
    U[:50, :30] = -4.5  # every angle at 0 after clipping
    U[50:100, :30] = 94.5  # every angle at 90
    X = problem.decode(U)
    assert np.all((X >= 0.0) & (X <= 90.0)) and np.all(np.diff(X, axis=1) > ORDER_THRESHOLD)


def test_level_variant_sets_the_second_objective() -> None:
    # f2 = (m u - m)^2 whenever the angles can reach the fundamental m u: not for u = 0, since the angles
    # cannot all reach 90 with their gaps.
    problem = SOPWMReformulation(30, variant="level")
    U = _genotypes(problem, 200, seed=11)
    u = np.clip(U[:, -1], 0.0, 1.0)
    F, _ = evaluate_population_with_constraints(problem, U)
    inside = u > 0.01
    assert inside.sum() > 150
    np.testing.assert_allclose(F[inside, 1], (0.32 * u[inside] - 0.32) ** 2, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("k", REFORMULATED)
def test_reformulations_resolve_the_original_reference_front(k: int) -> None:
    assert resolve_reference_front_path(f"rcm{k:02d}_reformulated", None) == resolve_reference_front_path(f"rcm{k:02d}", None)


@pytest.mark.parametrize("key", ["rcm22_reformulated", "rcm30_reformulated", "rcm36_reformulated", "rcm47_reformulated"])
def test_reformulations_run_with_nsgaii(key: str) -> None:
    result = optimize(key, algorithm="nsgaii", pop_size=12, max_evaluations=48, seed=5)
    problem = make_problem_selection(key).instantiate()
    assert isinstance(problem, RCMReformulation)
    assert result.X is not None and result.X.shape[1] == problem.n_var
    designs = problem.decode(result.X)
    decoded = np.isfinite(designs).all(axis=1)
    F_original, _ = _original(problem, designs[decoded])
    np.testing.assert_array_equal(result.F[decoded], F_original)
