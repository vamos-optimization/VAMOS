"""Properties that every operator of the configuration spaces keeps (VR-069).

Each crossover and mutation that a configuration space can select is built as the algorithms
build it, through ``VariationPipeline`` with the parameters of the space's default assignment,
and applied to a population of its encoding. The checks are the encoding invariants (bounds,
integer values, permutations), the probability (0 changes nothing, and the mutation rate follows
``prob``) and, for crossovers with two children, children that differ. Checks of a distribution
against its published formula come with the fix of that operator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.engine.test_operator_combo_validation import _build_full_assignment, _inject_special_params, operator_choices
from vamos.engine.operators.impl.real import DEMatingCrossover, VariationWorkspace
from vamos.engine.tuning.racing.bridge import config_from_assignment
from vamos.engine.variation import VariationPipeline
from vamos.foundation.problem.registry import make_problem_selection

_CHOICES = operator_choices()
_N_VAR = {"real": 12, "integer": 12, "binary": 40, "permutation": 20}
_BOUNDS = {"real": (-2.0, 5.0), "integer": (0.0, 10.0), "binary": (0.0, 1.0)}
# The mixed problem has real, integer and categorical segments; its operators are chosen per segment.
_MIXED_SEGMENTS = ("real", "int", "cat")
_MIXED_INDEX = {"real": "real_idx", "int": "int_idx", "cat": "cat_idx"}

_TWIN_CHILDREN = {
    ("integer", "crossover", "arithmetic"): "VR-229: the integer arithmetic crossover gives two identical children",
    ("mixed", "int_crossover", "arithmetic"): "VR-229: the integer arithmetic crossover gives two identical children",
    ("mixed", "real_crossover", "arithmetic"): "VR-229: the real crossover of the mixed encoding gives two identical children",
    ("permutation", "crossover", "aex"): "VR-324: AEX builds its second child from the first parent, like the first child",
}
# PCX and SPX vary all three parents whatever prob_crossover is, and UNDX gives two children per three parents.
_PROB_ZERO_NOT_A_COPY = {"pcx", "simplex", "undx"}
# Mutations that act on a whole row (a segment or a reordering), so ``prob`` is a per-row rate.
_ROW_MUTATIONS = {"segment_inversion"}


def _cases(role: str) -> list[tuple[str, str, str]]:
    cases = []
    for encoding, parameter, operator in _CHOICES:
        if encoding == "mixed":
            segment = parameter.split("_", 1)[0]
            if segment in _MIXED_SEGMENTS and parameter == f"{segment}_{role}":
                cases.append((encoding, parameter, operator))
        elif parameter == role and operator != "de":
            cases.append((encoding, parameter, operator))
    return cases


def _id(case: tuple[str, str, str]) -> str:
    return "-".join(case)


def _with_prob(entry: tuple[str, dict[str, Any]], role: str, parameter: str, prob: float) -> tuple[str, dict[str, Any]]:
    """``entry`` (the crossover or the mutation, ``role``) with probability ``prob``.

    In the mixed encoding only the tested segment's operator gets ``prob``; the others get 0.
    """
    name, params = entry
    if name != "mixed":
        return name, {**params, "prob": prob}
    rates = {f"{segment}_{role}_prob": prob if f"{segment}_{role}" == parameter else 0.0 for segment in (*_MIXED_SEGMENTS, "perm")}
    return name, {**params, **rates}


class _Setup:
    """The operators of one configuration-space choice and a population of its encoding."""

    def __init__(self, encoding: str, parameter: str, operator: str, *, cross_prob: float, mut_prob: float) -> None:
        algo_name, builder = _CHOICES[(encoding, parameter, operator)]
        assignment = _build_full_assignment(builder(), overrides={parameter: operator})
        _inject_special_params(algo_name, assignment)
        config = config_from_assignment(algo_name, assignment).to_dict()
        crossover, mutation = config["crossover"], config["mutation"]
        if operator == "whole_arithmetic":
            # alpha = 0.5, the middle of its tuning range, is the parents' mean twice by construction.
            crossover = (crossover[0], {**crossover[1], "alpha": 0.3})
        self.encoding, self.parameter = encoding, parameter
        self.problem = make_problem_selection("mixed_design").instantiate() if encoding == "mixed" else None
        if self.problem is not None:
            self.spec = {key: np.asarray(value) for key, value in self.problem.mixed_spec.items()}
            self.xl, self.xu = np.asarray(self.problem.xl, dtype=float), np.asarray(self.problem.xu, dtype=float)
        else:
            low, high = _BOUNDS.get(encoding, (0.0, _N_VAR[encoding] - 1.0))
            # Integer problems declare integer bounds; the integer mutations clip in the population's dtype.
            dtype = float if encoding == "real" else int
            self.xl, self.xu = np.full(_N_VAR[encoding], low, dtype=dtype), np.full(_N_VAR[encoding], high, dtype=dtype)
        self.pipeline = VariationPipeline(
            encoding=encoding,
            cross_method=crossover[0],
            cross_params=_with_prob(crossover, "crossover", parameter, cross_prob)[1],
            mut_method=mutation[0],
            mut_params=_with_prob(mutation, "mutation", parameter, mut_prob)[1],
            xl=self.xl,
            xu=self.xu,
            workspace=VariationWorkspace(),
            repair_cfg=config.get("repair", "auto"),
            problem=self.problem,
        )

    def columns(self) -> np.ndarray:
        """The columns that the tested operator varies."""
        if self.encoding != "mixed":
            return np.arange(self.xl.size)
        return self.spec[_MIXED_INDEX[self.parameter.split("_", 1)[0]]]

    def population(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.encoding == "permutation":
            return np.array([rng.permutation(self.xl.size) for _ in range(n)])
        if self.encoding in {"integer", "binary"}:
            return rng.integers(self.xl.astype(int), self.xu.astype(int) + 1, size=(n, self.xl.size))
        X = self.xl + rng.random((n, self.xl.size)) * (self.xu - self.xl)
        if self.encoding == "mixed":
            discrete = np.concatenate([self.spec["int_idx"], self.spec["cat_idx"]])
            X[:, discrete] = np.floor(X[:, discrete] + rng.random((n, discrete.size)) * 0.999)
            X[:, self.spec["cat_idx"]] = np.minimum(X[:, self.spec["cat_idx"]], self.spec["cat_cardinality"] - 1)
        return X

    def crossover(self, parents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._repaired(self.pipeline.crossover_op(parents.copy(), rng), rng)

    def mutation(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._repaired(self.pipeline.mutation_op(X.copy(), rng), rng)

    def _repaired(self, X: Any, rng: np.random.Generator) -> np.ndarray:
        # As in VariationPipeline.produce_offspring: real-coded variation is followed by the repair.
        if self.pipeline.repair_op is None:
            return np.asarray(X)
        return np.asarray(self.pipeline.repair_op(np.asarray(X), self.xl, self.xu, rng))

    def assert_valid(self, X: np.ndarray) -> None:
        assert X.shape[1] == self.xl.size and np.isfinite(X).all()
        if self.encoding == "permutation":
            assert (np.sort(X, axis=1) == np.arange(self.xl.size)).all()
            return
        assert (X >= self.xl).all() and (X <= self.xu).all()
        if self.encoding in {"integer", "binary"}:
            assert (X == np.round(X)).all()
        if self.encoding == "mixed":
            discrete = np.concatenate([self.spec["int_idx"], self.spec["cat_idx"]])
            assert (X[:, discrete] == np.round(X[:, discrete])).all()
            assert (X[:, self.spec["cat_idx"]] < self.spec["cat_cardinality"]).all()


@pytest.mark.parametrize("case", _cases("crossover"), ids=_id)
def test_crossover_children_keep_the_encoding(case: tuple[str, str, str]) -> None:
    setup = _Setup(*case, cross_prob=1.0, mut_prob=0.0)
    rng = np.random.default_rng(11)
    groups = 60
    children = setup.crossover(setup.population(groups * setup.pipeline.parents_per_group, rng), rng)
    assert children.shape[0] == groups * setup.pipeline.children_per_group
    setup.assert_valid(children)


@pytest.mark.parametrize("case", [c for c in _cases("crossover") if c[2] not in _PROB_ZERO_NOT_A_COPY], ids=_id)
def test_crossover_with_prob_zero_copies_the_parents(case: tuple[str, str, str]) -> None:
    setup = _Setup(*case, cross_prob=0.0, mut_prob=0.0)
    rng = np.random.default_rng(12)
    parents = setup.population(60 * setup.pipeline.parents_per_group, rng)
    np.testing.assert_array_equal(setup.crossover(parents, rng), parents)


def _two_child_cases() -> list[Any]:
    cases = []
    for case in _cases("crossover"):
        if case[2] in _PROB_ZERO_NOT_A_COPY or case[2] == "mean":  # "mean" is the parents' mean by definition
            continue
        reason = _TWIN_CHILDREN.get(case)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        cases.append(pytest.param(case, marks=marks, id=_id(case)))
    return cases


@pytest.mark.parametrize("case", _two_child_cases())
def test_the_two_children_of_a_pair_differ(case: tuple[str, str, str]) -> None:
    setup = _Setup(*case, cross_prob=1.0, mut_prob=0.0)
    rng = np.random.default_rng(13)
    children = setup.crossover(setup.population(2000, rng), rng)[:, setup.columns()]
    twins = np.all(children[0::2] == children[1::2], axis=1)
    assert twins.mean() <= 0.05


@pytest.mark.parametrize("case", _cases("mutation"), ids=_id)
def test_mutation_keeps_the_encoding_and_changes_nothing_at_prob_zero(case: tuple[str, str, str]) -> None:
    rng = np.random.default_rng(14)
    setup = _Setup(*case, cross_prob=0.0, mut_prob=1.0)
    X = setup.population(500, rng)
    setup.assert_valid(setup.mutation(X, rng))

    unchanged = _Setup(*case, cross_prob=0.0, mut_prob=0.0)
    np.testing.assert_array_equal(unchanged.mutation(X, rng), X)


@pytest.mark.parametrize("case", _cases("mutation"), ids=_id)
def test_mutation_rate_follows_prob(case: tuple[str, str, str]) -> None:
    rng = np.random.default_rng(15)
    setup = _Setup(*case, cross_prob=0.0, mut_prob=0.2)
    columns = setup.columns()
    X = setup.population(3000, rng)
    changed = (setup.mutation(X, rng) != X)[:, columns]
    per_row = setup.encoding == "permutation" or case[2] in _ROW_MUTATIONS
    rate = changed.any(axis=1).mean() if per_row else changed.mean()
    # A discrete mutation can redraw or round back to the current value, so only a continuous one
    # changes every gene it mutates.
    continuous = setup.encoding == "real" or case[1] == "real_mutation"
    assert abs(rate - 0.2) <= 0.02 if continuous else rate <= 0.22

    always = _Setup(*case, cross_prob=0.0, mut_prob=1.0)
    rows_changed = (always.mutation(X, rng) != X)[:, columns].any(axis=1).mean()
    # A reordering can draw the current order (the displacement and scramble of a short segment);
    # an integer polynomial step on a small range often rounds back.
    assert rows_changed >= (0.99 if continuous else 0.8 if per_row else 0.1)


def test_de_crossover_takes_the_mutant_genes_with_rate_cr_and_at_least_one() -> None:
    # MOEA/D's DE/rand/1/bin: trial = target, with genes of target + F (b - c) where a uniform draw < CR,
    # plus one forced gene.
    rng = np.random.default_rng(16)
    n_var = 8
    lower, upper = np.full(n_var, -2.0), np.full(n_var, 5.0)
    groups = lower + rng.random((3000, 3, n_var)) * (upper - lower)
    target, mutant = groups[:, 0], groups[:, 0] + 0.5 * (groups[:, 1] - groups[:, 2])
    for cr in (0.0, 0.5, 1.0):
        trial = np.asarray(DEMatingCrossover(F=0.5, CR=cr, lower=lower, upper=upper)(groups.copy(), rng))
        assert trial.shape == (3000, 1, n_var)
        trial = trial[:, 0]
        from_mutant = np.isclose(trial, mutant) & ~np.isclose(target, mutant)
        assert (np.isclose(trial, target) | from_mutant).all()
        assert from_mutant.sum(axis=1).min() >= 1
        assert abs(from_mutant.mean() - (cr + (1 - cr) / n_var)) <= 0.02
