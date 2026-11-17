"""MOEA/D runs that leave the options added after 1.3.0 unset give VAMOS 1.3.0's results, bit for bit.

Two kinds of checks:

- **Absolute, x86-64 only.** The digests below were computed with VAMOS 1.3.0 (``main`` at
  ``8b1a292a``) on x86-64, and hold there with NumPy, Numba and the Python neighbourhood
  fallback (Windows locally; the Linux and Windows CI legs). Their configurations avoid
  ``np.power`` (DE, uniform mutation, Tchebycheff aggregation, ZDT1 and a linear constraint),
  but the trajectories still depend on the platform: NumPy's reductions and BLAS round
  differently on arm64, and the macOS CI legs get other digests for all three runs, the same
  with both engines and with the options set or not. Those tests are skipped off x86-64.
- **Relative, every platform.** On the machine that runs the tests, the new options set to
  their defaults (``subproblem_order="random_permutation"``, the DE ``scheme="rand_1_bin"``
  and ``normalize=False``) give the bytes of the configuration without them, and the NumPy
  engine, the Numba engine and the Python neighbourhood fallback give the same bytes. None of
  these reads a pinned digest.

Each digest covers the final population, the checkpoint (subproblem order, cursor and ideal
point) and the state of the run's generator.
"""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem

X86_64 = platform.machine().lower() in {"amd64", "x86_64"}
x86_64_only = pytest.mark.skipif(
    not X86_64,
    reason="the 1.3.0 digests were computed on x86-64; arm64's NumPy reductions and BLAS round differently, so its trajectories differ",
)


class _LinearConstraintZDT1:
    """ZDT1 with six variables and the linear constraint ``x1 + x2 >= 0.4``."""

    n_var = 6
    n_obj = 2
    n_constraints = 1
    xl = np.zeros(6)
    xu = np.ones(6)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        ZDT1Problem(n_var=6).evaluate(X, out)
        out["G"] = (0.4 - X[:, 1] - X[:, 2])[:, None]


# case -> (batch size, constrained problem); the VAMOS 1.3.0 digests on x86-64.
CASES: dict[str, tuple[int, bool]] = {"steady-state": (1, False), "batch": (4, False), "constrained": (1, True)}
GOLDEN_X86_64: dict[str, str] = {
    "steady-state": "90c5da2196ee882c3b8639f805e2720af461a8e1895fa09f6cf0c87b4e24ba5a",
    "batch": "f90ba3a3c7127466a5d20c41ff63c2c9d39845ad0393ae2e557dca8bc92e3a98",
    "constrained": "34711faf14677145a7650c326aa61bba49f90e545ef1981c47c05ab769a3a9fc",
}


def _config(batch_size: int, *, explicit: bool) -> dict[str, Any]:
    builder = MOEADConfig.builder().pop_size(20).batch_size(batch_size).neighbor_size(5).delta(0.9).replace_limit(2)
    builder.crossover("de", cr=0.7, f=0.5, **({"scheme": "rand_1_bin"} if explicit else {}))
    builder.mutation("uniform", prob="1/n", perturb=0.2)
    builder.aggregation("tchebycheff", **({"normalize": False} if explicit else {}))
    builder.weight_vectors(divisions=19)
    if explicit:
        builder.subproblem_order("random_permutation")
    return builder.build().to_dict()


def _run_config(batch_size: int, *, explicit: bool) -> dict[str, Any]:
    """The configuration a run receives; the explicit one names the default order too, which ``to_dict()`` leaves out."""
    cfg = _config(batch_size, explicit=explicit)
    if explicit:
        cfg["subproblem_order"] = "random_permutation"
    return cfg


def _digest(result: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    population = result["population"]
    for key in ("X", "F", "G"):
        if key in population:
            digest.update(np.ascontiguousarray(population[key]).tobytes())
    checkpoint = result["checkpoint"]
    extra = checkpoint["extra"]
    digest.update(np.asarray(extra["subproblem_order"], dtype=np.int64).tobytes())
    digest.update(str(int(extra["subproblem_cursor"])).encode())
    digest.update(np.asarray(extra["ideal"], dtype=float).tobytes())
    digest.update(json.dumps(checkpoint["rng_state"], sort_keys=True, default=str).encode())
    digest.update(str(int(result["evaluations"])).encode())
    return digest.hexdigest()


def _run(cfg: dict[str, Any], constrained: bool, kernel: Any) -> dict[str, Any]:
    problem: Any = _LinearConstraintZDT1() if constrained else ZDT1Problem(n_var=8)
    return MOEAD(dict(cfg), kernel=kernel).run(problem, termination=("max_evaluations", 603), seed=11)


def _numba_kernel() -> Any:
    pytest.importorskip("numba")
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    return NumbaKernel()


# ---- Absolute: the results of VAMOS 1.3.0 on x86-64 -------------------------------------------


@x86_64_only
@pytest.mark.parametrize("explicit", [False, True], ids=["unset", "explicit-defaults"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_runs_give_the_results_of_1_3_0_on_x86_64(case: str, explicit: bool) -> None:
    batch_size, constrained = CASES[case]
    assert _digest(_run(_run_config(batch_size, explicit=explicit), constrained, NumPyKernel())) == GOLDEN_X86_64[case]


@x86_64_only
@pytest.mark.numba
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_numba_engine_gives_the_results_of_1_3_0_on_x86_64(case: str) -> None:
    batch_size, constrained = CASES[case]
    kernel = _numba_kernel()
    for explicit in (False, True):
        assert _digest(_run(_run_config(batch_size, explicit=explicit), constrained, kernel)) == GOLDEN_X86_64[case]


# ---- Relative: on the machine that runs the tests ----------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_new_options_at_their_defaults_change_nothing(case: str) -> None:
    batch_size, constrained = CASES[case]
    expected = _digest(_run(_run_config(batch_size, explicit=False), constrained, NumPyKernel()))
    assert _digest(_run(_run_config(batch_size, explicit=True), constrained, NumPyKernel())) == expected


@pytest.mark.numba
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_numba_engine_gives_the_same_runs(case: str) -> None:
    batch_size, constrained = CASES[case]
    expected = _digest(_run(_run_config(batch_size, explicit=False), constrained, NumPyKernel()))
    kernel = _numba_kernel()
    for explicit in (False, True):
        assert _digest(_run(_run_config(batch_size, explicit=explicit), constrained, kernel)) == expected


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_python_neighbourhood_update_gives_the_same_runs(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.engine.algorithm.moead import neighborhood

    batch_size, constrained = CASES[case]
    expected = _digest(_run(_run_config(batch_size, explicit=False), constrained, NumPyKernel()))
    # The update without numba: its Tchebycheff values are products and maxima, as the compiled update's.
    monkeypatch.setattr(neighborhood, "get_update_neighborhood_numba", lambda: None)
    monkeypatch.setattr(neighborhood, "get_update_neighborhood_batch_numba", lambda: None)
    for explicit in (False, True):
        assert _digest(_run(_run_config(batch_size, explicit=explicit), constrained, NumPyKernel())) == expected


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_the_default_configuration_is_unchanged_by_the_explicit_defaults(engine: str) -> None:
    kernel: Any = _numba_kernel() if engine == "numba" else NumPyKernel()
    plain = MOEADConfig.default(pop_size=30, n_var=8, n_obj=2).to_dict()
    explicit = dict(plain)
    explicit["subproblem_order"] = "random_permutation"
    explicit["crossover"] = ("de", {**plain["crossover"][1], "scheme": "rand_1_bin"})  # type: ignore[index]
    explicit["aggregation"] = ("pbi", {"theta": 5.0, "normalize": False})
    problem = ZDT1Problem(n_var=8)
    expected = MOEAD(plain, kernel=kernel).run(problem, termination=("max_evaluations", 1203), seed=3)
    actual = MOEAD(explicit, kernel=kernel).run(problem, termination=("max_evaluations", 1203), seed=3)
    assert _digest(actual) == _digest(expected)


def test_a_configuration_without_the_new_options_serializes_as_in_1_3_0() -> None:
    # The fields of VAMOS 1.3.0's MOEADConfig.to_dict(): those of 1.0, then subproblem_update_criterion and live_callback_mode.
    fields_1_3_0 = {
        "pop_size",
        "batch_size",
        "neighbor_size",
        "delta",
        "replace_limit",
        "crossover",
        "mutation",
        "aggregation",
        "weight_vectors",
        "constraint_mode",
        "repair",
        "initializer",
        "mutation_prob_factor",
        "use_numba_variation",
        "track_genealogy",
        "result_mode",
        "external_archive",
        "subproblem_update_criterion",
        "live_callback_mode",
    }
    assert set(MOEADConfig.default(pop_size=30, n_var=8, n_obj=2).to_dict()) == fields_1_3_0
    explicit = _config(1, explicit=True)
    assert "subproblem_order" not in explicit
    assert explicit == _config(1, explicit=False) | {
        "crossover": ("de", {"cr": 0.7, "f": 0.5, "scheme": "rand_1_bin"}),
        "aggregation": ("tchebycheff", {"normalize": False}),
    }
    assert MOEADConfig.from_dict(explicit).subproblem_order == "random_permutation"
