"""Tests for genealogy tracking across all algorithms."""

import pytest

from vamos.engine.algorithm.config import (
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.ibea import IBEA
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem


class TestGenealogyTracking:
    """Test genealogy tracking for all algorithms."""

    @pytest.fixture
    def small_problem(self):
        """Small ZDT1 problem for fast tests."""
        return ZDT1Problem(n_var=5)

    @pytest.fixture
    def many_obj_problem(self):
        """Small DTLZ2 problem for NSGA-III tests."""
        return DTLZ2Problem(n_var=6, n_obj=3)

    @pytest.fixture
    def kernel(self):
        """NumPy kernel instance."""
        return NumPyKernel()

    def test_nsgaii_genealogy_tracking(self, small_problem, kernel):
        """Test NSGA-II with genealogy tracking enabled."""
        cfg = (
            NSGAIIConfig.builder()
            .pop_size(10)
            .offspring_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
            .track_genealogy(True)
            .build()
        )
        alg = NSGAII(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        # Genealogy should contain tracking info (structure may vary)
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_moead_genealogy_tracking(self, small_problem, kernel):
        """Test MOEA/D with genealogy tracking enabled."""
        cfg = (
            MOEADConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .neighbor_size(5)
            .delta(0.9)
            .replace_limit(2)
            .aggregation("tchebycheff")
            .track_genealogy(True)
            .build()
        )
        alg = MOEAD(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_spea2_genealogy_tracking(self, small_problem, kernel):
        """Test SPEA2 with genealogy tracking enabled."""
        cfg = (
            SPEA2Config.builder()
            .pop_size(10)
            .archive_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
            .track_genealogy(True)
            .build()
        )
        alg = SPEA2(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_ibea_genealogy_tracking(self, small_problem, kernel):
        """Test IBEA with genealogy tracking enabled."""
        cfg = (
            IBEAConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
            .indicator("eps")
            .kappa(0.05)
            .track_genealogy(True)
            .build()
        )
        alg = IBEA(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_smsemoa_genealogy_tracking(self, small_problem, kernel):
        """Test SMS-EMOA with genealogy tracking enabled."""
        cfg = (
            SMSEMOAConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
            .track_genealogy(True)
            .build()
        )
        alg = SMSEMOA(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 20), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_smpso_genealogy_tracking(self, small_problem, kernel):
        """Test SMPSO with genealogy tracking enabled."""
        cfg = SMPSOConfig.builder().pop_size(10).archive_size(10).mutation("polynomial", prob="1/n", eta=20.0).track_genealogy(True).build()
        alg = SMPSO(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_nsgaiii_genealogy_tracking(self, many_obj_problem, kernel):
        """Test NSGA-III with genealogy tracking enabled."""
        cfg = (
            NSGAIIIConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
            .reference_directions(divisions=3)
            .track_genealogy(True)
            .build()
        )
        alg = NSGAIII(cfg.to_dict(), kernel)
        result = alg.run(many_obj_problem, termination=("max_evaluations", 36), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_genealogy_disabled_by_default(self, small_problem, kernel):
        """Test that genealogy is not tracked when disabled (default)."""
        cfg = (
            NSGAIIConfig.builder()
            .pop_size(10)
            .offspring_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
            .build()
        )
        alg = NSGAII(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        # Should not have genealogy when disabled
        assert result.get("genealogy") is None or "genealogy" not in result


# --- Lineages through survival, in every algorithm that tracks genealogy ---

_GENEALOGY_ALGORITHMS = ["nsgaii", "moead", "smsemoa", "nsgaiii", "spea2", "ibea", "smpso"]
_SBX_ALGORITHMS = ["nsgaii", "smsemoa", "nsgaiii", "spea2", "ibea"]


def _tracked_run(algorithm: str, pop_size: int, max_evaluations: int):
    import dataclasses

    from vamos.engine.algorithm import config as configs
    from vamos.engine.algorithm.registry import resolve_algorithm

    cls = {
        "nsgaii": configs.NSGAIIConfig,
        "moead": configs.MOEADConfig,
        "smsemoa": configs.SMSEMOAConfig,
        "nsgaiii": configs.NSGAIIIConfig,
        "spea2": configs.SPEA2Config,
        "ibea": configs.IBEAConfig,
        "smpso": configs.SMPSOConfig,
    }[algorithm]
    kwargs = {"pop_size": pop_size, "n_var": 30}
    if algorithm in {"moead", "nsgaiii"}:
        kwargs["n_obj"] = 2
    cfg = dataclasses.replace(cls.default(**kwargs), track_genealogy=True)
    instance = resolve_algorithm(algorithm)(cfg.to_dict(), NumPyKernel())
    result = instance.run(ZDT1Problem(n_var=30), termination=("max_evaluations", max_evaluations), seed=1)
    return instance, result


@pytest.mark.parametrize("algorithm", _GENEALOGY_ALGORITHMS)
def test_offspring_reach_final_lineages(algorithm):
    _instance, result = _tracked_run(algorithm, pop_size=40, max_evaluations=2000)

    genealogy = result["genealogy"]
    # In VAMOS 1.1.0, IBEA, MOEA/D and SPEA2 kept the initial population's ids after survival, so the
    # final front was marked on generation 0 and no offspring appeared in a final lineage.
    assert sum(int(row["uses_in_final_lineages"]) for row in genealogy["operator_stats"]) > 0
    generations_with_final_members = [row["generation"] for row in genealogy["generation_contributions"] if row["final_lineage"]]
    assert len(generations_with_final_members) > 1
    assert max(generations_with_final_members) > 0


@pytest.mark.parametrize("algorithm", _SBX_ALGORITHMS)
def test_sbx_offspring_record_two_parents(algorithm):
    instance, _result = _tracked_run(algorithm, pop_size=12, max_evaluations=60)

    offspring = [record for record in instance._st.genealogy_tracker.records.values() if record.generation > 0]
    assert offspring
    # In VAMOS 1.1.0, NSGA-III and IBEA recorded one parent per SBX child.
    assert {len(record.parents) for record in offspring} == {2}


@pytest.mark.parametrize("algorithm", _GENEALOGY_ALGORITHMS)
def test_build_algorithm_passes_track_genealogy(algorithm):
    from vamos.engine.algorithm.factory import build_algorithm
    from vamos.foundation.core.experiment_config import ExperimentConfig

    problem = ZDT1Problem(n_var=6)
    config = ExperimentConfig(population_size=20, offspring_population_size=20, max_evaluations=60)

    instance, _cfg = build_algorithm(algorithm, "numpy", problem, config, track_genealogy=True)
    result = instance.run(problem, termination=("max_evaluations", 60), seed=1)

    # In VAMOS 1.1.0 only NSGA-II received track_genealogy from build_algorithm.
    assert "genealogy" in result


@pytest.mark.parametrize("algorithm", ["agemoea", "rvea"])
def test_build_algorithm_rejects_genealogy_where_it_is_not_tracked(algorithm):
    from vamos.engine.algorithm.factory import build_algorithm
    from vamos.foundation.core.experiment_config import ExperimentConfig
    from vamos.foundation.exceptions import ConfigurationError

    config = ExperimentConfig(population_size=20, offspring_population_size=20, max_evaluations=60)

    with pytest.raises(ConfigurationError, match="track_genealogy"):
        build_algorithm(algorithm, "numpy", ZDT1Problem(n_var=6), config, track_genealogy=True)


@pytest.mark.parametrize("max_evaluations", [20, 60])
def test_spea2_genealogy_follows_an_archive_smaller_than_the_population(max_evaluations):
    import dataclasses

    cfg = dataclasses.replace(SPEA2Config.default(pop_size=20, n_var=6), track_genealogy=True, archive_size=10)

    result = SPEA2(cfg.to_dict(), NumPyKernel()).run(ZDT1Problem(n_var=6), termination=("max_evaluations", max_evaluations), seed=1)

    # The ids follow the 10-row archive, also when no generation runs after the initial population.
    assert result["genealogy"]["generation_contributions"]


def test_moead_genealogy_credits_the_child_that_replaced_each_subproblem(monkeypatch):
    import dataclasses

    import numpy as np

    from vamos.foundation.eval.backends import SerialEvalBackend

    problem = ZDT1Problem(n_var=6)
    cfg = dataclasses.replace(MOEADConfig.default(pop_size=4, n_var=6, n_obj=2), track_genealogy=True, batch_size=2)
    algorithm = MOEAD(cfg.to_dict(), NumPyKernel())
    algorithm.run(problem, termination=("max_evaluations", 4), seed=1)
    st = algorithm._st
    st.max_evals = 6  # room for one batch of two children
    algorithm.ask()
    first_id, second_id = (int(i) for i in st.pending_offspring_ids)

    # Two identical children on the true front, which beat any random initial solution: the first is
    # offered subproblem 0 only and the second subproblem 3 only, so each replaces one subproblem.
    child = np.zeros(6)
    child[0] = 0.5
    st.pending_offspring = np.vstack([child, child])
    monkeypatch.setattr(MOEAD, "_build_candidate_orders", lambda self, st, active, use: (np.array([[0], [3]]), np.array([1, 1])))
    algorithm.tell(SerialEvalBackend().evaluate(st.pending_offspring, problem))

    # Matching the children by their values credited the later identical child for both subproblems.
    assert (int(st.ids[0]), int(st.ids[3])) == (first_id, second_id)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_moead_tracking_genealogy_does_not_change_the_run(batch_size):
    import dataclasses

    import numpy as np

    results = []
    for track in (False, True):
        cfg = dataclasses.replace(MOEADConfig.default(pop_size=12, n_var=6, n_obj=2), track_genealogy=track, batch_size=batch_size)
        result = MOEAD(cfg.to_dict(), NumPyKernel()).run(ZDT1Problem(n_var=6), termination=("max_evaluations", 120), seed=3)
        results.append(np.asarray(result["F"]))
    np.testing.assert_array_equal(results[0], results[1])
