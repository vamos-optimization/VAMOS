from dataclasses import replace

import numpy as np
import pytest

from vamos.engine.algorithm.agemoea import AGEMOEA
from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    IBEAConfig,
    MOEADConfig,
    RVEAConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.ibea import IBEA
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.rvea import RVEA
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


class _ConstrainedBiObjectiveProblem:
    n_var = 2
    n_obj = 2
    n_constraints = 1
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])
    encoding = "real"

    def evaluate(self, X, out):
        out["F"] = np.column_stack([X[:, 0], 1.0 - X[:, 0] + X[:, 1]])
        out["G"] = (0.5 - X[:, 0])[:, None]


def _agemoea_builder(pop_size: int = 12):
    return AGEMOEAConfig.builder().pop_size(pop_size).crossover("sbx", prob=0.9, eta=15.0).mutation("polynomial", prob=0.1, eta=20.0)


def _rvea_builder(pop_size: int = 6, n_partitions: int = 5):
    return (
        RVEAConfig.builder()
        .pop_size(pop_size)
        .n_partitions(n_partitions)
        .alpha(2.0)
        .adapt_freq(0.1)
        .crossover("sbx", prob=1.0, eta=30.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
    )


def test_moead_archive_keeps_default_nondominated_result_mode():
    cfg = (
        MOEADConfig.builder()
        .pop_size(10)
        .neighbor_size(5)
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .external_archive(capacity=20)
        .build()
    )
    assert cfg.result_mode == "non_dominated"

    problem = ZDT1Problem(n_var=6)
    result = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", 20),
        seed=0,
    )

    assert result["archive"]["F"].shape[0] > 0
    assert result["population"]["F"].shape[0] == 10
    np.testing.assert_allclose(result["F"], result["archive"]["F"])
    np.testing.assert_allclose(result["X"], result["archive"]["X"])


def test_smsemoa_archive_keeps_default_nondominated_result_mode():
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(10)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .external_archive(capacity=20)
        .build()
    )
    assert cfg.result_mode == "non_dominated"

    problem = ZDT1Problem(n_var=6)
    result = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", 20),
        seed=0,
    )

    assert result["archive"]["F"].shape[0] > 0
    assert result["population"]["F"].shape[0] == 10
    np.testing.assert_allclose(result["F"], result["archive"]["F"])
    np.testing.assert_allclose(result["X"], result["archive"]["X"])


def test_moead_rejects_archive_result_mode():
    with pytest.raises(ValueError, match="result_mode must be 'non_dominated' or 'population'"):
        (
            MOEADConfig.builder()
            .pop_size(10)
            .neighbor_size(5)
            .delta(0.9)
            .replace_limit(2)
            .crossover("de", cr=1.0, f=0.5)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .aggregation("tchebycheff")
            .result_mode("archive")
            .build()
        )


def test_smsemoa_rejects_archive_result_mode():
    with pytest.raises(ValueError, match="result_mode must be 'non_dominated' or 'population'"):
        (
            SMSEMOAConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
            .result_mode("archive")
            .build()
        )


def test_agemoea_rejects_archive_result_mode():
    with pytest.raises(ValueError, match="result_mode must be 'non_dominated' or 'population'"):
        _agemoea_builder().result_mode("archive").build()


def test_rvea_rejects_archive_result_mode():
    with pytest.raises(ValueError, match="result_mode must be 'non_dominated' or 'population'"):
        _rvea_builder().result_mode("archive").build()


def test_agemoea_archive_keeps_default_result_and_exposes_population():
    pop_size = 12
    cfg = _agemoea_builder(pop_size).external_archive(capacity=pop_size * 2).build()
    problem = ZDT1Problem(n_var=6)
    result = AGEMOEA(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", pop_size * 2),
        seed=0,
    )

    assert result["population"]["F"].shape[0] == pop_size
    assert result["archive"]["F"].shape[0] > 0
    np.testing.assert_allclose(result["F"], result["archive"]["F"])
    np.testing.assert_allclose(result["X"], result["archive"]["X"])


def test_agemoea_population_result_mode_with_archive():
    pop_size = 12
    cfg = _agemoea_builder(pop_size).external_archive(capacity=pop_size * 2).result_mode("population").build()
    problem = ZDT1Problem(n_var=6)
    result = AGEMOEA(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", pop_size * 2),
        seed=0,
    )

    assert result["F"].shape == result["population"]["F"].shape
    assert result["archive"]["F"].shape[0] > 0


def test_rvea_archive_keeps_default_result_and_exposes_population():
    pop_size = 6
    cfg = _rvea_builder(pop_size=pop_size, n_partitions=5).external_archive(capacity=pop_size * 2).build()
    problem = ZDT1Problem(n_var=6)
    result = RVEA(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", pop_size * 2),
        seed=0,
    )

    assert result["archive"]["F"].shape[0] > 0
    np.testing.assert_allclose(result["F"], result["archive"]["F"])
    np.testing.assert_allclose(result["X"], result["archive"]["X"])


def test_rvea_population_result_mode_with_archive():
    pop_size = 6
    cfg = _rvea_builder(pop_size=pop_size, n_partitions=5).external_archive(capacity=pop_size * 2).result_mode("population").build()
    problem = ZDT1Problem(n_var=6)
    result = RVEA(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", pop_size * 2),
        seed=0,
    )

    assert result["F"].shape == result["population"]["F"].shape
    assert result["archive"]["F"].shape[0] > 0


@pytest.mark.parametrize(
    ("algorithm_cls", "config"),
    [
        (IBEA, replace(IBEAConfig.default(pop_size=10, n_var=6), result_mode="population")),
        (SPEA2, replace(SPEA2Config.default(pop_size=10, n_var=6), result_mode="population")),
    ],
)
def test_population_result_mode_honored_by_cli_tuning_algorithms_requiring_top_level_constraints(algorithm_cls, config):
    problem = ZDT1Problem(n_var=6)
    result = algorithm_cls(config.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", 20),
        seed=0,
    )

    np.testing.assert_allclose(result["F"], result["population"]["F"])
    np.testing.assert_allclose(result["X"], result["population"]["X"])


@pytest.mark.parametrize(
    ("algorithm_cls", "config"),
    [
        (IBEA, replace(IBEAConfig.default(pop_size=10, n_var=2), result_mode="population")),
        (SPEA2, replace(SPEA2Config.default(pop_size=10, n_var=2), result_mode="population")),
    ],
)
def test_population_result_mode_exposes_aligned_constraints_for_ibea_and_spea2(algorithm_cls, config):
    result = algorithm_cls(config.to_dict(), kernel=NumPyKernel()).run(
        _ConstrainedBiObjectiveProblem(),
        termination=("max_evaluations", 20),
        seed=0,
    )

    assert "G" in result
    assert result["G"].shape[0] == result["F"].shape[0]
    assert result["G"].shape[0] == result["population"]["F"].shape[0]
    np.testing.assert_allclose(result["F"], result["population"]["F"])
