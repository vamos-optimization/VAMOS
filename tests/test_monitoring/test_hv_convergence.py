from __future__ import annotations

from vamos.engine.hooks import HVConvergenceConfig, HVConvergenceMonitor


def test_hv_monitor_stops_on_plateau():
    cfg = HVConvergenceConfig(
        every_k=100,
        window=5,
        patience=3,
        epsilon=1e-6,
        epsilon_mode="abs",
        statistic="median",
        min_points=12,
        confidence=None,
        rng_seed=0,
    )
    mon = HVConvergenceMonitor(cfg)

    # Increase then plateau
    hv = [
        0.1,
        0.2,
        0.3,
        0.35,
        0.37,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
    ]
    stop = False
    for i, v in enumerate(hv):
        dec = mon.add_sample(evals=100 * (i + 1), hv=v)
        if dec.stop:
            stop = True
            assert dec.reason in ("converged_hv", "already_stopped")
            break
    assert stop


def test_hv_monitor_does_not_stop_while_improving():
    cfg = HVConvergenceConfig(
        every_k=100,
        window=5,
        patience=4,
        epsilon=1e-3,
        epsilon_mode="rel",
        statistic="mean",
        min_points=12,
        confidence=None,
        rng_seed=0,
    )
    mon = HVConvergenceMonitor(cfg)
    hv = [0.1 + 0.01 * i for i in range(30)]
    for i, v in enumerate(hv):
        dec = mon.add_sample(evals=100 * (i + 1), hv=v)
        assert not dec.stop


def test_auto_reference_point_is_frozen_at_the_first_sample():
    import numpy as np

    from vamos.engine.archive import ExternalArchiveConfig
    from vamos.engine.hooks import HookManager, HookManagerConfig

    manager = HookManager(
        HookManagerConfig(
            stopping_enabled=True,
            stop_cfg=HVConvergenceConfig(every_k=1, min_points=100),
            archive_enabled=False,
            archive_cfg=ExternalArchiveConfig(capacity=None),
        )
    )
    first = np.array([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]])
    # The second sample dominates the first but spans a smaller box, so a reference
    # recomputed from it would shrink and report a lower hypervolume.
    second = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    manager.on_checkpoint(evals=10, F=first)
    manager.on_checkpoint(evals=20, F=second)

    hv_first, hv_second = (entry["hv"] for entry in manager.metadata_payload()["stopping"]["trace"])
    assert hv_second >= hv_first
    assert manager.metadata_payload()["stopping"]["reference_point"] == [4.4, 4.4]


def test_hv_convergence_config_rejects_invalid_values():
    import pytest

    # Before VAMOS validated these values, a misspelled statistic ran as the median, an unknown
    # epsilon mode as "rel" and a confidence of 95 was clipped to 1.
    for kwargs, field in (
        ({"statistic": "medain"}, "statistic"),
        ({"epsilon_mode": "relative"}, "epsilon_mode"),
        ({"confidence": 95}, "confidence"),
        ({"every_k": "200"}, "every_k"),
        ({"window": True}, "window"),
        ({"epsilon": float("nan")}, "epsilon"),
    ):
        with pytest.raises((TypeError, ValueError), match=field):
            HVConvergenceConfig(**kwargs)


def test_hook_manager_rejects_reference_point_of_wrong_length():
    import pytest

    from vamos.engine.archive import ExternalArchiveConfig
    from vamos.engine.hooks import HookManager, HookManagerConfig

    for ref in ([1.1, 1.1, 1.1], [1.1]):
        # Before, every HV computation failed silently and the stop never triggered.
        with pytest.raises(ValueError, match="ref_point"):
            HookManager(
                HookManagerConfig(
                    stopping_enabled=True,
                    stop_cfg=HVConvergenceConfig(),
                    archive_enabled=False,
                    archive_cfg=ExternalArchiveConfig(capacity=None),
                    hv_ref_point=ref,
                )
            )


def test_hv_monitor_bootstrap_uses_numpy_generator():
    import numpy as np

    def upper_bound(cfg, run_seed):
        monitor = HVConvergenceMonitor(cfg, run_seed=run_seed)
        for index, value in enumerate([0.1, 0.2, 0.25, 0.27, 0.28, 0.285, 0.287, 0.288, 0.2885, 0.2887, 0.2888, 0.2889]):
            monitor.add_sample((index + 1) * 100, value)
        assert isinstance(monitor._rng, np.random.Generator)
        return monitor._bootstrap_ci_upper_recent(len(monitor._hv) - 1, conf=0.95)

    cfg = HVConvergenceConfig(window=6, min_points=8, confidence=0.95, bootstrap_samples=200)
    # Without rng_seed the bootstrap stream derives from the run seed; an explicit seed fixes it.
    assert cfg.rng_seed is None
    assert upper_bound(cfg, 3) == upper_bound(cfg, 3)
    fixed = HVConvergenceConfig(window=6, min_points=8, confidence=0.95, bootstrap_samples=200, rng_seed=7)
    assert upper_bound(fixed, 3) == upper_bound(fixed, 4)


def test_hv_monitor_accepts_a_negative_rng_seed():
    # Configuration validation accepts any integer seed, and VAMOS 1.2.1's random.Random took negative ones.
    # The NumPy generator needs a nonnegative seed, so a negative one is reduced as the run seed is (mod 2**64).
    def draws(seed: int) -> list[float]:
        monitor = HVConvergenceMonitor(HVConvergenceConfig(rng_seed=seed))
        return [float(value) for value in monitor._rng.random(3)]

    assert draws(-1) == draws(-1)
    assert draws(-1) == draws(2**64 - 1)
    assert draws(-1) != draws(1)
