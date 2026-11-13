import math

import numpy as np

from vamos.engine.tuning.racing.param_space import Int, ParamSpace, Real
from vamos.engine.tuning.racing.refill import make_neighbor_config
from vamos.engine.tuning.racing.sampler import ModelBasedSampler


def test_model_update_log_real():
    space = ParamSpace({"lr": Real("lr", 0.001, 1.0, log=True)})
    sampler = ModelBasedSampler(space, min_samples_to_model=5)

    # Generate data from log-normal distribution roughly centered at 0.01
    # log(0.01) ~= -4.6
    rng = np.random.default_rng(42)
    # log-space mean -4.6, std 0.1
    log_vals = rng.normal(-4.6, 0.1, 100)
    vals = np.exp(log_vals)

    configs = [{"lr": v} for v in vals]
    sampler.update(configs)

    model = sampler._real_models.get("lr")
    assert model is not None
    assert model["log"] is True
    assert np.isclose(model["mean"], -4.6, atol=0.2)
    assert np.isclose(model["std"], 0.1, atol=0.05)


def test_model_update_log_int():
    space = ParamSpace({"batch": Int("batch", 8, 1024, log=True)})
    sampler = ModelBasedSampler(space, min_samples_to_model=5)

    # Generate data around 64 (log(64) ~= 4.16)
    rng = np.random.default_rng(42)
    log_vals = rng.normal(4.16, 0.1, 100)
    vals = np.round(np.exp(log_vals)).astype(int)

    configs = [{"batch": v} for v in vals]
    sampler.update(configs)

    model = sampler._int_models.get("batch")
    assert model is not None
    assert model["log"] is True
    assert np.isclose(model["mean"], 4.16, atol=0.2)


def test_sample_log_real():
    space = ParamSpace({"lr": Real("lr", 0.001, 1.0, log=True)})
    sampler = ModelBasedSampler(space, exploration_prob=0.0)  # pure exploitation

    # Manually set model
    sampler._real_models["lr"] = {
        "mean": math.log(0.1),  # approx -2.3
        "std": 0.1,
        "low": 0.001,
        "high": 1.0,
        "log": True,
    }

    rng = np.random.default_rng(42)
    samples = []
    for _ in range(100):
        cfg = sampler.sample(rng)
        samples.append(cfg["lr"])

    # Check that samples are around 0.1
    arr = np.array(samples)
    assert np.all(arr >= 0.001)
    assert np.all(arr <= 1.0)
    # Geometric mean should be close to 0.1 (arithmetic mean of logs close to log(0.1))
    log_mean = np.mean(np.log(arr))
    assert np.isclose(log_mean, math.log(0.1), atol=0.2)


def test_refill_log_perturbation_real():
    space = ParamSpace({"lr": Real("lr", 1e-5, 1.0, log=True)})
    rng = np.random.default_rng(42)
    base_cfg = {"lr": 1e-4}

    # Check that we explore both directions in log space
    # 1e-4 is -9.21
    # range 1e-5 to 1.0 is huge

    generated = []
    for _ in range(200):
        new_cfg = make_neighbor_config(base_cfg, space, rng)
        generated.append(new_cfg["lr"])

    arr = np.array(generated)
    assert np.all(arr >= 1e-5)
    assert np.all(arr <= 1.0)

    # Check we have values smaller and larger
    assert np.sum(arr < 1e-4) > 10
    assert np.sum(arr > 1e-4) > 10

    # Check that the distribution of log-values is roughly symmetric around log(1e-4)
    log_arr = np.log(arr)
    assert np.abs(np.mean(log_arr) - math.log(1e-4)) < 1.0


def test_refill_log_perturbation_int():
    space = ParamSpace({"units": Int("units", 16, 1024, log=True)})
    rng = np.random.default_rng(42)
    base_cfg = {"units": 64}

    generated = []
    for _ in range(200):
        new_cfg = make_neighbor_config(base_cfg, space, rng)
        generated.append(new_cfg["units"])

    arr = np.array(generated)
    assert np.all(arr >= 16)
    assert np.all(arr <= 1024)

    # values should be ints
    assert all(isinstance(x, (int, np.integer)) for x in generated)

    assert np.sum(arr < 64) > 0
    assert np.sum(arr > 64) > 0


def test_model_based_sampler_joint_mode_preserves_numeric_correlation():
    space = ParamSpace({"x": Real("x", 0.0, 1.0), "y": Real("y", 0.0, 1.0)})
    sampler = ModelBasedSampler(
        space,
        exploration_prob=0.0,
        min_samples_to_model=3,
        min_samples_for_joint=3,
        joint_sampling_prob=1.0,
        neighbor_noise_scale=0.08,
    )

    rng_data = np.random.default_rng(123)
    good = []
    for _ in range(80):
        x = float(rng_data.uniform(0.1, 0.9))
        y = float(np.clip(x + rng_data.normal(0.0, 0.03), 0.0, 1.0))
        good.append({"x": x, "y": y})

    sampler.update(good)

    rng_sample = np.random.default_rng(99)
    xs = []
    ys = []
    for _ in range(250):
        cfg = sampler.sample(rng_sample)
        xs.append(float(cfg["x"]))
        ys.append(float(cfg["y"]))

    corr = float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])
    assert corr > 0.6
