import numpy as np
import pytest

from vamos.engine.tuning.racing.param_space import Boolean, Categorical, Condition, Int, ParamSpace, Real


def test_sample_omits_descendants_of_inactive_parameters_and_keeps_its_draws():
    # "budget" depends on "bounded", which is active only with an archive: the archive case of the built-in spaces.
    space = ParamSpace(
        params={"archive": Boolean("archive"), "bounded": Boolean("bounded"), "budget": Int("budget", 1, 9)},
        conditions=[Condition("bounded", "cfg['archive'] == True"), Condition("budget", "cfg['bounded'] == True")],
    )
    for seed in range(40):
        sample = space.sample(np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        drawn = {name: spec.sample(rng) for name, spec in space.params.items()}

        assert ("bounded" in sample) == bool(sample["archive"])
        assert "budget" not in sample or ("bounded" in sample and sample["bounded"])
        # Active parameters keep the values drawn in declaration order, so seeded searches are unchanged.
        assert all(sample[name] == drawn[name] for name in sample)
        assert list(sample) == [name for name in space.params if name in sample]


@pytest.mark.parametrize("enabled", [False, True])
def test_condition_accepts_numpy_boolean_parent(enabled):
    space = ParamSpace(
        params={"enabled": Boolean("enabled"), "value": Real("value", 0.0, 1.0)},
        conditions=[Condition("value", "cfg['enabled']")],
    )
    assert space.is_active("value", {"enabled": np.bool_(enabled)}) is enabled


def test_param_space_sample_and_validate_with_conditions():
    space = ParamSpace(
        params={
            "a": Real("a", 0.0, 1.0),
            "b": Int("b", 1, 5),
            "c": Categorical("c", ["x", "y"]),
            "d": Real("d", 0.0, 10.0),
        },
        conditions=[Condition("d", "cfg['c'] == 'y'")],
    )
    rng = np.random.default_rng(0)
    cfg = space.sample(rng)
    space.validate(cfg)  # should not raise
    cfg_inactive = dict(cfg)
    cfg_inactive["c"] = "x"
    cfg_inactive.pop("d", None)
    space.validate(cfg_inactive)  # d inactive, so still valid
