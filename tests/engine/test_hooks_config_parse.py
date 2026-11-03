import numpy as np
import pytest

from vamos.engine.hooks import HookManager, HookManagerConfig
from vamos.engine.hooks.config_parse import build_archive_cfg, parse_stopping_archive


def test_build_archive_cfg_accepts_simplified_fields():
    cfg = build_archive_cfg({"capacity": 120, "pruning": "knn"})
    assert cfg.capacity == 120
    assert cfg.pruning == "knn"


def test_build_archive_cfg_rejects_unknown_fields():
    with pytest.raises(TypeError, match="Unknown archive.external fields"):
        build_archive_cfg({"unexpected_option": True})


def test_problem_override_merges_with_default_stopping_block():
    spec = {
        "defaults": {
            "stopping": {"hv_convergence": {"enabled": True, "ref_point": [2.0, 2.0], "window": 4}},
            "archive": {"external": {"enabled": True, "capacity": 50}},
        },
        "archive": {"external": {"pruning": "hv"}},
        "problems": {"zdt1": {"stopping": {"hv_convergence": {"patience": 10}}}},
    }

    cfg = parse_stopping_archive(spec, problem_key="zdt1")

    # A partial problem override keeps the inherited keys instead of replacing the whole block.
    assert cfg["stopping_enabled"] is True
    assert cfg["hv_ref_point"] == [2.0, 2.0]
    assert cfg["stop_cfg"].patience == 10
    assert cfg["stop_cfg"].window == 4
    # A top-level block merges into the defaults one.
    assert cfg["archive_enabled"] is True
    assert cfg["archive_cfg"].capacity == 50
    assert cfg["archive_cfg"].pruning == "hv"


def test_problem_override_keys_are_case_insensitive():
    spec = {"problems": {"ZDT1": {"stopping": {"hv_convergence": {"enabled": True, "ref_point": [2.0, 2.0]}}}}}

    cfg = parse_stopping_archive(spec, problem_key="zdt1")

    assert cfg["stopping_enabled"] is True
    assert cfg["hv_ref_point"] == [2.0, 2.0]


@pytest.mark.parametrize("mode", ["decision", "both"])
def test_hook_archive_rejects_decision_space_deduplication(mode):
    # The hook archive stores objective vectors only, so there is no decision vector to compare.
    with pytest.raises(ValueError, match="deduplicate_in"):
        build_archive_cfg({"capacity": 20, "deduplicate_in": mode})


def test_hook_archive_without_capacity_is_unbounded():
    spec = {"archive": {"external": {"enabled": True, "capacity": None}}}
    cfg = parse_stopping_archive(spec, problem_key="zdt1")
    assert cfg["archive_cfg"].capacity is None

    hooks = HookManager(
        HookManagerConfig(
            stopping_enabled=False,
            stop_cfg=cfg["stop_cfg"],
            archive_enabled=True,
            archive_cfg=cfg["archive_cfg"],
        )
    )
    f1 = np.linspace(0.0, 1.0, 300)
    hooks.on_checkpoint(evals=300, F=np.column_stack([f1, 1.0 - f1]))

    assert hooks.metadata_payload()["archive"]["final_size"] == 300
