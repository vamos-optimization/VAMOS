import pytest

from vamos.engine.config.spec import allowed_override_keys, validate_experiment_spec
from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.experiment.cli.spec_args import parser_spec_keys
from vamos.foundation.core.experiment_config import ExperimentConfig


def _build_parser() -> object:
    default_config = ExperimentConfig()
    pre_parser = build_pre_parser()
    spec_defaults = load_spec_defaults(None)
    return build_parser(
        default_config=default_config,
        pre_parser=pre_parser,
        spec_defaults=spec_defaults,
    )


def _allowed_override_keys() -> set[str]:
    parser = _build_parser()
    return allowed_override_keys(parser_spec_keys(parser))


def test_validate_experiment_spec_accepts_minimal_spec() -> None:
    spec = {"version": "1"}
    validated = validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())
    assert validated["version"] == "1"


def test_validate_experiment_spec_rejects_unknown_top_level_key() -> None:
    spec = {"version": "1", "surprise": {}}
    with pytest.raises(ValueError, match="Unknown top-level keys"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_unknown_problem_key() -> None:
    spec = {"version": "1", "problems": {"zdt999": {}}}
    with pytest.raises(ValueError, match="Unknown problem key"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_invalid_operator_spec() -> None:
    spec = {"version": "1", "defaults": {"nsgaii": {"crossover": ["sbx"]}}}
    with pytest.raises(ValueError, match="tuple/list form must have exactly 2 elements"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_invalid_hv_ref_point() -> None:
    spec = {"version": "1", "stopping": {"hv_convergence": {"ref_point": ["bad"]}}}
    with pytest.raises(ValueError, match="ref_point"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_non_mapping_defaults() -> None:
    spec = {"version": "1", "defaults": 123}
    with pytest.raises(TypeError, match="defaults"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_accepts_external_archive_block() -> None:
    spec = {
        "version": "1",
        "archive": {
            "external": {
                "enabled": True,
                "capacity": 50,
                "pruning": "crowding",
            }
        },
    }
    validated = validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())
    assert validated["archive"]["external"]["capacity"] == 50


def test_validate_experiment_spec_rejects_unknown_archive_block() -> None:
    spec = {
        "version": "1",
        "archive": {
            "unexpected": {},
        },
    }
    with pytest.raises(ValueError, match="Unknown keys in 'archive'"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


@pytest.mark.parametrize(
    ("suffix", "text"),
    [
        (
            ".yaml",
            "version: '1'\nstopping:\n  hv_convergence: {enabled: true, ref_point: [2, 2]}\nstopping:\n  hv_convergence: {patience: 9}\n",
        ),
        (".yaml", "version: '1'\ndefaults:\n  max_evaluations: 100\n  max_evaluations: 200\n"),
        (".json", '{"version": "1", "defaults": {"max_evaluations": 100}, "defaults": {"max_evaluations": 200}}'),
    ],
)
def test_spec_loader_rejects_duplicate_keys(tmp_path, suffix: str, text: str) -> None:
    from vamos.engine.config.loader import load_experiment_spec

    path = tmp_path / f"spec{suffix}"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate key"):
        load_experiment_spec(str(path))


def test_spec_loader_keeps_yaml_merge_keys(tmp_path) -> None:
    from vamos.engine.config.loader import load_experiment_spec

    path = tmp_path / "spec.yaml"
    path.write_text(
        "version: '1'\nbase: &base {max_evaluations: 100, population_size: 20}\ndefaults:\n  <<: *base\n  max_evaluations: 300\n",
        encoding="utf-8",
    )

    assert load_experiment_spec(str(path))["defaults"] == {"max_evaluations": 300, "population_size": 20}


_INVALID_HOOK_VALUES = [
    ({"stopping": {"hv_convergence": {"window": 1}}}, "window"),
    ({"stopping": {"hv_convergence": {"every_k": "200"}}}, "every_k"),
    ({"stopping": {"hv_convergence": {"statistic": "medain"}}}, "statistic"),
    ({"stopping": {"hv_convergence": {"epsilon_mode": "relative"}}}, "epsilon_mode"),
    ({"stopping": {"hv_convergence": {"confidence": 95}}}, "confidence"),
    ({"stopping": {"hv_convergence": {"enabled": "yes please"}}}, "enabled"),
    ({"stopping": {"hv_convergence": {"ref_point": [1.1, 1.1, 1.1]}}}, "ref_point"),
    ({"archive": {"external": {"deduplicate_in": "decision"}}}, "deduplicate_in"),
    ({"archive": {"external": {"capacity": True}}}, "capacity"),
    ({"archive": {"external": {"capacity": 10.7}}}, "capacity"),
    ({"archive": {"external": {"pruning": "hypervolume"}}}, "pruning"),
    ({"archive": {"external": {"objective_tolerance": float("nan")}}}, "objective_tolerance"),
    ({"problems": {"zdt1": {"stopping": {"hv_convergence": {"window": 1}}}}}, "problems.zdt1.stopping"),
]


@pytest.mark.parametrize(("blocks", "field"), _INVALID_HOOK_VALUES)
def test_spec_rejects_invalid_hook_values(blocks: dict[str, object], field: str) -> None:
    spec = {"version": "1", **blocks}

    with pytest.raises(ValueError, match=field):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_spec_accepts_an_unbounded_hook_archive() -> None:
    spec = {"version": "1", "archive": {"external": {"enabled": True, "capacity": None}}}

    validated = validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())

    assert validated["archive"]["external"]["capacity"] is None


def test_cli_problem_overrides_are_case_insensitive(tmp_path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text("version: '1'\nproblems:\n  ZDT1:\n    max_evaluations: 400\n", encoding="utf-8")

    spec_defaults = load_spec_defaults(str(path))

    assert spec_defaults.problem_overrides["zdt1"] == {"max_evaluations": 400}
    assert spec_defaults.spec["problems"] == {"zdt1": {"max_evaluations": 400}}


def test_problem_keys_that_differ_only_in_case_are_rejected(tmp_path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(
        "version: '1'\nproblems:\n  ZDT1: {max_evaluations: 400}\n  zdt1: {max_evaluations: 800}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="zdt1"):
        load_spec_defaults(str(path))
