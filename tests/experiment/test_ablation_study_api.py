import json

import pytest

from vamos.engine.tuning.api import AblationVariant, build_ablation_plan
from vamos.experiment._execution_support import VariationConfigs
from vamos.experiment.ablation import run_ablation_plan, write_ablation_csv
from vamos.foundation.exceptions import ConfigurationError


def test_run_ablation_plan_uses_canonical_studies_and_summaries(tmp_path):
    variants = [
        AblationVariant(name="baseline"),
        AblationVariant(name="tuned", config_overrides={"population_size": 8}),
    ]
    plan = build_ablation_plan(
        problems=["zdt1"],
        variants=variants,
        seeds=[1],
        default_max_evals=12,
        engine="numpy",
    )
    variations = {
        "tuned": VariationConfigs(nsgaii={"crossover": ("sbx", {"prob": 0.9, "eta": 20.0})}),
    }

    result = run_ablation_plan(
        plan,
        algorithm="nsgaii",
        output=tmp_path / "ablation",
        base_config={"population_size": 6, "offspring_population_size": 6},
        variations_by_variant=variations,
    )

    assert len(result.studies) == 2
    assert len(set(result.study_ids)) == 2
    assert result.study_roots == tuple(item.study.root for item in result.studies)
    assert [item.variant for item in result.studies] == ["baseline", "tuned"]
    for execution in result.studies:
        assert execution.study.status == "completed"
        assert execution.report.study_id == execution.study.study_id
        assert execution.summary.study_id == execution.study.study_id
        assert len(execution.summary.rows) == 1
        row = execution.summary.rows[0]
        assert row.selected_run_id is not None
        assert row.run_manifest_path is not None
        assert row.run_manifest_sha256 is not None
        assert row.metrics is not None
        assert row.problem_id == "vamos.problem:zdt1@1"

    derived = result.summary_rows()
    assert {row["variant"] for row in derived} == {"baseline", "tuned"}
    assert all(row["study_id"] in result.study_ids for row in derived)
    assert all(isinstance(row["hv"], float) for row in derived)
    assert all(row["hv_reference"] for row in derived)

    output = write_ablation_csv(result, tmp_path / "derived" / "ablation.csv")
    original_bytes = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        write_ablation_csv(result, output)
    assert output.read_bytes() == original_bytes
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))

    for execution in result.studies:
        documents = [path for path in execution.study.root.rglob("*.json") if "runs" not in path.parts]
        for document in documents:
            assert not _contains_array_key(json.loads(document.read_text(encoding="utf-8")))


def _contains_array_key(value):
    if isinstance(value, dict):
        return bool({"F", "X"} & value.keys()) or any(_contains_array_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_array_key(item) for item in value)
    return False


def _resolved(execution):
    task = execution.study.plan.tasks[0]
    return task.resolved_run_spec["algorithm"]["config"], task.resolved_run_spec["operators"]


def test_ablation_variant_overrides_reach_the_study(tmp_path):
    variants = [
        AblationVariant(name="baseline"),
        AblationVariant(name="pop_size_20", config_overrides={"pop_size": 20}),
        AblationVariant(name="sbx_05", config_overrides={"crossover": ("sbx", {"prob": 0.5, "eta": 20.0})}),
        AblationVariant(name="pm_map", config_overrides={"mutation": {"method": "pm", "prob": 0.2, "eta": 5.0}}),
        AblationVariant(name="elsewhere", config_overrides={"output_root": str(tmp_path / "elsewhere")}),
    ]
    plan = build_ablation_plan(problems=["zdt1"], variants=variants, seeds=[1, 2], default_max_evals=400, engine="numpy")

    result = run_ablation_plan(plan, algorithm="nsgaii", output=tmp_path / "ablation")

    by_variant = {item.variant: item for item in result.studies}
    baseline_config, baseline_operators = _resolved(by_variant["baseline"])
    assert _resolved(by_variant["pop_size_20"])[0]["pop_size"] == 20 != baseline_config["pop_size"]
    assert dict(_resolved(by_variant["sbx_05"])[1]["crossover"]["config"]) == {"prob": 0.5, "eta": 20.0}
    assert dict(_resolved(by_variant["pm_map"])[1]["mutation"]["config"]) == {"prob": 0.2, "eta": 5.0}
    assert by_variant["elsewhere"].study.root.parent == (tmp_path / "elsewhere").absolute()

    hv = {}
    for row in result.summary_rows():
        hv.setdefault(row["variant"], []).append(row["hv"])
    for variant in ("pop_size_20", "sbx_05", "pm_map"):
        assert hv[variant] != hv["baseline"], f"{variant} ran the baseline configuration"


@pytest.mark.parametrize(
    ("base_config", "overrides", "message"),
    [
        (None, {"crossover_prob": 0.5}, r"(?s)variant 'tuned'.*'crossover_prob'.*Did you mean 'crossover'"),
        ({"pop_sise": 20}, {}, r"(?s)base_config.*'pop_sise'.*Did you mean 'pop_size'"),
    ],
)
def test_ablation_rejects_unknown_overrides(tmp_path, base_config, overrides, message):
    variants = [AblationVariant(name="baseline"), AblationVariant(name="tuned", config_overrides=overrides)]
    plan = build_ablation_plan(problems=["zdt1"], variants=variants, seeds=[1], default_max_evals=40, engine="numpy")

    with pytest.raises(ConfigurationError, match=message):
        run_ablation_plan(
            plan, algorithm="nsgaii", output=tmp_path / "ablation", base_config={"population_size": 10, **(base_config or {})}
        )

    assert not (tmp_path / "ablation").exists(), "no variant may run before every variant is planned"


@pytest.mark.parametrize(
    ("base_config", "overrides"),
    [
        ({"pop_size": 20, "offspring_size": 10}, {"population_size": 40, "offspring_population_size": 20}),
        ({"population_size": 20, "offspring_population_size": 10}, {"pop_size": 40, "offspring_size": 20}),
    ],
    ids=["base_fields_variant_settings", "base_settings_variant_fields"],
)
def test_variant_population_overrides_win_over_base_aliases(tmp_path, base_config, overrides):
    variants = [AblationVariant(name="baseline"), AblationVariant(name="larger", config_overrides=overrides)]
    plan = build_ablation_plan(problems=["zdt1"], variants=variants, seeds=[1], default_max_evals=40, engine="numpy")

    result = run_ablation_plan(plan, algorithm="nsgaii", output=tmp_path / "ablation", base_config=base_config)

    resolved = {item.variant: _resolved(item)[0] for item in result.studies}
    assert (resolved["baseline"]["pop_size"], resolved["baseline"]["offspring_size"]) == (20, 10)
    assert (resolved["larger"]["pop_size"], resolved["larger"]["offspring_size"]) == (40, 20)


def test_ablation_rejects_conflicting_population_aliases(tmp_path):
    variants = [AblationVariant(name="baseline", config_overrides={"pop_size": 20, "population_size": 30})]
    plan = build_ablation_plan(problems=["zdt1"], variants=variants, seeds=[1], default_max_evals=40, engine="numpy")

    with pytest.raises(ConfigurationError, match="(?s)variant 'baseline'.*'pop_size'.*'population_size'"):
        run_ablation_plan(plan, algorithm="nsgaii", output=tmp_path / "ablation")
