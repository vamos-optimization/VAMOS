"""The ``vamos.parameter-space`` 1.0.0 schema: the rules G1 to G6 that every space follows when it is read.

Each test breaks one rule in an otherwise valid space and checks the error names that rule and the
place. Rule G7 runs the algorithm and has its own tests (``test_parameter_space_default_check.py``).
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from vamos.engine.tuning.grammar import (
    SCHEMA,
    ParameterSpaceError,
    load_parameter_space,
    parameter_space_document,
    parameter_space_from_document,
    parameter_space_from_json,
    parameter_space_from_yaml,
    space_sha256,
)


def small_space() -> dict[str, Any]:
    """A valid real-coded NSGA-II space with each kind of node: categorical, real, integer and boolean."""
    return {
        "schema": "vamos.parameter-space",
        "schema_version": "1.0.0",
        "name": "nsgaii_small",
        "algorithm": "nsgaii",
        "encoding": "real",
        "description": "A small NSGA-II space for the schema tests.",
        "parameters": {
            "offspring_ratio": {"type": "categorical", "default": 1.0, "values": [0.5, 1.0], "role": "population"},
            "selection": {
                "type": "categorical",
                "default": "tournament",
                "values": {
                    "tournament": {"conditional_parameters": {"tournament_size": {"type": "int", "range": [2, 10], "default": 2}}},
                    "random": {},
                },
            },
            "crossover": {
                "type": "categorical",
                "default": "sbx",
                "global_sub_parameters": {"crossover_probability": {"type": "real", "range": [0.6, 1.0], "default": 0.9}},
                "values": {
                    "sbx": {"conditional_parameters": {"sbx_distribution_index": {"type": "real", "range": [5, 40], "default": 20}}},
                    "blx_alpha": {"conditional_parameters": {"crossover_alpha": {"type": "real", "range": [0.0, 1.0], "default": 0.5}}},
                },
            },
            "mutation": {
                "type": "categorical",
                "default": "polynomial",
                "global_sub_parameters": {
                    "mutation_probability_factor": {"type": "real", "range": [0.25, 3.0], "default": 1.0, "label": "Factor"}
                },
                "values": {
                    "polynomial": {
                        "conditional_parameters": {
                            "polynomial_mutation_distribution_index": {"type": "real", "range": [5.0, 40.0], "default": 20.0}
                        }
                    }
                },
            },
            "keep_duplicates": {"type": "boolean", "default": False},
        },
    }


def broken(edit: Any) -> dict[str, Any]:
    document = copy.deepcopy(small_space())
    edit(document)
    return document


def rejects(document: dict[str, Any], rule: str, where: str) -> ParameterSpaceError:
    with pytest.raises(ParameterSpaceError) as raised:
        parameter_space_from_document(document)
    assert raised.value.rule == rule, str(raised.value)
    assert where in raised.value.path, str(raised.value)
    assert str(raised.value).startswith(rule)
    return raised.value


def params(document: dict[str, Any]) -> dict[str, Any]:
    return document["parameters"]


def test_a_valid_space_reads_into_its_tree() -> None:
    space = parameter_space_from_document(small_space())

    assert [parameter.name for parameter in space.parameters] == [
        "offspring_ratio",
        "selection",
        "crossover",
        "mutation",
        "keep_duplicates",
    ]
    assert space.names() == (
        "offspring_ratio",
        "selection",
        "tournament_size",
        "crossover",
        "crossover_probability",
        "sbx_distribution_index",
        "crossover_alpha",
        "mutation",
        "mutation_probability_factor",
        "polynomial_mutation_distribution_index",
        "keep_duplicates",
    )
    # Real ranges and defaults are floats, integer ones integers.
    assert space.parameter("sbx_distribution_index").range == (5.0, 40.0)
    assert space.parameter("sbx_distribution_index").default == 20.0
    assert space.parameter("tournament_size").range == (2, 10)
    assert space.variant == "nsgaii"


# -- G1 ------------------------------------------------------------------------------------------


def test_g1_rejects_a_name_outside_the_pattern() -> None:
    def edit(document: dict[str, Any]) -> None:
        params(document)["Mutation"] = params(document).pop("mutation")

    error = rejects(broken(edit), "G1", "parameters.Mutation")
    assert "^[a-z][a-z0-9_]*$" in str(error)


def test_g1_rejects_a_name_used_twice_in_the_space() -> None:
    def edit(document: dict[str, Any]) -> None:
        params(document)["crossover"]["global_sub_parameters"]["keep_duplicates"] = {"type": "boolean", "default": True}

    error = rejects(broken(edit), "G1", "parameters.keep_duplicates")
    assert "unique" in str(error) and "parameters.crossover.global_sub_parameters.keep_duplicates" in str(error)


def test_g1_rejects_a_key_repeated_in_one_mapping() -> None:
    text = json.dumps(small_space())
    repeated = text.replace('"keep_duplicates": {', '"offspring_ratio": {"type": "boolean", "default": true}, "keep_duplicates": {', 1)

    with pytest.raises(ParameterSpaceError, match="G1") as raised:
        parameter_space_from_json(repeated)
    assert raised.value.path == "offspring_ratio"


# -- G2 ------------------------------------------------------------------------------------------


def test_g2_rejects_a_parameter_under_two_productions_of_one_node() -> None:
    def edit(document: dict[str, Any]) -> None:
        blx = params(document)["crossover"]["values"]["blx_alpha"]["conditional_parameters"]
        blx["sbx_distribution_index"] = {"type": "real", "range": [5.0, 40.0], "default": 20.0}

    error = rejects(broken(edit), "G2", "blx_alpha")
    assert "one parent" in str(error) and "own name" in str(error)


@pytest.mark.parametrize("key", ["condition", "conditions", "active_if", "depends_on", "parent"])
def test_g2_rejects_a_condition_expression(key: str) -> None:
    def edit(document: dict[str, Any]) -> None:
        params(document)["keep_duplicates"][key] = "cfg['selection'] == 'random'"

    error = rejects(broken(edit), "G2", f"keep_duplicates.{key}")
    assert "no condition expressions" in str(error)


# -- G3 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "default", "where"),
    [
        ("sbx_distribution_index", 41.0, "outside the range"),
        ("tournament_size", 2.5, "not an integer"),
        ("crossover", "pcx", "not one of the values"),
        ("keep_duplicates", 0, "not a boolean"),
        ("crossover_probability", True, "not a number"),
    ],
)
def test_g3_rejects_a_default_outside_its_domain(name: str, default: object, where: str) -> None:
    def edit(document: dict[str, Any]) -> None:
        _find(document, name)["default"] = default

    error = rejects(broken(edit), "G3", name)
    assert where in str(error)


def test_g3_rejects_a_parameter_without_a_default() -> None:
    def edit(document: dict[str, Any]) -> None:
        del params(document)["mutation"]["global_sub_parameters"]["mutation_probability_factor"]["default"]

    error = rejects(broken(edit), "G3", "mutation_probability_factor")
    assert "every parameter has a default" in str(error)


# -- G4 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "edit", "message"),
    [
        ("crossover_probability", {"range": [1.0, 0.6]}, "not ordered"),
        ("crossover_probability", {"range": [0.0, 1.0], "log": True, "default": 0.5}, "positive lower end"),
        ("tournament_size", {"range": [2.0, 10]}, "integer ends"),
        ("crossover_probability", {"range": ["1e-8", 1.0]}, "not a number"),
        ("crossover_probability", {"range": [0.6]}, "[low, high]"),
    ],
)
def test_g4_rejects_a_bad_range(name: str, edit: dict[str, Any], message: str) -> None:
    def change(document: dict[str, Any]) -> None:
        _find(document, name).update(edit)

    error = rejects(broken(change), "G4", f"{name}.range")
    assert message in str(error)


# -- G5 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "empty"),
        ([0.5, 1.0, 0.5], "listed twice"),
        ([0.5, [1.0]], "not a string, number or boolean"),
        ("1.0", "list of values"),
    ],
)
def test_g5_rejects_bad_values(values: object, message: str) -> None:
    def edit(document: dict[str, Any]) -> None:
        params(document)["offspring_ratio"]["values"] = values

    error = rejects(broken(edit), "G5", "offspring_ratio")
    assert message in str(error)


def test_g5_rejects_a_production_on_a_value_that_is_not_a_string() -> None:
    def edit(document: dict[str, Any]) -> None:
        params(document)["offspring_ratio"]["values"] = {
            0.5: {},
            1.0: {"conditional_parameters": {"x": {"type": "boolean", "default": True}}},
        }

    error = rejects(broken(edit), "G5", "offspring_ratio.values.1.0")
    assert "a value with a production is a string" in str(error)


def test_g5_rejects_a_number_beside_a_production() -> None:
    # The document of a categorical with a production is a mapping, whose keys are text: 1 would read back as "1".
    def edit(document: dict[str, Any]) -> None:
        params(document)["selection"]["values"] = {1: {}, "tournament": params(document)["selection"]["values"]["tournament"]}
        params(document)["selection"]["default"] = 1

    error = rejects(broken(edit), "G5", "selection.values.1")
    assert "string values only" in str(error)


@pytest.mark.parametrize(
    ("values", "default"),
    [([1, "1"], "1"), (["0.5", 0.5], 0.5), ([1.0, "1.0"], 1.0)],
    ids=["int-and-text", "text-and-float", "float-and-text"],
)
def test_g5_rejects_values_that_the_command_line_writes_alike(values: list[object], default: object) -> None:
    def edit(document: dict[str, Any]) -> None:
        params(document)["offspring_ratio"]["values"] = values
        params(document)["offspring_ratio"]["default"] = default

    error = rejects(broken(edit), "G5", "offspring_ratio")
    assert "written alike" in str(error)


def test_the_canonical_document_of_a_space_with_numbers_and_productions_reads_back() -> None:
    # Numbers stay in list-form categoricals, which keep their types; text keys keep theirs.
    space = parameter_space_from_document(small_space())
    document = json.loads(json.dumps(parameter_space_document(space)))

    assert parameter_space_from_document(document) == space


# -- G6 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema", "evolver.parameter-space", SCHEMA),
        ("schema_version", "2.0.0", "1.0.0"),
        ("algorithm", "nsga2", "not a VAMOS algorithm"),
        ("encoding", "float", "not a VAMOS encoding"),
    ],
)
def test_g6_rejects_an_unknown_schema_algorithm_or_encoding(key: str, value: str, message: str) -> None:
    def edit(document: dict[str, Any]) -> None:
        document[key] = value

    error = rejects(broken(edit), "G6", key)
    assert message in str(error)


def test_g6_rejects_an_encoding_the_algorithm_has_no_variant_for() -> None:
    def edit(document: dict[str, Any]) -> None:
        document |= {"algorithm": "ccmo", "encoding": "binary"}

    error = rejects(broken(edit), "G6", "encoding")
    assert "no binary variant of ccmo" in str(error)


# -- What the schema does not have ---------------------------------------------------------------


def test_the_schema_has_no_forbidden_clauses() -> None:
    def edit(document: dict[str, Any]) -> None:
        document["forbidden"] = [{"crossover": "blx_alpha", "selection": "random"}]

    error = rejects(broken(edit), "structure", "forbidden")
    assert "no forbidden clauses" in str(error)


def test_the_schema_has_one_range_per_parameter() -> None:
    def edit(document: dict[str, Any]) -> None:
        _find(document, "crossover_probability")["limits"] = [0.0, 1.0]

    error = rejects(broken(edit), "structure", "crossover_probability.limits")
    assert "one range per parameter" in str(error)


def test_unknown_keys_are_rejected() -> None:
    def edit(document: dict[str, Any]) -> None:
        _find(document, "tournament_size")["step"] = 1

    error = rejects(broken(edit), "structure", "tournament_size.step")
    assert "unknown key 'step'" in str(error)


# -- Canonical form and identity -----------------------------------------------------------------


def test_the_canonical_document_reads_back_into_the_same_space() -> None:
    space = parameter_space_from_document(small_space())
    document = parameter_space_document(space)

    assert parameter_space_from_document(document) == space
    assert parameter_space_document(parameter_space_from_document(document)) == document
    # Values without productions are written as a list; real numbers as floats.
    assert document["parameters"]["offspring_ratio"]["values"] == [0.5, 1.0]
    assert document["parameters"]["crossover"]["values"]["sbx"]["conditional_parameters"]["sbx_distribution_index"]["range"] == [5.0, 40.0]


def test_the_identity_ignores_labels_and_descriptions_and_follows_the_tree() -> None:
    space = parameter_space_from_document(small_space())
    relabelled = broken(
        lambda document: document.update(description="Another text.") or _find(document, "crossover_probability").update(label="p_c")
    )
    other_default = broken(lambda document: _find(document, "sbx_distribution_index").update(default=15.0))
    reordered = broken(lambda document: document.update(parameters=dict(reversed(list(params(document).items())))))

    assert space.identity.sha256 == space_sha256(space) and len(space.identity.sha256) == 64
    assert space_sha256(parameter_space_from_document(relabelled)) == space.identity.sha256
    assert space_sha256(parameter_space_from_document(other_default)) != space.identity.sha256
    # The order of the parameters is the depth-first order of the tree, so it is part of the identity.
    assert space_sha256(parameter_space_from_document(reordered)) != space.identity.sha256


def test_yaml_and_json_files_read_the_same_space(tmp_path: Any) -> None:
    import yaml

    json_file = tmp_path / "space.json"
    yaml_file = tmp_path / "space.yaml"
    json_file.write_text(json.dumps(small_space()), encoding="utf-8")
    yaml_file.write_text(yaml.safe_dump(small_space(), sort_keys=False), encoding="utf-8")

    assert load_parameter_space(json_file) == load_parameter_space(yaml_file) == parameter_space_from_document(small_space())


def test_a_yaml_key_repeated_in_one_mapping_is_rejected() -> None:

    text = "schema: vamos.parameter-space\nschema: vamos.parameter-space\n"

    with pytest.raises(ParameterSpaceError, match="G1.*repeated"):
        parameter_space_from_yaml(text)


def _find(document: dict[str, Any], name: str) -> dict[str, Any]:
    """The node called ``name`` anywhere in the document."""

    def search(nodes: dict[str, Any]) -> dict[str, Any] | None:
        for key, node in nodes.items():
            if key == name:
                return node
            children = dict(node.get("global_sub_parameters", {}))
            values = node.get("values")
            if isinstance(values, dict):
                for production in values.values():
                    children.update((production or {}).get("conditional_parameters", {}))
            found = search(children)
            if found is not None:
                return found
        return None

    found = search(params(document))
    assert found is not None, name
    return found
