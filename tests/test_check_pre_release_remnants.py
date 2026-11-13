from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tools.check_pre_release_remnants import guidance_remnant_tokens, scan


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_clean_fixture_passes(tmp_path: Path) -> None:
    _write(tmp_path, "src/vamos/example.py", "ENVIRONMENT_COMPATIBILITY = True\n")
    _write(tmp_path, "experiments/collector.py", 'for path in root.rglob("manifest.json"):\n    pass\n')

    assert scan(tmp_path) == []


def test_discarded_variation_import_fails(tmp_path: Path) -> None:
    _write(tmp_path, "src/vamos/example.py", "from vamos.engine.algorithm.components.variation import VariationPipeline\n")

    assert {finding.signature for finding in scan(tmp_path)} == {"discarded variation import"}


def test_discarded_run_file_fails(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/collector.py", 'path = run / "FUN.csv"\n')

    assert any(finding.signature.startswith("discarded run file") for finding in scan(tmp_path))


def test_discarded_archive_builder_fails(tmp_path: Path) -> None:
    _write(tmp_path, "src/vamos/example.py", "def build_bounded_archive_cfg(value):\n    return value\n")

    assert {finding.signature for finding in scan(tmp_path)} == {"discarded archive builder"}


def test_discarded_cli_alias_fails(tmp_path: Path) -> None:
    _write(tmp_path, "src/vamos/experiment/cli/main.py", 'if command == "--help-commands":\n    pass\n')

    assert {finding.signature for finding in scan(tmp_path)} == {"discarded CLI alias"}


def test_discarded_study_command_envelope_fails_in_docs(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide/studies.md", 'document_type = "vamos.study-plan-result"\n')

    assert {finding.signature for finding in scan(tmp_path)} == {"discarded study command envelope"}


@pytest.mark.parametrize(
    ("relative", "source", "signature"),
    [
        ("src/vamos/product.py", "StudyRunner()\n", "discarded study runtime symbol"),
        ("docs/guide/studies.md", "Call run_study.\n", "discarded study runtime symbol"),
        ("notebooks/studies.ipynb", '"source": ["StudyTask"]\n', "discarded study runtime symbol"),
        (
            "examples/study.py",
            "from vamos.experiment.study.runner import StudyResult\n",
            "discarded study runtime import",
        ),
        ("AGENTS.md", "Use study.migration for callers.\n", "discarded study caller transition API"),
    ],
)
def test_discarded_study_surface_fails_across_active_roots(tmp_path: Path, relative: str, source: str, signature: str) -> None:
    _write(tmp_path, relative, source)

    assert signature in {finding.signature for finding in scan(tmp_path)}


def test_superseded_study_file_fails_even_when_empty(tmp_path: Path) -> None:
    _write(tmp_path, "src/vamos/experiment/study/runner.py", "")

    assert {finding.signature for finding in scan(tmp_path)} == {"discarded path exists"}


def test_top_level_multi_seed_result_remains_semantically_distinct(tmp_path: Path) -> None:
    _write(tmp_path, "src/vamos/experiment/optimization_result/study.py", "class StudyResult:\n    pass\n")

    assert scan(tmp_path) == []


@pytest.mark.parametrize(
    "module",
    [
        "vamos.experiment.study.api",
        "vamos.experiment.study.persistence",
        "vamos.experiment.study.runner",
        "vamos.experiment.study.types",
    ],
)
def test_removed_study_modules_are_not_importable(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_website_shim_path_fails(tmp_path: Path) -> None:
    _write(tmp_path, "website/_compat/plugin.py", "class Plugin:\n    pass\n")

    assert {finding.signature for finding in scan(tmp_path)} == {"discarded path exists"}


def test_checker_fixture_vocabulary_is_narrowly_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_check_pre_release_remnants.py", 'DISCARDED = "legacy"\n')

    assert scan(tmp_path) == []


def test_shared_guidance_rules_classify_removed_paths_fields_and_cli_aliases() -> None:
    text = "\n".join(
        (
            "Import vamos.engine.algorithm.components.variation.",
            "Call build_bounded_archive_cfg with archive_type and size_cap.",
            "Run vamos benchmark --help and --quickstart.",
            "Read FUN.csv and metadata.json.",
            "Do not emit vamos.study-plan-result.",
        )
    )

    assert set(guidance_remnant_tokens(text)) == {
        "FUN.csv",
        "build_bounded_archive_cfg",
        "discarded archive field",
        "discarded CLI alias",
        "metadata.json",
        "vamos.study-plan-result",
        "vamos.engine.algorithm.components.variation",
    }


def test_repository_has_no_active_remnants() -> None:
    root = Path(__file__).resolve().parents[1]

    assert scan(root) == []


def test_untracked_local_configs_are_ignored(tmp_path: Path) -> None:
    import subprocess

    _write(tmp_path, "src/vamos/example.py", "ENVIRONMENT_COMPATIBILITY = True\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    # A developer's local, untracked experiment config is not part of the repository.
    _write(tmp_path, "experiments/configs/zz_local_untracked.yml", "archive_type: bounded\n")

    assert scan(tmp_path) == []
