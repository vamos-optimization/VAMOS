from __future__ import annotations

from pathlib import Path

from tools.check_agent_docs import check_repository

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    rules = "\n".join(f"- Bounded repository-wide rule {index} applies to every task." for index in range(105))
    _write(
        root / "AGENTS.md",
        "# Agent contract\n\n"
        f"{rules}\n\n"
        "```agent-docs\n"
        "path: src/vamos/__init__.py\n"
        "symbol: vamos:public_symbol\n"
        "cli: vamos ok --help\n"
        "command: python tools/check_agent_docs.py\n"
        "```\n",
    )
    _write(root / ".github" / "copilot-instructions.md", "Read [/AGENTS.md](/AGENTS.md).\n")
    _write(root / "tools" / "check_agent_docs.py", "# fixture path\n")
    _write(root / "src" / "vamos" / "__init__.py", "public_symbol = object()\n")
    _write(root / "src" / "vamos" / "experiment" / "__init__.py", "")
    _write(root / "src" / "vamos" / "experiment" / "cli" / "__init__.py", "")
    _write(
        root / "src" / "vamos" / "experiment" / "cli" / "main.py",
        "from __future__ import annotations\n"
        "import sys\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(0 if sys.argv[1:] == ['ok', '--help'] else 2)\n",
    )
    return root


def test_checker_accepts_minimal_valid_instruction_tree(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    assert check_repository(root) == []


def test_checker_rejects_missing_checked_path(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    agents = root / "AGENTS.md"
    agents.write_text(agents.read_text(encoding="utf-8").replace("src/vamos/__init__.py", "src/vamos/missing.py"), encoding="utf-8")

    assert any("checked path does not exist: src/vamos/missing.py" in error for error in check_repository(root))


def test_checker_rejects_removed_public_symbol(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    (root / "src" / "vamos" / "__init__.py").write_text("another_symbol = object()\n", encoding="utf-8")

    assert any("checked public symbol is unavailable: vamos:public_symbol" in error for error in check_repository(root))


def test_checker_rejects_obsolete_artifact_token(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    agents = root / "AGENTS.md"
    agents.write_text(agents.read_text(encoding="utf-8") + "Write FUN.csv after each run.\n", encoding="utf-8")

    assert any("forbidden obsolete guidance token: FUN.csv" in error for error in check_repository(root))


def test_checker_rejects_obsolete_variation_wrapper(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    agents = root / "AGENTS.md"
    agents.write_text(
        agents.read_text(encoding="utf-8") + "Import vamos.engine.algorithm.components.variation.\n",
        encoding="utf-8",
    )

    assert any("vamos.engine.algorithm.components.variation" in error for error in check_repository(root))


def test_checker_rejects_obsolete_cli_alias(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    agents = root / "AGENTS.md"
    agents.write_text(agents.read_text(encoding="utf-8") + "Run vamos benchmark --help.\n", encoding="utf-8")

    assert any("forbidden obsolete guidance token: discarded CLI alias" in error for error in check_repository(root))


def test_checker_rejects_nested_instruction_without_inheritance(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    _write(
        root / "src" / "vamos" / "AGENTS.md",
        "# Scope\n\nApplies only to `src/vamos/**`.\n\n- Local rule.\n",
    )

    assert any("src/vamos/AGENTS.md: missing inheritance from /AGENTS.md" in error for error in check_repository(root))


def test_checker_rejects_adapter_without_root_reference(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    (root / ".github" / "copilot-instructions.md").write_text("Use the local conventions.\n", encoding="utf-8")

    assert any("adapter does not reference root AGENTS.md" in error for error in check_repository(root))


def test_checker_rejects_invalid_checked_cli_command(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    agents = root / "AGENTS.md"
    agents.write_text(agents.read_text(encoding="utf-8").replace("vamos ok --help", "vamos unknown --help"), encoding="utf-8")

    assert any("checked CLI is not recognized: vamos unknown --help" in error for error in check_repository(root))


def test_checker_rejects_duplicate_global_instruction_body(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    root_text = (root / "AGENTS.md").read_text(encoding="utf-8")
    _write(
        root / "src" / "vamos" / "AGENTS.md",
        "# Scope\n\nApplies only to `src/vamos/**`.\n\nInherits all repository-wide rules from `/AGENTS.md`.\n\n" + root_text,
    )

    assert any("duplicates the global instruction body" in error for error in check_repository(root))


def test_checker_rejects_guidance_that_reintroduces_migration(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    agents = root / "AGENTS.md"
    agents.write_text(
        agents.read_text(encoding="utf-8") + "Implement a migration reader for discarded run formats.\n",
        encoding="utf-8",
    )

    assert any("guidance reintroduces discarded pre-release behavior" in error for error in check_repository(root))


def test_checker_rejects_nonexact_replay_claim(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    agents = root / "AGENTS.md"
    agents.write_text(agents.read_text(encoding="utf-8") + "VAMOS supports tolerant replay execution.\n", encoding="utf-8")

    assert any("non-exact replay mode" in error for error in check_repository(root))


def test_health_and_ci_use_the_same_agent_checker_arguments() -> None:
    health = (ROOT / "tools" / "health.py").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert '("Agent documentation", [python, "tools/check_agent_docs.py"])' in health
    assert "run: python tools/check_agent_docs.py" in ci


def test_ignored_agents_md_is_skipped(tmp_path: Path) -> None:
    import subprocess

    root = _fixture_repo(tmp_path)
    _write(root / ".gitignore", ".venv/\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    # A package inside an ignored virtual environment ships its own AGENTS.md; it is not this repository's.
    _write(root / ".venv" / "Lib" / "site-packages" / "somepkg" / "AGENTS.md", "# Someone else's instructions\n")

    assert check_repository(root) == []
