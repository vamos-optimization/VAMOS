# Changing the CLI

The `vamos` console entry point dispatches through `src/vamos/experiment/cli/main.py`. Base optimization parsing is split across `args.py`, `args_*.py`, `parser.py`, validation, and orchestration; focused subcommands own their parser in a focused module or subsystem.

The current canonical subcommands are `quickstart`, `create-problem`, `summarize`, `open-results`, `results`, `reproduce`, `study`, `ablation`, `assist`, `check`, `bench`, `studio`, `zoo`, `tune`, and `profile`. `vamos help` lists them; the standard optimization path uses top-level options without a subcommand.

## Workflow

1. Add or change the top-level dispatch and `_SUBCOMMANDS` help entry together.
2. Keep parsing and validation separate from execution. A command's `--help` path must not perform optimization, network access, file writes, or import an unavailable optional service.
3. Reuse `add_spec_argument(...)` for values shared with machine-readable experiment specs and keep defaults aligned with `ExperimentConfig`/algorithm config.
4. Delegate domain behavior to its owning layer; CLI modules translate inputs, render results, and return a meaningful exit status.
5. Add subprocess tests for help, valid invocation, invalid input, exit status, and outputs. Use `tmp_path` and tiny exact budgets.
6. Update [cli](../guide/cli.md) for user behavior and its docs smoke test when the command is published.

The canonical run commands have separate responsibilities:

- `vamos results inspect <run_dir>` reads a concise manifest summary.
- `vamos results verify <run_dir>` performs inert integrity/environment/component verification.
- `vamos reproduce <run_dir>` verifies, executes a supported built-in exact replay, compares arrays bitwise, and writes a new canonical run.
- `vamos study plan <study.json> [--output PATH] [--json]` resolves a durable study without creating or executing it.
- `vamos study create|run|inspect|resume|retry|summarize` delegates the current single-owner lifecycle to canonical StudyManifest services. Every JSON result uses `vamos.study-command-result` version `1.0.0`.

JSON mode reserves stdout for exactly one finite command-result document;
warnings and any explicitly requested progress remain on stderr.

Do not add an execution mode to inspection or verification. Do not infer success from a zero-byte or partial output directory.

## Required validation

```bash
vamos --help
vamos results inspect --help
vamos results verify --help
vamos reproduce --help
vamos study plan --help
vamos study create --help
vamos study run --help
vamos study inspect --help
vamos study resume --help
vamos study retry --help
vamos study summarize --help
python -m pytest -q tests/experiment/test_cli_consolidation.py tests/experiment/test_cli_config_validation.py tests/experiment/test_cli_run_artifacts.py
python -m pytest -q tests/docs/test_cli_docs_smoke.py
```

Add the affected command's focused tests and run the higher tier from `/AGENTS.md`.

```agent-docs
path: src/vamos/experiment/cli/main.py
path: src/vamos/experiment/cli/args.py
path: src/vamos/experiment/cli/parser.py
path: src/vamos/experiment/cli/orchestration.py
path: src/vamos/experiment/cli/run_artifact_cli.py
path: src/vamos/experiment/cli/study.py
path: src/vamos/experiment/cli/study_command.py
path: src/vamos/experiment/cli/study_command_result.py
path: src/vamos/experiment/cli/study_spec_io.py
path: src/vamos/experiment/cli/study_summary_output.py
path: docs/guide/cli.md
path: tests/experiment/test_cli_consolidation.py
path: tests/experiment/test_cli_config_validation.py
path: tests/experiment/test_cli_run_artifacts.py
path: tests/docs/test_cli_docs_smoke.py
symbol: vamos.experiment.cli.spec_args:add_spec_argument
cli: vamos --help
cli: vamos results inspect --help
cli: vamos results verify --help
cli: vamos reproduce --help
cli: vamos quickstart --help
cli: vamos create-problem --help
cli: vamos summarize --help
cli: vamos open-results --help
cli: vamos ablation --help
cli: vamos assist --help
cli: vamos check --help
cli: vamos bench --help
cli: vamos studio --help
cli: vamos zoo --help
cli: vamos tune --help
cli: vamos profile --help
cli: vamos --help
cli: vamos results inspect --help
cli: vamos results verify --help
cli: vamos reproduce --help
command: python -m pytest -q tests/experiment/test_cli_consolidation.py tests/experiment/test_cli_config_validation.py tests/experiment/test_cli_run_artifacts.py
command: python -m pytest -q tests/docs/test_cli_docs_smoke.py
```
