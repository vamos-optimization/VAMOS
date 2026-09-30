# VAMOS Assist

!!! warning "Experimental workflow in VAMOS 1.0.0"
    The Assist CLI, provider integrations, generated plans, and convenience
    artifacts are Experimental. They may change in a minor release; see
    [Stability and versioning](project/stability-and-versioning.md).

Start with the [core installation](guide/installation.md). Template-first
planning needs no provider extra.

VAMOS Assist provides a no-code workflow for planning, materializing, and running experiments from the command line. You can use deterministic template-first planning or optional LLM-assisted planning. Every step writes reproducible artifacts to disk, and generated configs are validated against the ExperimentSpec v1 schema before execution.

## Quick Start (Template-First, No API Keys)

```bash
vamos assist go "template-first example" --template demo --smoke
```

This path does not require OpenAI, external API keys, or network calls.

### Example (Template-first)

```bash
vamos assist go "template-first example" --template demo --smoke --json
```

Expected JSON output (example):

```json
{
  "plan_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx",
  "plan_paths": {
    "catalog": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\catalog.json",
    "config": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\config.json",
    "plan": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\plan.json",
    "prompt": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\prompt.txt"
  },
  "project_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\project",
  "recommended_commands": [
    "vamos --config results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\project\\config.json",
    "vamos --config results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\execution_config.json"
  ],
  "run": {
    "execution_mode": "in_process",
    "exit_code": 0,
    "execution_config_path": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\execution_config.json",
    "run_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke",
    "run_report_path": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\run_report.json",
    "status": "ok"
  }
}
```

Note: `execution_mode` can be `"in_process"` or `"subprocess"` (fallback). The actual mode is recorded in `run_report.json`.

## Auto Planning with OpenAI (Optional)

Install optional dependencies:

```bash
python -m pip install "vamos-optimization[openai]==1.0.0"
```

Alternative:

```bash
python -m pip install openai
```

Set your API key:

Windows:

```bash
setx OPENAI_API_KEY "..."
```

macOS/Linux:

```bash
export OPENAI_API_KEY="..."
```

Run diagnostics:

```bash
vamos assist doctor
```

Run auto planning:

```bash
vamos assist go "I have 10 continuous variables, minimize 2 objectives..." --mode auto --provider openai --smoke --json
```

VAMOS requests structured output from the provider (strict schema) and validates the resulting config before running.

### Example (Auto planning with mock provider — no API keys)

```bash
vamos assist go "optimize a 2-objective continuous benchmark quickly" --mode auto --provider mock --smoke --json
```

Expected JSON output (example):

```json
{
  "plan_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx",
  "plan_paths": {
    "catalog": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\catalog.json",
    "config": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\config.json",
    "plan": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\plan.json",
    "prompt": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\prompt.txt"
  },
  "project_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\project",
  "recommended_commands": [
    "vamos --config results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\project\\config.json",
    "vamos --config results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\execution_config.json"
  ],
  "run": {
    "execution_mode": "in_process",
    "exit_code": 0,
    "execution_config_path": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\execution_config.json",
    "run_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke",
    "run_report_path": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\run_report.json",
    "status": "ok"
  }
}
```

### Example (Auto planning with OpenAI provider)

Install optional dependency support:

```bash
python -m pip install "vamos-optimization[openai]==1.0.0"
```

Alternative:

```bash
python -m pip install openai
```

Set your key:

Windows:

```bash
setx OPENAI_API_KEY "..."
```

macOS/Linux:

```bash
export OPENAI_API_KEY="..."
```

Run diagnostics:

```bash
vamos assist doctor --json
```

Example diagnostics snippet:

```json
{
  "openai": {
    "sdk_available": true,
    "api_key_set": true,
    "model": "gpt-5.2",
    "temperature": 0.2,
    "max_output_tokens": 900
  }
}
```

Run auto planning:

```bash
vamos assist go "I have 10 continuous variables, minimize 2 objectives..." --mode auto --provider openai --smoke --json
```

Expected JSON output (example):

```json
{
  "plan_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx",
  "plan_paths": {
    "catalog": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\catalog.json",
    "config": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\config.json",
    "plan": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\plan.json",
    "prompt": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\prompt.txt"
  },
  "project_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\project",
  "recommended_commands": [
    "vamos --config results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\project\\config.json",
    "vamos --config results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\execution_config.json"
  ],
  "run": {
    "execution_mode": "in_process",
    "exit_code": 0,
    "execution_config_path": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\execution_config.json",
    "run_dir": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke",
    "run_report_path": "results\\assist_plans\\plan_YYYYMMDD_HHMMSS_xxxxxx\\runs\\run_YYYYMMDD_HHMMSS_xxxxxx_smoke\\run_report.json",
    "status": "ok"
  }
}
```

Note: OpenAI usage is billed to the key owner (BYOK). See [Billing (BYOK)](#billing-byok). In auto mode, VAMOS sends your prompt plus compact planning context (templates/algorithms/kernels/allowed override keys), not datasets or arbitrary local files.

## Billing (BYOK)

When using `--provider openai`, OpenAI API usage is billed to the account/organization associated with the `OPENAI_API_KEY` on the machine running VAMOS.

VAMOS does not ship any API key.

## Privacy

In auto mode, VAMOS sends the user prompt and compact planning context (for example templates, algorithms, kernels, and allowed override keys, potentially truncated for size) to the provider. VAMOS does not automatically upload datasets or local files.

Plan trace files (`provider_request.json`, `provider_response.json`) are written to the plan directory and do not store secrets such as `OPENAI_API_KEY`.

## Commands

- `vamos assist catalog`: show available algorithms, kernels, operators, and templates.
- `vamos assist plan`: create a plan directory (template mode or auto mode with provider).
- `vamos assist apply`: turn a plan into a runnable project directory.
- `vamos assist run`: execute a plan/project config; supports `--smoke` and `--smoke-evals`.
- `vamos assist explain`: summarize plan/run defaults, paths, and overrides.
- `vamos assist doctor`: show environment, catalog, and provider readiness diagnostics.
- `vamos assist go`: high-level happy path (`plan -> apply -> optional smoke run`).

## Artifacts (what gets created)

Plan directory (`assist plan` / `assist go`):

- `prompt.txt`
- `catalog.json`
- `config.json`
- `plan.json`
- `provider_request.json` and `provider_response.json` (auto mode only)

Project directory (`assist apply` / `assist go`):

- `project/config.json`
- `project/README_run.md`

Run directory (`assist run` / `assist go --smoke`):

- `runs/run_*/execution_config.json`
- `runs/run_*/run_report.json`
- results under the run output root (`defaults.output_root`, typically inside the run directory)

`run_report.json` records `execution_mode` (`in_process` or `subprocess` fallback).

## Troubleshooting

- Missing OpenAI SDK: install optional deps with `python -m pip install "vamos-optimization[openai]==1.0.0"` (or `python -m pip install openai`).
- Missing API key: set `OPENAI_API_KEY` and open a new shell session if needed.
- Run diagnostics first: `vamos assist doctor`.
- Invalid auto overrides: VAMOS filters/rejects unsupported keys and records requested/applied/rejected overrides in `plan.json`.

For development installs, you can also use the same `vamos assist ...` commands after following
[Install from a source checkout](guide/installation.md#install-from-a-source-checkout).
