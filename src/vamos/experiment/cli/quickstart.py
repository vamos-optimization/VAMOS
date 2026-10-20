from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from vamos.engine.algorithm.catalog import ENABLED_ALGORITHMS
from vamos.engine.config.spec import EXPERIMENT_SPEC_VERSION
from vamos.experiment.presentation import run_experiments_from_args
from vamos.experiment.runner_utils import problem_output_dir, run_output_dir
from vamos.experiment.runtime.catalog import EXPERIMENT_BACKENDS, resolve_engine
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.io_utils import ensure_dir
from vamos.foundation.problem.registry import available_problem_names, make_problem_selection

from .parser import parse_args
from .preflight import run_preflight_checks
from .quickstart_templates import (
    QuickstartTemplate,
    extra_hint,
    get_template,
    list_templates,
    missing_extras,
    template_keys,
)

DEFAULT_QUICKSTART_OUTPUT_ROOT = "results/quickstart"


def _print_list(label: str, items: list[str], *, width: int = 4) -> None:
    print(label)
    for idx in range(0, len(items), width):
        chunk = items[idx : idx + width]
        print("  " + "  ".join(chunk))


def _print_templates() -> None:
    print("Quickstart templates:")
    for template_key in available_templates():
        template = get_template(template_key)
        req = f" (requires: {', '.join(template.requires)})" if template.requires else ""
        print(f"  {template.key}: {template.label}{req}")
        print(f"    {template.description}")


def available_templates() -> list[str]:
    """Return the available quickstart template keys in deterministic order."""
    return sorted(template.key for template in list_templates())


def generate_quickstart_config(template: str, overrides: dict[str, object] | None = None) -> dict[str, object]:
    """Generate an ExperimentSpec v1 mapping from a quickstart template."""
    template_key = template.strip().lower()
    templates = available_templates()
    if template_key not in templates:
        available = ", ".join(templates)
        raise ValueError(f"Unknown template '{template}'. Available: {available}.")
    selected_template = get_template(template_key)
    defaults = selected_template.defaults
    spec_defaults: dict[str, object] = {
        "title": f"Quickstart: {selected_template.label}",
        "problem": defaults.problem,
        "algorithm": defaults.algorithm,
        "engine": defaults.engine,
        "max_evaluations": defaults.budget,
        "population_size": defaults.pop_size,
        "seed": defaults.seed,
        "output_root": DEFAULT_QUICKSTART_OUTPUT_ROOT,
        "plot": defaults.plot,
    }

    safe_keys = {
        "title",
        "problem",
        "algorithm",
        "engine",
        "max_evaluations",
        "population_size",
        "seed",
        "output_root",
        "plot",
    }
    if overrides:
        for key, value in overrides.items():
            if key == "defaults":
                if not isinstance(value, Mapping):
                    raise TypeError("Quickstart overrides['defaults'] must be a mapping when provided.")
                for nested_key, nested_value in value.items():
                    if nested_key not in safe_keys:
                        raise ValueError(f"Unsupported quickstart override key: defaults.{nested_key}")
                    spec_defaults[nested_key] = nested_value
                continue
            if key not in safe_keys:
                raise ValueError(f"Unsupported quickstart override key: {key}")
            spec_defaults[key] = value

    spec: dict[str, object] = {
        "version": EXPERIMENT_SPEC_VERSION,
        "defaults": spec_defaults,
    }
    return spec


def _prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def _prompt_choice(prompt: str, default: str, choices: set[str], *, allow_list: bool = False) -> str:
    while True:
        raw = _prompt_text(prompt, default).strip()
        if allow_list and raw.lower() == "list":
            _print_list("Available options:", sorted(choices))
            continue
        value = raw.lower()
        if value in choices:
            return value
        print(f"Please choose one of: {', '.join(sorted(choices))}.")


def _prompt_int(prompt: str, default: int, *, min_value: int = 1) -> int:
    while True:
        raw = _prompt_text(prompt, str(default)).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return value


def _resolve_output_root(raw: str | None) -> str:
    value = (raw or "").strip()
    if value:
        return value
    return DEFAULT_QUICKSTART_OUTPUT_ROOT


def _pick_problem(problem: str | None, *, default: str, interactive: bool) -> str:
    available = sorted(available_problem_names())
    if problem:
        return problem.lower()
    if not interactive:
        return default
    _print_list("Popular problems:", available[:12])
    return _prompt_choice("Problem (type 'list' for all)", default, set(available), allow_list=True)


def _pick_algorithm(algorithm: str | None, *, default: str, interactive: bool) -> str:
    choices = set(ENABLED_ALGORITHMS)
    if algorithm:
        return algorithm.lower()
    if not interactive:
        return default
    return _prompt_choice("Algorithm (type 'list' for all)", default, choices, allow_list=True)


def _pick_engine(engine: str | None, *, default: str, interactive: bool) -> str:
    choices = {*EXPERIMENT_BACKENDS, "auto"}
    if engine:
        return engine.lower()
    if not interactive:
        return default
    return _prompt_choice("Engine (type 'list' for all)", default, choices, allow_list=True)


def _pick_budget(budget: int | None, *, default: int, interactive: bool) -> int:
    if budget is not None and budget > 0:
        return budget
    if not interactive:
        return default
    return _prompt_int("Budget (max evaluations)", default, min_value=1)


def _pick_pop_size(pop_size: int | None, *, default: int, interactive: bool) -> int:
    if pop_size is not None and pop_size > 0:
        return pop_size
    if not interactive:
        return default
    return _prompt_int("Population size", default, min_value=1)


def _pick_seed(seed: int | None, *, default: int, interactive: bool) -> int:
    if seed is not None and seed >= 0:
        return seed
    if not interactive:
        return default
    return _prompt_int("Random seed", default, min_value=0)


def _pick_plot(plot: bool, *, interactive: bool) -> bool:
    if not interactive:
        return plot
    default = "y" if plot else "n"
    choice = _prompt_text("Save a Pareto front plot? (y/n)", default).strip().lower()
    if choice in {"n", "no"}:
        return False
    return True


def _write_spec(
    *,
    title: str,
    problem: str,
    algorithm: str,
    engine: str,
    budget: int,
    pop_size: int,
    seed: int,
    output_root: str,
    plot: bool,
    config_path: str | None,
) -> Path:
    spec: dict[str, object] = {
        "version": EXPERIMENT_SPEC_VERSION,
        "defaults": {
            "title": title,
            "problem": problem,
            "algorithm": algorithm,
            "engine": engine,
            "max_evaluations": budget,
            "population_size": pop_size,
            "seed": seed,
            "output_root": output_root,
            "plot": plot,
        },
    }
    return _write_spec_data(spec=spec, output_root=output_root, config_path=config_path)


def _write_spec_data(*, spec: Mapping[str, object], output_root: str, config_path: str | None) -> Path:
    ensure_dir(output_root)
    if config_path:
        path = Path(config_path)
        ensure_dir(path.parent)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(output_root) / f"quickstart_{timestamp}.json"
    path.write_text(json.dumps(dict(spec), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _latest_plot_path(problem_dir: str) -> str | None:
    paths = sorted(Path(problem_dir).glob("pareto_front_*.png"))
    if not paths:
        return None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _summary(
    *,
    output_dir: str,
    config_path: Path,
    problem_dir: str,
    plot_path: str | None,
    template: QuickstartTemplate,
) -> None:
    print("\nQuickstart complete.")
    print(f"Template: {template.label}")
    print(f"Results folder: {output_dir}")
    print(f"Config saved: {config_path}")
    print(f"Run manifest: {Path(output_dir) / 'manifest.json'}")
    if plot_path:
        print(f"Plot: {plot_path}")
    else:
        print("Plot: not created (install the 'analysis' extra to enable plotting).")
    print(f"Re-run: vamos --config {config_path}")
    print(f"Browse outputs: {problem_dir}")


def _select_template(key: str | None, *, interactive: bool) -> QuickstartTemplate:
    if key and key.lower() == "list":
        _print_templates()
        raise SystemExit(0)

    while True:
        template_key = key
        if template_key is None and interactive:
            template_key = _prompt_choice(
                "Template (type 'list' for all)",
                "demo",
                template_keys(),
                allow_list=True,
            )
        if template_key is None:
            template_key = "demo"
        template = get_template(template_key)
        missing = missing_extras(template)
        if not missing:
            return template
        print(f"Template '{template.key}' requires extras: {', '.join(missing)}.")
        for extra in missing:
            hint = extra_hint(extra)
            if hint:
                print(f"Install with: {hint}")
        if not interactive:
            raise SystemExit(1)
        key = None


def run_quickstart(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vamos quickstart",
        description="Guided quickstart for a single VAMOS run (writes config + outputs).",
    )
    parser.add_argument("--template", help="Template key (use --template list to see options).")
    parser.add_argument("--problem", help="Problem name (default: zdt1).")
    parser.add_argument("--algorithm", help="Algorithm name (default: nsgaii).")
    parser.add_argument(
        "--engine",
        help="Backend engine. Default is deterministic numpy. Use 'auto' to prefer available Numba, with a NumPy fallback.",
    )
    parser.add_argument("--budget", "--max-evaluations", dest="budget", type=int, help="Max evaluations (default: 5000).")
    parser.add_argument("--pop-size", "--population-size", dest="pop_size", type=int, help="Population size (default: 100).")
    parser.add_argument("--seed", type=int, help="Random seed (default: 42).")
    parser.add_argument("--output-root", help="Output root directory (default: results/quickstart).")
    parser.add_argument("--config-path", help="Write the generated config to this path.")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving a Pareto front plot.")
    parser.add_argument("--no-preflight", action="store_true", help="Skip optional dependency warnings.")
    parser.add_argument("--yes", action="store_true", help="Accept defaults without prompting.")
    args = parser.parse_args(argv)

    interactive = sys.stdin.isatty() and not args.yes
    if not sys.stdin.isatty() and not args.yes:
        print("Non-interactive shell detected; using defaults.")

    template = _select_template(args.template, interactive=interactive)
    defaults = template.defaults
    problem = _pick_problem(args.problem, default=defaults.problem, interactive=interactive)
    algorithm = _pick_algorithm(args.algorithm, default=defaults.algorithm, interactive=interactive)
    engine = _pick_engine(args.engine, default=defaults.engine, interactive=interactive)
    budget = _pick_budget(args.budget, default=defaults.budget, interactive=interactive)
    pop_size = _pick_pop_size(args.pop_size, default=defaults.pop_size, interactive=interactive)
    seed = _pick_seed(args.seed, default=defaults.seed, interactive=interactive)
    output_root = _resolve_output_root(args.output_root)
    plot_default = defaults.plot if not args.no_plot else False
    plot = _pick_plot(plot_default, interactive=interactive)

    spec = generate_quickstart_config(
        template.key,
        overrides={
            "title": f"Quickstart: {template.label}",
            "problem": problem,
            "algorithm": algorithm,
            "engine": engine,
            "max_evaluations": budget,
            "population_size": pop_size,
            "seed": seed,
            "output_root": output_root,
            "plot": plot,
        },
    )
    config_path = _write_spec_data(spec=spec, output_root=output_root, config_path=args.config_path)

    default_config = ExperimentConfig()
    parse_argv = ["--config", str(config_path)]
    if args.no_preflight:
        parse_argv.append("--no-preflight")
    parsed_args = parse_args(default_config, argv=parse_argv)
    config = ExperimentConfig(
        title=default_config.title,
        output_root=parsed_args.output_root,
        population_size=parsed_args.population_size,
        offspring_population_size=parsed_args.offspring_population_size,
        max_evaluations=parsed_args.max_evaluations,
        seed=parsed_args.seed,
        eval_strategy=parsed_args.eval_strategy,
        n_workers=parsed_args.n_workers,
        live_viz=parsed_args.live_viz,
        live_viz_interval=parsed_args.live_viz_interval,
        live_viz_max_points=parsed_args.live_viz_max_points,
    )
    if not getattr(parsed_args, "no_preflight", False):
        run_preflight_checks(parsed_args)
    try:
        run_experiments_from_args(parsed_args, config)
    except ImportError as exc:
        print(f"Quickstart failed: {exc}")
        print("Install missing extras and try again.")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Quickstart failed: {exc}")
        raise SystemExit(1) from exc

    selection = make_problem_selection(problem)
    resolved_engine = resolve_engine(engine, algorithm=algorithm)
    output_dir = run_output_dir(selection, algorithm, resolved_engine, seed, config)
    problem_dir = problem_output_dir(selection, config)
    plot_path = _latest_plot_path(problem_dir) if plot else None
    _summary(
        output_dir=output_dir,
        config_path=config_path,
        problem_dir=problem_dir,
        plot_path=plot_path,
        template=template,
    )


__all__ = ["run_quickstart", "list_templates", "get_template", "available_templates", "generate_quickstart_config"]
