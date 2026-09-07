"""Inspect retained paper data, run a small example, or rebuild in isolation.

Run ``python paper/review.py --help`` from a source checkout. The data command
uses only the Python standard library; other dependencies are imported lazily.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATA_FILES = (
    "benchmark_paper.csv",
    "benchmark_paper_agemoea.csv",
    "benchmark_paper_moead.csv",
    "benchmark_paper_nsgaii_archive.csv",
    "benchmark_paper_nsgaii_ss.csv",
    "benchmark_paper_nsgaiii.csv",
    "benchmark_paper_smsemoa.csv",
    "benchmark_paper_spea2.csv",
    "benchmark_zcat_scalability.csv",
    "convergence_paper.csv",
    "memory_benchmark.csv",
    "scaling_vectorization.csv",
)
KEY_COLUMNS = (
    "experiment",
    "algorithm",
    "framework",
    "engine",
    "problem",
    "n_obj",
    "n_var",
    "pop_size",
    "timing_policy",
    "n_evals",
    "seed",
)
REBUILD_SCRIPTS = (
    "05_run_statistical_tests.py",
    "19_plot_convergence.py",
    "26_plot_memory_comparison.py",
    "28_plot_runtime_heatmap.py",
    "27_plot_forest_ci.py",
    "33_update_accessibility_tables.py",
)
DOCUMENTS = ("main", "supplementary", "manuscript_anonymous", "title_page")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inspect_csv(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)
    required = {"problem", "seed", "n_evals"}
    if not required.issubset(columns) or not rows:
        raise ValueError(f"{path.name}: empty data or missing columns {sorted(required - set(columns))}")
    metrics = [name for name in ("runtime_seconds", "hypervolume", "igd_plus", "peak_memory_mb") if name in columns]
    keys = [name for name in KEY_COLUMNS if name in columns]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    group_columns = [name for name in keys if name != "seed"]
    for line, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path.name}:{line}: inconsistent column count")
        for name in metrics + ["n_evals", "seed"]:
            if name in metrics and row[name] == "":
                continue
            try:
                number = float(row[name])
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line}: invalid {name}") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{path.name}:{line}: non-finite or negative {name}")
        groups[tuple(row[name] for name in group_columns)].append(row)
    duplicates = sum(count - 1 for count in Counter(tuple(row[name] for name in keys) for row in rows).values())
    if duplicates:
        raise ValueError(f"{path.name}: {duplicates} duplicate observation keys ({', '.join(keys)})")
    budgets = Counter(row["n_evals"] for row in rows)
    warnings = []
    if path.name.startswith("benchmark_") and set(budgets) != {"50000"}:
        warnings.append(f"Mixed or non-publication evaluation budgets: {dict(budgets)}; keep budgets separate.")
    empty_fields = {name: sum(not row[name] for row in rows) for name in columns}
    empty_fields = {name: count for name, count in empty_fields.items() if count}
    if empty_fields:
        warnings.append(f"Empty fields: {empty_fields}; see experiments/REFERENCE_RESULTS.md for experimental field conventions.")
    summary = []
    for key, observations in sorted(groups.items()):
        item: dict[str, Any] = {"file": path.name, **dict(zip(group_columns, key)), "observations": len(observations)}
        for metric in metrics:
            values = [float(row[metric]) for row in observations if row[metric] != ""]
            item[f"valid_{metric}"] = len(values)
            item[f"median_{metric}"] = statistics.median(values) if values else None
        summary.append(item)
    return {
        "file": f"experiments/{path.name}",
        "rows": len(rows),
        "columns": columns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "problems": len({row["problem"] for row in rows}),
        "frameworks": sorted({row.get("framework", row.get("engine", "")) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "evaluation_budgets": dict(budgets),
        "warnings": warnings,
    }, summary


def inspect_data(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    datasets, summaries = [], []
    for name in DATA_FILES:
        dataset, summary = inspect_csv(root / "experiments" / name)
        datasets.append(dataset)
        summaries.extend(summary)
    return {"datasets": datasets, "scope": "Retained CSV structure and descriptive summaries; no optimization executed."}, summaries


def save_data_report(output: Path, report: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    write_json(output / "data-report.json", report)
    columns = list(dict.fromkeys(name for item in summaries for name in item))
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)


def run_smoke(root: Path, output: Path) -> None:
    if not shutil.which("git"):
        raise RuntimeError("Exact replay needs Git and a Git clone of VAMOS; see paper/README.md.")
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True, check=False)
    if revision.returncode != 0:
        raise RuntimeError("Git cannot read this checkout's revision. Use a Git clone and check that git status succeeds in it.")
    import numpy as np

    import vamos

    module_path = Path(vamos.__file__).resolve()
    if not module_path.is_relative_to(root / "src"):
        raise RuntimeError("VAMOS imports from another checkout. Set PYTHONPATH to this checkout's src directory.")
    result = vamos.optimize("zdt1", algorithm="nsgaii", pop_size=40, max_evaluations=200, engine="numpy", seed=42)
    if result.data.get("evaluations") != 200:
        raise RuntimeError("Smoke run did not honor the 200-evaluation budget.")
    if result.F is None or result.F.shape[1] != 2 or not np.isfinite(result.F).all():
        raise RuntimeError("Smoke run did not produce a finite two-objective front.")
    stored = vamos.save_result(result, output / "run")
    loaded = vamos.load_result(stored.root)
    np.testing.assert_array_equal(result.F, loaded.F)
    np.testing.assert_array_equal(result.X, loaded.X)
    verification = vamos.verify_run(stored.root)
    write_json(output / "verification.json", verification.as_dict())
    if verification.artifact_integrity != "valid" or verification.effective_replayability != "exact":
        raise RuntimeError(f"Smoke artifact did not pass verification for exact replay; inspect {output / 'verification.json'}.")
    replay = vamos.reproduce(stored.root, output=output / "replay")
    if not replay.exact:
        raise RuntimeError("Smoke replay did not match exactly.")
    write_json(
        output / "smoke-report.json",
        {
            "problem": "zdt1",
            "algorithm": "nsgaii",
            "engine": "numpy",
            "seed": 42,
            "max_evaluations": 200,
            "verification": verification.as_dict(),
            "replay": replay.as_dict(),
        },
    )
    print("PASS: ZDT1 / NSGA-II / NumPy, 200 evaluations; save, load, verify and exact replay.")


def stage_rebuild(root: Path, output: Path) -> Path:
    workspace = output / "workspace"
    sources = [root / "experiments" / name for name in DATA_FILES]
    sources += [root / "paper" / name for name in REBUILD_SCRIPTS]
    sources += [root / "paper" / "accessibility_proxy_snippets.json"]
    for name in (*DOCUMENTS, "revision_marks"):
        sources.append(root / "paper" / "manuscript" / f"{name}.tex")
    sources.append(root / "paper" / "manuscript" / "references.bib")
    for name in ("architecture_v2.png", "Representation.png", "PROTOCOL.png", "workflow_v3.png"):
        sources.append(root / "paper" / "manuscript" / "figures" / name)
    for source in sources:
        target = workspace / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(
        output / "source-hashes.json",
        {source.relative_to(root).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
    )
    return workspace


def rebuild(root: Path, output: Path, *, pdf: bool) -> None:
    workspace = stage_rebuild(root, output)
    env = {key: value for key, value in os.environ.items() if not key.startswith("VAMOS_PAPER_")}
    env.update(
        VAMOS_PAPER_ALGORITHM="all", VAMOS_PAPER_UPDATE_MAIN_TEX="0", VAMOS_PAPER_OVERWRITE="0", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8"
    )
    commands = [[sys.executable, str(workspace / "paper" / name)] for name in REBUILD_SCRIPTS]
    manuscript = workspace / "paper" / "manuscript"
    if pdf:
        commands += [
            ["latexmk", "-pdf", f"-outdir={output / 'pdf' / name}", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]
            for name in DOCUMENTS
        ]
    for index, command in enumerate(commands, start=1):
        name = Path(command[1]).stem if command[0] == sys.executable else Path(command[-1]).stem
        log_path = output / f"{index:02d}-{name}.log"
        print(f"Running {name}; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=manuscript if command[0] == "latexmk" else workspace, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    generated = workspace / "paper" / "generated"
    expected = [generated / "tables" / f"stats_{name}.tex" for name in ("nsgaii", "smsemoa", "moead")]
    expected += [generated / "tables" / "accessibility_proxy_details.tex"]
    expected += [
        generated / "figures" / f"{name}.png"
        for name in (
            "convergence",
            "memory_comparison",
            "runtime_heatmap",
            "forest_ci_nsgaii",
            "forest_ci_smsemoa",
            "forest_ci_moead",
        )
    ]
    if pdf:
        expected += [output / "pdf" / name / f"{name}.pdf" for name in DOCUMENTS]
    for artifact in expected:
        if not artifact.is_file() or not artifact.stat().st_size:
            raise RuntimeError(f"Rebuild did not produce {artifact}")
    write_json(
        output / "rebuild-report.json",
        {
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
            "outputs": [path.relative_to(output).as_posix() for path in expected],
            "scope": "Recomputed supplementary analyses from retained CSVs; main article tables preserved as submitted.",
        },
    )
    print(f"PASS: rebuild completed. Generated tables and figures: {generated}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("data", "smoke", "rebuild"))
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing paths are refused.")
    parser.add_argument("--pdf", action="store_true", help="Also compile four PDFs during rebuild; requires latexmk and TeX.")
    args = parser.parse_args(argv)
    if args.pdf and args.command != "rebuild":
        parser.error("--pdf requires rebuild")
    root = Path(__file__).resolve().parents[1]
    try:
        report, summaries = inspect_data(root)
        if args.pdf and not shutil.which("latexmk"):
            raise RuntimeError("Install latexmk and a TeX distribution before using --pdf; see paper/README.md.")
        output = args.output_dir
        if output is None and args.command != "data":
            output = root / "paper" / "generated" / f"reviewer-{args.command}"
        if output is not None:
            output = output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}; choose a new --output-dir.")
            output.mkdir(parents=True, exist_ok=False)
            save_data_report(output, report, summaries)
        for dataset in report["datasets"]:
            print(f"{dataset['file']}: {dataset['rows']} rows, {dataset['problems']} problems, {len(dataset['frameworks'])} frameworks")
            for warning in dataset["warnings"]:
                print(f"  NOTE: {warning}")
        if args.command == "smoke" and output is not None:
            run_smoke(root, output)
        elif args.command == "rebuild" and output is not None:
            rebuild(root, output, pdf=args.pdf)
        if output is not None:
            print(f"Reports: {output}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
