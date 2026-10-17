"""Generate or check T06–T08 downloads, observations, report and figures."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.metadata
import io
import json
import runpy
import shutil
import subprocess
import textwrap
from pathlib import Path

from build_fundamental_tutorials import digest, replace_block

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "examples/tutorials/study_workflow.py"
ASSETS = "docs/assets/studies"
PACKAGES = {
    "vamos-optimization": "1.0.1",
    "numpy": "2.1.3",
    "scipy": "1.15.3",
    "joblib": "1.4.2",
    "moocore": "0.3.2",
    "matplotlib": "3.10.9",
}
LESSONS = {
    "t06": ("t06-fair-study", ("study_spec", "plan_or_execute_study", "main_t06")),
    "t07": ("t07-quality-and-budget", ("metric_policy", "measure_front", "measure_studies", "main_t07")),
    "t08": (
        "t08-study-report",
        (
            "metric_policy",
            "measure_front",
            "summarize_measurements",
            "save_tutorial_figure",
            "draw_study_figures",
            "render_report",
            "main_t08",
        ),
    ),
}


def download_source(source: str, lesson: str) -> str:
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    functions = [ast.get_source_segment(source, nodes[name]) for name in LESSONS[lesson][1]]
    return (
        f'"""{lesson.upper()}: generated from {SOURCE}; see study-setup for dependencies."""\n\n\n'
        + "\n\n\n".join(functions)
        + f'\n\n\nif __name__ == "__main__":\n    main_{lesson}()\n'
    )


def code_block(code: str) -> str:
    return '??? example "Complete runnable script"\n\n' + textwrap.indent(f"```python\n{code.rstrip()}\n```", "    ")


def capture(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def requirements() -> str:
    return "".join(f"{name}=={version}\n" for name, version in PACKAGES.items())


def draw_execution(dataset: dict, path: Path, save_figure) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = sorted({(row["problem"], row["algorithm"]) for row in dataset["rows"]})
    with plt.rc_context({"svg.hashsalt": "vamos-study-execution", "svg.fonttype": "none", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        bottom = np.zeros(len(groups))
        for seed in (0, 1):
            values = [
                next(row["evaluations"] for row in dataset["rows"] if (row["problem"], row["algorithm"], row["seed"]) == (*key, seed))
                for key in groups
            ]
            ax.bar(range(len(groups)), values, bottom=bottom, label=f"Seed {seed}")
            bottom += values
        ax.set_xticks(range(len(groups)), [f"{p.upper()}\n{a}" for p, a in groups])
        ax.set(ylabel="Actual objective evaluations", title="T06 · Eight persisted runs, 1,600 evaluations")
        ax.legend()
        save_figure(fig, path)
        plt.close(fig)


def write(root: Path, work: Path) -> None:
    import vamos

    if Path(vamos.__file__).resolve().is_relative_to(root.resolve()):
        raise ValueError("Generate with the installed PyPI wheel, without PYTHONPATH or an editable checkout.")
    actual = {name: importlib.metadata.version(name) for name in PACKAGES}
    if actual != PACKAGES:
        raise ValueError(f"Use the declared versions: {PACKAGES}; found {actual}")
    work.mkdir(parents=True, exist_ok=False)
    source = (root / SOURCE).read_text(encoding="utf-8")
    assets = root / ASSETS
    assets.mkdir(parents=True, exist_ok=True)
    modules, codes = {}, {}
    for lesson, (slug, _) in LESSONS.items():
        code = download_source(source, lesson)
        target = assets / f"{slug}.py"
        target.write_text(code, encoding="utf-8")
        modules[lesson] = runpy.run_path(str(target))
        codes[lesson] = code
    _, out06 = capture(modules["t06"]["plan_or_execute_study"], work / "smoke-study", execute=True)
    smoke, out07 = capture(modules["t07"]["measure_studies"], [work / "smoke-study"], work / "smoke-metrics")
    _, out08 = capture(modules["t08"]["render_report"], work / "smoke-metrics/measurements.json", work / "smoke-report")
    draw_execution(smoke, assets / "study-execution.svg", modules["t08"]["save_tutorial_figure"])
    recorded_outputs, study_paths = [], []
    for budget in (200, 800, 2000):
        path = work / f"recorded-{budget}"
        print(f"Recording 40 tasks at {budget} evaluations each...", flush=True)
        _, stdout = capture(modules["t06"]["plan_or_execute_study"], path, profile="recorded", budget=budget, execute=True)
        study_paths.append(path)
        recorded_outputs.append(stdout)
    dataset, measured_output = capture(modules["t07"]["measure_studies"], study_paths, work / "recorded-metrics")
    dataset["documentation_provenance"] = {
        "canonical_source": SOURCE,
        "source_sha256": digest(source),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "command": "python tools/build_study_tutorials.py --write --work-dir <new-directory>",
        "measurement_script_sha256": digest(codes["t07"]),
        "retention": "Objective fronts and scalar task observations only; full canonical studies remain in the generation work directory.",
    }
    retained = assets / "recorded-measurements.json"
    retained.write_text(json.dumps(dataset, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary, report_output = capture(modules["t08"]["render_report"], retained, work / "recorded-report")
    for filename in ("metrics.csv", "summary.csv", "budget-comparison.svg", "final-distributions.svg"):
        shutil.copyfile(work / "recorded-report" / filename, assets / filename)
    report = (work / "recorded-report/report.md").read_text(encoding="utf-8")
    report = report.replace("# Descriptive VAMOS study report", "# Recorded study report")
    report = report.replace("(budget-comparison.svg)", "(../assets/studies/budget-comparison.svg)")
    report = report.replace("(final-distributions.svg)", "(../assets/studies/final-distributions.svg)")
    report += (
        "\n[Download the observations](../assets/studies/recorded-measurements.json) · "
        "[Task metrics](../assets/studies/metrics.csv) · [Grouped summaries](../assets/studies/summary.csv)\n\n"
        "Generated from the recorded protocol in [Study setup](study-setup.md#the-recorded-experiment). "
        "Follow [T08](t08-study-report.md) to rebuild this report without rerunning optimization.\n"
    )
    report_path = root / "docs/tutorials/recorded-study-report.md"
    report_path.write_text(report, encoding="utf-8")
    outputs = {"t06": out06, "t07": out07, "t08": out08}
    for lesson, (slug, _) in LESSONS.items():
        path = root / "docs/tutorials" / f"{slug}.md"
        page = replace_block(path.read_text(encoding="utf-8"), "code", code_block(codes[lesson]))
        page = replace_block(page, "output", f"```text\n{outputs[lesson].rstrip()}\n```")
        path.write_text(page, encoding="utf-8")
    (assets / "study-requirements.txt").write_text(requirements(), encoding="utf-8")
    owned = [f"{slug}.py" for slug, _ in LESSONS.values()] + [
        "recorded-measurements.json",
        "metrics.csv",
        "summary.csv",
        "budget-comparison.svg",
        "final-distributions.svg",
        "study-requirements.txt",
        "study-execution.svg",
    ]
    evidence = {
        "schema": "vamos.documentation.study-tutorial-evidence.v1",
        "source": SOURCE,
        "source_sha256": digest(source),
        "base_commit": dataset["documentation_provenance"]["base_commit"],
        "command": dataset["documentation_provenance"]["command"],
        "packages": actual,
        "python": dataset["python"],
        "outputs": outputs,
        "recorded_execution": recorded_outputs,
        "recorded_measurement": measured_output,
        "recorded_report": report_output,
        "planned_evaluations": 120000,
        "size_budget_bytes": {"recorded-measurements.json": 750000, "other_assets": 250000},
        "assets": {name: digest((assets / name).read_text(encoding="utf-8")) for name in owned},
        "report_sha256": digest(report),
        "final_budget_summary": [g for g in summary["groups"] if g["budget"] == summary["final_budget"]],
    }
    (assets / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def check(root: Path) -> None:
    source = (root / SOURCE).read_text(encoding="utf-8")
    assets = root / ASSETS
    evidence = json.loads((assets / "evidence.json").read_text(encoding="utf-8"))
    if digest(source) != evidence["source_sha256"]:
        raise ValueError("Study tutorial source changed: regenerate the owned evidence.")
    if (assets / "study-requirements.txt").read_text(encoding="utf-8") != requirements():
        raise ValueError("Study tutorial dependency recipe is stale.")
    for filename, expected in evidence["assets"].items():
        path = assets / filename
        if digest(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Changed study asset: {filename}")
        if path.stat().st_size > evidence["size_budget_bytes"].get(filename, evidence["size_budget_bytes"]["other_assets"]):
            raise ValueError(f"Oversized study asset: {filename}")
    for lesson, (slug, _) in LESSONS.items():
        code = download_source(source, lesson)
        if (assets / f"{slug}.py").read_text(encoding="utf-8") != code:
            raise ValueError(f"Stale {lesson} download")
        page = (root / "docs/tutorials" / f"{slug}.md").read_text(encoding="utf-8")
        expected = replace_block(page, "code", code_block(code))
        expected = replace_block(expected, "output", "```text\n" + evidence["outputs"][lesson].rstrip() + "\n```")
        if page != expected:
            raise ValueError(f"Stale {lesson} content")
    if digest((root / "docs/tutorials/recorded-study-report.md").read_text(encoding="utf-8")) != evidence["report_sha256"]:
        raise ValueError("Recorded report was edited outside its generator.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--work-dir", type=Path, help="New external/ignored directory; preserves full generated studies")
    args = parser.parse_args()
    if args.write:
        if args.work_dir is None:
            parser.error("--write requires --work-dir pointing to a new directory")
        write(ROOT, args.work_dir)
    check(ROOT)
    print("Study tutorial downloads, observations, report and figures are synchronized.")


if __name__ == "__main__":
    main()
