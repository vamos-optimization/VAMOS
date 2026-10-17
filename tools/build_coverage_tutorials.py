"""Generate/check verified algorithm, encoding, application and advanced lessons."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.metadata
import io
import json
import platform
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

from build_fundamental_tutorials import digest, replace_block

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("examples/tutorials/capability_examples.py", "examples/tutorials/advanced_workflows.py")
ASSETS = "docs/assets/coverage"
PACKAGES = {
    "vamos-optimization": "1.0.1",
    "numpy": "2.1.3",
    "scipy": "1.15.3",
    "joblib": "1.4.2",
    "moocore": "0.3.2",
    "matplotlib": "3.10.9",
    "scikit-learn": "1.7.2",
    "numba": "0.67.0",
}
ALGORITHMS = ("smsemoa", "spea2", "ibea", "smpso", "agemoea", "rvea")
ENCODINGS = ("binary", "integer", "permutation", "mixed")
APPLICATIONS = {"engineering": "engineering-design", "feature_selection": "feature-selection", "model_tuning": "model-hyperparameters"}
ADVANCED = {"tuning": "tuning-experiment", "ablation": "controlled-ablation", "performance": "measure-performance"}
LESSONS = {
    **{name: (f"docs/algorithms/{name}.md", "") for name in ALGORITHMS},
    **{name: ("docs/tutorials/encodings.md", f"-{name}") for name in ENCODINGS},
    **{name: (f"docs/tutorials/{slug}.md", "") for name, slug in APPLICATIONS.items()},
    **{name: (f"docs/tutorials/{slug}.md", "") for name, slug in ADVANCED.items()},
}


def download_source(sources: dict, lesson: str) -> str:
    source = sources[SOURCES[int(lesson in ADVANCED)]]
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    if lesson in ADVANCED:
        names = ("new_output", "write_observations", f"run_{lesson}_example", "advanced_cli")
        entry = f"advanced_cli(run_{lesson}_example)"
        wrapper = ""
    else:
        names = ("execute_example", "run_encoding" if lesson in ENCODINGS else f"run_{lesson}", "plot_example", "lesson_cli")
        wrapper = f'\n\n\ndef run_lesson(budget=400):\n    return run_encoding("{lesson}", budget)\n' if lesson in ENCODINGS else ""
        function = "run_lesson" if lesson in ENCODINGS else f"run_{lesson}"
        entry = f"lesson_cli({function}, default_budget={80 if lesson in ('feature_selection', 'model_tuning') else 400})"
    code = "\n\n\n".join(ast.get_source_segment(source, nodes[name]) for name in names)
    return f'"""Verified VAMOS 1.0.1 lesson: {lesson}. See coverage-setup for dependencies."""\n\n\n{code}{wrapper}\n\n\nif __name__ == "__main__":\n    {entry}\n'


def code_block(code):
    return '??? example "Complete runnable script"\n\n' + textwrap.indent(f"```python\n{code.rstrip()}\n```", "    ")


def draw_advanced(name, data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"svg.hashsalt": "vamos-coverage-advanced", "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        if name == "tuning":
            trials = data["trials"]
            ax.bar([str(row["trial_id"]) for row in trials], [row["score"] for row in trials], color="#007f7a")
            ax.set(xlabel="Random-search trial", ylabel="Training HV (higher is better)", title="Two trials · one training seed · ZDT1")
        elif name == "ablation":
            for index, variant in enumerate(("baseline", "without-mutation")):
                rows = [row for row in data["rows"] if row["variant"] == variant]
                ax.scatter([index] * len(rows), [row["hv"] for row in rows], c="#007f7a", s=40)
                for row, offset in zip(rows, ((7, 0), (-22, 6), (7, -10))):
                    ax.annotate(str(row["seed"]), (index, row["hv"]), xytext=offset, textcoords="offset points")
            ax.set_xticks([0, 1], ["Baseline", "Without mutation"])
            ax.set(xlim=(-0.6, 1.6), ylabel="HV (higher is better)", title="One component removed · three seeds · ZDT1")
        else:
            backends = list(dict.fromkeys(row["backend"] for row in data["rows"]))
            for index, backend in enumerate(backends):
                for phase, marker, color in (("first", "D", "#bc6c25"), ("subsequent", "o", "#007f7a")):
                    rows = [r for r in data["rows"] if r["backend"] == backend and r["phase"] == phase]
                    ax.scatter(
                        [index] * len(rows), [r["seconds"] for r in rows], marker=marker, c=color, label=phase if index == 0 else None
                    )
            ax.set_xticks(range(len(backends)), backends)
            ax.set(yscale="log", ylabel="Seconds per optimize() call (log scale)", title="Local measurements · 400 evaluations per call")
            ax.legend()
        fig.savefig(path, metadata={"Date": None}, bbox_inches="tight")
        plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")


def write(root: Path, work: Path):
    actual = {name: importlib.metadata.version(name) for name in PACKAGES}
    if actual != PACKAGES:
        raise ValueError(f"Use the recorded dependency versions: {PACKAGES}; installed: {actual}")
    work.mkdir(parents=True, exist_ok=False)
    assets = root / ASSETS
    assets.mkdir(parents=True, exist_ok=True)
    sources = {name: (root / name).read_text(encoding="utf-8") for name in SOURCES}
    outputs, observations = {}, {}
    for lesson, (page_path, suffix) in LESSONS.items():
        print(f"Recording {lesson}...", flush=True)
        code = download_source(sources, lesson)
        script = work / f"{lesson}.py"
        script.write_text(code, encoding="utf-8")
        module = runpy.run_path(str(script))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            if lesson in ADVANCED:
                if lesson == "performance":
                    completed = subprocess.run(
                        [sys.executable, str(script), "--output", str(work / lesson)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(completed.stdout, end="")
                    data = json.loads((work / lesson / "observations.json").read_text(encoding="utf-8"))
                else:
                    data = module[f"run_{lesson}_example"](work / lesson)
                draw_advanced(lesson, data, assets / f"{lesson}.svg")
                observations[lesson] = data
            else:
                function = module["run_lesson" if lesson in ENCODINGS else f"run_{lesson}"]
                data = function()
                figure = work / f"{lesson}.svg"
                module["plot_example"](data, figure)
                (assets / figure.name).write_bytes(figure.read_bytes())
                result = data["result"]
                observations[lesson] = {
                    "algorithm": data["algorithm"],
                    "configuration": data["config"].to_dict(),
                    "problem": type(data["problem"]).__name__,
                    "encoding": data["problem"].encoding,
                    "seed": 42,
                    "backend": "numpy",
                    "evaluations": int(result.data["evaluations"]),
                    "X": result.X.tolist(),
                    "F": result.F.tolist(),
                    "G": None if result.data.get("G") is None else result.data["G"].tolist(),
                    "front_rows": data["front_rows"].tolist(),
                    "selected_row": data.get("selected_row"),
                }
        outputs[lesson] = buffer.getvalue()
        (assets / script.name).write_text(code, encoding="utf-8")
        page = root / page_path
        content = replace_block(page.read_text(encoding="utf-8"), f"code{suffix}", code_block(code))
        content = replace_block(content, f"output{suffix}", f"```text\n{outputs[lesson].rstrip()}\n```")
        page.write_text(content, encoding="utf-8")
    (assets / "observations.json").write_text(json.dumps(observations, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (assets / "requirements.txt").write_text("".join(f"{n}=={v}\n" for n, v in PACKAGES.items()), encoding="utf-8")
    owned = [f"{name}.{ext}" for name in LESSONS for ext in ("py", "svg")] + ["observations.json", "requirements.txt"]
    evidence = {
        "schema": "vamos.documentation.coverage-evidence.v1",
        "python": platform.python_version(),
        "packages": actual,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "command": "python tools/build_coverage_tutorials.py --write --work-dir <new-directory>",
        "retention": "Derived teaching arrays and observations; canonical ablation studies and original tuner artifacts remain in the work directory.",
        "sources": {name: digest(content) for name, content in sources.items()},
        "outputs": outputs,
        "assets": {name: digest((assets / name).read_text(encoding="utf-8")) for name in owned},
        "size_budget_bytes": {"observations.json": 750000, "other": 250000},
    }
    (assets / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def check(root: Path):
    assets = root / ASSETS
    evidence = json.loads((assets / "evidence.json").read_text(encoding="utf-8"))
    sources = {name: (root / name).read_text(encoding="utf-8") for name in SOURCES}
    if evidence["sources"] != {name: digest(code) for name, code in sources.items()}:
        raise ValueError("Coverage source changed; regenerate owned evidence.")
    if evidence["packages"] != PACKAGES or (assets / "requirements.txt").read_text(encoding="utf-8") != "".join(
        f"{n}=={v}\n" for n, v in PACKAGES.items()
    ):
        raise ValueError("Coverage dependency recipe is stale.")
    for name, expected in evidence["assets"].items():
        path = assets / name
        if digest(path.read_text(encoding="utf-8")) != expected or path.stat().st_size > evidence["size_budget_bytes"].get(name, 250000):
            raise ValueError(f"Changed or oversized coverage asset: {name}")
    for lesson, (path, suffix) in LESSONS.items():
        code = download_source(sources, lesson)
        if (assets / f"{lesson}.py").read_text(encoding="utf-8") != code:
            raise ValueError(f"Stale download: {lesson}")
        page = (root / path).read_text(encoding="utf-8")
        expected = replace_block(page, f"code{suffix}", code_block(code))
        expected = replace_block(expected, f"output{suffix}", f"```text\n{evidence['outputs'][lesson].rstrip()}\n```")
        if expected != page:
            raise ValueError(f"Stale lesson: {lesson}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.write:
        if args.work_dir is None:
            parser.error("--write requires a new --work-dir")
        write(ROOT, args.work_dir)
    check(ROOT)
    print("Coverage lessons, downloads, observations and figures are synchronized.")


if __name__ == "__main__":
    main()
