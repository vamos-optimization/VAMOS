"""Build or check T01–T03 downloads, code blocks, observed output and figures."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "examples/tutorials/fundamentals.py"
ASSETS = "docs/assets/tutorials"
LESSONS = {
    "t01": ("t01-first-result", ("run_t01",)),
    "t02": ("t02-constrained-problem", ("process_objectives", "process_constraints", "solve_process", "run_t02")),
    "t03": ("t03-pareto-decision", ("process_objectives", "process_constraints", "solve_process", "run_t03")),
}
CORE = {"vamos-optimization": "1.0.1", "numpy": "2.1.3", "scipy": "1.15.3", "joblib": "1.4.2"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lesson_source(source: str, lesson: str) -> str:
    """Compose a standalone script from the canonical function definitions."""
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    functions = [ast.get_source_segment(source, nodes[name]) for name in LESSONS[lesson][1]]
    return (
        f'"""{lesson.upper()}: generated from {SOURCE}; VAMOS 1.0.1, core dependencies only."""\n\n\n'
        + "\n\n\n".join(functions)
        + f'\n\n\nif __name__ == "__main__":\n    run_{lesson}()\n'
    )


def replace_block(page: str, kind: str, body: str) -> str:
    start, end = f"<!-- tutorial:{kind} -->", f"<!-- /tutorial:{kind} -->"
    if page.count(start) != 1 or page.count(end) != 1:
        raise ValueError(f"Expected one {kind} block")
    before, rest = page.split(start)
    _, after = rest.split(end)
    return before + start + "\n" + body.rstrip() + "\n" + end + after


def draw_figure(lesson: str, data: dict, path: Path, source_hash: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context({"svg.hashsalt": "vamos-fundamentals", "svg.fonttype": "none", "font.size": 11}):
        fig, ax = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
        result = data["result"]
        if lesson == "t01":
            ax.scatter(*result.F.T, c="#929ca3", s=36, label="Returned population (40)")
            ax.scatter(
                *data["front_F"].T, c="#007f7a", marker="x", s=80, linewidths=2, label=f"Non-dominated subset ({len(data['front_F'])})"
            )
            ax.set(xlabel="Objective f1 (minimize)", ylabel="Objective f2 (minimize)", title="T01 · One short ZDT1 run")
        elif lesson == "t02":
            temperature = np.linspace(60, 100, 160)
            ax.fill_between(temperature, 450 / temperature, 10, color="#dff2ee", label="Feasible region")
            ax.plot(temperature, 450 / temperature, "--", color="#405568", label="Constraint boundary")
            ax.scatter(*result.X[data["feasible_rows"]].T, c="#007f7a", s=32, label="Feasible returned decisions")
            probes = np.array([[60, 2], [75, 6], [90, 8]])
            ax.scatter(*probes.T, marker="x", c="#9a5416", s=80, linewidths=2, label="Hand-checked probes")
            ax.set(
                xlabel="Temperature set point",
                ylabel="Residence time",
                xlim=(58, 102),
                ylim=(1, 11),
                title="T02 · Check feasibility in decision space",
            )
        else:
            ax.scatter(*data["front_F"].T, c="#007f7a", s=36, label="Feasible non-dominated candidates")
            chosen = result.F[data["selected_row"]]
            ax.scatter(*chosen, marker="*", c="#ac5d12", edgecolors="#42290d", s=240, label="Selected: weights (0.8, 0.2)", zorder=3)
            ax.set(
                xlabel="Energy score (minimize)",
                ylabel="Conversion-shortfall score (minimize)",
                title="T03 · A preference selects one alternative",
            )
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
        fig.savefig(path, metadata={"Date": None, "Description": f"Actual VAMOS 1.0.1 output; source sha256={source_hash}"})
        plt.close(fig)
    if path.suffix == ".svg":
        # Matplotlib emits trailing spaces in multiline SVG paths.
        svg = path.read_text(encoding="utf-8")
        path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")


def write(root: Path) -> None:
    import vamos

    if Path(vamos.__file__).resolve().is_relative_to(root.resolve()):
        raise ValueError("Generate evidence with the installed PyPI wheel, without an editable checkout or PYTHONPATH.")
    versions = {name: importlib.metadata.version(name) for name in (*CORE, "matplotlib")}
    if any(versions[name] != expected for name, expected in CORE.items()) or versions["matplotlib"] != "3.10.9":
        raise ValueError("Use the declared tutorial versions and matplotlib==3.10.9 to regenerate evidence.")
    source = (root / SOURCE).read_text(encoding="utf-8")
    source_hash = digest(source)
    assets = root / ASSETS
    assets.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema": "vamos.documentation.tutorial-evidence.v1",
        "purpose": "Small teaching outputs, not comparative or convergence evidence; no run arrays retained.",
        "source": SOURCE,
        "source_sha256": source_hash,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "command": "python tools/build_fundamental_tutorials.py --write",
        "python": platform.python_version(),
        "packages": versions,
        "settings": {"algorithm": "nsgaii", "engine": "numpy", "seed": 42, "population": 40, "max_evaluations": 400},
        "asset_size_budget_bytes": 250000,
        "lessons": {},
    }
    for lesson, (slug, _) in LESSONS.items():
        code = lesson_source(source, lesson)
        namespace: dict = {"__name__": "tutorial_example"}
        exec(compile(code, slug + ".py", "exec"), namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            data = namespace[f"run_{lesson}"]()
        stdout = output.getvalue()
        (assets / f"{slug}.py").write_text(code, encoding="utf-8")
        figure = assets / f"{slug}.svg"
        draw_figure(lesson, data, figure, source_hash)
        page_path = root / "docs/tutorials" / f"{slug}.md"
        page = page_path.read_text(encoding="utf-8")
        page = replace_block(page, "code", f"```python\n{code.rstrip()}\n```")
        page = replace_block(page, "output", f"```text\n{stdout.rstrip()}\n```")
        page_path.write_text(page, encoding="utf-8")
        evidence["lessons"][lesson] = {
            "download_sha256": digest(code),
            "figure_sha256": digest(figure.read_text(encoding="utf-8")),
            "stdout": stdout,
            "returned_rows": len(data["result"].F),
            "front_rows": len(data["front_F"]),
        }
    (assets / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    (assets / "tutorial-requirements.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in CORE.items()),
        encoding="utf-8",
    )


def check(root: Path) -> None:
    source = (root / SOURCE).read_text(encoding="utf-8")
    assets = root / ASSETS
    evidence = json.loads((assets / "evidence.json").read_text(encoding="utf-8"))
    if evidence["source_sha256"] != digest(source):
        raise ValueError("Canonical tutorial source changed: regenerate the evidence.")
    requirements = "".join(f"{name}=={version}\n" for name, version in CORE.items())
    if (assets / "tutorial-requirements.txt").read_text(encoding="utf-8") != requirements:
        raise ValueError("Tutorial dependency recipe is stale.")
    for lesson, (slug, _) in LESSONS.items():
        code = lesson_source(source, lesson)
        record = evidence["lessons"][lesson]
        if (assets / f"{slug}.py").read_text(encoding="utf-8") != code or record["download_sha256"] != digest(code):
            raise ValueError(f"Stale {lesson} download")
        figure = assets / f"{slug}.svg"
        if digest(figure.read_text(encoding="utf-8")) != record["figure_sha256"]:
            raise ValueError(f"Changed {lesson} figure")
        if figure.stat().st_size > evidence["asset_size_budget_bytes"]:
            raise ValueError(f"Oversized {lesson} figure")
        page = (root / "docs/tutorials" / f"{slug}.md").read_text(encoding="utf-8")
        expected = replace_block(page, "code", f"```python\n{code.rstrip()}\n```")
        expected = replace_block(expected, "output", "```text\n" + record["stdout"].rstrip() + "\n```")
        if page != expected:
            raise ValueError(f"Stale {lesson} code/output block")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Explicitly replace the owned generated assets and page blocks")
    mode.add_argument("--check", action="store_true", help="Check source/content synchronization without executing examples")
    args = parser.parse_args()
    if args.write:
        write(ROOT)
    check(ROOT)
    print("Fundamental tutorial sources, downloads, output blocks and figures are synchronized.")


if __name__ == "__main__":
    main()
