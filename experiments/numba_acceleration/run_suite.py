"""Run the NumPy/Numba benchmark suite for a baseline and a candidate source tree.

Each configuration runs ``benchmark.py`` in its own interpreter with ``PYTHONPATH``
pointing at one source tree. Use an environment *without* Numba for the NumPy
configurations so that availability-gated JIT paths cannot run, and an
environment with the ``compute`` extra for the Numba configurations. Install
MooCore in both: hypervolume contributions use MooCore whenever it is
importable, independently of the engine.

Example::

    python experiments/numba_acceleration/run_suite.py \\
        --numpy-python /path/to/no-numba-env/python \\
        --numba-python /path/to/compute-env/python \\
        --baseline-src /path/to/baseline-checkout/src \\
        --candidate-src src \\
        --output-dir results/numba_acceleration
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SINGLE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numpy-python", required=True, help="Interpreter of an environment without numba.")
    parser.add_argument("--numba-python", required=True, help="Interpreter of an environment with the compute extra.")
    parser.add_argument("--baseline-src", required=True, type=Path, help="src/ directory of the baseline checkout.")
    parser.add_argument("--candidate-src", required=True, type=Path, help="src/ directory of the candidate checkout.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--skip", nargs="*", default=[], choices=("runs", "startup", "kernels"))
    parser.add_argument("--cpu", type=int, default=None, help="Pin every benchmark process to one logical processor.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("numpy-old", args.numpy_python, args.baseline_src, "numpy"),
        ("numpy-new", args.numpy_python, args.candidate_src, "numpy"),
        ("numba-old", args.numba_python, args.baseline_src, "numba"),
        ("numba-new", args.numba_python, args.candidate_src, "numba"),
    ]
    outputs: list[Path] = []

    def run(label: str, python: str, src: Path, engine: str, command: str, extra: list[str]) -> None:
        env = dict(os.environ, PYTHONPATH=str(src.resolve()), **SINGLE_THREAD)
        env.pop("VAMOS_USE_NUMBA_MOEAD", None)
        env.pop("VAMOS_USE_NUMBA_VARIATION", None)
        output = args.output_dir / f"{command}-{label}.jsonl"
        if output not in outputs:
            if output.exists():
                output.unlink()
            outputs.append(output)
        cmd = [python, str(HERE / "benchmark.py"), command, "--label", label, "--engine", engine, "--output", str(output), *extra]
        if args.cpu is not None:
            cmd += ["--cpu", str(args.cpu)]
        subprocess.run(cmd, env=env, check=True)

    for command in ("kernels", "startup"):
        if command in args.skip:
            continue
        for label, python, src, engine in configs:
            print(f"== {label}: {command}", flush=True)
            run(label, python, src, engine, command, [])
    if "runs" not in args.skip:
        # Interleave configurations case by case (rotating their order) so slow drift in
        # machine state cannot favour one configuration.
        from benchmark import RUN_CASES  # noqa: PLC0415 - sibling script in this directory

        cases = [case[0] for case in RUN_CASES if not args.cases or case[0] in args.cases]
        for index, case in enumerate(cases):
            rotated = configs[index % len(configs) :] + configs[: index % len(configs)]
            for label, python, src, engine in rotated:
                print(f"== {label}: runs {case}", flush=True)
                run(label, python, src, engine, "runs", ["--repeat", "1", "--seeds", *map(str, args.seeds), "--cases", case])
    report = args.output_dir / "summary.md"
    subprocess.run(
        [sys.executable, str(HERE / "benchmark.py"), "report", "--inputs", *map(str, outputs), "--output", str(report)], check=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
