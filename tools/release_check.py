"""Canonical VAMOS release checker and evidence generator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from check_repository_hygiene import distribution_violations
from release_artifacts import (
    copy_distributions,
    create_runtime_environment,
    create_sbom,
    distributions,
    inspect_distributions,
    run_dependency_audit,
    write_release_manifests,
    write_runtime_lock,
)
from release_policy import (
    document_evidence,
    git,
    license_evidence,
    repository_identity,
    scan_files,
    tag_evidence,
    tracked_files,
    version_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
CHECK_NAMES = (
    "repository-identity",
    "canonical-repository-identity",
    "version-consistency",
    "release-documents",
    "license-metadata",
    "tag-state",
    "source-path-and-secret-scan",
    "repository-hygiene",
    "pre-release-remnants",
    "stable-api-cli-schema-fixtures",
    "strict-typing",
    "stable-api-typing",
    "full-source-ratchet",
    "health",
    "release-typing-policy",
    "full-source-zero-informational",
    "ruff-lint",
    "ruff-format",
    "compileall",
    "public-examples",
    "complete-tests",
    "documentation",
    "distribution-build",
    "distribution-hygiene",
    "twine-and-wheel-content",
    "wheel-content-policy",
    "distribution-inspection",
    "clean-wheel-install",
    "clean-wheel-install-and-smoke",
    "runtime-dependency-audit",
    "build-dependency-audit",
    "release-tool-dependency-audit",
    "cyclonedx-sbom",
    "checksums-and-provenance",
    "final-worktree-cleanliness",
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    duration_seconds: float
    details: Any


class ReleaseChecker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.version = str(args.version)
        self.output = Path(args.output_dir).resolve() if args.output_dir else self._default_output()
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty evidence directory: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs = self.output / "logs"
        self.logs.mkdir()
        self.results: list[CheckResult] = []
        self.failed = False
        self.identity: dict[str, Any] = {}
        self.dist = self.output / "dist"
        self.runtime_python: Path | None = None
        self.runtime_lock = self.output / "runtime-lock.txt"
        self.typing_python = Path(os.path.abspath(args.typing_python)) if args.typing_python else Path(sys.executable).resolve()
        if not self.typing_python.is_file():
            raise FileNotFoundError(f"Canonical typing Python does not exist: {self.typing_python}")

    def _default_output(self) -> Path:
        head = git(ROOT, "rev-parse", "--short=12", "HEAD")
        return ROOT.parent / f"{ROOT.name}-release-evidence-{self.version}-{head}"

    def check(self, name: str, operation: Callable[[], Any], *, status: str = "passed", critical: bool = True) -> None:
        started = time.perf_counter()
        try:
            details = operation()
            result_status = str(details.pop("_status", status)) if isinstance(details, dict) else status
        except Exception as exc:
            details = {"error": f"{type(exc).__name__}: {exc}"}
            result_status = "failed" if critical else "warning"
            if critical:
                self.failed = True
        result = CheckResult(name, result_status, round(time.perf_counter() - started, 3), details)
        self.results.append(result)
        if not self.args.json:
            print(f"[{result.status.upper()}] {name} ({result.duration_seconds:.3f}s)", flush=True)
            if result.status in {"failed", "warning"}:
                print(f"  {details}", flush=True)

    def command(
        self,
        name: str,
        arguments: list[str],
        *,
        expected: tuple[int, ...] = (0,),
        critical: bool = True,
        informational: bool = False,
    ) -> None:
        def operation() -> dict[str, Any]:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")
            environment["PYTHONUNBUFFERED"] = "1"
            completed = subprocess.run(
                arguments,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            log = self.logs / f"{len(self.results) + 1:02d}-{_slug(name)}.log"
            log.write_text(
                f"$ {_display_command(arguments)}\n\nSTDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}",
                encoding="utf-8",
                newline="\n",
            )
            if completed.returncode not in expected:
                tail = (completed.stdout + "\n" + completed.stderr).strip().splitlines()[-20:]
                raise RuntimeError(f"exit {completed.returncode}; see {log}; tail={tail}")
            return {
                "command": arguments,
                "exit_code": completed.returncode,
                "log": log.relative_to(self.output).as_posix(),
                "summary": _output_summary(completed.stdout + "\n" + completed.stderr),
            }

        self.check(name, operation, status="informational" if informational else "passed", critical=critical)

    def run(self) -> dict[str, Any]:
        self.check("repository-identity", self._identity)
        self.command("canonical-repository-identity", [sys.executable, "tools/check_repository_identity.py"])
        self.check("version-consistency", lambda: version_evidence(ROOT, self.version))
        self.check("release-documents", lambda: document_evidence(ROOT, self.version))
        self.check("license-metadata", lambda: license_evidence(ROOT))
        self.check("tag-state", lambda: tag_evidence(ROOT, self.version, self.args.tag_state))
        self.check("source-path-and-secret-scan", lambda: scan_files(tracked_files(ROOT), root=ROOT))
        self.command("repository-hygiene", [sys.executable, "tools/check_repository_hygiene.py"])
        self.command("pre-release-remnants", [sys.executable, "tools/check_pre_release_remnants.py"])
        self.command(
            "stable-api-cli-schema-fixtures",
            [sys.executable, "-m", "pytest", "-q", "tests/compatibility"],
        )
        typing_python = str(self.typing_python)
        self.command("strict-typing", [typing_python, "tools/typecheck.py", "--scope", "strict"])
        self.command("stable-api-typing", [typing_python, "tools/typecheck.py", "--scope", "stable"])
        self.command("full-source-ratchet", [typing_python, "tools/typecheck.py", "--scope", "full"])
        self.command("health", [typing_python, "tools/health.py"])
        self.command("release-typing-policy", [typing_python, "tools/typecheck.py", "--scope", "release"])
        self.command(
            "full-source-zero-informational",
            [typing_python, "tools/typecheck.py", "--scope", "full-zero"],
            expected=(0, 1),
            critical=False,
            informational=True,
        )
        self.command("ruff-lint", [sys.executable, "-m", "ruff", "check", "src/vamos", "tests", "tools"])
        self.command(
            "ruff-format",
            [sys.executable, "-m", "pytest", "-q", "tests/architecture/test_ruff_format_gate.py"],
        )
        self.command("compileall", [sys.executable, "-m", "compileall", "-q", "src/vamos", "tests", "tools"])
        self.command("public-examples", [sys.executable, "-m", "pytest", "-q", "tests/docs"])
        self.command("complete-tests", [sys.executable, "-m", "pytest", "-q"])
        self.command(
            "documentation",
            [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(self.output / "site")],
        )
        self.check("distribution-build", self._prepare_distributions)
        wheel = self._wheel()
        self.check("distribution-hygiene", self._distribution_hygiene)
        artifact_arguments = [str(path) for path in distributions(self.dist)]
        self.command("twine-and-wheel-content", [sys.executable, "-m", "twine", "check", *artifact_arguments])
        self.command(
            "wheel-content-policy",
            [
                sys.executable,
                "-m",
                "check_wheel_contents",
                "--ignore",
                "W002",
                "--toplevel",
                "vamos,vamos_contrib",
                str(wheel),
            ],
        )
        self.check("distribution-inspection", lambda: inspect_distributions(self.dist, self.version))
        self._runtime_checks(wheel)
        self.check("runtime-dependency-audit", self._runtime_audit)
        self.check("build-dependency-audit", self._build_audit, critical=False)
        self.check("release-tool-dependency-audit", self._release_tool_audit, critical=False)
        sbom = self.output / "SBOM.cdx.json"
        self.check("cyclonedx-sbom", lambda: create_sbom(Path(sys.executable), self.runtime_lock, wheel, self.version, sbom))
        self.check(
            "checksums-and-provenance",
            lambda: write_release_manifests(self.output, self.dist, sbom, self.version, str(self.identity.get("commit", ""))),
        )
        self.check("final-worktree-cleanliness", self._final_clean)
        report = self.report()
        report_path = self.output / "release-check-report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        report["report"] = str(report_path)
        return report

    def _identity(self) -> dict[str, Any]:
        self.identity = repository_identity(ROOT, self.version, self.args.expected_branch, self.args.expected_commit)
        return self.identity

    def _prepare_distributions(self) -> dict[str, Any]:
        if self.args.artifacts:
            wheel, sdist = copy_distributions(Path(self.args.artifacts).resolve(), self.dist)
            return {"source": "frozen-input", "wheel": wheel.name, "sdist": sdist.name}
        self.dist.mkdir()
        completed = subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(self.dist)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        log = self.logs / "distribution-build.log"
        log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8", newline="\n")
        if completed.returncode != 0:
            raise RuntimeError(f"Build failed; see {log}")
        deprecations = [line for line in (completed.stdout + completed.stderr).splitlines() if "SetuptoolsDeprecationWarning" in line]
        if deprecations:
            raise AssertionError(f"Setuptools emitted deprecation warnings: {deprecations[:5]}")
        wheel, sdist = distributions(self.dist)
        return {"source": "isolated-build", "wheel": wheel.name, "sdist": sdist.name, "log": log.name}

    def _wheel(self) -> Path:
        return distributions(self.dist)[0]

    def _distribution_hygiene(self) -> dict[str, Any]:
        archives = list(distributions(self.dist))
        violations = distribution_violations(archives)
        if violations:
            details = [f"{item.path}: {item.message}" for item in violations]
            raise AssertionError(f"Forbidden distribution content: {details}")
        return {"archives": [path.name for path in archives], "violations": 0}

    def _runtime_checks(self, wheel: Path) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="vamos-release-runtime-", dir=self.output.parent)
        self._runtime_temporary = temporary

        def install() -> dict[str, Any]:
            _, python = create_runtime_environment(
                Path(temporary.name) / "venv",
                wheel,
                ROOT / "constraints" / "ci.txt",
                extras="compute",
            )
            self.runtime_python = python
            lock = write_runtime_lock(python, self.runtime_lock)
            return {"python": str(python), "runtime_lock": lock}

        self.check("clean-wheel-install", install)

        def smoke() -> dict[str, Any]:
            if self.runtime_python is None:
                raise AssertionError("Clean wheel environment was not created.")
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PATH"] = os.pathsep.join((str(self.runtime_python.parent), environment.get("PATH", "")))
            completed = subprocess.run(
                [str(self.runtime_python), str(ROOT / "tools" / "release_smoke.py"), "--version", self.version, "--mode", "full"],
                cwd=Path(temporary.name),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=600,
            )
            log = self.logs / "clean-wheel-release-smoke.log"
            log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8", newline="\n")
            if completed.returncode != 0:
                raise RuntimeError(f"Clean-wheel release smoke failed; see {log}")
            return {"log": log.relative_to(self.output).as_posix(), "evidence": json.loads(completed.stdout)}

        self.check("clean-wheel-install-and-smoke", smoke)

    def _runtime_audit(self) -> dict[str, Any]:
        return run_dependency_audit(Path(sys.executable), self.runtime_lock, self.output / "dependency-audit-runtime.json", blocking=True)

    def _build_audit(self) -> dict[str, Any]:
        result = run_dependency_audit(
            Path(sys.executable),
            ROOT / "release" / "requirements-build.txt",
            self.output / "dependency-audit-build.json",
            blocking=False,
        )
        if result["findings"]:
            result["_status"] = "warning"
        return result

    def _release_tool_audit(self) -> dict[str, Any]:
        result = run_dependency_audit(
            Path(sys.executable),
            ROOT / "release" / "requirements-tools.txt",
            self.output / "dependency-audit-release-tools.json",
            blocking=False,
        )
        if result["findings"]:
            result["_status"] = "warning"
        return result

    def _final_clean(self) -> dict[str, Any]:
        status = git(ROOT, "status", "--porcelain", "--untracked-files=all")
        if status:
            raise AssertionError(f"Release checks dirtied the worktree:\n{status}")
        return {"clean": True, "commit": git(ROOT, "rev-parse", "HEAD")}

    def report(self) -> dict[str, Any]:
        return {
            "document_type": "vamos.release-check",
            "schema_version": "1.0.0",
            "project": "vamos-optimization",
            "version": self.version,
            "status": "failed" if self.failed else "passed",
            "source_commit": self.identity.get("commit"),
            "tag_state": self.args.tag_state,
            "typing": {
                "strict": _status(self.results, "strict-typing"),
                "stable_public_api": _status(self.results, "stable-api-typing"),
                "full_source_ratchet": _status(self.results, "full-source-ratchet"),
                "full_source_zero": _status(self.results, "full-source-zero-informational"),
            },
            "checks": [asdict(result) for result in self.results],
        }


def _status(results: list[CheckResult], name: str) -> str | None:
    return next((result.status for result in results if result.name == name), None)


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-").lower()


def _display_command(arguments: list[str]) -> str:
    return " ".join(json.dumps(item) if any(character.isspace() for character in item) else item for item in arguments)


def _output_summary(output: str) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-10:]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--expected-branch")
    parser.add_argument("--expected-commit")
    parser.add_argument(
        "--tag-state",
        choices=("pre-normalization", "pre-tag", "pre-release", "normalized", "ignore"),
        default="pre-release",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--artifacts", help="Directory containing one already-frozen wheel and sdist.")
    parser.add_argument(
        "--typing-python",
        help="Python executable from the canonical dependency-minimal typing environment.",
    )
    parser.add_argument("--json", action="store_true", help="Emit exactly one JSON result document to stdout.")
    parser.add_argument("--list-checks", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list_checks:
        payload = {"document_type": "vamos.release-check-inventory", "checks": list(CHECK_NAMES)}
        print(json.dumps(payload, sort_keys=True) if args.json else "\n".join(CHECK_NAMES))
        return
    checker: ReleaseChecker | None = None
    try:
        checker = ReleaseChecker(args)
        report = checker.run()
    except Exception as exc:
        report = {
            "document_type": "vamos.release-check",
            "schema_version": "1.0.0",
            "project": "vamos-optimization",
            "version": args.version,
            "status": "failed",
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "checks": [asdict(result) for result in checker.results] if checker is not None else [],
        }
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Release check: {report['status'].upper()}")
        if report.get("report"):
            print(f"Evidence: {report['report']}")
        if report.get("fatal_error"):
            print(report["fatal_error"])
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
