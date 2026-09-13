"""Verify the live Cloudflare documentation host and clean current-route contract."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

PRIMARY_HOST = "vamos-optimization.org"
REDIRECT_HOSTS = (
    "www.vamos-optimization.org",
    "vamos-optimization.dev",
    "www.vamos-optimization.dev",
)
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_CANONICAL_RE = re.compile(r'<link\s+rel="canonical"\s+href="([^"]+)"')


class LiveDocsCheckError(RuntimeError):
    """Raised when the public documentation host violates the route contract."""


@dataclass(frozen=True, slots=True)
class ResponseSnapshot:
    status: int
    headers: Mapping[str, str]
    body: bytes


Requester = Callable[[str, str, float], ResponseSnapshot]


def request_https(host: str, target: str, timeout: float) -> ResponseSnapshot:
    """Fetch one HTTPS resource without following redirects."""
    connection = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        connection.request(
            "GET",
            target,
            headers={"User-Agent": "VAMOS-documentation-cutover-check/1"},
        )
        response = connection.getresponse()
        body = response.read()
        return ResponseSnapshot(
            status=response.status,
            headers={name.lower(): value for name, value in response.getheaders()},
            body=body,
        )
    finally:
        connection.close()


def _expect_status(snapshot: ResponseSnapshot, expected: int, label: str) -> None:
    if snapshot.status != expected:
        raise LiveDocsCheckError(
            f"{label}: expected HTTP {expected}, got {snapshot.status}"
        )


def _expect_redirect(
    requester: Requester,
    *,
    host: str,
    target: str,
    expected_location: str,
    timeout: float,
) -> None:
    snapshot = requester(host, target, timeout)
    _expect_status(snapshot, 308, f"https://{host}{target}")
    location = snapshot.headers.get("location")
    if location != expected_location:
        raise LiveDocsCheckError(
            f"https://{host}{target}: expected Location {expected_location!r}, "
            f"got {location!r}"
        )


def _expect_canonical(
    snapshot: ResponseSnapshot,
    *,
    expected_url: str,
    label: str,
) -> None:
    _expect_status(snapshot, 200, label)
    text = snapshot.body.decode("utf-8", errors="replace")
    canonicals = _CANONICAL_RE.findall(text)
    if expected_url not in canonicals:
        raise LiveDocsCheckError(
            f"{label}: expected canonical URL {expected_url!r}, got {canonicals!r}"
        )


def check_live(
    version: str,
    *,
    requester: Requester = request_https,
    timeout: float = 15.0,
) -> dict[str, object]:
    """Validate clean current URLs, aliases, immutable archives, and host redirects."""
    if _VERSION_RE.fullmatch(version) is None:
        raise LiveDocsCheckError(
            "Documentation version must be numeric major.minor.patch"
        )

    base_url = f"https://{PRIMARY_HOST}/"
    immutable_url = f"{base_url}docs/{version}/"
    algorithm_url = f"{base_url}algorithms/nsgaii/"
    checked: list[str] = []

    root = requester(PRIMARY_HOST, "/", timeout)
    _expect_canonical(root, expected_url=base_url, label=base_url)
    checked.append(base_url)

    algorithm = requester(PRIMARY_HOST, "/algorithms/nsgaii/", timeout)
    _expect_canonical(
        algorithm,
        expected_url=algorithm_url,
        label=algorithm_url,
    )
    checked.append(algorithm_url)

    for legacy_target in ("/docs/?cutover=1", "/docs/stable/?cutover=1", "/latest/?cutover=1"):
        _expect_redirect(
            requester,
            host=PRIMARY_HOST,
            target=legacy_target,
            expected_location=f"{base_url}?cutover=1",
            timeout=timeout,
        )
        checked.append(f"https://{PRIMARY_HOST}{legacy_target}")

    _expect_redirect(
        requester,
        host=PRIMARY_HOST,
        target="/docs/stable/algorithms/nsgaii/?cutover=1",
        expected_location=f"{algorithm_url}?cutover=1",
        timeout=timeout,
    )
    checked.append(f"https://{PRIMARY_HOST}/docs/stable/algorithms/nsgaii/?cutover=1")

    _expect_redirect(
        requester,
        host=PRIMARY_HOST,
        target=f"/{version}/?cutover=1",
        expected_location=f"{immutable_url}?cutover=1",
        timeout=timeout,
    )
    checked.append(f"https://{PRIMARY_HOST}/{version}/?cutover=1")

    for host in REDIRECT_HOSTS:
        target = "/algorithms/nsgaii/?cutover=1"
        _expect_redirect(
            requester,
            host=host,
            target=target,
            expected_location=f"{algorithm_url}?cutover=1",
            timeout=timeout,
        )
        checked.append(f"https://{host}{target}")

    manifest_url = f"https://{PRIMARY_HOST}/docs/versions.json"
    manifest = requester(PRIMARY_HOST, "/docs/versions.json", timeout)
    _expect_status(manifest, 200, manifest_url)
    try:
        payload = json.loads(manifest.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveDocsCheckError("docs/versions.json is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LiveDocsCheckError("docs/versions.json must contain a JSON object")
    versions = payload.get("versions")
    if payload.get("stable") != version or not isinstance(versions, list):
        raise LiveDocsCheckError(
            "docs/versions.json does not identify the requested version as stable"
        )
    if version not in versions:
        raise LiveDocsCheckError(
            "docs/versions.json does not retain the requested immutable version"
        )
    checked.append(manifest_url)

    immutable = requester(PRIMARY_HOST, f"/docs/{version}/", timeout)
    _expect_canonical(
        immutable,
        expected_url=immutable_url,
        label=f"https://{PRIMARY_HOST}/docs/{version}/",
    )
    checked.append(f"https://{PRIMARY_HOST}/docs/{version}/")

    return {
        "primary": base_url,
        "stable": version,
        "checked": checked,
    }


def check_live_with_retry(
    version: str,
    *,
    attempts: int,
    delay: float,
    timeout: float,
    requester: Requester = request_https,
) -> dict[str, object]:
    """Retry the live contract while DNS, certificates, and custom domains settle."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if delay < 0:
        raise ValueError("delay must be >= 0")
    if timeout <= 0:
        raise ValueError("timeout must be > 0")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return check_live(version, requester=requester, timeout=timeout)
        except (LiveDocsCheckError, OSError, http.client.HTTPException) as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"Live documentation check attempt {attempt}/{attempts} failed: "
                f"{exc}. Retrying in {delay:g}s.",
                flush=True,
            )
            time.sleep(delay)

    raise LiveDocsCheckError(
        f"Live documentation check failed after {attempts} attempt(s): {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--delay", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    try:
        result = check_live_with_retry(
            args.version,
            attempts=args.attempts,
            delay=args.delay,
            timeout=args.timeout,
        )
    except (LiveDocsCheckError, ValueError) as exc:
        print(f"Cloudflare documentation cutover check failed: {exc}")
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
