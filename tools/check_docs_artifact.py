"""Bind documentation bytes to a source commit and compare deployed current content."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from check_docs_portal import _current_relative_files

SCHEMA = "vamos.docs-artifact.v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not portal assets: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    if not result:
        raise ValueError("Cannot attest an empty portal")
    return result


def create_manifest(root: Path, source_sha: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("A full source commit SHA is required")
    files = inventory(root)
    # Current content is checked byte-for-byte after deployment. Compatibility
    # redirects and preserved archives retain their separate route/hash gates.
    current = sorted(p.as_posix() for p in _current_relative_files(root))
    return {"schema": SCHEMA, "source_sha": source_sha, "files": files, "current": current}


def read_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("Unknown documentation artifact schema")
    if not isinstance(payload.get("source_sha"), str) or re.fullmatch(r"[0-9a-f]{40}", payload["source_sha"]) is None:
        raise ValueError("Invalid artifact source SHA")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Artifact inventory is empty or invalid")
    for name, checksum in files.items():
        path_value = PurePosixPath(name)
        if (
            not name
            or path_value.is_absolute()
            or ".." in path_value.parts
            or path_value.as_posix() != name
            or "\\" in name
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise ValueError(f"Invalid artifact entry: {name}")
    current = payload.get("current")
    if not isinstance(current, list) or not current or any(not isinstance(p, str) or p not in files for p in current):
        raise ValueError("Invalid current-content inventory")
    if len(current) != len(set(current)):
        raise ValueError("Duplicate current-content entries")
    return payload


def verify_manifest(root: Path, payload: dict, expected_sha: str | None = None) -> int:
    if expected_sha is not None and payload["source_sha"] != expected_sha:
        raise ValueError("Artifact does not match the expected source commit")
    if inventory(root) != payload["files"]:
        raise ValueError("Portal bytes or file inventory differ from the recorded artifact")
    if sorted(p.as_posix() for p in _current_relative_files(root)) != payload["current"]:
        raise ValueError("Current-content inventory differs from the portal")
    return len(payload["files"])


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "VAMOS-documentation-artifact-check/1", "Accept-Encoding": "identity"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status at {url}: {response.status}")
            return response.read()
    except HTTPError as exc:
        # The real recovery document is deliberately served with HTTP 404.
        if exc.code == 404 and urlsplit(url).path.endswith("/404.html"):
            return exc.read()
        raise


def verify_live(payload: dict, base_url: str, *, fetch=fetch_bytes, timeout: float = 15, attempts: int = 3) -> int:
    base = urlsplit(base_url)
    if base.scheme != "https" or not base.netloc or base.query or base.fragment or base.username or base.password:
        raise ValueError("An HTTPS site base without credentials, query or fragment is required")
    if attempts < 1 or timeout <= 0:
        raise ValueError("Attempts and timeout must be positive")

    def check(name: str) -> None:
        relative = name.removesuffix("index.html") if name.endswith("/index.html") or name == "index.html" else name
        url = base_url.rstrip("/") + "/" + quote(relative, safe="/")
        for attempt in range(attempts):
            try:
                actual = digest(fetch(url, timeout))
                if actual == payload["files"][name]:
                    return
                raise ValueError(f"Deployed bytes differ from approved artifact: {name}")
            except (OSError, URLError, ValueError):
                if attempt + 1 == attempts:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, payload["current"]))
    return len(payload["current"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify", "live"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--base-url", default="https://vamos-optimization.org/")
    args = parser.parse_args()
    if args.mode == "create":
        if args.source_sha is None:
            parser.error("create requires --source-sha")
        if args.manifest.resolve().is_relative_to(args.root.resolve()):
            parser.error("Keep the manifest outside the portal tree")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x", encoding="utf-8") as handle:
            json.dump(create_manifest(args.root, args.source_sha), handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        payload = read_manifest(args.manifest)
        count = verify_manifest(args.root, payload, args.source_sha)
        live = verify_live(payload, args.base_url) if args.mode == "live" else 0
        print(json.dumps({"source_sha": payload["source_sha"], "artifact_files": count, "live_current_files": live}))


if __name__ == "__main__":
    main()
