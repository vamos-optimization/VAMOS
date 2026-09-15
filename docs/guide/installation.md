# Installation

VAMOS 1.0.0 requires Python 3.10, 3.11, or 3.12 on an operating system covered
by the release CI matrix.

## Core package

Create an isolated environment and install the package from PyPI:

=== "Linux and macOS"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install vamos-optimization
    ```

=== "Windows PowerShell"

    ```powershell
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install vamos-optimization
    ```

Verify the installation:

```bash
python -c "import vamos; print(vamos.__version__)"
vamos check
```

The version command must print `1.0.0` for this release.

## Optional extras

Install only the capability groups you use:

| Extra | Purpose |
| --- | --- |
| `compute` | Numba kernels, MooCore indicators, and Dask evaluation |
| `research` | Third-party research baselines and benchmarks |
| `analysis` | Data frames, plotting, notebooks, and analysis helpers |
| `tuning` | Optional model-based tuning backends |
| `studio` | Experimental Panel-based local Studio |
| `examples` | Dependencies used by selected examples |
| `docs` | Documentation toolchain |
| `dev` | Tests, linting, typing, building, and docs checks |
| `all` | All optional and development groups |

For example:

```bash
python -m pip install "vamos-optimization[compute,analysis]"
```

An explicit optional backend fails clearly when its dependency is unavailable.
VAMOS does not install packages during optimization, verification, or replay.

## Install from a source checkout

For development from the repository root:

```bash
python -m pip install -e ".[dev]"
```
