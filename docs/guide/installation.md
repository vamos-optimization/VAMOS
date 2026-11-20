---
description: Install VAMOS 1.5.2 and select the optional dependencies needed for your optimization workflow.
search:
  boost: 2
---

# Installation

The examples on this site target **VAMOS 1.5.2**. Use an isolated Python
environment and the pinned commands below to follow the same release.

## Supported Python and operating systems

| Operating system | Python versions covered by CI | Installed-wheel release checks |
| --- | --- | --- |
| Linux | 3.11, 3.12, 3.13, 3.14 | Core on 3.11 and 3.13; core and compute on 3.12 and 3.14 |
| Windows | 3.12, 3.13, 3.14 | Core on 3.12 |
| macOS | 3.12, 3.13, 3.14 | Core on 3.12 |

These are the combinations in the project's CI and release verification
matrices, not a promise for every version of every optional dependency. Other
interpreter/platform combinations are unclaimed. See
[Known limitations](../project/known-limitations.md#backends-and-optional-dependencies).

**VAMOS 1.2.0:** CI tests Python 3.13 and 3.14 on the three
systems, and the release checks install the wheel on both. VAMOS 1.1.0 was
tested on Python 3.10, 3.11 and 3.12 only.

**VAMOS 1.3.0:** VAMOS requires Python 3.11 or newer, as announced
with 1.2.0, and CI no longer tests Python 3.10. pip does not install it on
Python 3.10, where an unpinned install selects the newest 1.2 release; see
[Stability and versioning](../project/stability-and-versioning.md#python-and-operating-system-support).

## Core package

Create an isolated environment and install the package from PyPI:

=== "Linux and macOS"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install "vamos-optimization==1.5.2"
    ```

=== "Windows PowerShell"

    ```powershell
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install "vamos-optimization==1.5.2"
    ```

Verify the installation:

```bash
python -c "import vamos; print(vamos.__version__)"
vamos check
```

The version command must print `1.5.2` for this release. To install the latest
published version instead, use `python -m pip install --upgrade vamos-optimization`
and consult the documentation for that version. Pinning VAMOS alone does not
freeze dependencies; retain the environment record for scientific runs (see
[Run artifacts and replay](run-artifacts.md)).

## Optional extras

Install only the capability groups you use:

| Extra | Purpose | Interface status |
| --- | --- | --- |
| `compute` | Numba kernels, MooCore indicators, and Dask evaluation | Stable backend selection; Experimental Dask integration |
| `research` | Third-party research baselines and benchmarks | Experimental |
| `analysis` | Data frames, plotting, notebooks, and analysis helpers | Experimental helpers |
| `tuning` | Optional model-based tuning backends | Experimental |
| `studio` | Panel-based local Studio | Experimental; see [Studio](studio.md) for release limitations |
| `examples` | Dependencies used by selected examples | Depends on the example's interfaces |
| `modact` | **VAMOS 1.5.1:** the authors' `modact` package (GPL-3.0), which evaluates the [MODAct problems](../reference/problems.md#modact-actuator-design-evaluated-by-the-modact-package) | Registered problems; `all` leaves it out |
| `docs` | Documentation toolchain | Developer tooling |
| `dev` | Tests, linting, typing, building, and docs checks | Developer tooling |
| `all` | All optional and development groups | Mixed; does not change interface guarantees |

**VAMOS 1.2.0:** the extras install only what VAMOS and its
examples use, and no other optimization framework. The `research` extra
(pymoo, jMetalPy, DEAP, Platypus and pygmo) is removed, with the baselines it
served. `analysis` installs pandas, matplotlib, Plotly and scikit-learn.
`examples` adds seaborn, NetworkX, IPython, ipykernel and ipywidgets for the
examples and notebooks; in VAMOS 1.1.0 they came with `analysis`, so a notebook
now needs `analysis` and `examples`. `all` installs the user extras (`compute`,
`analysis`, `tuning`, `studio`, `examples`, `openai`, `anthropic` and
`gemini`) and no longer the `dev` and `docs` tooling.

**VAMOS 1.4.0:** the `openai`, `anthropic` and `gemini` extras
are removed, with `vamos assist` and Studio's Problem Builder, the only parts
of VAMOS that used them. No extra, `all` included, installs an LLM client.

**VAMOS 1.5.1:** the `modact` extra installs the `modact`
package of Picard and Schiffmann, which evaluates the 20 MODAct problems. It is
licensed under the GPL-3.0, unlike VAMOS (MIT), so `all` does not install it:
install the extra, or `python -m pip install modact`, to run those problems.
VAMOS contains none of its code and imports it only when a MODAct problem is
created.

For example:

```bash
python -m pip install "vamos-optimization[compute,analysis]==1.5.2"
```

An explicit optional backend fails clearly when its dependency is unavailable.
VAMOS does not install packages during optimization, verification, or replay.
Installing an extra does not make its APIs stable. The
[stability policy](../project/stability-and-versioning.md) defines **Stable**,
**Experimental**, and **Internal** interfaces.

## Install from a source checkout

Only use an editable install when developing VAMOS or modifying repository
examples. First obtain the source and enter its root:

```bash
git clone https://github.com/vamos-optimization/VAMOS.git
cd VAMOS
python -m pip install -e ".[dev]"
```

Use the environment setup above before this command. The default branch may
contain work beyond the published release; record the checked-out commit. To
inspect the release source, check out the `v1.5.2` tag before installing.
For ordinary use, prefer the PyPI installation under **Core package**.
