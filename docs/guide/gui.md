# VAMOS GUI (experimental)

`vamos gui` starts a local web interface built with [NiceGUI](https://nicegui.io)
and Plotly. It is a thin client of canonical run artifacts and the public run API:

- the **Explorer** lists the canonical runs under a results directory and lets you
  inspect their Pareto fronts interactively;
- **New run** launches a built-in problem/algorithm run in a separate worker
  process, refreshes the current front and a progress hypervolume while the
  generations advance, and opens the stored front when the run finishes.

The GUI is experimental: its options and pages may change without a deprecation
cycle (see [Stability and versioning](../project/stability-and-versioning.md)).

## Install and launch

```bash
pip install "vamos-optimization[gui]"
vamos gui results
```

The positional argument is the directory scanned for runs (default: `./results`).
Runs launched from the GUI are stored under `<results>/gui-runs/` unless
`--jobs-dir` is given.

| Option | Meaning |
|--------|---------|
| `--port 8080` | Port to listen on. |
| `--no-browser` | Do not open a browser tab on start. |
| `--native` | Open a native window instead of a browser tab (requires `pywebview`). |
| `--address 127.0.0.1` | Bind address; loopback by default. |
| `--allow-remote-binding` | Required to bind a non-loopback address. |

## Explorer

The Explorer finds every directory that contains a `manifest.json` (hidden and
staging directories are skipped) and summarizes it with the canonical, data-only
`load_run` reader. Runs that cannot be read are listed with the reason instead of
being hidden. Selecting a succeeded run loads its result with `load_result` and
shows:

- a **2-D projection** on any two objectives with lasso or box selection, plus a
  **3-D** view and **parallel coordinates** for three or more objectives;
- a table with the objectives and the first decision variables of the selected
  solutions, and **Download CSV** with all variables of the selected solutions
  (or of every solution when nothing is selected);
- **Verify integrity**, which runs `verify_run` and reports artifact integrity,
  path safety, and the effective replayability.

## New run with live progress

Choose a built-in problem, an algorithm (or `auto`), the evaluation budget, an
optional population size, the seed, and the engine, then press **Launch**. While
the worker runs, the page refreshes about three times per second:

- the **current non-dominated front** (2-D for two objectives, 3-D for three, and
  parallel coordinates for more), down-sampled to at most 500 points;
- the **progress hypervolume**, computed against a reference point fixed from the
  first reported population. It shows the progress of this run only and is not
  comparable across runs. It is reported for up to three objectives, or up to
  eight when the `moocore` backend is installed.

Zoom, camera, and selections persist while the figures refresh. **Cancel**
terminates the worker, and nothing is stored for a cancelled run. When the run
succeeds, the stored canonical run is shown with the same interactive tools as
the Explorer, and it appears in the Explorer list.

## Security and trust boundary

- The server binds to loopback by default. A non-loopback address requires
  `--allow-remote-binding`. The GUI has **no authentication**: anyone who can reach
  it can launch runs on the machine with your operating-system permissions.
- The GUI runs registered built-in problems and algorithms only. It never executes
  Python code typed in the browser; custom problems are not supported yet.
- Stored runs are read through the canonical inert readers (`load_run`,
  `load_result`, and `verify_run`), which use no pickle.

## How it works

Each launched run gets its own job directory:

```text
gui-runs/<timestamp>-<algorithm>-<problem>/
    request.json   validated run request
    events.jsonl   progress events
    worker.log     worker output
    run/           canonical stored run (written on success)
```

The worker (`python -m vamos.experiment.gui.worker JOB_DIR`) calls
`optimize(..., live_viz=...)` with a progress writer that appends time-throttled
JSON Lines (`vamos.gui-events`, version 1): one `start` event, `progress` events
with the current front and the progress hypervolume, and one terminal `end` event
whose status is `succeeded`, `failed`, or `cancelled`. The page polls this file.
Running each optimization out of process keeps the interface responsive and makes
cancellation a process termination.

## Limitations

- A single-machine, single-user tool without authentication or multi-user
  isolation.
- Built-in problems and algorithms with their default operator settings;
  algorithm-specific parameters are not exposed yet.
- Cancellation terminates the worker process; there is no cooperative stop and no
  partial result.
- Progress history is not part of the stored run, so the Explorer shows final
  fronts only.
