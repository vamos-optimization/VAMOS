# Set up the fundamental tutorials

T01–T03 run with the published **VAMOS 1.0.1** package. They use the Stable
public facades, the NumPy backend, and the core dependencies. No repository
checkout, notebook server, plotting package, API key, or optional backend is
needed to execute the downloadable examples.

## Prepare an environment

Follow [Installation](../guide/installation.md) to create and activate an
isolated Python environment. VAMOS supports Python 3.10 or newer; these
recorded tutorial outputs were generated with Python 3.12.

For the exact tested numerical dependencies, download
[tutorial-requirements.txt](../assets/tutorials/tutorial-requirements.txt)
and install from the directory containing it:

```bash
python -m pip install -r tutorial-requirements.txt
```

That file pins VAMOS 1.0.1, NumPy 2.1.3, SciPy 1.15.3, and joblib 1.4.2.
It is a reproducible tutorial recipe, not a change to VAMOS's supported
dependency ranges. If you use a different environment, compare the scientific
invariants (shapes, budget, feasibility, and row alignment) rather than
assuming identical last digits or a fixed non-dominated count.

## Download and run

Each Python file is standalone; save it with its `.py` filename. You can also
copy its complete code block into a new file. The new tutorials are supplied
with this documentation; they are not part of the older `v1.0.1` source ZIP.

| Lesson | Download | Command |
| --- | --- | --- |
| [T01. First result](t01-first-result.md) | [t01-first-result.py](../assets/tutorials/t01-first-result.py) | `python t01-first-result.py` |
| [T02. Constrained problem](t02-constrained-problem.md) | [t02-constrained-problem.py](../assets/tutorials/t02-constrained-problem.py) | `python t02-constrained-problem.py` |
| [T03. Decision](t03-pareto-decision.md) | [t03-pareto-decision.py](../assets/tutorials/t03-pareto-decision.py) | `python t03-pareto-decision.py` |

The scripts print their results and do not write run directories or figures.
The shown figures are generated from those same functions with Matplotlib
3.10.9 as a documentation-maintenance dependency.

## Tested environment and evidence

The [evidence record](../assets/tutorials/evidence.json) records the precise
Python and package versions, source and download hashes, generation command,
settings, actual printed output, and figure hashes. The three lessons use
NSGA-II, seed 42, a population of 40, and a budget of 400 evaluations.

Only small teaching summaries and figures are retained. They are not a
benchmark dataset, a convergence claim, or evidence for algorithm rankings.
No cross-platform bitwise reproducibility is promised. Package version,
example revision, and numerical environment are distinct parts of the record.

To update the maintained sources and figures, follow
[Maintaining fundamental tutorials](../dev/fundamental-tutorials.md). User
workflows for saving and verifying scientific runs remain in
[Run artifacts](../guide/run-artifacts.md).
