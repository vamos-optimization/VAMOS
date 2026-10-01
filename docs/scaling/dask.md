# Scaling with Dask

> **Experimental integration.** Dask evaluation is an optional third-party
> integration and is **outside the VAMOS 1.0 stable compatibility surface**.
> Its behavior and integration details may change in a minor release while the
> distributed execution contract is hardened.

VAMOS can distribute expensive objective-function evaluations through an active
Dask `Client`. The maintained user path is the public `optimize(...)` API with
`eval_strategy="dask"`; user code does not need to import VAMOS evaluation
backend internals.

## Installation

Install VAMOS with the optional distributed-compute dependencies:

```bash
python -m pip install "vamos-optimization[compute]==1.0.1"
```

For environment setup, see [Installation](../guide/installation.md). For an
editable development installation, follow the separate
[source-checkout instructions](../guide/installation.md#install-from-a-source-checkout).

## Quick Start

Create the Dask client first, then ask VAMOS to use the experimental Dask
strategy. VAMOS discovers the active client and does not own or close it.
Save the following as a Python script and run it with `python dask_demo.py`.
The `if __name__ == "__main__"` guard prevents worker processes from starting
the cluster again when they import the script.

```python
from dask.distributed import Client, LocalCluster

from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig

def main():
    problem = make_problem_selection("zdt1", n_var=30).instantiate()
    algo_cfg = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)

    with LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        dashboard_address=None,
    ) as cluster:
        with Client(cluster):
            result = optimize(
                problem,
                algorithm="nsgaii",
                algorithm_config=algo_cfg,
                max_evaluations=10_000,
                seed=42,
                engine="numpy",
                eval_strategy="dask",
            )


if __name__ == "__main__":
    main()
```

## Connecting to an Existing Cluster

The same public path works with an existing scheduler. The `Client` context is
responsible for its own lifecycle.

```python
from dask.distributed import Client

from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig

problem = make_problem_selection("zdt1", n_var=30).instantiate()
algo_cfg = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)

with Client("tcp://scheduler.example.com:8786"):
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        max_evaluations=50_000,
        seed=42,
        engine="numpy",
        eval_strategy="dask",
    )
```

## Failure Behavior

The maintained public Dask path does **not** silently fall back to serial
execution. If no active Dask client is available, or the scheduler cannot be
used, the run fails explicitly. This prevents a distributed benchmark from
quietly producing serial timings.

Create or connect a `Client` before calling `optimize(..., eval_strategy="dask")`.
If Dask is not installed, install the `compute` extra shown above.

## Kubernetes Deployment

The following is a conceptual resource fragment, **not a complete deployment
manifest**. Supply the worker and scheduler configuration required by your
cluster operator before applying it.

```yaml
# dask-cluster.yaml
apiVersion: kubernetes.dask.org/v1
kind: DaskCluster
metadata:
  name: vamos-cluster
spec:
  worker:
    replicas: 10
    resources:
      limits:
        memory: "4Gi"
        cpu: "2"
```

After connecting a `Client` to the cluster scheduler, use the same
`eval_strategy="dask"` call shown above.

## When to Use Distributed Evaluation

| Scenario | Recommended |
|----------|-------------|
| Cheap objectives (<1 ms) | No |
| Medium objectives (10-100 ms) | Maybe; benchmark first |
| Expensive objectives (>1 s) | Yes |
| Large populations with expensive evaluations | Yes |

Dask adds scheduling and serialization overhead. Measure end-to-end wall time on
the actual objective function rather than assuming more workers will improve a
cheap workload.

## Complete Example

See the [Dask example source](https://github.com/vamos-optimization/VAMOS/blob/main/examples/distributed/dask_cluster.py)
for an executable experimental example. Obtain the repository files as shown
in [Examples](../examples.md#obtain-the-example-files), then run from its root.

```bash
# Local comparison
python examples/distributed/dask_cluster.py --compare

# Existing scheduler
python examples/distributed/dask_cluster.py --scheduler tcp://scheduler.example.com:8786
```
