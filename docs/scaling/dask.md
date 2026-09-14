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
pip install "vamos-optimization[compute]"
```

For a local editable checkout, the equivalent developer command is
`pip install -e ".[compute]"`.

## Quick Start

Create the Dask client first, then ask VAMOS to use the experimental Dask
strategy. VAMOS discovers the active client and does not own or close it.

```python
from dask.distributed import Client, LocalCluster

from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig

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

See `examples/distributed/dask_cluster.py` for an executable experimental
example.

```bash
# Local comparison
python examples/distributed/dask_cluster.py --compare

# Existing scheduler
python examples/distributed/dask_cluster.py --scheduler tcp://scheduler.example.com:8786
```
