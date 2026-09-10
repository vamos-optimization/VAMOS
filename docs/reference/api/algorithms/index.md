# Algorithm configuration

Choose the configuration class for the algorithm passed to `optimize`. These classes are exported through `vamos.algorithms`; their documented fields follow the [1.x stability policy](../../../project/stability-and-versioning.md).

| Algorithm identifier | Public configuration |
| --- | --- |
| `nsgaii` | [`NSGAIIConfig`](nsgaii.md) |
| `nsgaiii` | [`NSGAIIIConfig`](nsgaiii.md) |
| `moead` | [`MOEADConfig`](moead.md) |
| `smsemoa` | [`SMSEMOAConfig`](smsemoa.md) |
| `spea2` | [`SPEA2Config`](spea2.md) |
| `ibea` | [`IBEAConfig`](ibea.md) |
| `smpso` | [`SMPSOConfig`](smpso.md) |
| `agemoea` | [`AGEMOEAConfig`](agemoea.md) |
| `rvea` | [`RVEAConfig`](rvea.md) |

## Explicit configuration

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
config = NSGAIIConfig.default(pop_size=40, n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=config,
    max_evaluations=400,
    seed=42,
)
```

Use a configuration that matches the selected algorithm. Read the generated signature on its page for defaults and construction methods; use `Config.builder()` rather than importing a private builder class.

[Algorithms and backends](../../algorithms.md) explains result modes, archive behavior, capabilities, and probability shorthand such as `"1/n"`. [Optimization](../optimization.md) documents how the configuration is passed to the run.
