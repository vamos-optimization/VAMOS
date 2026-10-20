# NumPy and Numba engine benchmarks

This directory holds the reproducible benchmark harness used to evaluate the
compiled (Numba) kernels against the NumPy reference, together with the raw
measurements and a summary of one recorded campaign.

## What is measured

`benchmark.py` measures whichever `vamos` package is importable, so two source
trees are compared by changing `PYTHONPATH`:

| Subcommand | Measurement |
| --- | --- |
| `runs` | Complete `vamos.optimize` runs with the JIT already compiled in the process. One warm-up run per case is excluded. Every seed is timed separately, and a separate pass records the Python peak allocation with `tracemalloc`. Each record stores a digest of the final objective matrix so identical trajectories can be checked across configurations. |
| `startup` | Fresh processes: import plus two short runs, first with an empty Numba cache (`cold`), then with the cache written by the previous process (`cached`). |
| `kernels` | Per-operation timings of ranking, NSGA-II survival, tournament selection, SBX and polynomial mutation. |
| `report` | Markdown summary with medians and interquartile ranges. |

`run_suite.py` runs four configurations:

| Label | Interpreter | Source | Engine |
| --- | --- | --- | --- |
| `numpy-old` | environment **without** Numba | baseline checkout | `numpy` |
| `numpy-new` | environment **without** Numba | candidate checkout | `numpy` |
| `numba-old` | environment with the `compute` extra | baseline checkout | `numba` |
| `numba-new` | environment with the `compute` extra | candidate checkout | `numba` |

The NumPy configurations use an environment without Numba because two compiled
paths are selected by Numba availability rather than by the engine: MOEA/D
neighbourhood updates (`VAMOS_USE_NUMBA_MOEAD`, enabled by default) and
single-pair SBX. Running the NumPy engine where Numba is installed would
compare two configurations that partly share the same compiled kernels. The
suite also removes `VAMOS_USE_NUMBA_MOEAD`/`VAMOS_USE_NUMBA_VARIATION` from the
environment and pins BLAS/OpenMP/Numba to one thread.

## Reproduce

```bash
python experiments/numba_acceleration/run_suite.py \
    --numpy-python <no-numba-env>/python \
    --numba-python <compute-env>/python \
    --baseline-src <baseline-checkout>/src \
    --candidate-src src \
    --output-dir results/numba_acceleration
```

Install both environments from `constraints/ci.txt`: `-e ".[dev]" moocore`
for the NumPy environment and `-e ".[dev,compute]"` for the Numba environment.
MooCore must be present in both, because hypervolume contributions use it
whenever it is importable, whatever the engine; without it, three-objective
SMS-EMOA falls back to a far slower exact Python computation. Outputs go to the
ignored `results/` directory; copy a reviewed campaign here.

## Campaign 2026-10-18

Raw JSON lines and the generated summary are in
[`results/2026-10-18/`](results/2026-10-18/summary.md).

**Sources.** Baseline: `ffa15b96` (`main` before this work). Candidate:
`857b6f15` (all source changes of this work). Both clean checkouts.

**Environment.** Windows 11 (10.0.26300), Intel Core Ultra 9 185H (hybrid
performance/efficiency cores), AC power, balanced power plan. Python 3.12.3,
NumPy 2.1.3, Numba 0.67.0 (Numba configurations only), MooCore 0.3.2 in both
environments. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`NUMBA_NUM_THREADS` were 1. Every benchmark process was pinned to logical
processor 14 (a performance core); without pinning, the scheduler moved runs
between core types and timings varied by up to 2x. Full-run cases were
interleaved across the four configurations with a rotating order.

**Method.** Five seeds per case, one run per seed after an excluded warm-up run
in the same process; median and interquartile range are reported. Startup was
measured in fresh processes with an empty Numba cache (`cold`) and with the
cache written by the previous process (`cached`, three repetitions).

### Full optimizations (median seconds, warm JIT)

| Case | NumPy old | NumPy new | Numba old | Numba new | Numba new vs old | Numba new vs NumPy old |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NSGA-II ZDT1, N=100 | 0.528 | 0.469 | 0.134 | 0.078 | 1.72x | 6.8x |
| NSGA-II ZDT1, N=1000 | 6.115 | 5.551 | 0.076 | 0.037 | 2.04x | 164x |
| NSGA-II DTLZ2 M=3, N=100 | 0.556 | 0.477 | 0.182 | 0.083 | 2.20x | 6.7x |
| NSGA-II DTLZ2 M=3, N=500 | 8.649 | 8.088 | 1.070 | 0.237 | 4.52x | 36.5x |
| NSGA-II DTLZ2 M=5, N=200 | 1.140 | 0.974 | 0.298 | 0.107 | 2.78x | 10.6x |
| NSGA-II DTLZ2 M=10, N=200 | 1.353 | 1.145 | 0.405 | 0.195 | 2.07x | 6.9x |
| NSGA-II C1-DTLZ1 M=3, N=100 | 0.376 | 0.230 | 0.124 | 0.069 | 1.79x | 5.4x |
| NSGA-II expensive DTLZ2 (0.6 ms/eval) | 3.281 | 3.019 | 3.052 | 2.930 | 1.04x | 1.12x |
| NSGA-II steady state ZDT1 | 3.126 | 2.052 | 1.412 | 1.150 | 1.23x | 2.7x |
| NSGA-II steady state DTLZ2 M=3 | 2.891 | 1.827 | 1.347 | 1.249 | 1.08x | 2.3x |
| MOEA/D ZDT1 | 6.873 | 6.610 | 2.552 | 1.607 | 1.59x | 4.3x |
| MOEA/D DTLZ2 M=3 | 7.700 | 7.682 | 2.802 | 1.858 | 1.51x | 4.1x |
| MOEA/D C1-DTLZ1 M=3 | 3.572 | 3.591 | 1.583 | 1.066 | 1.48x | 3.4x |
| NSGA-III DTLZ2 M=3 | 0.590 | 0.595 | 0.300 | 0.228 | 1.32x | 2.6x |
| SMS-EMOA ZDT1 | 2.596 | 1.503 | 1.230 | 1.130 | 1.09x | 2.3x |
| SMS-EMOA DTLZ2 M=3 | 2.369 | 1.313 | 1.012 | 0.885 | 1.14x | 2.7x |
| SPEA2 DTLZ2 M=3 | 0.682 | 0.692 | 0.700 | 0.386 | 1.82x | 1.8x |
| IBEA ZDT1 | 0.228 | 0.227 | 0.225 | 0.142 | 1.59x | 1.6x |
| SMPSO DTLZ2 M=3 | 0.241 | 0.238 | 0.246 | 0.169 | 1.45x | 1.4x |
| AGE-MOEA DTLZ2 M=3 | 0.607 | 0.609 | 0.445 | 0.201 | 2.21x | 3.0x |
| RVEA DTLZ2 M=3 | 0.054 | 0.056 | 0.052 | 0.038 | 1.39x | 1.4x |

Final fronts were bitwise identical for every seed between NumPy old, NumPy new
and Numba new in all 21 cases. Numba old differs wherever its tournament or its
single-pair SBX consumed different random draws (fixed by this work).

The NumPy engine became faster in two places: NSGA-II no longer ranks an
unchanged population twice per generation (and no longer computes a discarded
ranking under constraints), and steady-state runs without Numba no longer retry
a failing `import numba` for every single-pair SBX call.

### Compilation and startup (median seconds, NSGA-II ZDT1, 2000 evaluations)

| | Cold process | Import | First run, cold | Process, cached | First run, cached |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy new | 1.11 | 0.83 | 0.07 | 1.06 | 0.07 |
| Numba old | 3.64 | 0.96 | 2.22 | 1.60 | 0.42 |
| Numba new | 4.88 | 0.78 | 3.67 | 1.54 | 0.44 |

Without a cache, the additional kernels lengthen the first run by 0.3-1.5 s
(0.9 s for NSGA-II DTLZ2, 0.3 s for MOEA/D ZDT1; see the summary). This is a
one-time cost: with the on-disk cache, startup is unchanged.

### Measured trade-offs

- SBX on a single mating pair is 2.5-4 us slower with Numba than the previous
  single-pair kernel, and the binary tournament is 2-4 us slower; both now
  consume random draws exactly like NumPy.
- Two-objective ranking was already compiled and is unchanged; the sorting gain
  starts at three objectives.
- Python peak allocation (`tracemalloc`) did not increase; it decreased slightly
  for Numba NSGA-II because survival no longer copies into persistent buffers.

## Automatic selection after 1.0.2

`runs` and `startup` also accept `--engine auto`. Run them separately for
`numpy`, `numba`, and `auto`, using the same cases, seeds and thread settings.
Run records include the effective engine. `startup --all-algorithms` covers all
nine built-ins, with an empty cache in a fresh process, disk-cache startup in
another process, and a subsequent call with warm JIT. Selection itself does not
compile or draw random values. Keep generated reports under ignored
`artifacts/performance/`; the existing 2026-10-18 evidence remains historical.

### 2026-10-20 selection measurements

Measured on 2026-10-20: Linux x86-64, AMD EPYC 9V74, Python 3.12.14, NumPy 2.2.6, Numba 0.67.0 / llvmlite 0.49.0. One logical processor (CPU 0), one BLAS/Numba thread; sequential processes after the test suites completed. Results describe this machine and dependency build, not a universal speedup.

Warm optimization: median seconds over seeds 0–4, one measured run per seed after a separate excluded warm-up. The separate tracemalloc pass is excluded from timings. Final objective digests match all three engines for **50/50 case-seed pairs**. All auto calls resolved to Numba.

| Workload | Evaluations | NumPy | Numba | auto | NumPy / auto |
| --- | ---: | ---: | ---: | ---: | ---: |
| nsgaii-zdt1-m2-p100 | 20,000 | 0.842 | 0.152 | 0.128 | 6.59x |
| nsgaii-dtlz2-m3-p500 | 50,000 | 14.643 | 0.334 | 0.299 | 48.94x |
| moead-zdt1-m2-p100 | 20,000 | 2.535 | 2.442 | 2.499 | 1.01x |
| nsgaiii-dtlz2-m3-p91 | 20,000 | 0.872 | 0.282 | 0.364 | 2.40x |
| smsemoa-zdt1-m2-p100 | 5,000 | 1.667 | 1.800 | 1.732 | 0.96x |
| spea2-dtlz2-m3-p100 | 10,000 | 1.174 | 0.676 | 0.688 | 1.71x |
| ibea-zdt1-m2-p100 | 10,000 | 0.344 | 0.242 | 0.261 | 1.32x |
| smpso-dtlz2-m3-p100 | 10,000 | 0.265 | 0.189 | 0.198 | 1.34x |
| agemoea-dtlz2-m3-p100 | 10,000 | 0.899 | 0.207 | 0.241 | 3.73x |
| rvea-dtlz2-m3-p91 | 10,000 | 0.054 | 0.063 | 0.063 | 0.85x |

Startup uses **2,000 evaluations** per call (N=100, except NSGA-III/RVEA N=91). These are different budgets from the warm workloads above. Each cold measurement uses a fresh process and empty Numba cache. The warm value below is the median second-call time across the cold process and two new disk-cache processes. Disk-cache first-call values are medians of those two fresh processes. Times wrap optimize(), including backend load/selection and compilation when reached; they exclude Python/VAMOS import, which is recorded separately by the harness.

| Startup case | NumPy cold / warm | Numba cold / warm | auto cold / warm | auto disk-cache first |
| --- | ---: | ---: | ---: | ---: |
| nsgaii-zdt1 | 0.102 / 0.087 | 3.373 / 0.017 | 3.466 / 0.016 | 0.366 |
| nsgaii-dtlz2 | 0.084 / 0.078 | 2.880 / 0.013 | 2.890 / 0.015 | 0.341 |
| moead-zdt1 | 3.036 / 0.234 | 2.809 / 0.198 | 3.120 / 0.193 | 0.520 |
| nsgaiii-dtlz2 | 0.092 / 0.083 | 2.551 / 0.028 | 2.457 / 0.029 | 0.405 |
| smsemoa-dtlz2 | 0.921 / 0.591 | 3.003 / 0.647 | 2.864 / 0.710 | 1.013 |
| spea2-dtlz2 | 0.119 / 0.101 | 1.499 / 0.102 | 1.458 / 0.092 | 0.391 |
| ibea-dtlz2 | 0.072 / 0.060 | 1.409 / 0.049 | 1.415 / 0.049 | 0.352 |
| smpso-dtlz2 | 0.042 / 0.037 | 1.706 / 0.029 | 1.907 / 0.029 | 0.308 |
| agemoea-dtlz2 | 0.129 / 0.137 | 2.881 / 0.029 | 2.971 / 0.030 | 0.309 |
| rvea-dtlz2 | 0.029 / 0.014 | 0.449 / 0.017 | 0.418 / 0.014 | 0.318 |

The compiled-power guard disabled SBX/PM acceleration on this NumPy build; the remaining Numba kernels stayed active. NumPy timings use the same environment, which already permits the existing Numba single-pair/MOEA-D shortcuts. Thus they are not timings from the separate core-only installation. Short-run startup can outweigh warm throughput gains, and algorithms with little active compiled work can show no gain. Auto and explicit Numba execute the same algorithm implementation; differences in these sequential timings include machine variability and are not attributed to resolver performance.

Reproduce with the existing harness (set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and NUMBA_NUM_THREADS to 1 before starting Python):

```bash
python experiments/numba_acceleration/benchmark.py startup --engine auto --label auto --cpu 0 --all-algorithms --repeat 2 --output artifacts/performance/startup-auto.jsonl
python experiments/numba_acceleration/benchmark.py runs --engine auto --label auto --cpu 0 --repeat 1 --seeds 0 1 2 3 4 --cases nsgaii-zdt1-m2-p100 nsgaii-dtlz2-m3-p500 moead-zdt1-m2-p100 nsgaiii-dtlz2-m3-p91 smsemoa-zdt1-m2-p100 spea2-dtlz2-m3-p100 ibea-zdt1-m2-p100 smpso-dtlz2-m3-p100 agemoea-dtlz2-m3-p100 rvea-dtlz2-m3-p91 --output artifacts/performance/runs-auto.jsonl
```

Repeat both commands with `--engine numpy` / `--label numpy` and `--engine numba` / `--label numba`, each with a distinct output filename. Use an available logical CPU on your host. Raw JSON-lines and generated reports remain in ignored `artifacts/performance/`; maintained conclusions are recorded here.

### Check with rotating engine order

To investigate small slowdowns in the sequential measurements, the same warm cases below were repeated in one pinned process over seeds 0–8, rotating the order of NumPy, Numba and auto each seed. Each engine received an excluded warm-up. All 27 case-seed comparisons had identical final objective digests. Median seconds:

| Case | NumPy | Numba | auto |
| --- | ---: | ---: | ---: |
| rvea-dtlz2-m3-p91 | 0.057 | 0.056 | 0.058 |
| smsemoa-zdt1-m2-p100 | 1.668 | 1.712 | 1.752 |
| nsgaiii-dtlz2-m3-p91 | 0.902 | 0.339 | 0.290 |

Warm auto resolution itself took 1.87 microseconds per call (median of five batches of 10,000 calls). RVEA was essentially flat in this follow-up; SMS-EMOA remained about 5% slower with auto on this build. Automatic Numba preference is not a performance guarantee for every workload, especially when compiled variation is disabled. The omitted NumPy default remains available without any behavior change.
