# VAMOS 1.0.2 release notes

Released 2026-10-18.

VAMOS 1.0.2 extends the optional Numba engine to every built-in algorithm,
speeds up NSGA-II and MOEA/D, and fixes defects that made results depend on the
selected engine or on whether Numba was installed. The stable 1.x API and the
1.0.0 run and study schemas remain the compatibility baseline. The default
engine, the `engine="auto"` policy and the optional status of Numba are
unchanged.

## Engine parity fixes

- The Numba binary tournament now draws tie-break coins exactly like the NumPy
  reference. For the same seed in the same environment, NSGA-II, NSGA-III,
  IBEA, AGE-MOEA and SMS-EMOA with tournament selection now follow the same
  trajectory with `engine="numba"` and `engine="numpy"`.
- Single-mating-pair SBX, which uses compiled arithmetic whenever Numba is
  installed, consumes random draws exactly like the NumPy implementation.
  Steady-state runs (MOEA/D, SMS-EMOA and steady-state NSGA-II) no longer
  depend on whether Numba is installed when two parents are identical.
- `NumbaKernel.nsga2_survival` returns arrays owned by the caller. A later call
  with the same kernel, for example a second run of the same algorithm object,
  no longer overwrites previously returned populations.
- `NumbaKernel.polynomial_mutation` reproduces NumPy for `eta == 1`.

## Numba engine coverage and performance

With `engine="numba"`:

- SBX crossover and polynomial mutation use compiled arithmetic in NSGA-II,
  NSGA-III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO turbulence, AGE-MOEA and RVEA.
  The compiled operators are enabled only after a one-time check confirms that
  the compiled power function matches `np.power` bit for bit; otherwise the
  NumPy operators are used.
- The AGE-MOEA survival scores, SPEA2 archive truncation, IBEA indicator-based
  removal and crowding-distance archive truncation (SMPSO leaders and crowding
  external archives) run as compiled loops.
- Populations with three or more objectives are ranked by sequential-search
  non-dominated sorting, and NSGA-II survival computes crowding only for the
  front it truncates.
- MOEA/D computes its differential-evolution children with a compiled kernel.

Both engines also benefit from cheaper bookkeeping: NSGA-II reuses the
generation-end ranking for the next parent selection and no longer computes a
discarded ranking under constraints, and MOEA/D samples the parents of a single
offspring without vectorized overhead.

On the recorded benchmark campaign, complete runs with the Numba engine were
1.04x to 4.5x faster than with VAMOS 1.0.1, for example NSGA-II on
three-objective DTLZ2 with 500 individuals (1.07 s to 0.24 s) and MOEA/D on ZDT1
(2.55 s to 1.61 s). Final fronts were identical between engines in every
measured case. The first run without a Numba cache compiles more kernels and
takes 0.3 to 1.5 s longer; cached startup is unchanged. Speedups depend on the
problem, the population size and the hardware.

## Installation and compatibility

```bash
python -m pip install --upgrade "vamos-optimization==1.0.2"
```

Python 3.10 or newer remains required. The 1.0.0 compatibility fixtures and
schemas are preserved. Runs with `engine="numba"` that use tournament
selection, and steady-state runs with Numba installed whose parents coincide,
can follow different trajectories than in 1.0.1 because their random draws now
match the NumPy reference. Exact replay requires a matching material
environment; keep the original environment when replaying an older recorded
experiment.

See [stability and versioning](stability-and-versioning.md),
[known limitations](known-limitations.md),
[the 1.0.1 release notes](release-notes-1.0.1.md), and
[release verification](../release_smoke.md).
