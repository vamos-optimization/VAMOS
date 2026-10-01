# VAMOS 1.0.1 release notes

Released 2026-10-01.

VAMOS 1.0.1 packages the fixes and documentation updates merged after the
first public release. The stable 1.x API and the 1.0.0 run and study schemas
remain the compatibility baseline.

## Algorithm and evaluation fixes

- NSGA-II applies the requested steady-state configuration, including with
  the Numba kernel.
- SMPSO honors population and non-dominated result modes and stores an
  explicitly configured external result archive separately from its internal
  leaders. Result storage does not change leader selection or swarm dynamics.
- AGE-MOEA and RVEA apply their configured constraint handling.
- Experimental Dask evaluation can discover the active client. Serial fallback
  still requires explicit opt-in.

## Tuning and documentation

CLI tuning scores a consistent final-population source, validates constrained
algorithm support before execution, and namespaces persistent Optuna studies
for the corrected scoring contract. This experimental tuning change can alter
selected configurations; record the package version with scientific results.

The documentation includes verified installation and troubleshooting recipes,
algorithm and problem decision tables, MOEA/D and NSGA-III tutorials, and real
result plots. The retired multilingual site and internal audit and architecture
decision pages have been removed. Canonical contributor contracts remain in
Architecture Health and the linked guides.

## Installation and compatibility

```bash
python -m pip install --upgrade "vamos-optimization==1.0.1"
```

Python 3.10 or newer remains required. The 1.0.0 compatibility fixtures and
schemas are preserved. Exact replay requires a matching material environment;
an upgrade is not a promise of identical trajectories for algorithms whose
behavior was corrected. Keep the original environment when replaying an older
recorded experiment.

See [stability and versioning](stability-and-versioning.md),
[known limitations](known-limitations.md), and
[release verification](../release_smoke.md).
