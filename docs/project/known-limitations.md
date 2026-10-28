# Known limitations in VAMOS 1.1.0

These limitations are part of the current release scope. They describe
unsupported behavior rather than future compatibility promises.

## Algorithms and numerical behavior

- VAMOS minimizes every objective. Users must transform maximization
  objectives themselves.
- A valid population size may depend on an algorithm's reference directions or
  weight lattice. Incompatible sizes fail instead of being adjusted silently.
- The strict evaluation budget must be at least the resolved population size.
- Deterministic same-environment runs do not imply bitwise equality across
  different kernels, BLAS implementations, operating systems, architectures,
  or optional third-party engines.

## Benchmark validation

- `welded_beam` is a six-variable mixed design example with its own formulas,
  materials and stiffeners. Its constraint metadata is fixed, but this does
  not certify equivalence to the canonical four-variable welded-beam benchmark.
- **Development after 1.1.0:** `cec2009_uf3` searches the CEC2009 domain
  `[0, 1]^n`. VAMOS 1.1.0 and earlier search `[-1, 1]` for variables 2 to n, so
  UF3 results from those versions do not correspond to the CEC2009 instance.
- Independent algorithm-conformance validation for AGE-MOEA, RVEA and SMPSO
  remains pending. Passing API and regression tests does not establish
  convergence equivalence with a reference implementation.
- IBEA normalizes the objectives over the population for both indicators,
  as adaptive IBEA does.
  - **Effect on multimodal problems:** far-dominated solutions in early
    populations widen the normalization range, which lowers the resolution
    among the other solutions. This slows convergence on problems such as ZDT4
    and DTLZ1; jMetal's IBEA behaves the same.
  - **Earlier versions:**
    - VAMOS 1.1.0 used the epsilon indicator unscaled, so objective units
      shaped its selection.
    - VAMOS 1.0.2 and earlier also inverted the hypervolume indicator and
      counted overflowing contributions as zero; see the
      [IBEA indicator corrections](../reference/algorithms.md#ibea-indicator-corrections-in-110).
  - **Overflow:** with the normalized indicators, contributions exceed the
    float64 range only for `kappa` below about 1/709, and selection then works
    in the log domain.
  - **Infinite objective values:** they give infinite contributions, and
    solutions with such contributions are not ordered among themselves.

## Backends and optional dependencies

- NumPy is the reference path. Numba accelerates selected kernels rather than
  every operation; MooCore accelerates selected indicators rather than whole
  algorithms.
- Explicitly selecting an unavailable backend is an error. VAMOS does not
  install dependencies during a run, verification, or replay.
- Optional research frameworks and model-based tuning packages have their own
  Python and operating-system constraints.
- The release CI claims Python 3.10, 3.11, and 3.12 on Linux, and Python 3.12
  on Windows and macOS. Other interpreter/platform combinations are unclaimed.

## Run artifacts and replay

- Public readers support the 1.0.0 run schema. Internal pre-public formats are
  unsupported and must be regenerated.
- Exact replay is available only for reconstructable registered built-in
  components with exact material environment compatibility.
- Custom Python, plugins, cross-backend execution, and best-effort environment
  matches are not exact-replay targets.
- SHA-256 evidence detects modification; it is not an authenticity signature.

## Durable studies

- Study mutation is single-owner. Do not run, resume, or retry the same study
  concurrently from multiple processes.
- Study execution is sequential in 1.1.0. Distributed workers, multiprocess
  ownership, and cross-process cancellation are unsupported.
- Cancellation is cooperative and local to the process that owns execution.
  Studio's token-addressed worker stop ends the worker and requires reconciliation
  on resume; it is not durable cross-process cancellation of the study API.

## Analysis and tuning

- Statistical analysis, visualization, MCDM helpers, tuning, and racing APIs
  remain experimental and may change in a minor release.
- Users are responsible for selecting indicators, reference points, sample
  sizes, and statistical tests appropriate for their scientific claim.

## Studio and generated code

- Studio is experimental and is intended for trusted local use. It is not a
  multi-user hosted service.
- **Development after 1.1.0:** the Statistical Comparison counts a run without
  feasible solutions as the worst IGD+ (∞). VAMOS 1.1.0 and earlier leave those
  runs out of IGD+, which favors algorithms that are rarely feasible; see
  [Statistical comparison](../guide/studio.md#statistical-comparison).
- AST checks, restricted builtins, process isolation, resource limits, and
  timeouts are best-effort controls, not a security sandbox.
- Reviewed Python executes with the current operating-system user's
  permissions. Remote binding increases exposure and requires explicit opt-in.
- LLM-generated code is displayed for review and is never executed
  automatically.

## Plugins and providers

- Plugin descriptors, custom component interfaces, and LLM-provider
  integrations are experimental and are not covered by the 1.x stable API.
- Loading and verification never import recorded plugins or contact providers.

## Static typing

- The stable API and strict release scope pass mypy. Full-source typing still
  has a frozen diagnostic ratchet and is not yet zero-error; new or increased
  diagnostics fail the release gate.
