# VAMOS

<div class="vamos-hero">
  <div class="vamos-hero__copy">
    <p class="vamos-eyebrow">Vectorized Architecture for Multiobjective Optimization Studies</p>
    <h1>Multi-objective optimization<br><span>built for reproducible studies.</span></h1>
    <p class="vamos-hero__lead">A scientific Python framework with one optimization API, vectorized kernels, durable experiments, and explicit reproducibility contracts.</p>
    <div class="vamos-hero__actions">
      <a class="vamos-button vamos-button--primary" href="guide/getting-started/">Get started</a>
      <a class="vamos-button vamos-button--secondary" href="reference/api_reference/">API reference</a>
    </div>
    <div class="vamos-install" aria-label="Install VAMOS with pip">
      <span class="vamos-install__prompt">$</span><code>pip install vamos-optimization</code>
    </div>
  </div>
  <div class="vamos-hero__code" aria-label="Minimal VAMOS example">
    <div class="vamos-code-window__bar"><span></span><span></span><span></span></div>
    <pre><code>from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=400,
    seed=42,
)

print(result.F.shape)</code></pre>
    <div class="vamos-code-window__result">one API · explicit seed · durable results</div>
  </div>
</div>

<div class="vamos-journeys" aria-label="Choose a VAMOS learning path">
  <a class="vamos-journey" href="guide/zero_to_hero/">
    <span class="vamos-journey__number">01</span>
    <h2>Try VAMOS</h2>
    <p>Install the package, optimize a built-in problem, and inspect the Pareto approximation in a few minutes.</p>
    <span class="vamos-journey__link">Quickstart →</span>
  </a>
  <a class="vamos-journey" href="guide/custom-problem/">
    <span class="vamos-journey__number">02</span>
    <h2>Solve my problem</h2>
    <p>Define objectives, bounds, encodings, and constraints through the public problem builder.</p>
    <span class="vamos-journey__link">Custom problems →</span>
  </a>
  <a class="vamos-journey" href="guide/studies/">
    <span class="vamos-journey__number">03</span>
    <h2>Run a reproducible study</h2>
    <p>Freeze a problem–algorithm–seed matrix, persist canonical runs, inspect results, resume work, and verify artifacts.</p>
    <span class="vamos-journey__link">Durable studies →</span>
  </a>
</div>

## Scientific workflows, not just algorithms

VAMOS 1.0 distinguishes stable optimization, run-artifact, and single-owner study surfaces from experimental features such as Studio, provider integrations, and tuning. Check [Stability and versioning](project/stability-and-versioning.md) before depending on an API as a 1.x compatibility commitment.

<div class="vamos-capabilities">
  <div class="vamos-capability"><strong>Algorithms</strong><span>NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA.</span></div>
  <div class="vamos-capability"><strong>Encodings</strong><span>Real, integer, binary, permutation, and mixed decision variables where supported.</span></div>
  <div class="vamos-capability"><strong>Backends</strong><span>NumPy reference execution with optional Numba kernel and MooCore indicator acceleration.</span></div>
  <div class="vamos-capability"><strong>Research tooling</strong><span>Studies, benchmarking, tuning, result inspection, verification, replay, and analysis.</span></div>
</div>

For precise signatures and configuration contracts, use the [API reference](reference/api_reference.md), [algorithm reference](reference/algorithms.md), and [problem reference](reference/problems.md).

## Project and citation

Citation metadata is maintained in [`CITATION.cff`](https://github.com/vamos-optimization/VAMOS/blob/main/CITATION.cff). Security policy and supported reporting channels are maintained in [`SECURITY.md`](https://github.com/vamos-optimization/VAMOS/blob/main/SECURITY.md). See [Known limitations](project/known-limitations.md), the [roadmap](roadmap.md), and [repository governance](project/repository-governance.md) for project-level information.
