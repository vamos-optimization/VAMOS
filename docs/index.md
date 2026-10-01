---
title: Multi-objective optimization in Python
description: Compare algorithms, solve custom problems, and run reproducible optimization studies with VAMOS, a scientific Python framework.
hide:
  - navigation
  - toc
---

<div class="vamos-hero-shell">
<div class="vamos-hero">
  <div class="vamos-hero__copy">
    <p class="vamos-eyebrow">VAMOS · Scientific Python</p>
    <h1><span class="vamos-hero__headline-main"><span class="vamos-hero__nowrap">Multi-objective</span> optimization</span><span class="vamos-hero__headline-accent">built for reproducible studies.</span></h1>
    <p class="vamos-hero__lead">Compare algorithms, solve your own problems, and preserve the evidence behind every run.</p>
    <div class="vamos-hero__actions">
      <a class="vamos-button vamos-button--primary" href="guide/getting-started/">Get started</a>
      <a class="vamos-button vamos-button--secondary" href="reference/api_reference/">API reference</a>
    </div>
    <div class="vamos-install" aria-label="Install VAMOS with pip">
      <span class="vamos-install__prompt">$</span><code>python -m pip install "vamos-optimization==1.0.1"</code>
    </div>
  </div>
  <div class="vamos-hero__demo" aria-label="An optimization and its actual result">
  <div class="vamos-hero__code" aria-label="Minimal VAMOS example">
    <div class="vamos-code-window__bar"><span></span><span></span><span></span></div>
    <pre><code>from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    pop_size=40,
    max_evaluations=400,
    engine="numpy",
    seed=42,
)

front = result.front()</code></pre>
    <div class="vamos-code-window__result">40 solutions · 2 objectives</div>
  </div>
  <figure class="vamos-hero__result">
    <a href="guide/understanding-results/" aria-label="Understand this optimization result">
      <img src="assets/results/understanding-results.svg" width="720" height="500" alt="Actual NSGA-II run on ZDT1: grey circles show the final population, and teal crosses identify its non-dominated subset." fetchpriority="high">
    </a>
    <figcaption>Real output from the code shown: 400 evaluations, seed 42. A first run, not a convergence benchmark. <a href="guide/understanding-results/">Read and reproduce this result →</a></figcaption>
  </figure>
  </div>
</div>
</div>

<div class="vamos-journeys" aria-label="Choose a VAMOS learning path">
  <a class="vamos-journey" href="guide/zero_to_hero/">
    <span class="vamos-journey__number">01</span>
    <h2>Try VAMOS</h2>
    <p>Run a built-in problem and inspect its trade-offs.</p>
    <span class="vamos-journey__link">Quickstart →</span>
  </a>
  <a class="vamos-journey" href="guide/custom-problem/">
    <span class="vamos-journey__number">02</span>
    <h2>Solve my problem</h2>
    <p>Define your decisions, objectives, bounds, and constraints.</p>
    <span class="vamos-journey__link">Custom problems →</span>
  </a>
  <a class="vamos-journey" href="guide/studies/">
    <span class="vamos-journey__number">03</span>
    <h2>Run a reproducible study</h2>
    <p>Plan a study, check its budget, and trace every result.</p>
    <span class="vamos-journey__link">Reproducible studies →</span>
  </a>
</div>

**Prefer runnable source?** [Run the three executable journeys](examples.md#three-executable-journeys).

## From a first run to a scientific study

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
