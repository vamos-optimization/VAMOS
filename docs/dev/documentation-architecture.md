# Documentation architecture

This page defines the source-of-truth boundaries for VAMOS documentation while the public portal is consolidated around the canonical `docs/` source tree and validated with Zensical.

## Canonical sources

| Content | Canonical source |
| --- | --- |
| Installation | `docs/guide/installation.md` |
| First optimization and progression to saved runs/studies | `docs/guide/zero_to_hero.md` |
| Navigation between beginner workflows | `docs/guide/getting-started.md` |
| User-defined objective functions and constraints | `docs/guide/custom-problem.md` |
| Durable studies | `docs/guide/studies.md` |
| Individual run persistence, verification, and replay | `docs/guide/run-artifacts.md` |
| Public API navigation and preserved bookmarks | `docs/reference/api_reference.md` |
| Public API signatures | Python code and docstrings rendered through the focused pages in `docs/reference/api/` |
| API documentation maintenance | [Maintaining the API documentation](api-documentation.md) |
| Algorithm/problem contracts | `docs/reference/` plus the implementation and tests they describe |
| Executable scripts | `examples/` |
| Executable notebooks | `notebooks/` |
| Citation metadata | `CITATION.cff` |
| Stable compatibility commitments | `docs/project/stability-and-versioning.md` |
| Contributor contracts | `docs/dev/` and accepted ADRs |

A user-facing page may summarize another source, but it should link to the canonical page instead of copying a second installation procedure, API table, or lifecycle description.

## User navigation model

The public information architecture is task-first:

1. **Get Started** — install VAMOS and complete a first run.
2. **Guides** — solve a custom problem, persist runs, and conduct reproducible studies.
3. **Examples** — locate executable scripts, notebooks, and cookbook recipes.
4. **Reference** — inspect exact API, algorithm, problem, constraint, and stopping contracts.
5. **Developer** — roadmap, extension contracts, tests, architecture, and maintenance material.

Roadmap is the first entry under Developer. Project policy and release pages retain their existing URLs and contextual links without a separate navigation section.

The homepage should route readers into three primary scientific journeys: trying VAMOS, solving their own problem, and running a reproducible study.

## Single public source

`docs/` is the only public documentation source. The former multilingual `website/` tree and its `/website/` route were retired; no parallel content tree or manually maintained copy of API signatures is published. Language publication will be designed only when there is reviewed translated source content and an explicit URL plan.

Architecture decision records in `docs/dev/adr/` remain the repository's accepted decisions, but `exclude_docs` in `mkdocs.yml` keeps them off the published site; contributor pages cite them by repository path.

Raw audit evidence, validation transcripts, and Goal handoffs are not documentation pages. They live outside the product tree or in CI artifact storage, as described in [Architecture Health](architecture_health.md).

## Versioned portal boundary

The portal exposes the release currently designated stable directly at clean root URLs such as `/guide/...`, `/reference/...`, and `/algorithms/...`. Immutable release snapshots remain under `docs/<version>/`. The former moving aliases `docs/stable/...` and `latest/...` are compatibility redirects to the equivalent clean current route, while legacy root-level `<version>/...` routes redirect to `docs/<version>/...`. The exact contract and archive-preservation input are documented in [Documentation versions and archive](../project/documentation-versioning.md).

The versioning layer changes publication layout, not editorial ownership: current user guidance still comes from the repository `docs/` source tree, executable material from `examples/` and `notebooks/`, and historical release trees are frozen build artifacts.

## Zensical migration boundary

The existing MkDocs release builder remains the production publication path until the Zensical migration is explicitly approved. The isolated `Zensical compatibility` workflow validates the canonical root documentation with a separate compatibility dependency stack.

Goal 2 changes information architecture and canonical ownership; it does not deploy to Cloudflare, change the custom domains, replace the versioned release builder, or delete historical URLs. Goal 5 prepares the versioned portal artifact and redirect/archive contract, but merging it likewise does not deploy to Cloudflare or switch the production generator.

## Change rules

- Preserve existing document paths unless a redirect is defined and tested.
- Preserve immutable released documentation when publishing a newer stable release.
- Prefer public facades in user examples.
- Keep scientific claims and citations attached to their maintained source.
- Do not expose historical audits as current user guidance merely because they remain in the repository.
- Add maintained user pages to `mkdocs.yml` so readers can discover them.
- Keep temporary branch previews separate from the current/immutable archive surfaces.
- Run both the existing strict MkDocs build and the Zensical compatibility build for changes to the canonical portal.
