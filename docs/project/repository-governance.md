# Repository governance

The canonical source is [vamos-optimization/VAMOS](https://github.com/vamos-optimization/VAMOS).
The organization repository owns authoritative `main`, issues, pull requests,
Actions and CI, the [roadmap](../roadmap.md), security reporting, official tags,
GitHub Releases, release artifacts, and PyPI/TestPyPI publishing.

Personal mirror: [NicolasRodriguezUribe/VAMOS](https://github.com/NicolasRodriguezUribe/VAMOS).

The personal repository receives canonical `main` through normal fast-forward
updates. It preserves the same source history and tree; independent development
belongs in the organization repository. Official tags may be copied only after
successful publication. A mirror or fork is never an authorized package
publisher or source of release artifacts. Repository guards in the shared
workflows enforce this even when both repositories contain the same commit.

Use the canonical [issues](https://github.com/vamos-optimization/VAMOS/issues),
[pull requests](https://github.com/vamos-optimization/VAMOS/pulls), and
[releases](https://github.com/vamos-optimization/VAMOS/releases).
Documentation is hosted at [https://vamos-optimization.org/](https://vamos-optimization.org/).
For security reports, use the organization's
[private vulnerability reporting](https://github.com/vamos-optimization/VAMOS/security/advisories/new).

`python tools/check_repository_identity.py` checks tracked references and the
publication guard. The personal-mirror declaration above is the only permitted
project link to the personal repository. The checker's owner identifier is a
policy constant, not a project link. Historical audit evidence stays outside
the repository. Health, CI, and release validation enforce this policy.

See [release verification](../release_smoke.md) for the publisher identity,
the GitHub Pages redirect, and the required fresh official artifact freeze.
