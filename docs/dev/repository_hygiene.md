# Repository hygiene

This contract keeps the product tree reviewable and makes generated output
ownership explicit. `release/repository-hygiene-policy.json` is the
machine-readable policy; `release/repository-hygiene-exceptions.json` contains
the narrow path-specific exceptions.

Run the focused gate with:

```bash
python tools/check_repository_hygiene.py
```

The default scan considers Git-tracked files only. It does not inspect external
Goal evidence, CI artifact storage, ignored local outputs, or unrelated
worktrees.

## Root responsibilities

The root contains only canonical project, community, packaging, build,
documentation and agent metadata. Each current root file is named and owned in
the machine policy. A new standard metadata file may use a documented metadata
extension, but the same reviewed change must add its exact name and purpose to
the policy. Numerical results, plots, tables, archives, compiled papers, logs,
audit handoffs and scratch files never belong at root.

Each tracked top-level directory also has one concise owner in the policy. Add a
new one only when no existing owner fits, and update the ownership map and this
document in the same change.

Cloudflare hosting is an explicit infrastructure responsibility: `cloudflare/`
owns the trusted Worker runtime and edge-routing code, while `wrangler.jsonc`
and `wrangler.preview.jsonc` are reviewed root-level deployment metadata for the
production and isolated preview Workers. They are configuration source, not
built-site output.

## Generated outputs and reports

Local and example output belongs below ignored `artifacts/`, `results/`,
`build/`, or `dist/` directories, or at a caller-supplied path. A producer must
create parents safely and must not silently replace an existing output unless
the caller explicitly opts in. Tests use temporary directories.

One-host performance runs are observations, not portable claims. Tools write
them below `artifacts/performance/`; CI uploads that directory as a workflow
artifact. A benchmark baseline may be tracked only when it has a schema,
generation command, source commit, environment and machine context, date,
interpretation and reviewable size budget. Deterministic Markdown views should
be generated from the authoritative structured data rather than retained as a
second source.

Raw audit logs, Goal handoffs and validation transcripts live in an external
Goal-audit directory or CI artifact. Durable public conclusions may be edited
into maintained documentation under `docs/project/` or `docs/topics/`; raw
evidence and placeholder aliases are not product content.

## Scientific data and fixtures

Preserve benchmark definitions, reference fronts, weights, explicit experiment
configs, compatibility fixtures and irreplaceable curated inputs. Generated
runs, tuning databases, traces, temporary campaign slices, backups and derived
plots/tables are external or ignored outputs.

An intentionally tracked reference result needs a nearby manifest recording
its source commit, generation command, schema, expected use and size budget.
`experiments/REFERENCE_RESULTS.md` governs the small result set retained for the
paper. It is scientific evidence, not a VAMOS RunManifest or StudyManifest.
Scientific values are never edited merely to satisfy a hygiene check.

Fixtures live in an explicit test or documented example-fixture directory.
They must state the behavior they stabilize and remain small enough for routine
tests. A generated file does not become a fixture merely because it is useful
once.

## Publication outputs

`paper/` owns manuscript/supplementary source, rebuild scripts, the minimal
curated data, and source figures that cannot be recreated deterministically.
Compiled PDFs, generated plots/tables, submission ZIPs and expanded submission
copies are not tracked. Local paper builds use ignored `paper/build/`,
`paper/generated/` and `paper/dist/` paths. Final deliverables belong on the
relevant release or scholarly archive.

Before removing unique publication or scientific-looking material, preserve an
external manifest and, when reproducibility is uncertain, a verified external
safety archive. Do not create an internal junk/archive directory. External
archives must not contain credentials or private reviewer correspondence.

## Notebooks and documentation assets

Learning notebooks use the `python3` kernel with display name
`Python 3 (VAMOS)`, have no stored outputs or execution counts, avoid personal
absolute paths, and must not create root output. A small pedagogically essential
output requires an explicit exception and test.

`docs/assets/` is authoritative for public documentation images;
`website/` owns website source and does not duplicate those binaries. Built
MkDocs sites are ignored. Keep only referenced assets or explicitly documented
source artwork.

## Large files, duplicates and exceptions

The checker rejects tracked files above the configured size limit and exact
duplicate content above the duplicate threshold. Semantic aliases needed for a
stable package-resource lookup may be excepted, but each exact path must record
category, owner, reason, size and review condition. Wildcard exceptions are
forbidden. Archive-plus-expanded-directory duplication is always reviewed.

Request an exception by editing
`release/repository-hygiene-exceptions.json`, documenting all required fields,
and adding a focused test or consumer reference. Exceptions are re-reviewed
when size, ownership, use or generation provenance changes.

## Command ownership

- `tools/` contains repository validation, maintenance and benchmark commands.
- `release/` contains frozen release requirements and machine-readable release
  or hygiene policy data.
- `scripts/` contains only documented user-facing platform launchers.

Health, CI and the release checker invoke the same no-argument hygiene command.
Release validation additionally applies its distribution-content function to
the built wheel and sdist.
