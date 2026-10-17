# Documentation content map

This is the maintained inventory of canonical Markdown pages under `docs/`.
Use it with [Documentation architecture](documentation-architecture.md) when
adding a page or changing navigation. Roles describe the reader's main purpose;
an index inherits the role of the material it routes to. A page can be linked
from several routes while retaining one source and one primary navigation entry.

## Placement rules

- **Tutorial:** follow a worked example or learning path. Beginner setup remains
  under Get Started; algorithm lessons and the examples gallery are in Tutorials.
- **How-to:** complete a specific operation. The How-to Guides index also links
  directly to relevant sections of tutorials and reference pages.
- **Reference:** look up exact interfaces, capabilities, support, or version
  contracts. Internal implementation APIs stay in Developer.
- **Development:** contribute, maintain, or validate the project. Roadmap comes
  first; extension guides, contracts, internal APIs, and Maintainers are grouped
  separately. Project policies remain contextual links, without a Project tab.

The logo links to the homepage. Each user section opens with an index; the
three homepage journeys lead directly to T01, T02, and T06.

## Page boundaries and editorial decisions

| Material | Decision | Boundary to preserve |
| --- | --- | --- |
| Existing pages | Keep every source path and existing heading/explicit anchor. Move only navigation entries. | Bookmarks and historical releases retain their identity. |
| Algorithm lessons | Move into Tutorials; keep the capability matrix in Reference. | A lesson explains a run; configuration signatures remain generated API content. |
| Algorithm and API maintenance | Regroup under Developer → Maintainers and use explicit maintenance labels. | Readers learning an algorithm should reach its lesson. |
| Examples | Keep as a gallery linked from Tutorials, How-to Guides, and the homepage. | Scripts remain in `examples/`; notebooks remain in `notebooks/`. |
| Quickstart | Keep the complete page and anchors; route focused first-run learning to T01–T03. | Preserve its numbered section anchors and link onward instead of duplicating executable sources. |
| Cookbook | Keep the recipe index now; split selected tasks into dedicated guides as they are developed. | Retain the recipe anchors as routing points when content is extracted. |
| Study and analysis material | Keep the lifecycle and analysis pages; connect T06–T08 and their retained study report. | Keep the synthetic API matrix distinct from actual study observations. |
| Merging pages | No page merge is needed for this navigation change. | Prefer cross-links between complementary tutorial, how-to, and reference roles. |

## Complete page inventory

“Keep” preserves placement; “Regroup” changes a menu position or label only;
“Add” introduces a maintained page. Contextual pages remain published
and reachable through the named parent pages even when absent from the menu.

| Page (relative to `docs/`) | Role | Primary entry or contextual access | Decision |
| --- | --- | --- | --- |
| [algorithms/moead.md](../algorithms/moead.md) | Tutorial | Tutorials → Algorithms → MOEA/D | Regroup |
| [algorithms/nsgaii.md](../algorithms/nsgaii.md) | Tutorial | Tutorials → Algorithms → NSGA-II | Regroup |
| [algorithms/nsgaiii.md](../algorithms/nsgaiii.md) | Tutorial | Tutorials → Algorithms → NSGA-III | Regroup |
| [assist.md](../assist.md) | How-to | How-to Guides → Experimental → VAMOS Assist | Regroup |
| [dev/add_algorithm.md](add_algorithm.md) | Development | Developer → Extending VAMOS → Add an Algorithm | Regroup |
| [dev/add_backend.md](add_backend.md) | Development | Developer → Extending VAMOS → Add a Backend | Regroup |
| [dev/add_metric.md](add_metric.md) | Development | Developer → Extending VAMOS → Add a Metric | Regroup |
| [dev/add_operator.md](add_operator.md) | Development | Developer → Extending VAMOS → Add an Operator | Regroup |
| [dev/add_problem.md](add_problem.md) | Development | Developer → Extending VAMOS → Add a Problem | Regroup |
| [dev/algorithm-documentation.md](algorithm-documentation.md) | Development | Developer → Maintainers → Maintaining Algorithm Tutorials | Regroup |
| [dev/api-documentation.md](api-documentation.md) | Development | Developer → Maintainers → Maintaining API Documentation | Regroup |
| [dev/architecture_health.md](architecture_health.md) | Development | Developer → Maintainers → Architecture Health | Regroup |
| [dev/cli.md](cli.md) | Development | Developer → Extending VAMOS → Change the CLI | Regroup |
| [dev/documentation-architecture.md](documentation-architecture.md) | Development | Developer → Maintainers → Documentation Architecture | Regroup |
| [dev/documentation-delivery.md](documentation-delivery.md) | Development | Developer → Maintainers → Documentation Delivery | Regroup |
| [dev/documentation-navigation.md](documentation-navigation.md) | Development | Developer → Maintainers → Documentation Content Map | Add |
| [dev/fundamental-tutorials.md](fundamental-tutorials.md) | Development | Developer → Maintainers → Maintaining Fundamental Tutorials | Add |
| [dev/maintainers.md](maintainers.md) | Development | Developer → Maintainers → Overview | Add |
| [dev/repository_hygiene.md](repository_hygiene.md) | Development | Developer → Maintainers → Repository Hygiene | Regroup |
| [dev/run_artifact_acceptance_tests.md](run_artifact_acceptance_tests.md) | Development | Developer → Contracts → Run Artifact Acceptance Tests | Regroup |
| [dev/run_artifact_contract.md](run_artifact_contract.md) | Development | Developer → Contracts → Run Artifact Contract | Regroup |
| [dev/run_artifact_examples/README.md](run_artifact_examples/README.md) | Development | Run artifact contract links | Keep; contextual links |
| [dev/run_artifacts_and_replay.md](run_artifacts_and_replay.md) | Development | Developer → Extending VAMOS → Run Artifacts and Replay | Regroup |
| [dev/studies.md](studies.md) | Development | Developer → Extending VAMOS → Change Studies | Regroup |
| [dev/study-tutorials.md](study-tutorials.md) | Development | Developer → Maintainers → Maintaining Study Tutorials | Add |
| [dev/study_manifest_acceptance_tests.md](study_manifest_acceptance_tests.md) | Development | Developer → Contracts → StudyManifest Acceptance | Regroup |
| [dev/study_manifest_contract.md](study_manifest_contract.md) | Development | Developer → Contracts → StudyManifest v1 Contract | Regroup |
| [dev/study_manifest_examples/README.md](study_manifest_examples/README.md) | Development | StudyManifest contract links | Keep; contextual links |
| [dev/study_plan_acceptance_tests.md](study_plan_acceptance_tests.md) | Development | Developer → Contracts → Study Planning Acceptance | Regroup |
| [dev/testing.md](testing.md) | Development | Developer → Maintainers → Testing and Validation | Regroup |
| [dev/typing.md](typing.md) | Development | Developer → Maintainers → Typing Policy | Regroup |
| [examples.md](../examples.md) | Tutorial | Tutorials → Examples Gallery | Regroup |
| [experiment/stopping_and_archive.md](../experiment/stopping_and_archive.md) | Reference | Reference → Stopping & Archive | Keep |
| [guide/cli.md](../guide/cli.md) | How-to | How-to Guides → CLI & Config | Regroup |
| [guide/cookbook.md](../guide/cookbook.md) | How-to | How-to Guides → Cookbook | Regroup |
| [guide/custom-problem.md](../guide/custom-problem.md) | Tutorial | Tutorials → First results → Solve Your Own Problem | Regroup |
| [guide/getting-started.md](../guide/getting-started.md) | Tutorial | Get Started → Overview | Keep |
| [guide/index.md](../guide/index.md) | How-to | How-to Guides → Overview | Add |
| [guide/installation.md](../guide/installation.md) | How-to | Get Started → Installation | Keep |
| [guide/minimal-python.md](../guide/minimal-python.md) | Tutorial | Get Started → Minimal Python Track | Keep |
| [guide/run-artifacts.md](../guide/run-artifacts.md) | How-to | How-to Guides → Run Artifacts & Replay | Regroup |
| [guide/studies.md](../guide/studies.md) | Tutorial | Tutorials → From study to report → Study Lifecycle Guide | Regroup |
| [guide/studio.md](../guide/studio.md) | How-to | How-to Guides → Experimental → VAMOS Studio | Regroup |
| [guide/troubleshooting.md](../guide/troubleshooting.md) | How-to | How-to Guides → Troubleshooting | Regroup |
| [guide/understanding-results.md](../guide/understanding-results.md) | Tutorial | Tutorials → First results → Understanding Results | Regroup |
| [guide/zero_to_hero.md](../guide/zero_to_hero.md) | Tutorial | Get Started → Quickstart | Keep |
| [index.md](../index.md) | Tutorial | Logo; homepage | Regroup; logo entry |
| [project/documentation-versioning.md](../project/documentation-versioning.md) | Reference | Reference overview; Maintainers → Documentation Delivery | Keep; contextual links |
| [project/hosting.md](../project/hosting.md) | Development | Maintainers overview; Documentation Delivery | Keep; contextual links |
| [project/known-limitations.md](../project/known-limitations.md) | Reference | Homepage; Reference overview | Keep; contextual links |
| [project/release-notes-1.0.0.md](../project/release-notes-1.0.0.md) | Reference | Reference overview; Documentation versions | Keep; contextual links |
| [project/release-notes-1.0.1.md](../project/release-notes-1.0.1.md) | Reference | Reference overview; Documentation versions | Keep; contextual links |
| [project/repository-governance.md](../project/repository-governance.md) | Development | Homepage; Maintainers overview | Keep; contextual links |
| [project/stability-and-versioning.md](../project/stability-and-versioning.md) | Reference | Homepage; Reference overview; learning indexes | Keep; contextual links |
| [reference/algorithms.md](../reference/algorithms.md) | Reference | Reference → Algorithms & Backends | Regroup |
| [reference/api/algorithms/agemoea.md](../reference/api/algorithms/agemoea.md) | Reference | Reference → API Reference → Algorithm configuration → AGE-MOEA | Keep |
| [reference/api/algorithms/ibea.md](../reference/api/algorithms/ibea.md) | Reference | Reference → API Reference → Algorithm configuration → IBEA | Keep |
| [reference/api/algorithms/index.md](../reference/api/algorithms/index.md) | Reference | Reference → API Reference → Algorithm configuration → Overview | Keep |
| [reference/api/algorithms/moead.md](../reference/api/algorithms/moead.md) | Reference | Reference → API Reference → Algorithm configuration → MOEA/D | Keep |
| [reference/api/algorithms/nsgaii.md](../reference/api/algorithms/nsgaii.md) | Reference | Reference → API Reference → Algorithm configuration → NSGA-II | Keep |
| [reference/api/algorithms/nsgaiii.md](../reference/api/algorithms/nsgaiii.md) | Reference | Reference → API Reference → Algorithm configuration → NSGA-III | Keep |
| [reference/api/algorithms/rvea.md](../reference/api/algorithms/rvea.md) | Reference | Reference → API Reference → Algorithm configuration → RVEA | Keep |
| [reference/api/algorithms/smpso.md](../reference/api/algorithms/smpso.md) | Reference | Reference → API Reference → Algorithm configuration → SMPSO | Keep |
| [reference/api/algorithms/smsemoa.md](../reference/api/algorithms/smsemoa.md) | Reference | Reference → API Reference → Algorithm configuration → SMS-EMOA | Keep |
| [reference/api/algorithms/spea2.md](../reference/api/algorithms/spea2.md) | Reference | Reference → API Reference → Algorithm configuration → SPEA2 | Keep |
| [reference/api/discovery.md](../reference/api/discovery.md) | Reference | Reference → API Reference → Discovery | Keep |
| [reference/api/experimental/diagnostics.md](../reference/api/experimental/diagnostics.md) | Reference | Reference → API Reference → Experimental APIs → Diagnostics | Keep |
| [reference/api/experimental/parameter-spaces.md](../reference/api/experimental/parameter-spaces.md) | Reference | Reference → API Reference → Experimental APIs → Parameter spaces | Keep |
| [reference/api/experimental/tuning.md](../reference/api/experimental/tuning.md) | Reference | Reference → API Reference → Experimental APIs → Tuning and racing | Keep |
| [reference/api/internal/constraints.md](../reference/api/internal/constraints.md) | Development | Developer → Internal API → Constraint implementation | Keep |
| [reference/api/internal/generic-configuration.md](../reference/api/internal/generic-configuration.md) | Development | Developer → Internal API → Generic configuration | Keep |
| [reference/api/internal/problem-types.md](../reference/api/internal/problem-types.md) | Development | Developer → Internal API → Problem adapters and encodings | Keep |
| [reference/api/optimization.md](../reference/api/optimization.md) | Reference | Reference → API Reference → Optimization | Keep |
| [reference/api/problem-definition.md](../reference/api/problem-definition.md) | Reference | Reference → API Reference → Problem definition | Keep |
| [reference/api/results.md](../reference/api/results.md) | Reference | Reference → API Reference → Results | Keep |
| [reference/api/run-models.md](../reference/api/run-models.md) | Reference | Reference → API Reference → Run artifacts → Stored runs and reports | Keep |
| [reference/api/runs.md](../reference/api/runs.md) | Reference | Reference → API Reference → Run artifacts → Save, load, and replay | Keep |
| [reference/api/studies.md](../reference/api/studies.md) | Reference | Reference → API Reference → Durable studies → Lifecycle | Keep |
| [reference/api/study-models.md](../reference/api/study-models.md) | Reference | Reference → API Reference → Durable studies → Limits and reports | Keep |
| [reference/api/study-specification.md](../reference/api/study-specification.md) | Reference | Reference → API Reference → Durable studies → Specification and planning | Keep |
| [reference/api_reference.md](../reference/api_reference.md) | Reference | Reference → API Reference → Overview | Keep |
| [reference/constraints.md](../reference/constraints.md) | Reference | Reference → Constraints | Keep |
| [reference/index.md](../reference/index.md) | Reference | Reference → Overview | Add |
| [reference/problems.md](../reference/problems.md) | Reference | Reference → Problems | Keep |
| [release_smoke.md](../release_smoke.md) | Development | Developer → Maintainers → Release Verification | Regroup |
| [roadmap.md](../roadmap.md) | Development | Developer → Roadmap | Keep |
| [scaling/dask.md](../scaling/dask.md) | How-to | How-to Guides → Experimental → Scaling with Dask | Regroup |
| [topics/analysis.md](../topics/analysis.md) | How-to | How-to Guides → Experimental → Analysis & Visualization | Regroup |
| [topics/extending.md](../topics/extending.md) | Development | Developer → Extending VAMOS → Overview | Regroup |
| [topics/plugin_guide.md](../topics/plugin_guide.md) | Development | Developer → Extending VAMOS → Plugin Guide | Regroup |
| [topics/tuning.md](../topics/tuning.md) | How-to | How-to Guides → Experimental → Hyperparameter Tuning | Regroup |
| [tutorials/index.md](../tutorials/index.md) | Tutorial | Tutorials → Overview | Add |
| [tutorials/recorded-study-report.md](../tutorials/recorded-study-report.md) | Tutorial | Tutorials → From study to report → Recorded Study Report | Add |
| [tutorials/setup.md](../tutorials/setup.md) | How-to | Tutorials → Tutorial Setup | Add |
| [tutorials/study-setup.md](../tutorials/study-setup.md) | How-to | Tutorials → From study to report → Study Setup | Add |
| [tutorials/t01-first-result.md](../tutorials/t01-first-result.md) | Tutorial | Tutorials → First results → T01. Your First Result | Add |
| [tutorials/t02-constrained-problem.md](../tutorials/t02-constrained-problem.md) | Tutorial | Tutorials → First results → T02. Your Constrained Problem | Add |
| [tutorials/t03-pareto-decision.md](../tutorials/t03-pareto-decision.md) | Tutorial | Tutorials → First results → T03. From Pareto Set to Decision | Add |
| [tutorials/t06-fair-study.md](../tutorials/t06-fair-study.md) | Tutorial | Tutorials → From study to report → T06. A Fair Comparative Study | Add |
| [tutorials/t07-quality-and-budget.md](../tutorials/t07-quality-and-budget.md) | Tutorial | Tutorials → From study to report → T07. Quality and Budget | Add |
| [tutorials/t08-study-report.md](../tutorials/t08-study-report.md) | Tutorial | Tutorials → From study to report → T08. From Results to Report | Add |
