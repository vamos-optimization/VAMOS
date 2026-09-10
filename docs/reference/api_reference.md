# API Reference

Find a public function, configuration, or result type without browsing the entire implementation. Signatures and parameter descriptions on the linked pages are generated from Python docstrings.

New to VAMOS? Start with the [Quickstart](../guide/zero_to_hero.md) or [solve your own problem](../guide/custom-problem.md). For commands rather than Python imports, see [CLI and configuration](../guide/cli.md).

## Core optimization

| I need to… | Reference |
| --- | --- |
| Run an algorithm | [Optimization](api/optimization.md) |
| Define objectives, bounds, and constraints | [Problem definition](api/problem-definition.md) |
| Inspect decision variables and objective values | [Results](api/results.md) |
| Find supported problem, algorithm, or operator identifiers | [Discovery](api/discovery.md) |
| Set algorithm parameters | [Algorithm configuration](api/algorithms/index.md) |

## Reproducible runs and studies

| I need to… | Reference |
| --- | --- |
| Save, load, verify, or replay a run | [Run lifecycle](api/runs.md) |
| Inspect a stored run or verification report | [Run models](api/run-models.md) |
| Define and preview an experiment matrix | [Study specification and planning](api/study-specification.md) |
| Create, run, resume, or retry a study | [Study lifecycle](api/studies.md) |
| Inspect study limits, reports, and summaries | [Study models](api/study-models.md) |

## Stability and advanced APIs

The references above use the public facades `vamos`, `vamos.algorithms`, `vamos.run_artifacts`, and `vamos.study_artifacts`. Only the surfaces listed in [Stability and versioning](../project/stability-and-versioning.md) and frozen under `tests/compatibility/v1_0_0/` have a 1.x compatibility commitment.

[Tuning](api/experimental/tuning.md), [parameter spaces](api/experimental/parameter-spaces.md), and [development diagnostics](api/experimental/diagnostics.md) are **experimental**. [Constraint internals](api/internal/constraints.md), [problem adapters and encodings](api/internal/problem-types.md), and [generic configuration](api/internal/generic-configuration.md) are **implementation references**, not stable imports.

## Existing bookmarks

Old section and symbol bookmarks are forwarded to their new page within the same documentation version. With JavaScript disabled, expand the directory below and follow the corresponding link. Unknown fragments stay on this index.

<details id="api-legacy-bookmarks">
<summary>Legacy bookmark directory</summary>
<ul>
<li><a id="unified-api" data-api-legacy href="../api/optimization/">unified-api</a></li>
<li><a id="vamos.experiment.unified" data-api-legacy href="../api/optimization/">vamos.experiment.unified</a></li>
<li><a id="vamos.experiment.unified.optimize" data-api-legacy href="../api/optimization/#vamos.optimize">vamos.experiment.unified.optimize</a></li>
<li><a id="vamos.experiment.unified.optimize--automl-mode-zero-config" data-api-legacy href="../api/optimization/#vamos.optimize--automl-mode-zero-config">vamos.experiment.unified.optimize--automl-mode-zero-config</a></li>
<li><a id="vamos.experiment.unified.optimize--specify-algorithm" data-api-legacy href="../api/optimization/#vamos.optimize--specify-algorithm">vamos.experiment.unified.optimize--specify-algorithm</a></li>
<li><a id="vamos.experiment.unified.optimize--multi-seed-study" data-api-legacy href="../api/optimization/#vamos.optimize--multi-seed-study">vamos.experiment.unified.optimize--multi-seed-study</a></li>
<li><a id="problem-definition" data-api-legacy href="../api/problem-definition/">problem-definition</a></li>
<li><a id="vamos.foundation.problem.base" data-api-legacy href="../api/problem-definition/">vamos.foundation.problem.base</a></li>
<li><a id="vamos.foundation.problem.base.Problem" data-api-legacy href="../api/problem-definition/#vamos.Problem">vamos.foundation.problem.base.Problem</a></li>
<li><a id="vamos.foundation.problem.base.Problem.encoding" data-api-legacy href="../api/problem-definition/#vamos.Problem.encoding">vamos.foundation.problem.base.Problem.encoding</a></li>
<li><a id="vamos.foundation.problem.base.Problem.n_constraints" data-api-legacy href="../api/problem-definition/#vamos.Problem.n_constraints">vamos.foundation.problem.base.Problem.n_constraints</a></li>
<li><a id="vamos.foundation.problem.base.Problem.constraints" data-api-legacy href="../api/problem-definition/#vamos.Problem.constraints">vamos.foundation.problem.base.Problem.constraints</a></li>
<li><a id="vamos.foundation.problem.base.Problem.evaluate" data-api-legacy href="../api/problem-definition/#vamos.Problem.evaluate">vamos.foundation.problem.base.Problem.evaluate</a></li>
<li><a id="vamos.foundation.problem.base.Problem.objectives" data-api-legacy href="../api/problem-definition/#vamos.Problem.objectives">vamos.foundation.problem.base.Problem.objectives</a></li>
<li><a id="vamos.foundation.problem.builder" data-api-legacy href="../api/problem-definition/">vamos.foundation.problem.builder</a></li>
<li><a id="vamos.foundation.problem.builder.FunctionalProblem" data-api-legacy href="../api/internal/problem-types/#vamos.foundation.problem.builder.FunctionalProblem">vamos.foundation.problem.builder.FunctionalProblem</a></li>
<li><a id="vamos.foundation.problem.builder.FunctionalProblem.evaluate" data-api-legacy href="../api/internal/problem-types/#vamos.foundation.problem.builder.FunctionalProblem.evaluate">vamos.foundation.problem.builder.FunctionalProblem.evaluate</a></li>
<li><a id="vamos.foundation.problem.builder.make_problem" data-api-legacy href="../api/problem-definition/#vamos.make_problem">vamos.foundation.problem.builder.make_problem</a></li>
<li><a id="vamos.foundation.problem.types" data-api-legacy href="../api/internal/problem-types/">vamos.foundation.problem.types</a></li>
<li><a id="results" data-api-legacy href="../api/results/">results</a></li>
<li><a id="vamos.experiment.optimization_result" data-api-legacy href="../api/results/">vamos.experiment.optimization_result</a></li>
<li><a id="vamos.experiment.optimization_result.OptimizationResult" data-api-legacy href="../api/results/#vamos.OptimizationResult">vamos.experiment.optimization_result.OptimizationResult</a></li>
<li><a id="vamos.experiment.optimization_result.OptimizationResult.manifest" data-api-legacy href="../api/results/#vamos.OptimizationResult.manifest">vamos.experiment.optimization_result.OptimizationResult.manifest</a></li>
<li><a id="vamos.experiment.optimization_result.StudyResult" data-api-legacy href="../api/results/#vamos.StudyResult">vamos.experiment.optimization_result.StudyResult</a></li>
<li><a id="vamos.experiment.optimization_result.StudyResult.runs" data-api-legacy href="../api/results/#vamos.StudyResult.runs">vamos.experiment.optimization_result.StudyResult.runs</a></li>
<li><a id="vamos.experiment.optimization_result.StudyResult.best_run" data-api-legacy href="../api/results/#vamos.StudyResult.best_run">vamos.experiment.optimization_result.StudyResult.best_run</a></li>
<li><a id="vamos.experiment.optimization_result.StudyResult.mean" data-api-legacy href="../api/results/#vamos.StudyResult.mean">vamos.experiment.optimization_result.StudyResult.mean</a></li>
<li><a id="vamos.experiment.optimization_result.StudyResult.metric_values" data-api-legacy href="../api/results/#vamos.StudyResult.metric_values">vamos.experiment.optimization_result.StudyResult.metric_values</a></li>
<li><a id="vamos.experiment.optimization_result.StudyResult.std" data-api-legacy href="../api/results/#vamos.StudyResult.std">vamos.experiment.optimization_result.StudyResult.std</a></li>
<li><a id="run-artifacts" data-api-legacy href="../api/runs/">run-artifacts</a></li>
<li><a id="vamos.run_artifacts" data-api-legacy href="../api/runs/">vamos.run_artifacts</a></li>
<li><a id="vamos.run_artifacts.StoredRun" data-api-legacy href="../api/run-models/#vamos.run_artifacts.StoredRun">vamos.run_artifacts.StoredRun</a></li>
<li><a id="vamos.run_artifacts.StoredRun.environment" data-api-legacy href="../api/run-models/#vamos.run_artifacts.StoredRun.environment">vamos.run_artifacts.StoredRun.environment</a></li>
<li><a id="vamos.run_artifacts.StoredRun.result" data-api-legacy href="../api/run-models/#vamos.run_artifacts.StoredRun.result">vamos.run_artifacts.StoredRun.result</a></li>
<li><a id="vamos.run_artifacts.RunManifest" data-api-legacy href="../api/run-models/#vamos.run_artifacts.RunManifest">vamos.run_artifacts.RunManifest</a></li>
<li><a id="vamos.run_artifacts.RunManifest.artifact" data-api-legacy href="../api/run-models/#vamos.run_artifacts.RunManifest.artifact">vamos.run_artifacts.RunManifest.artifact</a></li>
<li><a id="vamos.run_artifacts.LoadLimits" data-api-legacy href="../api/run-models/#vamos.run_artifacts.LoadLimits">vamos.run_artifacts.LoadLimits</a></li>
<li><a id="vamos.run_artifacts.CompatibilityReport" data-api-legacy href="../api/run-models/#vamos.run_artifacts.CompatibilityReport">vamos.run_artifacts.CompatibilityReport</a></li>
<li><a id="vamos.run_artifacts.VerificationReport" data-api-legacy href="../api/run-models/#vamos.run_artifacts.VerificationReport">vamos.run_artifacts.VerificationReport</a></li>
<li><a id="vamos.run_artifacts.ReplayReport" data-api-legacy href="../api/run-models/#vamos.run_artifacts.ReplayReport">vamos.run_artifacts.ReplayReport</a></li>
<li><a id="vamos.run_artifacts.save_result" data-api-legacy href="../api/runs/#vamos.run_artifacts.save_result">vamos.run_artifacts.save_result</a></li>
<li><a id="vamos.run_artifacts.load_run" data-api-legacy href="../api/runs/#vamos.run_artifacts.load_run">vamos.run_artifacts.load_run</a></li>
<li><a id="vamos.run_artifacts.load_result" data-api-legacy href="../api/runs/#vamos.run_artifacts.load_result">vamos.run_artifacts.load_result</a></li>
<li><a id="vamos.run_artifacts.verify_run" data-api-legacy href="../api/runs/#vamos.run_artifacts.verify_run">vamos.run_artifacts.verify_run</a></li>
<li><a id="vamos.run_artifacts.reproduce" data-api-legacy href="../api/runs/#vamos.run_artifacts.reproduce">vamos.run_artifacts.reproduce</a></li>
<li><a id="durable-studies" data-api-legacy href="../api/study-specification/">durable-studies</a></li>
<li><a id="vamos.study_artifacts" data-api-legacy href="../api/studies/">vamos.study_artifacts</a></li>
<li><a id="vamos.study_artifacts.StudySpec" data-api-legacy href="../api/study-specification/#vamos.study_artifacts.StudySpec">vamos.study_artifacts.StudySpec</a></li>
<li><a id="vamos.study_artifacts.StudySpec.as_intent_dict" data-api-legacy href="../api/study-specification/#vamos.study_artifacts.StudySpec.as_intent_dict">vamos.study_artifacts.StudySpec.as_intent_dict</a></li>
<li><a id="vamos.study_artifacts.StudyPlanReport" data-api-legacy href="../api/study-specification/#vamos.study_artifacts.StudyPlanReport">vamos.study_artifacts.StudyPlanReport</a></li>
<li><a id="vamos.study_artifacts.StudyPlanReport.as_dict" data-api-legacy href="../api/study-specification/#vamos.study_artifacts.StudyPlanReport.as_dict">vamos.study_artifacts.StudyPlanReport.as_dict</a></li>
<li><a id="vamos.study_artifacts.Study" data-api-legacy href="../api/studies/#vamos.study_artifacts.Study">vamos.study_artifacts.Study</a></li>
<li><a id="vamos.study_artifacts.Study.cancel" data-api-legacy href="../api/studies/#vamos.study_artifacts.Study.cancel">vamos.study_artifacts.Study.cancel</a></li>
<li><a id="vamos.study_artifacts.Study.inspect" data-api-legacy href="../api/studies/#vamos.study_artifacts.Study.inspect">vamos.study_artifacts.Study.inspect</a></li>
<li><a id="vamos.study_artifacts.Study.resume" data-api-legacy href="../api/studies/#vamos.study_artifacts.Study.resume">vamos.study_artifacts.Study.resume</a></li>
<li><a id="vamos.study_artifacts.Study.retry" data-api-legacy href="../api/studies/#vamos.study_artifacts.Study.retry">vamos.study_artifacts.Study.retry</a></li>
<li><a id="vamos.study_artifacts.Study.run" data-api-legacy href="../api/studies/#vamos.study_artifacts.Study.run">vamos.study_artifacts.Study.run</a></li>
<li><a id="vamos.study_artifacts.Study.summarize" data-api-legacy href="../api/studies/#vamos.study_artifacts.Study.summarize">vamos.study_artifacts.Study.summarize</a></li>
<li><a id="vamos.study_artifacts.StudyLoadLimits" data-api-legacy href="../api/study-models/#vamos.study_artifacts.StudyLoadLimits">vamos.study_artifacts.StudyLoadLimits</a></li>
<li><a id="vamos.study_artifacts.StudyReport" data-api-legacy href="../api/study-models/#vamos.study_artifacts.StudyReport">vamos.study_artifacts.StudyReport</a></li>
<li><a id="vamos.study_artifacts.StudySummary" data-api-legacy href="../api/study-models/#vamos.study_artifacts.StudySummary">vamos.study_artifacts.StudySummary</a></li>
<li><a id="vamos.study_artifacts.plan_study" data-api-legacy href="../api/study-specification/#vamos.study_artifacts.plan_study">vamos.study_artifacts.plan_study</a></li>
<li><a id="vamos.study_artifacts.create_study" data-api-legacy href="../api/studies/#vamos.study_artifacts.create_study">vamos.study_artifacts.create_study</a></li>
<li><a id="vamos.study_artifacts.load_study" data-api-legacy href="../api/studies/#vamos.study_artifacts.load_study">vamos.study_artifacts.load_study</a></li>
<li><a id="algorithm-configuration" data-api-legacy href="../api/algorithms/">algorithm-configuration</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaii" data-api-legacy href="../api/algorithms/nsgaii/">vamos.engine.algorithm.config.nsgaii</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaii.NSGAIIConfig" data-api-legacy href="../api/algorithms/nsgaii/#vamos.algorithms.NSGAIIConfig">vamos.engine.algorithm.config.nsgaii.NSGAIIConfig</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaii.NSGAIIConfig.default" data-api-legacy href="../api/algorithms/nsgaii/#vamos.algorithms.NSGAIIConfig.default">vamos.engine.algorithm.config.nsgaii.NSGAIIConfig.default</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaii._NSGAIIConfigBuilder" data-api-legacy href="../api/algorithms/nsgaii/#builder-methods">vamos.engine.algorithm.config.nsgaii._NSGAIIConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaii._NSGAIIConfigBuilder.steady_state" data-api-legacy href="../api/algorithms/nsgaii/#vamos.engine.algorithm.config.nsgaii._NSGAIIConfigBuilder.steady_state">vamos.engine.algorithm.config.nsgaii._NSGAIIConfigBuilder.steady_state</a></li>
<li><a id="vamos.engine.algorithm.config.moead" data-api-legacy href="../api/algorithms/moead/">vamos.engine.algorithm.config.moead</a></li>
<li><a id="vamos.engine.algorithm.config.moead.MOEADConfig" data-api-legacy href="../api/algorithms/moead/#vamos.algorithms.MOEADConfig">vamos.engine.algorithm.config.moead.MOEADConfig</a></li>
<li><a id="vamos.engine.algorithm.config.moead.MOEADConfig.default" data-api-legacy href="../api/algorithms/moead/#vamos.algorithms.MOEADConfig.default">vamos.engine.algorithm.config.moead.MOEADConfig.default</a></li>
<li><a id="vamos.engine.algorithm.config.moead._MOEADConfigBuilder" data-api-legacy href="../api/algorithms/moead/#builder-methods">vamos.engine.algorithm.config.moead._MOEADConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaiii" data-api-legacy href="../api/algorithms/nsgaiii/">vamos.engine.algorithm.config.nsgaiii</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaiii.NSGAIIIConfig" data-api-legacy href="../api/algorithms/nsgaiii/#vamos.algorithms.NSGAIIIConfig">vamos.engine.algorithm.config.nsgaiii.NSGAIIIConfig</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaiii.NSGAIIIConfig.default" data-api-legacy href="../api/algorithms/nsgaiii/#vamos.algorithms.NSGAIIIConfig.default">vamos.engine.algorithm.config.nsgaiii.NSGAIIIConfig.default</a></li>
<li><a id="vamos.engine.algorithm.config.nsgaiii._NSGAIIIConfigBuilder" data-api-legacy href="../api/algorithms/nsgaiii/#builder-methods">vamos.engine.algorithm.config.nsgaiii._NSGAIIIConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.smsemoa" data-api-legacy href="../api/algorithms/smsemoa/">vamos.engine.algorithm.config.smsemoa</a></li>
<li><a id="vamos.engine.algorithm.config.smsemoa.SMSEMOAConfig" data-api-legacy href="../api/algorithms/smsemoa/#vamos.algorithms.SMSEMOAConfig">vamos.engine.algorithm.config.smsemoa.SMSEMOAConfig</a></li>
<li><a id="vamos.engine.algorithm.config.smsemoa.SMSEMOAConfig.default" data-api-legacy href="../api/algorithms/smsemoa/#vamos.algorithms.SMSEMOAConfig.default">vamos.engine.algorithm.config.smsemoa.SMSEMOAConfig.default</a></li>
<li><a id="vamos.engine.algorithm.config.smsemoa._SMSEMOAConfigBuilder" data-api-legacy href="../api/algorithms/smsemoa/#builder-methods">vamos.engine.algorithm.config.smsemoa._SMSEMOAConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.spea2" data-api-legacy href="../api/algorithms/spea2/">vamos.engine.algorithm.config.spea2</a></li>
<li><a id="vamos.engine.algorithm.config.spea2.SPEA2Config" data-api-legacy href="../api/algorithms/spea2/#vamos.algorithms.SPEA2Config">vamos.engine.algorithm.config.spea2.SPEA2Config</a></li>
<li><a id="vamos.engine.algorithm.config.spea2.SPEA2Config.default" data-api-legacy href="../api/algorithms/spea2/#vamos.algorithms.SPEA2Config.default">vamos.engine.algorithm.config.spea2.SPEA2Config.default</a></li>
<li><a id="vamos.engine.algorithm.config.spea2._SPEA2ConfigBuilder" data-api-legacy href="../api/algorithms/spea2/#builder-methods">vamos.engine.algorithm.config.spea2._SPEA2ConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.spea2._SPEA2ConfigBuilder.external_archive" data-api-legacy href="../api/algorithms/spea2/#vamos.engine.algorithm.config.spea2._SPEA2ConfigBuilder.external_archive">vamos.engine.algorithm.config.spea2._SPEA2ConfigBuilder.external_archive</a></li>
<li><a id="vamos.engine.algorithm.config.ibea" data-api-legacy href="../api/algorithms/ibea/">vamos.engine.algorithm.config.ibea</a></li>
<li><a id="vamos.engine.algorithm.config.ibea.IBEAConfig" data-api-legacy href="../api/algorithms/ibea/#vamos.algorithms.IBEAConfig">vamos.engine.algorithm.config.ibea.IBEAConfig</a></li>
<li><a id="vamos.engine.algorithm.config.ibea.IBEAConfig.default" data-api-legacy href="../api/algorithms/ibea/#vamos.algorithms.IBEAConfig.default">vamos.engine.algorithm.config.ibea.IBEAConfig.default</a></li>
<li><a id="vamos.engine.algorithm.config.ibea._IBEAConfigBuilder" data-api-legacy href="../api/algorithms/ibea/#builder-methods">vamos.engine.algorithm.config.ibea._IBEAConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.smpso" data-api-legacy href="../api/algorithms/smpso/">vamos.engine.algorithm.config.smpso</a></li>
<li><a id="vamos.engine.algorithm.config.smpso.SMPSOConfig" data-api-legacy href="../api/algorithms/smpso/#vamos.algorithms.SMPSOConfig">vamos.engine.algorithm.config.smpso.SMPSOConfig</a></li>
<li><a id="vamos.engine.algorithm.config.smpso.SMPSOConfig.default" data-api-legacy href="../api/algorithms/smpso/#vamos.algorithms.SMPSOConfig.default">vamos.engine.algorithm.config.smpso.SMPSOConfig.default</a></li>
<li><a id="vamos.engine.algorithm.config.smpso._SMPSOConfigBuilder" data-api-legacy href="../api/algorithms/smpso/#builder-methods">vamos.engine.algorithm.config.smpso._SMPSOConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.smpso._SMPSOConfigBuilder.external_archive" data-api-legacy href="../api/algorithms/smpso/#vamos.engine.algorithm.config.smpso._SMPSOConfigBuilder.external_archive">vamos.engine.algorithm.config.smpso._SMPSOConfigBuilder.external_archive</a></li>
<li><a id="vamos.engine.algorithm.config.agemoea" data-api-legacy href="../api/algorithms/agemoea/">vamos.engine.algorithm.config.agemoea</a></li>
<li><a id="vamos.engine.algorithm.config.agemoea.AGEMOEAConfig" data-api-legacy href="../api/algorithms/agemoea/#vamos.algorithms.AGEMOEAConfig">vamos.engine.algorithm.config.agemoea.AGEMOEAConfig</a></li>
<li><a id="vamos.engine.algorithm.config.agemoea.AGEMOEAConfig.default" data-api-legacy href="../api/algorithms/agemoea/#vamos.algorithms.AGEMOEAConfig.default">vamos.engine.algorithm.config.agemoea.AGEMOEAConfig.default</a></li>
<li><a id="vamos.engine.algorithm.config.agemoea._AGEMOEAConfigBuilder" data-api-legacy href="../api/algorithms/agemoea/#builder-methods">vamos.engine.algorithm.config.agemoea._AGEMOEAConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.rvea" data-api-legacy href="../api/algorithms/rvea/">vamos.engine.algorithm.config.rvea</a></li>
<li><a id="vamos.engine.algorithm.config.rvea.RVEAConfig" data-api-legacy href="../api/algorithms/rvea/#vamos.algorithms.RVEAConfig">vamos.engine.algorithm.config.rvea.RVEAConfig</a></li>
<li><a id="vamos.engine.algorithm.config.rvea.RVEAConfig.default" data-api-legacy href="../api/algorithms/rvea/#vamos.algorithms.RVEAConfig.default">vamos.engine.algorithm.config.rvea.RVEAConfig.default</a></li>
<li><a id="vamos.engine.algorithm.config.rvea._RVEAConfigBuilder" data-api-legacy href="../api/algorithms/rvea/#builder-methods">vamos.engine.algorithm.config.rvea._RVEAConfigBuilder</a></li>
<li><a id="vamos.engine.algorithm.config.generic" data-api-legacy href="../api/internal/generic-configuration/">vamos.engine.algorithm.config.generic</a></li>
<li><a id="vamos.engine.algorithm.config.generic.GenericAlgorithmConfig" data-api-legacy href="../api/internal/generic-configuration/#vamos.engine.algorithm.config.generic.GenericAlgorithmConfig">vamos.engine.algorithm.config.generic.GenericAlgorithmConfig</a></li>
<li><a id="constraint-handling" data-api-legacy href="../api/internal/constraints/">constraint-handling</a></li>
<li><a id="vamos.foundation.constraints" data-api-legacy href="../api/internal/constraints/">vamos.foundation.constraints</a></li>
<li><a id="vamos.foundation.constraints.ConstraintInfo" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.ConstraintInfo">vamos.foundation.constraints.ConstraintInfo</a></li>
<li><a id="vamos.foundation.constraints.FeasibilityFirstStrategy" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.FeasibilityFirstStrategy">vamos.foundation.constraints.FeasibilityFirstStrategy</a></li>
<li><a id="vamos.foundation.constraints.PenaltyCVStrategy" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.PenaltyCVStrategy">vamos.foundation.constraints.PenaltyCVStrategy</a></li>
<li><a id="vamos.foundation.constraints.CVAsObjectiveStrategy" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.CVAsObjectiveStrategy">vamos.foundation.constraints.CVAsObjectiveStrategy</a></li>
<li><a id="vamos.foundation.constraints.EpsilonConstraintStrategy" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.EpsilonConstraintStrategy">vamos.foundation.constraints.EpsilonConstraintStrategy</a></li>
<li><a id="vamos.foundation.constraints.compute_constraint_info" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.compute_constraint_info">vamos.foundation.constraints.compute_constraint_info</a></li>
<li><a id="vamos.foundation.constraints.get_constraint_strategy" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.get_constraint_strategy">vamos.foundation.constraints.get_constraint_strategy</a></li>
<li><a id="vamos.foundation.constraints.utils" data-api-legacy href="../api/internal/constraints/">vamos.foundation.constraints.utils</a></li>
<li><a id="vamos.foundation.constraints.utils.compute_violation" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.utils.compute_violation">vamos.foundation.constraints.utils.compute_violation</a></li>
<li><a id="vamos.foundation.constraints.utils.is_feasible" data-api-legacy href="../api/internal/constraints/#vamos.foundation.constraints.utils.is_feasible">vamos.foundation.constraints.utils.is_feasible</a></li>
<li><a id="encoding" data-api-legacy href="../api/internal/problem-types/">encoding</a></li>
<li><a id="vamos.foundation.encoding" data-api-legacy href="../api/internal/problem-types/">vamos.foundation.encoding</a></li>
<li><a id="vamos.foundation.encoding.normalize_encoding" data-api-legacy href="../api/internal/problem-types/#vamos.foundation.encoding.normalize_encoding">vamos.foundation.encoding.normalize_encoding</a></li>
<li><a id="problem-registry" data-api-legacy href="../api/discovery/">problem-registry</a></li>
<li><a id="vamos.foundation.problem.registry" data-api-legacy href="../api/discovery/">vamos.foundation.problem.registry</a></li>
<li><a id="vamos.foundation.problem.registry.available_problem_names" data-api-legacy href="../api/discovery/#vamos.available_problem_names">vamos.foundation.problem.registry.available_problem_names</a></li>
<li><a id="vamos.foundation.problem.registry.make_problem_selection" data-api-legacy href="../api/discovery/#vamos.make_problem_selection">vamos.foundation.problem.registry.make_problem_selection</a></li>
<li><a id="tuning-experimental" data-api-legacy href="../api/experimental/tuning/">tuning-experimental</a></li>
<li><a id="vamos.engine.tuning.racing.core" data-api-legacy href="../api/experimental/tuning/">vamos.engine.tuning.racing.core</a></li>
<li><a id="vamos.engine.tuning.racing.core.RacingTuner" data-api-legacy href="../api/experimental/tuning/#vamos.engine.tuning.racing.core.RacingTuner">vamos.engine.tuning.racing.core.RacingTuner</a></li>
<li><a id="vamos.engine.tuning.racing.core.RacingTuner._check_convergence" data-api-legacy href="../api/experimental/tuning/#vamos.engine.tuning.racing.core.RacingTuner._check_convergence">vamos.engine.tuning.racing.core.RacingTuner._check_convergence</a></li>
<li><a id="vamos.engine.tuning.racing.core.RacingTuner._get_current_best_score" data-api-legacy href="../api/experimental/tuning/#vamos.engine.tuning.racing.core.RacingTuner._get_current_best_score">vamos.engine.tuning.racing.core.RacingTuner._get_current_best_score</a></li>
<li><a id="vamos.engine.tuning.racing.param_space" data-api-legacy href="../api/experimental/parameter-spaces/">vamos.engine.tuning.racing.param_space</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Boolean" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Boolean">vamos.engine.tuning.racing.param_space.Boolean</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Categorical" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Categorical">vamos.engine.tuning.racing.param_space.Categorical</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Categorical.from_unit" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Categorical.from_unit">vamos.engine.tuning.racing.param_space.Categorical.from_unit</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Categorical.to_unit" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Categorical.to_unit">vamos.engine.tuning.racing.param_space.Categorical.to_unit</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Condition" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Condition">vamos.engine.tuning.racing.param_space.Condition</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.ConditionalBlock" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.ConditionalBlock">vamos.engine.tuning.racing.param_space.ConditionalBlock</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Int" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Int">vamos.engine.tuning.racing.param_space.Int</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Int.from_unit" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Int.from_unit">vamos.engine.tuning.racing.param_space.Int.from_unit</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Int.to_unit" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Int.to_unit">vamos.engine.tuning.racing.param_space.Int.to_unit</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.ParamSpace" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.ParamSpace">vamos.engine.tuning.racing.param_space.ParamSpace</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.ParamSpace.is_active" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.ParamSpace.is_active">vamos.engine.tuning.racing.param_space.ParamSpace.is_active</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.ParamSpace.sample" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.ParamSpace.sample">vamos.engine.tuning.racing.param_space.ParamSpace.sample</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.ParamSpace.validate" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.ParamSpace.validate">vamos.engine.tuning.racing.param_space.ParamSpace.validate</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Real" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Real">vamos.engine.tuning.racing.param_space.Real</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Real.from_unit" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Real.from_unit">vamos.engine.tuning.racing.param_space.Real.from_unit</a></li>
<li><a id="vamos.engine.tuning.racing.param_space.Real.to_unit" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space.Real.to_unit">vamos.engine.tuning.racing.param_space.Real.to_unit</a></li>
<li><a id="vamos.engine.tuning.racing.param_space._MissingKeyError" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space._MissingKeyError">vamos.engine.tuning.racing.param_space._MissingKeyError</a></li>
<li><a id="vamos.engine.tuning.racing.param_space._safe_eval_condition" data-api-legacy href="../api/experimental/parameter-spaces/#vamos.engine.tuning.racing.param_space._safe_eval_condition">vamos.engine.tuning.racing.param_space._safe_eval_condition</a></li>
<li><a id="diagnostics" data-api-legacy href="../api/experimental/diagnostics/">diagnostics</a></li>
<li><a id="vamos.experiment.diagnostics.self_check" data-api-legacy href="../api/experimental/diagnostics/">vamos.experiment.diagnostics.self_check</a></li>
<li><a id="vamos.experiment.diagnostics.self_check.run_self_check" data-api-legacy href="../api/experimental/diagnostics/#vamos.experiment.diagnostics.self_check.run_self_check">vamos.experiment.diagnostics.self_check.run_self_check</a></li>
</ul>
</details>
