from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.factory import build_algorithm
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.config.spec import ExperimentSpec, SpecBlock
from vamos.engine.hooks import LiveVisualization
from vamos.experiment._execution_support import VariationConfigs
from vamos.experiment.execution import run_single as execute_run_single
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import ProblemSelection


def run_single(
    engine_name: str,
    algorithm_name: str,
    selection: ProblemSelection,
    config: ExperimentConfig,
    *,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    external_archive: ExternalArchiveConfig | None = None,
    selection_pressure: int = 2,
    variations: VariationConfigs | None = None,
    hv_stop_config: dict[str, object] | None = None,
    evaluator: object | None = None,
    termination: tuple[str, object] | None = None,
    config_source: str | None = None,
    config_spec: ExperimentSpec | None = None,
    problem_override: SpecBlock | None = None,
    track_genealogy: bool = False,
    live_viz: LiveVisualization | None = None,
    cli_request: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    problem = selection.instantiate()
    if track_genealogy and algorithm_name.lower() != "nsgaii":
        raise ValueError("track_genealogy is currently supported only for NSGA-II in the experiment/CLI runner.")
    variations = variations or VariationConfigs()
    algorithm, cfg_data = build_algorithm(
        algorithm_name,
        engine_name,
        problem,
        config,
        algorithm_config=algorithm_config,
        external_archive=external_archive,
        selection_pressure=selection_pressure,
        variations={
            "nsgaii": variations.nsgaii,
            "moead": variations.moead,
            "smsemoa": variations.smsemoa,
            "nsgaiii": variations.nsgaiii,
            "spea2": variations.spea2,
            "ibea": variations.ibea,
            "smpso": variations.smpso,
            "agemoea": variations.agemoea,
            "rvea": variations.rvea,
        },
        track_genealogy=track_genealogy,
    )
    return execute_run_single(
        engine_name,
        algorithm_name,
        selection,
        config,
        algorithm=algorithm,
        cfg_data=cfg_data,
        problem=problem,
        external_archive=external_archive,
        selection_pressure=selection_pressure,
        variations=variations,
        hv_stop_config=hv_stop_config,
        evaluator=evaluator,
        termination=termination,
        config_source=config_source,
        config_spec=config_spec,
        problem_override=problem_override,
        track_genealogy=track_genealogy,
        live_viz=live_viz,
        cli_request=cli_request,
    )
