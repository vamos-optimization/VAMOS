"""Thin compatibility layer for algorithm builder entrypoints."""

from __future__ import annotations

from vamos.engine.algorithm._builder_adaptive import (
    build_agemoea_algorithm,
    build_smpso_algorithm,
)
from vamos.engine.algorithm._builder_archive_population import (
    build_ibea_algorithm,
    build_nsgaii_algorithm,
    build_smsemoa_algorithm,
    build_spea2_algorithm,
)
from vamos.engine.algorithm._builder_constrained import build_ccmo_algorithm, build_ctaea_algorithm, build_top_algorithm
from vamos.engine.algorithm._builder_decomposition import (
    build_moead_algorithm,
    build_nsgaiii_algorithm,
    build_rvea_algorithm,
)

__all__ = [
    "build_nsgaii_algorithm",
    "build_moead_algorithm",
    "build_smsemoa_algorithm",
    "build_nsgaiii_algorithm",
    "build_spea2_algorithm",
    "build_ibea_algorithm",
    "build_smpso_algorithm",
    "build_agemoea_algorithm",
    "build_rvea_algorithm",
    "build_ccmo_algorithm",
    "build_ctaea_algorithm",
    "build_top_algorithm",
]
