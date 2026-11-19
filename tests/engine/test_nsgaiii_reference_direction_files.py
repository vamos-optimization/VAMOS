"""NSGA-III's ``reference_directions={"path": ...}``: jMetal's weight files, with whitespace or commas, and a missing file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vamos.engine.algorithm.config import NSGAIIIConfig
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.resources import weight_path


def _config(pop_size: int, path: str) -> dict[str, object]:
    return (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=30.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament")
        .reference_directions(path=path)
        .build()
    ).to_dict()


def _run(pop_size: int, path: str) -> dict[str, object]:
    algorithm = NSGAIII(_config(pop_size, path), kernel=NumPyKernel())
    return algorithm.run(DTLZ2Problem(n_var=12, n_obj=3), termination=("max_evaluations", 2 * pop_size), seed=1)


def test_nsgaiii_reads_the_whitespace_separated_jmetal_file() -> None:
    result = _run(100, str(weight_path("W3D_100.dat")))

    np.testing.assert_array_equal(result["reference_directions"], np.loadtxt(weight_path("W3D_100.dat")))
    assert result["evaluations"] == 200


@pytest.mark.parametrize("separator", [",", " "])
def test_nsgaiii_reads_comma_and_whitespace_separated_files_alike(tmp_path: Path, separator: str) -> None:
    stored = np.loadtxt(weight_path("W3D_91.dat"))
    path = tmp_path / "directions.csv"
    np.savetxt(path, stored, delimiter=separator)

    result = _run(91, str(path))

    np.testing.assert_array_equal(result["reference_directions"], stored)


def test_a_missing_reference_direction_file_raises_and_writes_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "W3D_91.dat"

    with pytest.raises(ConfigurationError, match="does not exist"):
        _run(91, str(missing))
    # VAMOS 1.4.0 generated the lattice and saved it there.
    assert not missing.exists() and not missing.parent.exists()


def test_a_reference_direction_file_with_more_directions_than_individuals_raises() -> None:
    with pytest.raises(ValueError, match="pop_size >= 100"):
        _run(91, str(weight_path("W3D_100.dat")))
