from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vamos.experiment.gui.events import (
    EVENT_SCHEMA,
    ProgressEventWriter,
    ProgressHypervolume,
    append_event,
    downsample,
    finite_rows,
    has_terminal_event,
    nondominated,
    read_events,
)

pytestmark = pytest.mark.gui


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _front(n_points: int) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, n_points)
    return np.c_[f1, 1.0 - np.sqrt(f1)]


def _progress(path: Path) -> list[dict[str, object]]:
    return [event for event in read_events(path)[0] if event["type"] == "progress"]


def test_writer_throttles_generations_by_time(tmp_path: Path) -> None:
    clock = FakeClock()
    path = tmp_path / "events.jsonl"
    writer = ProgressEventWriter(path, request={"problem": "zdt1"}, min_interval=1.0, clock=clock)
    for generation in range(1, 6):
        clock.now = generation * 0.3
        writer.on_generation(generation, F=_front(10) + generation, stats={"evals": generation * 10})
    writer.on_end()
    writer.write_end("succeeded", run_dir="run", evaluations=50)

    events, offset, malformed = read_events(path)
    assert malformed == 0
    assert offset == path.stat().st_size
    assert events[0]["type"] == "start"
    assert events[0]["schema"] == EVENT_SCHEMA
    assert events[0]["request"] == {"problem": "zdt1"}
    assert [event["generation"] for event in _progress(path)] == [1, 5]
    assert _progress(path)[-1]["evaluations"] == 50
    assert events[-1]["type"] == "end"
    assert (events[-1]["status"], events[-1]["run_dir"], events[-1]["evaluations"]) == ("succeeded", "run", 50)


def test_on_end_flushes_the_latest_coalesced_generation(tmp_path: Path) -> None:
    clock = FakeClock()
    path = tmp_path / "events.jsonl"
    writer = ProgressEventWriter(path, min_interval=10.0, clock=clock)
    writer.on_generation(1, F=_front(5))
    clock.now = 1.0
    writer.on_generation(2, F=_front(6), stats={"evaluations": 12})
    writer.on_end()
    progress = _progress(path)
    assert [event["generation"] for event in progress] == [1, 2]
    assert (progress[-1]["front_size"], progress[-1]["evaluations"]) == (6, 12)


def test_progress_front_is_finite_nondominated_and_downsampled(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    F = np.vstack([_front(40), [[2.0, 2.0]], [[np.nan, 0.0]], [[np.inf, 1.0]]])
    ProgressEventWriter(path, max_points=10).on_generation(3, F=F)
    (progress,) = _progress(path)
    front = np.asarray(progress["front"])
    assert progress["front_size"] == 40
    assert front.shape == (10, 2)
    assert np.all(np.isfinite(front))
    assert (front[:, 0].min(), front[:, 0].max()) == (0.0, 1.0)
    assert progress["hypervolume"] > 0.0
    assert progress["evaluations"] is None
    assert progress["update"] == 1


def test_progress_hypervolume_keeps_its_first_reference_point() -> None:
    indicator = ProgressHypervolume()
    population = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    first = indicator(population, population[:2])
    assert indicator.reference is not None
    np.testing.assert_allclose(indicator.reference, [1.1, 1.1])
    improved = indicator(np.array([[5.0, 5.0]]), np.array([[0.0, 0.5], [0.5, 0.0]]))
    np.testing.assert_allclose(indicator.reference, [1.1, 1.1])
    assert improved > first > 0.0
    assert indicator(population, np.array([[5.0, 5.0]])) == 0.0
    assert ProgressHypervolume.supported(2) and ProgressHypervolume.supported(3)


def test_reader_leaves_partial_lines_and_counts_malformed_ones(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_event(path, {"type": "start"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('not json\n{"type": "progress", "update": 1}\n{"type": "end"')
    events, offset, malformed = read_events(path)
    assert [event["type"] for event in events] == ["start", "progress"]
    assert malformed == 1
    assert not has_terminal_event(path)

    append_event(path, {"type": "end", "status": "cancelled"})
    later, _, later_malformed = read_events(path, offset)
    assert [event.get("status") for event in later] == ["cancelled"]
    assert later_malformed == 1
    assert has_terminal_event(path)
    assert read_events(tmp_path / "missing.jsonl") == ([], 0, 0)


def test_events_are_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_event(path, {"type": "progress", "hypervolume": float("nan"), "value": np.float64(1.5), "count": np.int64(3)})
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw) == {"type": "progress", "hypervolume": None, "value": 1.5, "count": 3}


def test_terminal_event_is_validated_and_written_once(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = ProgressEventWriter(path)
    with pytest.raises(ValueError, match="Unknown terminal status"):
        writer.write_end("done")
    writer.write_end("failed", error="boom")
    writer.write_end("succeeded")
    ends = [event for event in read_events(path)[0] if event["type"] == "end"]
    assert [(event["status"], event["error"]) for event in ends] == [("failed", "boom")]


def test_writer_rejects_invalid_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_interval"):
        ProgressEventWriter(tmp_path / "e.jsonl", min_interval=-1.0)
    with pytest.raises(ValueError, match="max_points"):
        ProgressEventWriter(tmp_path / "e.jsonl", max_points=0)


def test_front_helpers() -> None:
    assert finite_rows([[1.0, np.nan], [1.0, 2.0]]).tolist() == [[1.0, 2.0]]
    assert nondominated(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])).shape == (2, 2)
    assert downsample(_front(3), 10).shape == (3, 2)
    with pytest.raises(ValueError, match="max_points"):
        downsample(_front(3), 0)
