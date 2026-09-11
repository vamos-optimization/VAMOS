from pathlib import Path


def test_dask_guide_is_explicitly_experimental_and_uses_public_api() -> None:
    text = Path("docs/scaling/dask.md").read_text(encoding="utf-8")

    assert "**Experimental integration.**" in text
    assert "outside the VAMOS 1.0 stable compatibility surface" in text
    assert 'pip install "vamos-optimization[compute]"' in text
    assert 'eval_strategy="dask"' in text
    assert "vamos.foundation.eval.backends" not in text
    assert "DaskEvalBackend" not in text
    assert "does **not** silently fall back to serial" in text


def test_dask_example_stays_experimental_and_avoids_backend_internals() -> None:
    text = Path("examples/distributed/dask_cluster.py").read_text(encoding="utf-8")

    assert "Experimental Dask" in text
    assert "outside the VAMOS 1.0" in text
    assert 'eval_strategy="dask"' in text
    assert "vamos.foundation.eval.backends" not in text
    assert "DaskEvalBackend" not in text
