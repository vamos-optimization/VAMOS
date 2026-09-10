from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "topics" / "analysis.md"


def test_analysis_docs_match_public_contract() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Experimental surface in VAMOS 1.0.0" in text
    assert "(n_problems, n_algorithms)" in text
    assert "raw Wilcoxon p-values" in text
    assert "does not apply\nHolm" in text
    assert "MCDMResult" in text
    assert "knee_point_scores(F)" in text

    assert "topsis_scores" not in text
    assert "num_datasets=" not in text
    assert "filename=" not in text
