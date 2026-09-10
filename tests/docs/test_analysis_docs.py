from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "topics" / "analysis.md"
PYTHON_FENCE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


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


@pytest.mark.smoke
def test_analysis_python_snippets_execute(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    text = DOC_PATH.read_text(encoding="utf-8")
    blocks = PYTHON_FENCE.findall(text)
    assert len(blocks) == 4

    script = tmp_path / "analysis_snippets.py"
    script.write_text("\n\n".join(blocks), encoding="utf-8")

    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (tmp_path / "cd_plot.png").exists()
    assert (tmp_path / "front.png").exists()
