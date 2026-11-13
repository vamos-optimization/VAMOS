from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem

# Tests that take more than about 10 s and that another gate or the nightly slow tier covers well
# enough; they are marked `slow` here, so pull requests and the merge queue skip them (docs/dev/testing.md).
# An entry without "[...]" covers every parameter of the test. Tests that are the only check of a
# stable contract (compatibility snapshots, the study CLI, resume and retry) stay in the fast tier.
SLOW_TESTS = frozenset(
    {
        # Health and CI's architecture step run these two files directly.
        "tests/architecture/test_build_smoke.py::test_build_smoke",
        "tests/architecture/test_py_typed_present.py::test_py_typed_present",
        # Documentation portal and tutorials.
        "tests/docs/test_release_pages_layout.py::test_versioned_pages_canonical_urls_resolve_to_deployed_files",
        "tests/docs/test_modular_api.py::test_every_legacy_bookmark_resolves_in_current_and_immutable_portal",
        "tests/docs/test_study_tutorials.py::test_measurements_trace_to_verified_runs_and_actual_objectives",
        "tests/docs/test_study_tutorials.py::test_all_downloads_run_standalone",
        "tests/docs/test_study_tutorials.py::test_duplicate_or_incompatible_studies_are_not_pooled",
        "tests/docs/test_docs_smoke.py::test_doc_smoke_cases[study_tutorial_t06]",
        "tests/docs/test_docs_smoke.py::test_doc_smoke_cases[study_tutorial_t07]",
        "tests/docs/test_docs_smoke.py::test_doc_smoke_cases[journey_reproducible_study]",
        "tests/docs/test_docs_smoke.py::test_doc_smoke_cases[coverage_tutorial_ablation]",
        "tests/docs/test_cookbook_recipes.py::test_minimal_python_cli_path",
        "tests/docs/test_reproducible_study_journey.py::test_executable_journey_plans_budget_and_traces_every_run",
        # Studio journeys; the real encoding of the batch test stays in the fast tier.
        "tests/ux/test_studio_batch_and_ranking.py::test_every_algorithm_offered_for_an_encoding_runs_in_a_batch[binary]",
        "tests/ux/test_studio_batch_and_ranking.py::test_every_algorithm_offered_for_an_encoding_runs_in_a_batch[integer]",
        "tests/ux/test_studio_batch_and_ranking.py::test_every_algorithm_offered_for_an_encoding_runs_in_a_batch[mixed]",
        "tests/ux/test_studio_batch_and_ranking.py::test_every_algorithm_offered_for_an_encoding_runs_in_a_batch[permutation]",
        "tests/ux/test_studio_comparison.py::test_comparison_of_a_study",
        "tests/ux/test_studio_panel_pages.py::test_compare_page_analyzes_a_study",
        # Release tooling, which tools/release_check.py exercises in full.
        "tests/test_release_automation.py::test_clean_wheel_install_and_lock_ignore_checkout_metadata",
    }
)


def pytest_configure() -> None:
    # Numba caches compiled kernels next to the source unless told otherwise; keep them out of src/.
    # Runs before any test imports Numba, and xdist workers and subprocesses inherit it.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "vamos-tests-numba-cache"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        if nodeid in SLOW_TESTS or nodeid.split("[", 1)[0] in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def zdt1_problem() -> ZDT1Problem:
    return ZDT1Problem(n_var=8)


@pytest.fixture
def dtlz2_problem() -> DTLZ2Problem:
    return DTLZ2Problem(n_var=12, n_obj=3)
