import pytest

from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.registry.specs import available_problem_names
from vamos.problems import CEC2009CF1


def test_problem_selection_defaults_and_encoding():
    selection = make_problem_selection("zdt1")
    assert selection.n_var == 30
    assert selection.n_obj == 2
    problem = selection.instantiate()
    assert problem.encoding == "real"


def test_problem_selection_zdt5_binary_encoding():
    selection = make_problem_selection("zdt5")
    assert selection.n_var == 80
    assert selection.n_obj == 2
    problem = selection.instantiate()
    assert getattr(problem, "encoding", "") == "binary"


@pytest.mark.parametrize("name", ["dtlz5", "dtlz6"])
def test_problem_selection_dtlz56_defaults(name: str):
    selection = make_problem_selection(name)
    assert selection.n_var == 12
    assert selection.n_obj == 3


@pytest.mark.parametrize("name", ["lsmop1", "lsmop9"])
def test_problem_selection_lsmop_defaults(name: str):
    selection = make_problem_selection(name)
    assert selection.n_var == 300
    assert selection.n_obj == 3


@pytest.mark.parametrize("name", ["c1dtlz1", "dc2dtlz3", "mw1"])
def test_problem_selection_constrained_many_defaults(name: str):
    selection = make_problem_selection(name)
    assert selection.n_var > 0
    assert selection.n_obj >= 2


@pytest.mark.parametrize("name", ["zcat1", "zcat14", "zcat20"])
def test_problem_selection_zcat_defaults(name: str):
    selection = make_problem_selection(name)
    assert selection.n_var == 30
    assert selection.n_obj == 2


def test_problem_selection_zcat_n_obj_override():
    selection = make_problem_selection("zcat6", n_var=40, n_obj=4)
    assert selection.n_var == 40
    assert selection.n_obj == 4


def test_tsplib_selection_encoding():
    selection = make_problem_selection("kroa100")
    assert selection.n_var == 100
    assert selection.n_obj == 2
    problem = selection.instantiate()
    assert getattr(problem, "encoding", "") == "permutation"


def test_invalid_problem_raises():
    with pytest.raises(KeyError):
        make_problem_selection("does_not_exist")


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("re21", {"n_var": 10}),
        ("rwa1", {"n_var": 9}),
        ("kroa100", {"n_var": 10}),
        ("welded_beam", {"n_var": 3}),
        ("mw1", {"n_obj": 4}),
    ],
)
def test_fixed_dimension_problems_reject_mismatched_overrides(name: str, overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="fixed"):
        make_problem_selection(name, **overrides)


@pytest.mark.parametrize("name", ["re21", "rwa1", "kroa100", "welded_beam", "mw1"])
def test_fixed_dimension_problems_accept_their_own_dimensions(name: str) -> None:
    default = make_problem_selection(name)
    explicit = make_problem_selection(name, n_var=default.n_var, n_obj=default.n_obj)
    assert (explicit.n_var, explicit.n_obj) == (default.n_var, default.n_obj)


@pytest.mark.parametrize("name", available_problem_names())
def test_registry_default_dimensions_match_instances(name: str) -> None:
    selection = make_problem_selection(name)
    problem = selection.instantiate()
    assert (problem.n_var, problem.n_obj) == (selection.n_var, selection.n_obj)


@pytest.mark.parametrize("name", ["mw4", "mw8", "mw14"])
def test_scalable_mw_problems_keep_their_objective_override(name: str) -> None:
    selection = make_problem_selection(name, n_obj=4)
    assert selection.instantiate().n_obj == 4


def test_cec2009_cf1_defaults_to_ten_variables() -> None:
    assert CEC2009CF1().n_var == 10
    assert make_problem_selection("cec2009_cf1").instantiate().n_var == 10
    assert CEC2009CF1(n_var=30).n_var == 30
