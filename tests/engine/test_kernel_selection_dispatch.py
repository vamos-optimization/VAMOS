import pytest

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


class _TrackingNumPyKernel(NumPyKernel):
    def __init__(self) -> None:
        super().__init__()
        self.tournament_calls = 0

    def tournament_selection(self, ranks, crowding, pressure, rng, n_parents):  # noqa: ANN001 - kernel protocol signature
        self.tournament_calls += 1
        return super().tournament_selection(ranks, crowding, pressure, rng, n_parents)


def test_nsgaii_mating_dispatches_to_kernel_tournament_selection() -> None:
    pop_size = 8
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    kernel = _TrackingNumPyKernel()
    algorithm = NSGAII(cfg.to_dict(), kernel=kernel)
    problem = ZDT1Problem(n_var=6)

    algorithm.run(problem, termination=("max_evaluations", pop_size * 2), seed=11)

    assert kernel.tournament_calls > 0


def test_default_kernel_is_numpy() -> None:
    from vamos.foundation.kernel import default_kernel

    # Without an engine VAMOS uses NumPy, the stable 1.x default; Numba is opt-in.
    assert isinstance(default_kernel(), NumPyKernel)
    cfg = NSGAIIConfig.builder().pop_size(8).crossover("sbx", prob=0.9, eta=20.0).mutation("pm", prob="1/n", eta=20.0).build()
    assert NSGAII(cfg.to_dict()).kernel.name == "numpy"


def test_importing_moocore_backend_does_not_import_numba() -> None:
    import subprocess
    import sys

    pytest.importorskip("moocore")
    code = "import sys; import vamos.foundation.kernel.moocore_backend; print('numba' in sys.modules)"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
