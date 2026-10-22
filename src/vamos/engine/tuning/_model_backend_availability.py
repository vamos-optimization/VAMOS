from __future__ import annotations

import importlib.util
import shutil


def available_model_based_backends() -> dict[str, bool]:
    """Return availability flags for optional model-based tuning backends.

    The ``irace`` flag only reports an ``Rscript`` on PATH; the R package is checked when the
    backend runs, because probing it needs an R process.
    """
    has_optuna = importlib.util.find_spec("optuna") is not None
    has_smac = importlib.util.find_spec("smac") is not None
    has_cs = importlib.util.find_spec("ConfigSpace") is not None
    has_hpbandster = importlib.util.find_spec("hpbandster") is not None
    return {
        "optuna": has_optuna,
        "bohb_optuna": has_optuna,
        "smac3": has_smac and has_cs,
        "bohb": has_hpbandster and has_cs,
        "irace": shutil.which("Rscript") is not None,
    }


def require_backend(name: str) -> None:
    """Raise a clear runtime error when an optional backend is unavailable."""
    available = available_model_based_backends()
    ok = bool(available.get(name, False))
    if ok:
        return
    if name == "optuna":
        raise RuntimeError("Backend 'optuna' requires optional dependency `optuna`.")
    if name == "bohb_optuna":
        raise RuntimeError("Backend 'bohb_optuna' requires optional dependency `optuna`.")
    if name == "smac3":
        raise RuntimeError("Backend 'smac3' requires optional dependencies `smac` and `ConfigSpace`.")
    if name == "bohb":
        raise RuntimeError("Backend 'bohb' requires optional dependencies `hpbandster` and `ConfigSpace`.")
    if name == "irace":
        raise RuntimeError("Backend 'irace' requires R with `Rscript` on PATH and the R package `irace` >= 4.0.")
    raise ValueError(f"Unknown backend '{name}'.")


__all__ = ["available_model_based_backends", "require_backend"]
