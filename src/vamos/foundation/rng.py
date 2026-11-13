"""Explicit random generators for stochastic components."""

from __future__ import annotations

import numpy as np

from vamos.foundation.exceptions import ConfigurationError


def require_generator(rng: object, owner: str) -> np.random.Generator:
    """Return ``rng`` when it is a ``numpy.random.Generator``; raise otherwise.

    Stochastic production paths receive and own an explicit generator. A component
    that fell back to an unseeded ``np.random.default_rng()`` gave results that no
    seed could reproduce, so a missing generator is a configuration error.
    """
    if not isinstance(rng, np.random.Generator):
        raise ConfigurationError(f"{owner} needs an explicit numpy.random.Generator (rng=...); got {type(rng).__name__}.")
    return rng


__all__ = ["require_generator"]
