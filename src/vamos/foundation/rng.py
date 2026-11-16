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


# Spawn keys of the streams that components derive from the run seed; each is a child of the
# run's SeedSequence that the algorithm's own generator, ``default_rng(seed)``, never draws from.
_DERIVED_STREAMS = {"external_archive": 1, "hv_convergence": 2}


def derived_seed(run_seed: int, stream: str) -> int:
    """Seed of the component ``stream`` of a run, derived from ``run_seed``.

    The value is ``SeedSequence(run_seed, spawn_key=(k,))`` for the stream's fixed key ``k``,
    the child that ``SeedSequence(run_seed).spawn`` would give, so it differs between run seeds,
    repeats for the same seed, and does not consume the algorithm's stream.
    """
    sequence = np.random.SeedSequence(int(run_seed) % 2**64, spawn_key=(_DERIVED_STREAMS[stream],))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


__all__ = ["derived_seed", "require_generator"]
