from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


def configure_vamos_logging(*, level: int = logging.INFO) -> None:
    """
    Configure a minimal console logger for VAMOS.

    Notes:
        - This is intentionally opt-in (library code must not call logging.basicConfig()).
        - The handler is only attached if neither the root logger nor the "vamos" logger has handlers.
    """
    root = logging.getLogger()
    vamos_logger = logging.getLogger("vamos")

    # If the user already configured logging, don't interfere.
    if root.handlers or vamos_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    vamos_logger.addHandler(handler)
    vamos_logger.setLevel(level)
    vamos_logger.propagate = False


@contextmanager
def vamos_logging(*, enabled: bool = True, level: int = logging.INFO) -> Iterator[None]:
    """:func:`configure_vamos_logging` for the duration of a block, which then restores the "vamos" logger.

    Nothing changes when ``enabled`` is false or when the root or "vamos" logger already has handlers.
    """
    root = logging.getLogger()
    vamos_logger = logging.getLogger("vamos")
    if not enabled or root.handlers or vamos_logger.handlers:
        yield
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level, previous_propagate = vamos_logger.level, vamos_logger.propagate
    vamos_logger.addHandler(handler)
    vamos_logger.setLevel(level)
    vamos_logger.propagate = False
    try:
        yield
    finally:
        vamos_logger.removeHandler(handler)
        vamos_logger.setLevel(previous_level)
        vamos_logger.propagate = previous_propagate


__all__ = ["configure_vamos_logging", "vamos_logging"]
