from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output
from vamos.foundation.problem.base import Problem


class ZDT5Problem(Problem):
    """
    ZDT5: Binary-coded ZDT benchmark.

    Canonical definition (Zitzler et al. 2000):
    - Decision vector is a bitstring of length 80.
      The first 30 bits form x1; the remaining 50 bits are split into 10 groups of 5 bits.
    - Objectives:
        f1 = 1 + u1, where u1 is the number of ones in the first 30 bits.
        g  = sum_{i=2..11} v_i, where v_i = 1 if u_i == 5 else 2 + u_i.
        f2 = g / f1
    """

    def __init__(self, n_var: int = 80, *, first_block: int = 30, block_size: int = 5) -> None:
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        if first_block <= 0:
            raise ValueError("first_block must be positive.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if n_var < first_block + block_size:
            raise ValueError("ZDT5 requires at least one tail block.")
        if (n_var - first_block) % block_size != 0:
            raise ValueError("ZDT5 requires (n_var - first_block) to be divisible by block_size.")

        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = 0
        self.xu = 1
        self.encoding = "binary"

        self._first_block = int(first_block)
        self._block_size = int(block_size)
        self._n_blocks = (self.n_var - self._first_block) // self._block_size

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected input shape (N, {self.n_var}), got {X.shape}.")

        bits = (X > 0.5).astype(np.int8, copy=False)
        head = bits[:, : self._first_block]
        tail = bits[:, self._first_block :]
        # u0: ones in first block, u_tail: ones per tail block
        u0 = head.sum(axis=1).astype(float)
        u_tail = tail.reshape(X.shape[0], self._n_blocks, self._block_size).sum(axis=2).astype(float)

        # v_i = 1 if u_i == block_size else 2 + u_i
        v_tail = np.where(u_tail == float(self._block_size), 1.0, 2.0 + u_tail)
        g = v_tail.sum(axis=1)

        f1 = 1.0 + u0
        f2 = g / f1

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = f1
        F[:, 1] = f2


__all__ = ["ZDT5Problem"]
