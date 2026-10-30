from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _rwa7_xcc(
    alpha: NDArray[np.float64], dha: NDArray[np.float64], doa: NDArray[np.float64], optt: NDArray[np.float64]
) -> NDArray[np.float64]:
    val = (
        0.153
        - 0.322 * alpha
        + 0.396 * dha
        + 0.424 * doa
        + 0.0226 * optt
        + 0.175 * alpha * alpha
        + 0.0185 * dha * alpha
        - 0.0701 * dha * dha
        - 0.251 * doa * alpha
        + 0.179 * doa * dha
        + 0.0150 * doa * doa
        + 0.0134 * optt * alpha
        + 0.0296 * optt * dha
        + 0.0752 * optt * doa
        + 0.0192 * optt * optt
    )
    return np.asarray(val, dtype=float)


def _rwa7_tfmax(
    alpha: NDArray[np.float64], dha: NDArray[np.float64], doa: NDArray[np.float64], optt: NDArray[np.float64]
) -> NDArray[np.float64]:
    val = (
        0.692
        + 0.477 * alpha
        - 0.687 * dha
        - 0.080 * doa
        - 0.0650 * optt
        - 0.167 * alpha * alpha
        - 0.0129 * dha * alpha
        + 0.0796 * dha * dha
        - 0.0634 * doa * alpha
        - 0.0257 * doa * dha
        + 0.0877 * doa * doa
        - 0.0521 * optt * alpha
        + 0.00156 * optt * dha
        + 0.00198 * optt * doa
        + 0.0184 * optt * optt
    )
    return np.asarray(val, dtype=float)


def _rwa7_ttmax(
    alpha: NDArray[np.float64], dha: NDArray[np.float64], doa: NDArray[np.float64], optt: NDArray[np.float64]
) -> NDArray[np.float64]:
    val = (
        0.370
        - 0.205 * alpha
        + 0.0307 * dha
        + 0.108 * doa
        + 1.019 * optt
        - 0.135 * alpha * alpha
        + 0.0141 * dha * alpha
        + 0.0998 * dha * dha
        + 0.208 * doa * alpha
        - 0.0301 * doa * dha
        - 0.226 * doa * doa
        + 0.353 * optt * alpha
        - 0.0497 * optt * doa
        - 0.423 * optt * optt
        + 0.202 * dha * alpha * alpha
        - 0.281 * doa * alpha * alpha
        - 0.342 * dha * dha * alpha
        - 0.245 * dha * dha * doa
        + 0.281 * doa * doa * dha
        - 0.184 * optt * optt * alpha
        - 0.281 * dha * alpha * doa
    )
    return np.asarray(val, dtype=float)


class RWA7Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA7: Liquid-rocket single element injector design.

    Objectives (minimize): Xcc, TFmax, TTmax (paper, Eq. 8). They are the response
    surfaces of Goel et al. (2007), which RE37 also uses, in another order.

    The variables are the hydrogen flow angle alpha, the hydrogen area dHA, the
    oxygen area dOA and the oxidizer post tip thickness OPTT. The response surfaces
    are fitted in variables normalized to [0, 1], so the search space is [0, 1]^4,
    as in RE37 and jMetal's RWA7. The normalized values 0 and 1 are the ends of the
    design ranges: alpha in [0, 20], dHA in [0, 25], dOA in [-40, 0] and OPTT in
    [0.01, 0.02].
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.zeros(4, dtype=float)
        self.xu = np.ones(4, dtype=float)

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        alpha, dha, doa, optt = (X[:, i] for i in range(4))

        xcc = _rwa7_xcc(alpha, dha, doa, optt)
        tfmax = _rwa7_tfmax(alpha, dha, doa, optt)
        ttmax = _rwa7_ttmax(alpha, dha, doa, optt)

        F = out["F"]
        F[:, 0] = xcc
        F[:, 1] = tfmax
        F[:, 2] = ttmax


class RWA8Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA8: Liquid-rocket single element injector design.

    Objectives (minimize): TFmax, TW4, TTmax, Xcc (paper, Eq. 9). It adds the
    response surface TW4 to the three of RWA7 and searches the same normalized
    domain [0, 1]^4 (see RWA7Problem), as jMetal's RWA8 does.
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 4
        self.encoding = "continuous"
        self.xl = np.zeros(4, dtype=float)
        self.xu = np.ones(4, dtype=float)

    def evaluate(self, X: NDArray[np.float64], out: dict[str, NDArray[np.float64]]) -> None:
        alpha, dha, doa, optt = (X[:, i] for i in range(4))

        tfmax = _rwa7_tfmax(alpha, dha, doa, optt)
        tw4 = (
            0.758
            + 0.358 * alpha
            - 0.807 * dha
            + 0.0925 * doa
            - 0.0468 * optt
            - 0.172 * alpha * alpha
            + 0.0106 * dha * alpha
            + 0.0697 * dha * dha
            - 0.146 * doa * alpha
            - 0.0416 * doa * dha
            + 0.102 * doa * doa
            - 0.0694 * optt * alpha
            - 0.00503 * optt * dha
            + 0.0151 * optt * doa
            + 0.0173 * optt * optt
        )
        ttmax = _rwa7_ttmax(alpha, dha, doa, optt)
        xcc = _rwa7_xcc(alpha, dha, doa, optt)

        F = out["F"]
        F[:, 0] = tfmax
        F[:, 1] = tw4
        F[:, 2] = ttmax
        F[:, 3] = xcc


__all__ = ["RWA7Problem", "RWA8Problem"]
