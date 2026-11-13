from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output


class RWA1Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA1: Honeycomb heat sink.

    Minimization formulation (paper, Eq. 2):
      f1(x) = -Nu (maximize Nu)
      f2(x) =  FF (minimize friction factor)
    """

    def __init__(self) -> None:
        self.n_var = 5
        self.n_obj = 2
        self.encoding = "continuous"
        self.xl = np.array([20.0, 6.0, 20.0, 0.0, 8000.0], dtype=float)
        self.xu = np.array([60.0, 15.0, 40.0, 30.0, 25000.0], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        h, t, sy, theta, re = (X[:, i] for i in range(5))

        nu = (
            89.027
            + 0.300 * h
            - 0.096 * t
            - 1.124 * sy
            - 0.968 * theta
            + 4.148e-3 * re
            + 0.0464 * h * t
            - 0.0244 * h * sy
            + 0.0159 * h * theta
            + 4.151e-5 * h * re
            + 0.1111 * t * sy
            - 4.121e-5 * sy * re
            + 4.192e-5 * theta * re
        )
        ff = (
            0.4753
            - 0.0181 * h
            + 0.0420 * t
            + 5.481e-3 * sy
            - 0.0191 * theta
            - 3.416e-6 * re
            - 8.851e-4 * h * sy
            + 8.702e-4 * h * theta
            + 1.536e-3 * t * theta
            - 2.761e-6 * t * re
            - 4.400e-4 * sy * theta
            + 9.714e-7 * sy * re
            + 6.777e-4 * h * h
        )

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = -nu
        F[:, 1] = ff


class RWA2Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA2: Crashworthiness design of vehicles.

    Objectives (minimize):
      f1: Mass
      f2: Ain
      f3: Intrusion
    """

    def __init__(self) -> None:
        self.n_var = 5
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.full(self.n_var, 1.0, dtype=float)
        self.xu = np.full(self.n_var, 3.0, dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        t1, t2, t3, t4, t5 = (X[:, i] for i in range(5))

        mass = 1640.2823 + 2.3573285 * t1 + 2.3220035 * t2 + 4.5688768 * t3 + 7.7213633 * t4 + 4.4559504 * t5
        ain = (
            6.5856
            + 1.15 * t1
            - 1.0427 * t2
            + 0.9738 * t3
            + 0.8364 * t4
            - 0.3695 * t1 * t4
            + 0.0861 * t1 * t5
            + 0.3628 * t2 * t4
            - 0.1106 * t1 * t1
            - 0.3437 * t3 * t3
            + 0.1764 * t4 * t4
        )
        intrusion = (
            -0.0551
            + 0.0181 * t1
            + 0.1024 * t2
            + 0.0421 * t3
            - 0.0073 * t1 * t2
            + 0.0240 * t2 * t3
            - 0.0118 * t2 * t4
            - 0.0204 * t3 * t4
            - 0.0080 * t3 * t5
            - 0.0241 * t2 * t2
            + 0.0109 * t4 * t4
        )

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = mass
        F[:, 1] = ain
        F[:, 2] = intrusion


class RWA3Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA3: Synthesis gas production.

    Minimization formulation (paper, Eq. 4):
      f1(x) = -CH4
      f2(x) = -CO
      f3(x) =  H2CO
    """

    def __init__(self) -> None:
        self.n_var = 3
        self.n_obj = 3
        self.encoding = "continuous"
        self.xl = np.array([0.25, 10000.0, 600.0], dtype=float)
        self.xu = np.array([0.55, 20000.0, 1100.0], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        o2ch4 = X[:, 0]
        gv = X[:, 1]
        temp = X[:, 2]

        ch4 = -8.87e-6 * (86.74 + 14.6 * o2ch4 - 3.06 * gv + 18.82 * temp + 3.14 * o2ch4 * gv - 6.91 * o2ch4 * o2ch4 - 13.31 * temp * temp)
        co = (
            -2.152e-9
            * (
                39.46
                + 5.98 * o2ch4
                - 2.4 * gv
                + 13.06 * temp
                + 2.5 * o2ch4 * gv
                + 1.64 * gv * temp
                - 3.9 * o2ch4 * o2ch4
                - 10.15 * temp * temp
                - 3.69 * gv * gv * o2ch4
            )
            + 45.7
        )
        h2co = (
            4.425e-10
            * (
                1.29
                - 0.45 * temp
                - 0.112 * o2ch4 * gv
                - 0.142 * temp * gv
                + 0.109 * o2ch4 * o2ch4
                + 0.405 * temp * temp
                + 0.167 * temp * temp * gv
            )
            + 0.18
        )

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = -ch4
        F[:, 1] = -co
        F[:, 2] = h2co


class RWA4Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA4: Wire electrical discharge machining.

    Minimization formulation (paper, Eq. 5):
      f1(x) = -CR (maximize cutting rate)
      f2(x) =  Ra (minimize surface roughness)
      f3(x) =  DD (minimize dimensional deviation)
    """

    def __init__(self) -> None:
        self.n_var = 5
        self.n_obj = 3
        self.encoding = "continuous"
        # x = (Ton, Toff, WT, SV, SF)
        self.xl = np.array([1.0, 10.0, 850.0, 20.0, 4.0], dtype=float)
        self.xu = np.array([1.4, 26.0, 1650.0, 40.0, 8.0], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        x1, x2, x3, x4, x5 = (X[:, i] for i in range(5))

        cr = (
            1.74
            + 0.42 * x1
            - 0.27 * x2
            + 0.087 * x3
            - 0.19 * x4
            + 0.18 * x5
            + 0.11 * x1 * x1
            + 0.036 * x4 * x4
            - 0.025 * x5 * x5
            + 0.044 * x1 * x2
            + 0.034 * x1 * x4
            + 0.17 * x1 * x5
            - 0.028 * x2 * x4
            + 0.093 * x3 * x4
            - 0.033 * x4 * x5
        )
        ra = (
            2.19
            + 0.26 * x1
            - 0.088 * x2
            + 0.037 * x3
            - 0.16 * x4
            + 0.069 * x5
            + 0.036 * x1 * x1
            + 0.11 * x1 * x3
            - 0.077 * x1 * x4
            - 0.075 * x2 * x3
            + 0.054 * x2 * x4
            + 0.090 * x3 * x5
            + 0.041 * x4 * x5
        )
        dd = (
            0.095
            + 0.013 * x1
            - 8.625e-3 * x2
            - 5.458e-3 * x3
            - 0.012 * x4
            + 1.462e-3 * x1 * x1
            - 6.635e-4 * x2 * x2
            - 1.788e-3 * x4 * x4
            - 0.011 * x1 * x2
            - 6.188e-3 * x1 * x3
            + 8.937e-3 * x1 * x4
            - 4.563e-3 * x1 * x5
            - 0.012 * x2 * x3
            - 1.063e-3 * x2 * x4
            + 2.438e-3 * x2 * x5
            - 1.937e-3 * x3 * x4
            - 1.188e-3 * x3 * x5
            - 3.312e-3 * x4 * x5
        )

        F = objective_output(out, len(X), self.n_obj)
        F[:, 0] = -cr
        F[:, 1] = ra
        F[:, 2] = dd


__all__ = ["RWA1Problem", "RWA2Problem", "RWA3Problem", "RWA4Problem"]
