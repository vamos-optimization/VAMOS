from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem._output import objective_output

from .tanabe_ishibuchi_utils import sum_violations_gte0


class RE41Problem:
    """
    Tanabe & Ishibuchi (2020) RE41 / RE4-7-1: Car side impact design.
    """

    def __init__(self) -> None:
        self.n_var = 7
        self.n_obj = 4
        self.encoding = "continuous"
        self.xl = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], dtype=float)
        self.xu = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        x1, x2, x3, x4, x5, x6, x7 = (X[:, i] for i in range(7))

        f1 = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f3 = 0.5 * (vmbp + vfd)

        g1 = 1.0 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g2 = 0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
        g3 = 0.32 - (
            0.214
            + 0.00817 * x5
            - 0.045195 * x1
            - 0.0135168 * x1
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007176 * x3
            + 0.023232 * x3
            - 0.00364 * x5 * x6
            - 0.018 * x2 * x2
        )
        g4 = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g5 = 32.0 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g6 = 32.0 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g7 = 32.0 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g8 = 4.0 - f2
        g9 = 9.9 - vmbp
        g10 = 15.7 - vfd
        f4 = sum_violations_gte0(g1, g2, g3, g4, g5, g6, g7, g8, g9, g10)

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3
        Fout[:, 3] = f4


class RE42Problem:
    """
    Tanabe & Ishibuchi (2020) RE42 / RE4-6-2: Conceptual marine design.
    """

    def __init__(self) -> None:
        self.n_var = 6
        self.n_obj = 4
        self.encoding = "continuous"
        self.xl = np.array([150.0, 20.0, 13.0, 10.0, 14.0, 0.63], dtype=float)
        self.xu = np.array([274.32, 32.31, 25.0, 11.71, 18.0, 0.75], dtype=float)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        L, B, D, T, Vk, CB = (X[:, i] for i in range(6))

        displacement = 1.025 * L * B * T * CB
        V = 0.5144 * Vk
        g = 9.8065
        Fn = V / np.power(g * L, 0.5)
        a = (4977.06 * CB * CB) - (8105.61 * CB) + 4456.51
        b = (-10847.2 * CB * CB) + (12817.0 * CB) - 6960.32

        denom = np.maximum(a + b * Fn, 1e-12)
        power = (np.power(displacement, 2.0 / 3.0) * np.power(Vk, 3.0)) / denom

        outfit_weight = np.power(L, 0.8) * np.power(B, 0.6) * np.power(D, 0.3) * np.power(CB, 0.1)
        steel_weight = 0.034 * np.power(L, 1.7) * np.power(B, 0.7) * np.power(D, 0.4) * np.power(CB, 0.5)
        machinery_weight = 0.17 * np.power(np.maximum(power, 0.0), 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * (
            (2000.0 * np.power(np.maximum(steel_weight, 0.0), 0.85))
            + (3500.0 * outfit_weight)
            + (2400.0 * np.power(np.maximum(power, 0.0), 0.8))
        )
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight
        DWT_pos = np.maximum(DWT, 1e-12)

        running_costs = 40000.0 * np.power(DWT_pos, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT_pos, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT_pos, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA
        annual_cargo_safe = np.where(np.abs(annual_cargo) < 1e-12, 1e-12, annual_cargo)

        f1 = annual_costs / annual_cargo_safe
        f2 = light_ship_weight
        f3 = -annual_cargo

        c1 = (L / B) - 6.0
        c2 = -(L / D) + 15.0
        c3 = -(L / T) + 19.0
        c4 = 0.45 * np.power(DWT_pos, 0.31) - T
        c5 = 0.7 * D + 0.7 - T
        c6 = 500000.0 - DWT
        c7 = DWT - 3000.0
        c8 = 0.32 - Fn

        KB = 0.53 * T
        BMT = ((0.085 * CB - 0.002) * B * B) / (T * CB)
        KG = 1.0 + 0.52 * D
        c9 = (KB + BMT - KG) - (0.07 * B)
        f4 = sum_violations_gte0(c1, c2, c3, c4, c5, c6, c7, c8, c9)

        Fout = objective_output(out, len(X), self.n_obj)
        Fout[:, 0] = f1
        Fout[:, 1] = f2
        Fout[:, 2] = f3
        Fout[:, 3] = f4


__all__ = ["RE41Problem", "RE42Problem"]
