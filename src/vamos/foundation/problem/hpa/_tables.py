"""What the analysis of a wing (:mod:`.wing`) computes once per process: its grids and the moduli of its plies.

The lifting-line grid and the sines of its equations depend only on the number of stations, the structural grid
on its number of points, and the moduli of the CFRP plies (the reference code's classical lamination theory) on the
angles of their fibres. They are computed by the NumPy operations of the reference code, once.
"""

from __future__ import annotations

from functools import cache, lru_cache
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]

E_L, E_T, G_LT = 170e9, 6.0e9, 3.3e9  # [Pa]
NU_L, NU_T = 0.28, 0.01


def _frozen(*arrays: Array) -> tuple[Array, ...]:
    for array in arrays:
        array.flags.writeable = False
    return arrays


@lru_cache(maxsize=8)
def lifting_line_grid(n_aero: int) -> tuple[Array, ...]:
    """The lifting-line grid and the sines of its equations, which depend only on ``n_aero``.

    ``theta`` and ``y0`` are the stations from the root to the tip; ``m``, ``ms``, ``sin(theta)``, ``sin(thetas)``
    and ``sin(ms * thetas)`` are those of :meth:`Wing.lifting_line`, computed by the same NumPy operations.
    """
    theta = 0.5 * np.pi * np.linspace(0, 1, n_aero + 1)
    y = np.cos(theta)
    order = np.argsort(y, kind="stable")
    theta_aero = theta[order]
    stations = theta_aero[:-1]
    m = np.arange(1, 2 * n_aero, 2)
    ms = np.tile(m, [n_aero, 1])
    thetas = np.tile(stations, [n_aero, 1]).T
    return _frozen(theta_aero, y[order], m, ms, np.sin(stations), np.sin(thetas), np.sin(ms * thetas))


@lru_cache(maxsize=8)
def structural_grid(n_struc: int) -> Array:
    return _frozen(np.linspace(0, 1, n_struc))[0]


def flange_modulus(axis: Any, Ex0: float, nui: float, ETL: float) -> Any:
    """The axial modulus of a flange ply whose fibres are at ``axis`` radians, as the reference code computes it."""
    if axis == 0:
        return Ex0
    sin, cos = np.sin(axis), np.cos(axis)
    Exx_j = E_T * nui * sin**4 + 2 * ETL * (sin * cos) ** 2 + E_L * nui * cos**4
    Eyy_j = E_L * nui * sin**4 + 2 * ETL * (sin * cos) ** 2 + E_T * nui * cos**4
    Exy_j = E_T * NU_L * nui * (sin**4 + cos**4) + ((E_L + E_T) * nui - 4 * G_LT) * (sin * cos) ** 2
    return Exx_j - Exy_j**2 / Eyy_j


@cache
def entire_moduli(axis_angle: float) -> tuple[Any, Any, Any, Any]:
    """The factors of ``Exx``, ``Eyy``, ``Exy`` and ``Gxy`` of a ply at ``axis_angle`` degrees around the whole pipe."""
    nui, ETL, _ = moduli()
    axis = np.deg2rad(axis_angle)
    sin, cos = np.sin(axis), np.cos(axis)
    return (
        E_T * nui * sin**4 + 2 * ETL * (sin * cos) ** 2 + E_L * nui * cos**4,
        E_L * nui * sin**4 + 2 * ETL * (sin * cos) ** 2 + E_T * nui * cos**4,
        E_T * NU_L * nui * (sin**4 + cos**4) + ((E_L + E_T) * nui - 4 * G_LT) * (sin * cos) ** 2,
        0.25 * (E_L + E_T * (1 - 2 * NU_L)) * nui * np.sin(2 * axis) ** 2 + G_LT * np.cos(2 * axis) ** 2,
    )


@lru_cache(maxsize=1)
def moduli() -> tuple[float, float, float]:
    """``nui``, ``ETL`` and ``Ex0`` (the axial modulus of a ply along the spar) of :meth:`Wing.stiffness`."""
    nui = 1 / (1 - NU_L * NU_T)
    ETL = E_T * NU_L * nui + 2 * G_LT
    Exx0 = E_L * nui
    Eyy0 = E_T * nui
    Exy0 = E_T * NU_L * nui
    return nui, ETL, Exx0 - Exy0**2 / Eyy0


__all__ = ["E_L", "E_T", "G_LT", "NU_L", "NU_T", "entire_moduli", "flange_modulus", "lifting_line_grid", "moduli", "structural_grid"]
