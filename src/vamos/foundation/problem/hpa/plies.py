"""The sizing of the flange plies of the main spar, below level 2 (Namura's Algorithm 1).

This is ``HPADesigner._optimize_ply`` of Namura's reference code (https://github.com/Nobuo-Namura/hpa, commit
``3524115``, ``hpa/designer.py``; BSD-3-Clause-Clear, copyright 2023 Fujitsu Limited), in its benchmark mode
(``FINE_MODE=False``). From the tip segment to the root one, the flanges widen until the strains, reduced by the
strain ratios of the design variables, are below the allowable strain; then each flange is cut to the part of its
segment that needs it, from the outer flange to the inner one, and one end of each is moved to an end of its pipe.
The reference code's quirks are kept, since they define the benchmark: the second search of a cut reuses the limits
the first one returned, and the side to which the flanges are aligned compares sums that both use their roots.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._kernels import Kernels
from .wing import ALLOWABLE_STRAIN, Wing

Array = np.ndarray[Any, Any]


def _cut(
    wing: Wing, i: int, j: int, inside: Array, strain_max: float, segment: tuple[float, float], limits: tuple[float, float]
) -> tuple[float, float, float, float]:
    """The root and tip of flange ``j`` of segment ``i`` where it is needed, and the limits for the next flange."""
    flange = j - 4
    strain = np.abs(wing.moment / (wing.EI - wing.EI_ply[:, flange]) * 0.5 * wing.diameter) + wing.axial_force / (
        wing.EA - wing.EA_ply[:, flange]
    )
    strain_zerolift = np.abs(wing.moment_zerolift / (wing.EI - wing.EI_ply[:, flange]) * 0.5 * wing.diameter)
    strain = np.max([strain / wing.strain_ratios[0, i], strain_zerolift / wing.strain_ratios[1, i]], axis=0)
    strain = np.where(inside, strain, 0)
    violated = np.where(strain >= strain_max, 1, 0)
    i_min = int(np.argmax(inside))
    i_max = len(inside) - int(np.argmax(inside[::-1])) - 1
    if violated.max() > 0:
        root_index = int(np.clip(np.argmax(violated), i_min, i_max))
        tip_index = int(np.clip(len(violated) - np.argmax(violated[::-1]), i_min, i_max))
    else:
        root_index, tip_index = i_min, i_min
    return _adjust(wing, root_index, tip_index, len(violated), segment, limits)


def _adjust(
    wing: Wing, i_root: int, i_tip: int, n: int, segment: tuple[float, float], limits: tuple[float, float]
) -> tuple[float, float, float, float]:
    """The root and tip at grid indices ``i_root`` and ``i_tip``, moved to the ends of the pipe and to the limits."""
    y0_start, y0_end = segment
    root_limit, tip_limit = limits
    root = wing.y0[i_root]
    tip = wing.y0[i_tip]
    # An end on the grid point nearest to an end of the pipe moves to that end.
    if i_root > 0 and wing.y0[i_root - 1] < y0_start:
        root = y0_start
    if i_tip > 0 and wing.y0[i_tip - 1] < y0_start:
        tip = y0_start
    if i_tip + 1 < n and wing.y0[i_tip + 1] >= y0_end:
        tip = y0_end
    # An inner flange must be at least as long as an outer one.
    if tip - root > 0:
        if root > root_limit:
            root = root_limit
        else:
            root_limit = root
        if tip < tip_limit:
            tip = tip_limit
        else:
            tip_limit = tip
    return root, tip, root_limit, tip_limit


#: The trial widenings of a segment that :func:`_first_fit` computes at once, before the others.
FIRST_TRIALS = 8


def _first_fit(kernels: Kernels, dphi: Array, phis: Array, *args: Any) -> int:
    """The first trial widening ``dphi`` of :func:`_widen` whose strains fit, or -1: the kernel's, a few at a time.

    The trials are independent, so computing the sines of the first ones, and the others only when none of those
    fits, gives the trial that computing them all gives.
    """
    sin_phi = np.sin(phis)
    for low, high in ((0, FIRST_TRIALS), (FIRST_TRIALS, dphi.size)):
        if low >= dphi.size:
            break
        sin_trial = np.sin(phis[None, :] + dphi[low:high, None])
        fit = kernels.widen_trials(dphi[low:high], sin_trial, sin_phi, *args)
        if fit >= 0:
            return int(low + fit)
    return -1


def _widen(wing: Wing, strain_max: float) -> bool:
    """Widen the flanges of every segment whose strain is too large; whether none was."""
    feasible = True
    j: int | None = None  # as in the reference code, an empty search keeps the previous segment's widening
    for i in range(len(wing.ply_wing) - 1, -1, -1):
        plies = wing.ply_wing[i]
        n_entire = np.sum(plies[:, 1] >= 180)
        a, b, iy1, peak1, iy2, peak2 = (
            wing._kernels.segment_peaks(wing.y0, wing.y_div[i], wing.y_div[i + 1], wing.strain, wing.strain_zerolift)
            if wing._kernels is not None
            else (0, 0, 0, 0.0, 0, 0.0)
        )
        if b > a:
            strain1 = peak1 / wing.strain_ratios[0, i]
            strain2 = peak2 / wing.strain_ratios[1, i]
            iy1, iy2 = a + iy1, a + iy2
        else:  # the NumPy code (and its error for a segment without grid points)
            inside = (wing.y0 >= wing.y_div[i]) & (wing.y0 < wing.y_div[i + 1])
            strain1 = np.abs(wing.strain[inside]).max() / wing.strain_ratios[0, i]
            strain2 = np.abs(wing.strain_zerolift[inside]).max() / wing.strain_ratios[1, i]
            iy1 = inside.argmax() + np.abs(wing.strain[inside]).argmax()
            iy2 = inside.argmax() + np.abs(wing.strain_zerolift[inside]).argmax()
        if not max(strain1, strain2) > strain_max:
            continue
        feasible = False
        M1 = wing.moment[iy1]
        AF1 = wing.axial_force[iy1]
        M2 = -wing.moment_zerolift[iy2]
        phis = np.deg2rad(plies[n_entire:, 1])
        Ex = wing.Ex_ply[i, :]
        Ex = Ex[Ex > 0]
        tr3_1 = plies[n_entire:, 4] * (0.5 * wing.diameter[iy1]) ** 3
        tr_1 = plies[n_entire:, 4] * (0.5 * wing.diameter[iy1])
        tr3_2 = plies[n_entire:, 4] * (0.5 * wing.diameter[iy2]) ** 3
        # The reference code tries one more degree at a time until the strains fit; every trial is computed at
        # once, element by element as the loop computes it (NumPy sums each row as it sums a vector), and the
        # first that fits is taken, or the last.
        widenings = np.arange(1, 180 - int(phis.max()), 1)
        if widenings.size and wing._kernels is not None and Ex.size == phis.size:
            fit = _first_fit(
                wing._kernels, np.deg2rad(widenings), phis, Ex, tr3_1, tr_1, tr3_2, M1, AF1, M2, wing.EI[iy1], wing.EA[iy1],
                wing.EI[iy2], wing.diameter[iy1], wing.diameter[iy2], wing.strain_ratios[0, i], wing.strain_ratios[1, i], strain_max,
            )  # fmt: skip
            j = int(widenings[fit] if fit >= 0 else widenings[-1])
        elif widenings.size:
            dphi = np.deg2rad(widenings)[:, None]
            gain = dphi + np.sin(phis[None, :] + dphi) - np.sin(phis)[None, :]
            dIy1 = gain * tr3_1[None, :]
            A1 = 2 * dphi * tr_1[None, :]
            dIy2 = gain * tr3_2[None, :]
            strain1 = (
                np.abs(M1 / (wing.EI[iy1] + np.sum(Ex[None, :] * dIy1, axis=1)) * 0.5 * wing.diameter[iy1])
                + AF1 / (wing.EA[iy1] + np.sum(Ex[None, :] * A1, axis=1))
            ) / wing.strain_ratios[0, i]
            strain2 = np.abs(M2 / (wing.EI[iy2] + np.sum(Ex[None, :] * dIy2, axis=1)) * 0.5 * wing.diameter[iy2]) / wing.strain_ratios[1, i]
            # Python's max(strain1, strain2) is strain2 only when strain2 > strain1.
            fits = np.where(strain2 > strain1, strain2, strain1) <= strain_max
            j = int(widenings[int(np.argmax(fits))] if fits.any() else widenings[-1])
        if j is None:
            raise ValueError("the flanges of the HPA wing cannot widen: they already cover the whole pipe")
        wing.ply_wing[i][n_entire:, 1] += j
        wing.ply_wing[i][n_entire:, 1] = np.clip(wing.ply_wing[i][n_entire:, 1], 0, 179)
    return feasible


def _align(wing: Wing, i: int, segment: tuple[float, float]) -> None:
    """Move one end of every flange of segment ``i`` to an end of its pipe, restoring the stiffness it adds."""
    y0_start, y0_end = segment
    plies = wing.ply_wing[i]
    not0 = plies[:, 3] > plies[:, 2]
    sum_root = plies[not0, 2].sum()
    sum_tip = (1 - plies[not0, 2]).sum()
    if not sum_root * sum_tip > 0:
        return
    inside = (wing.y0 >= y0_start) & (wing.y0 < y0_end)
    for j in range(plies.shape[0] - 1, 3, -1):
        if sum_root <= sum_tip:
            if not (plies[j, 3] > plies[j, 2] and plies[j, 2] > 0):
                continue
            root = plies[j, 2] * (y0_end - y0_start) + y0_start
            on = inside & (wing.y0 < root)
        else:
            if not (plies[j, 3] > plies[j, 2] and plies[j, 3] < 1):
                continue
            tip = plies[j, 3] * (y0_end - y0_start) + y0_start
            on = inside & (wing.y0 >= tip)
        wing.EI[on] += wing.EI_ply[on, j - 4]
        wing.EA[on] += wing.EA_ply[on, j - 4]
        wing.EI[-1] = wing.EI[-2]
        wing.EA[-1] = wing.EA[-2]
    if sum_root <= sum_tip:
        plies[not0, 2] = 0.0
    else:
        plies[not0, 3] = 1.0


def _cut_segments(wing: Wing, kernels: Kernels, strain_max: float) -> None:
    """The cuts and alignments of :func:`optimize_plies`, segment by segment, with the compiled kernel."""
    for i in range(len(wing.ply_wing) - 1, -1, -1):
        plies = wing.ply_wing[i]
        y0_start, y0_end = wing.y_div[i], wing.y_div[i + 1]
        limits = np.array([1.1, -0.1])
        j, first = plies.shape[0] - 1, True
        while j >= 0:
            # The kernel stops before the first flange cut after many plies were removed: the wing is updated.
            j = kernels.cut_flanges(
                plies, j, first, limits, y0_start, y0_end, wing.y0, wing.moment, wing.moment_zerolift, wing.axial_force,
                wing.EI, wing.EA, wing.EI_ply, wing.EA_ply, wing.diameter, wing.strain_ratios[0, i], wing.strain_ratios[1, i],
                strain_max,
            )  # fmt: skip
            if j >= 0:
                wing.update(stiffness=False)
                first = False
        wing.update(stiffness=False)


def optimize_plies(wing: Wing) -> None:
    """Size the flanges of ``wing`` in place, and update its weight, lift and deflections (``_optimize_ply``)."""
    # 0.99 because cutting plies off lightens the wing, which can raise the strain above the maximum.
    strain_max = 0.99 * ALLOWABLE_STRAIN
    if not _widen(wing, strain_max):
        wing.update(stiffness=True)
    if wing._kernels is not None:
        _cut_segments(wing, wing._kernels, strain_max)
        return
    for i in range(len(wing.ply_wing) - 1, -1, -1):
        plies = wing.ply_wing[i]
        segment = (wing.y_div[i], wing.y_div[i + 1])
        y0_start, y0_end = segment
        limits = (1.1, -0.1)
        first = True
        for j in range(plies.shape[0] - 1, 3, -1):
            ply = plies[j]
            inside = (wing.y0 >= ply[2] * (y0_end - y0_start) + y0_start) & (wing.y0 < ply[3] * (y0_end - y0_start) + y0_start)
            root, tip, *new_limits = _cut(wing, i, j, inside, strain_max, segment, limits)
            limits = (new_limits[0], new_limits[1])
            # Removing many plies at once lightens the wing: its strains are computed again before the cut.
            if first and tip > root:
                first = False
                if plies.shape[0] - 1 - j > 4:
                    wing.update(stiffness=False)
                    root, tip, *new_limits = _cut(wing, i, j, inside, strain_max, segment, limits)
                    limits = (new_limits[0], new_limits[1])
            plies[j, 2] = (root - y0_start) / (y0_end - y0_start)
            plies[j, 3] = (tip - y0_start) / (y0_end - y0_start)
            off = inside & ((wing.y0 < root) | (wing.y0 >= tip))
            wing.EI[off] -= wing.EI_ply[off, j - 4]
            wing.EA[off] -= wing.EA_ply[off, j - 4]
            wing.EI[-1] = wing.EI[-2]
            wing.EA[-1] = wing.EA[-2]
        # The reference code notes that a practical design can skip this alignment.
        _align(wing, i, segment)
        wing.update(stiffness=False)


__all__ = ["optimize_plies"]
