"""RCM12-RCM21: mechanical design problems of the CEC 2021 suite, as jMetal's classes define them."""

from __future__ import annotations

import numpy as np

from vamos.foundation.problem.rcm._common import Array, Entry, java_round
from vamos.foundation.problem.rcm.mechanical_a import speed_reducer

# RCM15's wire diameters, indexed by the rounded third variable (1-42). The 25th, 0.0105, breaks the
# increasing order (0.105 would not); it is the value of the reference code and of jMetal.
_WIRE_DIAMETERS = """
0.009 0.0095 0.0104 0.0118 0.0128 0.0132 0.014 0.015 0.0162 0.0173 0.018 0.020 0.023 0.025
0.028 0.032 0.035 0.041 0.047 0.054 0.063 0.072 0.080 0.092 0.0105 0.120 0.135 0.148
0.162 0.177 0.192 0.207 0.225 0.244 0.263 0.283 0.307 0.331 0.362 0.394 0.4375 0.500
"""


def _rcm12(x: Array) -> tuple[list[Array], list[Array]]:
    """Simply supported I-beam design."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    p, l, e = 600.0, 200.0, 2e4
    inertia = x3 * (x1 - 2 * x4) ** 3 + 2 * x2 * x4 * (4 * x4 * x4 + 3 * x1 * (x1 - 2 * x4))
    f1 = 2 * x2 * x4 + x3 * (x1 - 2 * x4)
    f2 = p * l**3 / (48 * e * inertia / 12)
    g1 = -16 + 180000 * x1 / inertia + 15000 * x2 / ((x1 - 2 * x4) * x3**3 + 2 * x4 * x2**3)
    return [f1, f2], [-g1]


def _rcm13(x: Array) -> tuple[list[Array], list[Array]]:
    """Gear box design: RCM06 with the stress of the second shaft as a third objective, bounded by 1100."""
    (f1, f2, f3), c = speed_reducer(x)
    return [f1, f2, f3], [*c, -(f3 - 1100)]


def _rcm14(x: Array) -> tuple[list[Array], list[Array]]:
    """Multiple-disk clutch brake design."""
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    mf, ms, iz, n, tmax = 3.0, 40.0, 55.0, 250.0, 15.0
    s, delta, vsrmax, rho, pmax = 1.5, 0.5, 10.0, 0.0000078, 1.0
    mu, lmax, del_r = 0.6, 30.0, 20.0
    rsr = 2.0 / 3 * (x2**3 - x1**3) / (x2 * x2 * x1 * x1)
    vsr = np.pi * rsr * n / 30
    a = np.pi * (x2 * x2 - x1 * x1)
    prz = x4 / a
    w = np.pi * n / 30
    mh = 2.0 / 3 * mu * x4 * x5 * (x2**3 - x1**3) / (x2 * x2 - x1 * x1)
    t = iz * w / (mh + mf)
    f1 = np.pi * (x2 * x2 - x1 * x1) * x3 * (x5 + 1) * rho
    g = [
        -x2 + x1 + del_r,
        (x5 + 1) * (x3 + delta) - lmax,
        prz - pmax,
        prz * vsr - pmax * vsrmax,
        vsr - vsrmax,
        t - tmax,
        s * ms - mh,
        -t,
    ]
    return [f1, t], [-gi for gi in g]


def _rcm15(x: Array) -> tuple[list[Array], list[Array]]:
    """Spring design; the number of coils x1 is rounded and x3 picks one of 42 wire diameters."""
    x1, x2 = java_round(x[0]), x[1]
    index = np.clip(java_round(x[2]), 1, 42).astype(int)
    x3 = np.array(_WIRE_DIAMETERS.split(), dtype=float)[index - 1]
    cf = (4 * x2 / x3 - 1) / (4 * x2 / x3 - 4) + 0.615 * x3 / x2
    k = (11.5e6 * x3**4) / (8 * x1 * x2**3)
    lf = 1000 / k + 1.05 * (x1 + 2) * x3
    sigp = 300 / k
    f1 = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4
    f2 = (8000 * cf * x2) / (np.pi * x3**3)
    g = [
        (8000 * cf * x2) / (np.pi * x3**3) - 189000,
        lf - 14,
        0.2 - x3,
        x2 - 3,
        3 - x2 / x3,
        sigp - 6,
        sigp + 700 / k + 1.05 * (x1 + 2) * x3 - lf,
        1.25 - 700 / k,
    ]
    return [f1, f2], [-gi for gi in g]


def _rcm16(x: Array) -> tuple[list[Array], list[Array]]:
    """Cantilever beam design."""
    x1, x2 = x[0], x[1]
    p, e, sy, delta_max, rho = 1.0, 207000000.0, 300000.0, 0.005, 7800.0
    f1 = 0.25 * rho * np.pi * x2 * x1 * x1
    f2 = (64 * p * x2**3) / (3 * e * np.pi * x1**4)
    g1 = -sy + (32 * p * x2) / (np.pi * x1**3)
    g2 = -delta_max + (64 * p * x2**3) / (3 * e * np.pi * x1**4)
    return [f1, f2], [-g1, -g2]


def _rcm17(x: Array) -> tuple[list[Array], list[Array]]:
    """Bulk carrier design, as the reference code writes it (CRE32 is the original formulation)."""
    l, b, d, t, v_k, c_b = x[0], x[1], x[2], x[3], x[4], x[5]
    a = 4977.06 * c_b * c_b - 8105.61 * c_b + 4456.51
    bb = -10847.2 * c_b * c_b + 12817 * c_b - 6960.32
    f_n = 0.5144 / np.sqrt(9.8065 * l)
    p = (np.power(1.025 * l * b * t * c_b, 2.0 / 3) * v_k**3) / (a + bb * f_n)
    w_s = 0.034 * np.power(l, 1.7) * np.power(b, 0.6) * np.power(d, 0.4) * np.power(c_b, 0.5)
    w_o = np.power(l, 0.8) * np.power(b, 0.6) * np.power(d, 0.3) * np.power(c_b, 0.1)
    w_m = 0.17 * np.power(p, 0.9)
    ls = w_s + w_o + w_m
    d_wt = 1.025 * l * b * t * c_b - ls
    f_c = 0.19 * 24 * p / 1000 + 0.2
    d_cwt = d_wt - f_c * ((5000 * v_k) / 24 + 5) - 2 * np.sqrt(d_wt)
    r_trp = 350 / ((5000 * v_k) / 24 + 2 * (d_cwt / 8000 + 0.5))
    ac = d_cwt * r_trp
    s_d = 5000 * v_k / 24
    c_c = 0.2 * 1.3 * (2000 * np.power(w_s, 0.85) + 3500 * w_o + 2400 * np.power(p, 0.8))
    c_r = 40000 * np.power(d_wt, 0.3)
    c_v = (1.05 * 100 * f_c * s_d + 6.3 * np.power(d_wt, 0.8)) * r_trp
    c = [
        l / b - 6,
        15 - l / d,
        19 - l / t,
        0.45 * np.power(d_wt, 0.31) - t,
        0.7 * d + 0.7 - t,
        0.32 - f_n,
        0.53 * t + ((0.085 * c_b - 0.002) * b * b) / (t * c_b) - (1 + 0.52 * d) - 0.07 * b,
        d_wt - 3000,
        500000 - d_wt,
    ]
    return [(c_c + c_r + c_v) / ac, ls, -ac], c


def _rcm18(x: Array) -> tuple[list[Array], list[Array]]:
    """Front rail design."""
    hh, w, t = x[0], x[1], x[2]
    ea, fa = 14496.5, 234.9
    e = (
        -70973.4
        + 958.656 * w
        + 614.173 * hh
        - 3.827 * w * hh
        + 57.023 * w * t
        + 63.274 * hh * t
        - 3.582 * w * w
        - 1.4842 * hh * hh
        - 1890.174 * t * t
    )
    f = 111.854 - 20.210 * w + 7.560 * hh - 0.025 * w * hh + 2.731 * w * t - 1.479 * hh * t + 0.165 * w * w
    return [ea / e, f / fa], [(hh - 136) * (146 - hh), (w - 58) * (66 - w), (t - 1.4) * (2.2 - t)]


def _rcm19(x: Array) -> tuple[list[Array], list[Array]]:
    """Multi-product batch plant; the numbers of units n1, n2 and n3 are rounded."""
    n1, n2, n3 = java_round(x[0]), java_round(x[1]), java_round(x[2])
    v1, v2, v3, tl1, tl2, b1, b2 = x[3], x[4], x[5], x[6], x[7], x[8], x[9]
    s11, s21, s12, s22, s13, s23 = 2.0, 4.0, 3.0, 6.0, 4.0, 3.0
    t11, t21, t12, t22, t13, t23 = 8.0, 16.0, 20.0, 4.0, 8.0, 4.0
    h, alp, beta, q1, q2 = 6000.0, 250.0, 0.6, 40000.0, 20000.0
    f1 = alp * (n1 * v1**beta + n2 * v2**beta + n3 * v3**beta)
    f2 = 65 * (q1 / b1 + q2 / b2) + 0.08 * q1 + 0.1 * q2
    f3 = q1 * tl1 / b1 + q2 * tl2 / b2
    g = [
        q1 * tl1 / b1 + q2 * tl2 / b2 - h,
        s11 * b1 + s21 * b2 - v1,
        s12 * b1 + s22 * b2 - v2,
        s13 * b1 + s23 * b2 - v3,
        t11 - n1 * tl1,
        t12 - n2 * tl1,
        t13 - n3 * tl1,
        t21 - n1 * tl2,
        t22 - n2 * tl2,
        t23 - n3 * tl2,
    ]
    return [f1, f2, f3], [-gi for gi in g]


def _rcm20(x: Array) -> tuple[list[Array], list[Array]]:
    """Hydrostatic thrust bearing design."""
    r, ro, mu, q = x[0], x[1], x[2], x[3]
    gamma, c, n, c1, ws = 0.0307, 0.5, -3.55, 10.04, 101000.0
    pmax, del_tmax, hmin, gg, n_speed = 1000.0, 50.0, 0.001, 386.4, 750.0
    p = (np.log10(np.log10(8.122e6 * mu + 0.8)) - c1) / n
    del_t = 2 * (np.power(10.0, p) - 560)
    ef = 9336 * q * gamma * c * del_t
    h = (2 * np.pi * n_speed / 60) ** 2 * 2 * np.pi * mu / ef * (r**4 / 4 - ro**4 / 4) - 1e-5
    po = (6 * mu * q / (np.pi * h**3)) * np.log(r / ro)
    w = np.pi * po / 2 * (r * r - ro * ro) / (np.log(r / ro) - 1e-5)
    f2 = gamma / (gg * po) * (q / (2 * np.pi * r * h))
    g = [ws - w, po - pmax, del_t - del_tmax, hmin - h, ro - r, f2 - 0.001, w / (np.pi * (r * r - ro * ro) + 1e-5) - 5000]
    return [(q * po / 0.7 + ef) / 12, f2], [-gi for gi in g]


def _rcm21(x: Array) -> tuple[list[Array], list[Array]]:
    """Crash energy management for a high-speed train: two quadratic response surfaces, both in bounds."""
    x1, x2, x3, x4, x5, x6 = x[0], x[1], x[2], x[3], x[4], x[5]
    f1 = (
        1.3667145844797
        - 0.00904459793976106 * x1
        - 0.0016193573938033 * x2
        - 0.00758531275221425 * x3
        - 0.00440727360327102 * x4
        - 0.00572216860791644 * x5
        - 0.00936039926190721 * x6
        + 2.62510221107328e-6 * (x1 * x1)
        + 4.92982681358861e-7 * (x2 * x2)
        + 2.25524989067108e-6 * (x3 * x3)
        + 1.84605439400301e-6 * (x4 * x4)
        + 2.17175358243416e-6 * (x5 * x5)
        + 3.90158043948054e-6 * (x6 * x6)
        + 4.55276994245781e-7 * x1 * x2
        - 6.37013576290982e-7 * x1 * x3
        + 8.26736480446359e-7 * x1 * x4
        + 5.66352809442276e-8 * x1 * x5
        - 3.20213897443278e-7 * x1 * x6
        + 1.18015467772812e-8 * x2 * x3
        + 9.25820391546515e-8 * x2 * x4
        - 1.05705364119837e-7 * x2 * x5
        - 4.74797783014687e-7 * x2 * x6
        - 5.02319867013788e-7 * x3 * x4
        + 9.54284258085225e-7 * x3 * x5
        + 1.80533309229454e-7 * x3 * x6
        - 1.07938022118477e-6 * x4 * x5
        - 1.81370642220182e-7 * x4 * x6
        - 2.24238851688047e-7 * x5 * x6
    )
    f2 = (
        -1.19896668942683
        + 3.04107017009774 * x1
        + 1.23535701600191 * x2
        + 2.13882039381528 * x3
        + 2.33495178382303 * x4
        + 2.68632494801975 * x5
        + 3.43918953617606 * x6
        - 7.89144544980703e-4 * (x1 * x1)
        - 2.06085185698215e-4 * (x2 * x2)
        - 7.15269900037858e-4 * (x3 * x3)
        - 7.8449237573837e-4 * (x4 * x4)
        - 9.31396896237177e-4 * (x5 * x5)
        - 1.40826531972195e-3 * (x6 * x6)
        - 1.60434988248392e-4 * x1 * x2
        + 2.0824655419411e-4 * x1 * x3
        - 3.0530659653553e-4 * x1 * x4
        - 8.10145973591615e-5 * x1 * x5
        + 6.94728759651311e-5 * x1 * x6
        + 1.18015467772812e-8 * x2 * x3
        + 9.25820391546515e-8 * x2 * x4
        - 1.05705364119837e-7 * x2 * x5
        + 1.69935290196781e-4 * x2 * x6
        + 2.32421829190088e-5 * x3 * x4
        - 2.0808624041163476e-4 * x3 * x5
        + 1.75576341867273e-5 * x3 * x6
        + 2.68422081654044e-4 * x4 * x5
        + 4.39852066801981e-5 * x4 * x6
        + 2.96785446021357e-5 * x5 * x6
    )
    return [f1, f2], [-(f1 - 5), f1, -(f2 - 28), f2]


PROBLEMS: dict[str, Entry] = {
    "rcm12": (_rcm12, (10.0, 10.0, 0.9, 0.9), (80.0, 50.0, 5.0, 5.0), 2, 1),
    "rcm13": (_rcm13, (2.6, 0.7, 16.51, 7.3, 7.3, 2.9, 5.0), (3.6, 0.8, 28.49, 8.3, 8.3, 3.9, 5.5), 3, 11),
    "rcm14": (_rcm14, (60.0, 90.0, 1.0, 0.0, 2.0), (80.0, 110.0, 3.0, 1000.0, 9.0), 2, 8),
    "rcm15": (_rcm15, (0.51, 0.6, 0.51), (70.49, 3.0, 42.49), 2, 8),
    "rcm16": (_rcm16, (0.01, 0.2), (0.05, 1.0), 2, 2),
    "rcm17": (_rcm17, (150.0, 20.0, 13.0, 10.0, 14.0, 0.63), (274.32, 32.31, 25.0, 11.71, 18.0, 0.75), 3, 9),
    "rcm18": (_rcm18, (136.0, 56.0, 1.4), (146.0, 68.0, 2.2), 2, 3),
    "rcm19": (
        _rcm19,
        (0.51, 0.51, 0.51, 250.0, 250.0, 250.0, 6.0, 4.0, 40.0, 10.0),
        (3.49, 3.49, 3.49, 2500.0, 2500.0, 2500.0, 20.0, 16.0, 700.0, 450.0),
        3,
        10,
    ),
    "rcm20": (_rcm20, (1.0, 1.0, 1e-6, 1.0), (16.0, 16.0, 16e-6, 16.0), 2, 7),
    "rcm21": (_rcm21, (1.3, 2.5, 1.3, 1.3, 1.3, 1.3), (1.7, 3.5, 1.7, 1.7, 1.7, 1.7), 2, 4),
}

__all__ = ["PROBLEMS"]
