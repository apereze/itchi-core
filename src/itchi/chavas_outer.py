"""
Outer tropical cyclone wind profile of Emanuel (2004) as used by Chavas et
al. (2015) (hereafter E04/C15), constrained by the observed R34.

The non-convecting outer region satisfies

    dM/dr = (2 C_d / w_cool) (r V)^2 / (r0^2 - r^2),   M = r V + f r^2 / 2,

with V(r0) = 0. Given R34 (V = 34 kt at r = R34), r0 is found by root finding
and any outer radius R(V) follows from the same profile.

This is an independent numerical implementation for cross-checking
``tcwindprofile`` (Chavas), which uses an analytic approximation of the same
outer model. Units: km, m s-1, degrees.

References
----------
Emanuel, K. (2004). Tropical cyclone energetics and structure. In Atmospheric
    Turbulence and Mesoscale Meteorology, Cambridge Univ. Press, 165-192.
    https://doi.org/10.1017/CBO9780511735035.010
Chavas, D. R., Lin, N., and Emanuel, K. (2015). J. Atmos. Sci., 72,
    3647-3662. https://doi.org/10.1175/JAS-D-15-0014.1
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

OMEGA: float = 7.292e-5
CD_DEFAULT: float = 1.5e-3
W_COOL_DEFAULT: float = 2.0e-3  # m s-1
V34_MS: float = 34.0 * 0.514444


def coriolis(lat: float) -> float:
    return 2.0 * OMEGA * np.sin(np.deg2rad(abs(lat)))


def e04_outer_wind(
    radius_km: np.ndarray,
    r0_km: float,
    lat: float,
    cd: float = CD_DEFAULT,
    w_cool: float = W_COOL_DEFAULT,
    eps: float = 1.0e-5,
) -> np.ndarray:
    """
    E04 outer-model wind speed (m s-1) at radius_km (< r0_km) for a given r0.
    """
    f = coriolis(lat)
    r0 = r0_km * 1.0e3
    k = 2.0 * cd / w_cool

    def rhs(r, m):
        rv = m - 0.5 * f * r**2
        return k * rv**2 / (r0**2 - r**2)

    radius = np.atleast_1d(np.asarray(radius_km, dtype=float)) * 1.0e3
    r_start = r0 * (1.0 - eps)
    wind = np.zeros_like(radius)
    inside = np.isfinite(radius) & (radius < r_start) & (radius > 0)

    if inside.any():
        sol = solve_ivp(
            rhs,
            (r_start, float(radius[inside].min())),
            [0.5 * f * r0**2],
            method="DOP853",
            dense_output=True,
            rtol=1e-10,
            atol=1e-6,
        )
        r_in = radius[inside]
        wind[inside] = (sol.sol(r_in)[0] - 0.5 * f * r_in**2) / r_in

    return wind  # V = 0 for r >= r0 (1 - eps)


def e04_r0_from_r34(
    r34_km: float,
    lat: float,
    v34_ms: float = V34_MS,
    cd: float = CD_DEFAULT,
    w_cool: float = W_COOL_DEFAULT,
    r0_max_km: float = 8000.0,
) -> float:
    """
    Outer radius r0 (km) such that the E04 profile gives V(R34) = v34_ms.

    Returns NaN for missing/non-positive R34 or when no root is bracketed.
    """
    if not (np.isfinite(r34_km) and r34_km > 0 and np.isfinite(lat)):
        return np.nan

    def residual(r0_km: float) -> float:
        wind = e04_outer_wind(np.array([r34_km]), r0_km, lat, cd, w_cool)
        return float(wind[0]) - v34_ms

    lo, hi = r34_km * 1.001, r0_max_km
    if residual(lo) >= 0 or residual(hi) <= 0:
        return np.nan

    return brentq(residual, lo, hi, xtol=1e-3)


def e04_radius_of_wind(
    r34_km: float,
    lat: float,
    wind_ms: float,
    cd: float = CD_DEFAULT,
    w_cool: float = W_COOL_DEFAULT,
) -> tuple[float, float]:
    """
    Return (r0_km, R(wind_ms)_km) from observed R34 with the E04 outer model.
    """
    r0 = e04_r0_from_r34(r34_km, lat, cd=cd, w_cool=w_cool)
    if not np.isfinite(r0) or wind_ms >= V34_MS:
        return r0, np.nan

    def residual(r_km: float) -> float:
        return float(e04_outer_wind(np.array([r_km]), r0, lat, cd, w_cool)[0]) - wind_ms

    return r0, brentq(residual, r34_km, r0 * (1 - 1e-5), xtol=1e-3)
