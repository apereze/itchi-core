"""
Parametric radial wind profiles for ITCHI.

This module implements the sectionally continuous profile of Willoughby et
al. (2006, hereafter W06), used by Pérez-Alarcón et al. (2021, hereafter PA21)
to estimate critical wind radii and the outer tropical cyclone size from
best-track position and intensity only.

Conventions
-----------
- Radii in kilometers, wind speeds in m s-1, latitude in degrees.
- All functions are vectorized with NumPy broadcasting: parameters may be
  scalars (one track record) or 1-D arrays (many records).

References
----------
Willoughby, H. E., Darling, R. W. R., and Rahn, M. E. (2006).
    Mon. Wea. Rev., 134, 1102-1120. https://doi.org/10.1175/MWR3106.1
Pérez-Alarcón, A., et al. (2021). Weather Clim. Extremes, 33, 100366.
    https://doi.org/10.1016/j.wace.2021.100366
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

KNOT_TO_MS: float = 0.514444

V_OUTER_PA21_MS: float = 2.0  # PA21 outer-size threshold (~3.9 kt)
V_R5_MS: float = 5.0 * KNOT_TO_MS  # 5 kt, Knaff et al. (2014) R5
V_R34_MS: float = 34.0 * KNOT_TO_MS  # 34 kt

ArrayLike = float | np.ndarray


def bisect_decreasing(
    func: Callable[[np.ndarray], np.ndarray],
    lower: ArrayLike,
    upper: ArrayLike,
    n_iter: int = 60,
) -> np.ndarray:
    """
    Vectorized bisection assuming func(lower) > 0 >= func(upper) elementwise.
    """
    lo = np.array(lower, dtype=float, copy=True)
    hi = np.array(upper, dtype=float, copy=True)
    lo, hi = np.broadcast_arrays(lo, hi)
    lo, hi = lo.copy(), hi.copy()

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        positive = func(mid) > 0
        lo = np.where(positive, mid, lo)
        hi = np.where(positive, hi, mid)

    return 0.5 * (lo + hi)


def w06_rmax_km(vmax_ms: ArrayLike, lat: ArrayLike, variant: str = "w06") -> np.ndarray:
    """
    Empirical radius of maximum wind in km.

    variant="w06"  : original W06 regression.
    variant="pa21" : coefficients as printed in PA21, Eq. (6).
    """
    phi = np.abs(np.asarray(lat, dtype=float))
    vmax = np.asarray(vmax_ms, dtype=float)

    if variant == "w06":
        return 46.4 * np.exp(-0.0155 * vmax + 0.0169 * phi)

    if variant == "pa21":
        return 46.6 * np.exp(-0.015 * vmax + 0.0169 * phi)

    raise ValueError(f"Unsupported Rmax variant: {variant}")


def w06_shape_parameters(
    vmax_ms: ArrayLike,
    lat: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return W06 regression shape parameters n, X1 (km) and A (dual exponential).
    """
    phi = np.abs(np.asarray(lat, dtype=float))
    vmax = np.asarray(vmax_ms, dtype=float)

    n = 0.4067 + 0.0144 * vmax - 0.0038 * phi
    x1 = 317.1 - 2.026 * vmax + 1.915 * phi
    a = np.maximum(0.0696 + 0.0049 * vmax - 0.0064 * phi, 0.0)  # A < 0 -> 0

    return n, x1, a


def w06_ramp(xi: ArrayLike) -> np.ndarray:
    """
    W06 polynomial ramp, C4-continuous with w(0) = 0 and w(1) = 1.
    """
    x = np.clip(np.asarray(xi, dtype=float), 0.0, 1.0)
    return x**5 * (126.0 - 420.0 * x + 540.0 * x**2 - 315.0 * x**3 + 70.0 * x**4)


class Willoughby2006Profile:
    """
    W06 dual-exponential profile with polynomial transition zone.

    Parameters
    ----------
    vmax_ms, lat : float or numpy.ndarray
        Maximum wind (m s-1) and latitude (degrees).
    rmax_km : float, numpy.ndarray or None
        Observed Rmax in km. NaN or non-positive values are filled with the
        empirical regression.
    x1_km : float, numpy.ndarray or None
        Outer decay length in km. None uses the W06 regression.
    x2_km : float
        Second decay length in km (fixed, W06).
    rmax_variant : {"w06", "pa21"}
        Empirical Rmax regression used for missing values.
    transition : "w06" or float
        Transition width R2 - R1 in km. "w06" uses 25 km if Rmax > 20 km and
        15 km otherwise.
    w_condition : {"exact", "linear"}
        "exact" imposes dV/dr = 0 at Rmax for the dual exponential.
        "linear" uses X_eff = (1 - A) X1 + A X2, as in some implementations.
    """

    def __init__(
        self,
        vmax_ms: ArrayLike,
        lat: ArrayLike,
        rmax_km: ArrayLike | None = None,
        x1_km: ArrayLike | None = None,
        x2_km: float = 25.0,
        rmax_variant: str = "w06",
        transition: str | float = "w06",
        w_condition: str = "exact",
    ) -> None:
        self.vmax = np.asarray(vmax_ms, dtype=float)
        self.lat = np.asarray(lat, dtype=float)

        rmax_empirical = w06_rmax_km(self.vmax, self.lat, rmax_variant)

        if rmax_km is None:
            self.rmax = rmax_empirical
        else:
            rmax = np.asarray(rmax_km, dtype=float)
            valid = np.isfinite(rmax) & (rmax > 0.0)
            self.rmax = np.where(valid, rmax, rmax_empirical)

        self.n, x1_regression, self.a = w06_shape_parameters(self.vmax, self.lat)
        self.x1 = x1_regression if x1_km is None else np.asarray(x1_km, dtype=float)
        self.x2 = float(x2_km)

        if transition == "w06":
            self.width = np.where(self.rmax > 20.0, 25.0, 15.0)
        else:
            self.width = np.full_like(self.rmax, float(transition))

        if w_condition == "exact":
            decay = (1.0 - self.a) / self.x1 + self.a / self.x2
            w_at_rmax = self.n / (self.n + self.rmax * decay)
        elif w_condition == "linear":
            x_eff = (1.0 - self.a) * self.x1 + self.a * self.x2
            w_at_rmax = self.n * x_eff / (self.n * x_eff + self.rmax)
        else:
            raise ValueError(f"Unsupported w_condition: {w_condition}")

        xi_at_rmax = bisect_decreasing(
            lambda x: w_at_rmax - w06_ramp(x),
            np.zeros_like(w_at_rmax),
            np.ones_like(w_at_rmax),
            n_iter=50,
        )

        self.r1 = np.maximum(self.rmax - xi_at_rmax * self.width, 0.0)
        self.r2 = self.r1 + self.width

    def __call__(self, radius_km: ArrayLike) -> np.ndarray:
        """
        Evaluate tangential wind speed (m s-1) at radius_km.
        """
        r = np.asarray(radius_km, dtype=float)

        inner = self.vmax * (r / self.rmax) ** self.n
        dr = r - self.rmax
        outer = self.vmax * (
            (1.0 - self.a) * np.exp(-dr / self.x1) + self.a * np.exp(-dr / self.x2)
        )
        weight = w06_ramp((r - self.r1) / self.width)

        return inner * (1.0 - weight) + outer * weight


def radius_of_wind_km(
    profile: Willoughby2006Profile,
    threshold_ms: ArrayLike,
    search_limit_km: float = 5000.0,
    n_iter: int = 60,
) -> np.ndarray:
    """
    Outer radius (km) where the profile equals threshold_ms.

    Returns NaN where vmax <= threshold (radius undefined) or where the wind at
    search_limit_km still exceeds the threshold.
    """
    threshold = np.asarray(threshold_ms, dtype=float) + np.zeros_like(profile.vmax)
    upper = np.full_like(profile.rmax, search_limit_km)

    with np.errstate(invalid="ignore"):
        radius = bisect_decreasing(
            lambda r: profile(r) - threshold,
            profile.rmax,
            upper,
            n_iter=n_iter,
        )
        valid = (profile.vmax > threshold) & (profile(upper) <= threshold)

    return np.where(valid, radius, np.nan)
