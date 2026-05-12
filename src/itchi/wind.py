"""
Wind-hazard utilities for ITCHI.

This module provides utilities to estimate and normalize wind hazard fields.

For ITCHI v0.1, wind hazard is represented as a normalized field V* in [0, 1].
The final activation of wind hazard inside the direct region is handled later
by components.py using the direct-region mask.

Main functions
--------------
compute_radial_wind_profile:
    Builds a simple parametric radial wind profile from Vmax and RMW.

normalize_wind_hazard:
    Converts wind speed in knots to normalized wind hazard V*.

compute_wind_hazard_from_profile:
    Convenience function that builds a radial wind profile and normalizes it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import DEFAULT_EPSILON, TROPICAL_STORM_WIND_KT

ArrayLike = np.ndarray | xr.DataArray


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def _clip_unit_interval(values: ArrayLike) -> ArrayLike:
    """
    Clip values to [0, 1] preserving NumPy or xarray type.
    """
    if _contains_xarray_object(values):
        return values.clip(min=0.0, max=1.0)

    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _validate_positive_scalar(value: float, name: str) -> None:
    """
    Validate that a scalar value is finite and positive.
    """
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive. Received: {value}")


def compute_radial_wind_profile(
    radius_km: ArrayLike,
    vmax_kt: float,
    rmw_km: float,
    inner_exponent: float = 1.0,
    outer_decay_exponent: float = 0.5,
    outer_radius_km: float | None = None,
    minimum_radius_km: float = DEFAULT_EPSILON,
) -> ArrayLike:
    """
    Compute a simple radial wind profile.

    The profile is intentionally simple and intended as a first-order
    approximation for ITCHI v0.1 development.

    For r <= RMW:

        V(r) = Vmax * (r / RMW)^inner_exponent

    For r > RMW:

        V(r) = Vmax * (RMW / r)^outer_decay_exponent

    If outer_radius_km is provided, wind is set to zero outside that radius.

    Parameters
    ----------
    radius_km : numpy.ndarray or xarray.DataArray
        Radial distance from cyclone center in kilometers.
    vmax_kt : float
        Maximum sustained wind in knots.
    rmw_km : float
        Radius of maximum wind in kilometers.
    inner_exponent : float, default=1.0
        Shape parameter inside RMW.
    outer_decay_exponent : float, default=0.5
        Decay parameter outside RMW.
    outer_radius_km : float or None, default=None
        Optional outer radius beyond which wind is set to zero.
    minimum_radius_km : float, default=1e-6
        Minimum radius used for numerical stability.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Estimated wind speed in knots.
    """
    _validate_positive_scalar(vmax_kt, "vmax_kt")
    _validate_positive_scalar(rmw_km, "rmw_km")
    _validate_positive_scalar(inner_exponent, "inner_exponent")
    _validate_positive_scalar(outer_decay_exponent, "outer_decay_exponent")

    if outer_radius_km is not None:
        _validate_positive_scalar(outer_radius_km, "outer_radius_km")

    if _contains_xarray_object(radius_km):
        radius = radius_km.clip(min=minimum_radius_km)
        inside_rmw = radius <= rmw_km

        inner_wind = vmax_kt * (radius / rmw_km) ** inner_exponent
        outer_wind = vmax_kt * (rmw_km / radius) ** outer_decay_exponent

        wind = xr.where(inside_rmw, inner_wind, outer_wind)

        if outer_radius_km is not None:
            wind = xr.where(radius_km <= outer_radius_km, wind, 0.0)

        return wind.clip(min=0.0, max=vmax_kt)

    radius = np.maximum(np.asarray(radius_km, dtype=float), minimum_radius_km)
    inside_rmw = radius <= rmw_km

    inner_wind = vmax_kt * (radius / rmw_km) ** inner_exponent
    outer_wind = vmax_kt * (rmw_km / radius) ** outer_decay_exponent

    wind = np.where(inside_rmw, inner_wind, outer_wind)

    if outer_radius_km is not None:
        wind = np.where(
            np.asarray(radius_km, dtype=float) <= outer_radius_km, wind, 0.0
        )

    return np.clip(wind, 0.0, vmax_kt)


def normalize_wind_hazard(
    wind_kt: ArrayLike,
    vmax_kt: float,
    reference_wind_kt: float = TROPICAL_STORM_WIND_KT,
    below_threshold_mode: str = "relative_to_vmax",
    epsilon: float = DEFAULT_EPSILON,
) -> ArrayLike:
    """
    Normalize wind speed into wind hazard V*.

    For systems with Vmax > reference_wind_kt, the default normalization is:

        V* = clip((V - reference_wind_kt) / (Vmax - reference_wind_kt), 0, 1)

    For systems with Vmax <= reference_wind_kt, such as tropical depressions,
    the standard tropical-storm-force normalization cannot be applied because
    the denominator is zero or negative. In that case, below_threshold_mode
    controls the behavior:

    - "relative_to_vmax":
        V* = clip(V / Vmax, 0, 1)

    - "zero":
        V* = 0 everywhere

    Parameters
    ----------
    wind_kt : numpy.ndarray or xarray.DataArray
        Wind speed field in knots.
    vmax_kt : float
        Maximum sustained wind in knots.
    reference_wind_kt : float, default=34.0
        Reference wind threshold in knots.
    below_threshold_mode : {"relative_to_vmax", "zero"}, default="relative_to_vmax"
        Normalization rule when Vmax <= reference_wind_kt.
    epsilon : float, default=1e-6
        Small value used for numerical stability.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Normalized wind hazard V* in [0, 1].
    """
    _validate_positive_scalar(vmax_kt, "vmax_kt")
    _validate_positive_scalar(reference_wind_kt, "reference_wind_kt")

    mode = below_threshold_mode.lower().strip()

    if float(vmax_kt) > float(reference_wind_kt):
        denominator = max(float(vmax_kt) - float(reference_wind_kt), epsilon)
        hazard = (wind_kt - reference_wind_kt) / denominator
        return _clip_unit_interval(hazard)

    if mode == "relative_to_vmax":
        denominator = max(float(vmax_kt), epsilon)
        hazard = wind_kt / denominator
        return _clip_unit_interval(hazard)

    if mode == "zero":
        if _contains_xarray_object(wind_kt):
            return xr.zeros_like(wind_kt, dtype=float)

        return np.zeros_like(np.asarray(wind_kt, dtype=float), dtype=float)

    raise ValueError(
        "Unsupported below_threshold_mode. " "Expected 'relative_to_vmax' or 'zero'."
    )


def apply_wind_mask(
    wind_hazard_normalized: ArrayLike,
    mask: np.ndarray | xr.DataArray,
    fill_value: float = 0.0,
) -> ArrayLike:
    """
    Apply a Boolean mask to normalized wind hazard.

    Parameters
    ----------
    wind_hazard_normalized : numpy.ndarray or xarray.DataArray
        Normalized wind hazard V*.
    mask : numpy.ndarray or xarray.DataArray
        Boolean mask where True indicates active cells.
    fill_value : float, default=0.0
        Value assigned outside the active mask.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Masked wind hazard.
    """
    wind_hazard_normalized = _clip_unit_interval(wind_hazard_normalized)

    if _contains_xarray_object(wind_hazard_normalized, mask):
        return xr.where(mask, wind_hazard_normalized, fill_value)

    return np.where(
        np.asarray(mask, dtype=bool),
        np.asarray(wind_hazard_normalized, dtype=float),
        fill_value,
    )


def compute_wind_hazard_from_profile(
    radius_km: ArrayLike,
    vmax_kt: float,
    rmw_km: float,
    reference_wind_kt: float = TROPICAL_STORM_WIND_KT,
    below_threshold_mode: str = "relative_to_vmax",
    inner_exponent: float = 1.0,
    outer_decay_exponent: float = 0.5,
    outer_radius_km: float | None = None,
    active_mask: np.ndarray | xr.DataArray | None = None,
) -> ArrayLike:
    """
    Compute normalized wind hazard from a radial wind profile.

    Parameters
    ----------
    radius_km : numpy.ndarray or xarray.DataArray
        Radial distance from cyclone center in kilometers.
    vmax_kt : float
        Maximum sustained wind in knots.
    rmw_km : float
        Radius of maximum wind in kilometers.
    reference_wind_kt : float, default=34.0
        Reference wind threshold in knots.
    below_threshold_mode : {"relative_to_vmax", "zero"}, default="relative_to_vmax"
        Normalization rule when Vmax <= reference_wind_kt.
    inner_exponent : float, default=1.0
        Shape parameter inside RMW.
    outer_decay_exponent : float, default=0.5
        Decay parameter outside RMW.
    outer_radius_km : float or None, default=None
        Optional outer radius beyond which wind is set to zero.
    active_mask : numpy.ndarray, xarray.DataArray or None, default=None
        Optional mask used to deactivate wind hazard outside a target region.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Normalized wind hazard V* in [0, 1].
    """
    wind_kt = compute_radial_wind_profile(
        radius_km=radius_km,
        vmax_kt=vmax_kt,
        rmw_km=rmw_km,
        inner_exponent=inner_exponent,
        outer_decay_exponent=outer_decay_exponent,
        outer_radius_km=outer_radius_km,
    )

    wind_hazard = normalize_wind_hazard(
        wind_kt=wind_kt,
        vmax_kt=vmax_kt,
        reference_wind_kt=reference_wind_kt,
        below_threshold_mode=below_threshold_mode,
    )

    if active_mask is not None:
        return apply_wind_mask(
            wind_hazard_normalized=wind_hazard,
            mask=active_mask,
            fill_value=0.0,
        )

    return wind_hazard
