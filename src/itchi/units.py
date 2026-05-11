"""
Unit-conversion utilities for ITCHI.

This module provides explicit conversions used by the ITCHI workflow.

The main use case is converting IBTrACS wind radii from nautical miles
to kilometers, because ITCHI geometry functions compute radial distance
in kilometers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import QUADRANTS

NAUTICAL_MILE_TO_KM: float = 1.852


ArrayLike = float | np.ndarray | xr.DataArray


_QUADRANT_ALIASES: dict[str, str] = {
    "NE": "RNE",
    "SE": "RSE",
    "SW": "RSW",
    "NW": "RNW",
    "RNE": "RNE",
    "RSE": "RSE",
    "RSW": "RSW",
    "RNW": "RNW",
}


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def normalize_quadrant_key(key: str) -> str:
    """
    Normalize quadrant labels to the ITCHI convention.

    ITCHI internal convention:

    - RNE
    - RSE
    - RSW
    - RNW

    The function also accepts shorter aliases:

    - NE -> RNE
    - SE -> RSE
    - SW -> RSW
    - NW -> RNW

    Parameters
    ----------
    key : str
        Quadrant label.

    Returns
    -------
    str
        Normalized ITCHI quadrant label.

    Raises
    ------
    KeyError
        If the quadrant label is not recognized.
    """
    key_upper = str(key).upper().strip()

    if key_upper not in _QUADRANT_ALIASES:
        raise KeyError(f"Unknown quadrant label: {key}")

    return _QUADRANT_ALIASES[key_upper]


def nautical_miles_to_km(values: ArrayLike) -> ArrayLike:
    """
    Convert nautical miles to kilometers.

    Parameters
    ----------
    values : float, numpy.ndarray or xarray.DataArray
        Value or values in nautical miles.

    Returns
    -------
    float, numpy.ndarray or xarray.DataArray
        Value or values in kilometers.
    """
    return values * NAUTICAL_MILE_TO_KM


def km_to_nautical_miles(values: ArrayLike) -> ArrayLike:
    """
    Convert kilometers to nautical miles.

    Parameters
    ----------
    values : float, numpy.ndarray or xarray.DataArray
        Value or values in kilometers.

    Returns
    -------
    float, numpy.ndarray or xarray.DataArray
        Value or values in nautical miles.
    """
    return values / NAUTICAL_MILE_TO_KM


def convert_radius_to_km(
    radius: ArrayLike,
    input_unit: str,
) -> ArrayLike:
    """
    Convert a radius field to kilometers.

    Accepted input units:

    - "km"
    - "kilometer"
    - "kilometers"
    - "nm"
    - "nmi"
    - "nautical_mile"
    - "nautical_miles"

    Parameters
    ----------
    radius : float, numpy.ndarray or xarray.DataArray
        Radius value or field.
    input_unit : str
        Unit of the input radius.

    Returns
    -------
    float, numpy.ndarray or xarray.DataArray
        Radius converted to kilometers.

    Raises
    ------
    ValueError
        If the input unit is not recognized.
    """
    unit = input_unit.lower().strip()

    km_units = {"km", "kilometer", "kilometers", "kilometre", "kilometres"}
    nautical_mile_units = {
        "nm",
        "nmi",
        "nautical_mile",
        "nautical_miles",
        "nautical mile",
        "nautical miles",
    }

    if unit in km_units:
        return radius

    if unit in nautical_mile_units:
        return nautical_miles_to_km(radius)

    raise ValueError(f"Unsupported radius unit: {input_unit}")


def _clean_radius_value(value: float | int | None) -> float:
    """
    Clean a scalar radius value.

    Missing or invalid radii are converted to NaN.

    This is useful for best-track datasets, where missing radii can appear
    as negative values or special codes.
    """
    if value is None:
        return np.nan

    value_float = float(value)

    if not np.isfinite(value_float):
        return np.nan

    if value_float < 0:
        return np.nan

    return value_float


def convert_quadrant_radii_to_km(
    radii_by_quadrant: Mapping[str, float | int | None],
    input_unit: str,
) -> dict[str, float]:
    """
    Convert quadrant-specific radii to kilometers.

    Parameters
    ----------
    radii_by_quadrant : Mapping[str, float, int or None]
        Dictionary-like object with quadrant radii.
        Accepted keys are RNE, RSE, RSW, RNW or NE, SE, SW, NW.
    input_unit : str
        Unit of the input radii.

    Returns
    -------
    dict[str, float]
        Dictionary using ITCHI quadrant labels and radius values in kilometers.

    Raises
    ------
    KeyError
        If any required quadrant is missing.
    ValueError
        If the input unit is not recognized.
    """
    standardized: dict[str, float] = {}

    for key, value in radii_by_quadrant.items():
        quadrant = normalize_quadrant_key(key)
        standardized[quadrant] = _clean_radius_value(value)

    required = set(QUADRANTS)
    missing = required.difference(standardized)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing quadrant radius/radii: {missing_text}")

    return {
        quadrant: float(convert_radius_to_km(standardized[quadrant], input_unit))
        for quadrant in QUADRANTS
    }
