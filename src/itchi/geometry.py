"""
Cyclone-centered geometry utilities for ITCHI.

This module computes:

- radial distance from each grid cell to the tropical cyclone center;
- relative quadrant of each grid cell with respect to the cyclone center;
- quadrant-specific radius fields, such as R34_q and ROCLOUD_q.

The functions are designed to work with NumPy arrays and xarray DataArrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import QUADRANTS

EARTH_RADIUS_KM: float = 6371.0088


ArrayLike = np.ndarray | xr.DataArray


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def normalize_longitude_delta(
    delta_lon_deg: float | ArrayLike,
) -> float | ArrayLike:
    """
    Normalize longitude differences to the interval [-180, 180).

    This avoids problems when a storm or grid crosses the dateline.

    Parameters
    ----------
    delta_lon_deg : float, numpy.ndarray or xarray.DataArray
        Longitude difference in degrees.

    Returns
    -------
    float, numpy.ndarray or xarray.DataArray
        Normalized longitude difference in degrees.
    """
    return ((delta_lon_deg + 180.0) % 360.0) - 180.0


def _clip_unit_interval(values: ArrayLike) -> ArrayLike:
    """
    Clip values to [0, 1] preserving NumPy or xarray type.
    """
    if _contains_xarray_object(values):
        return values.clip(min=0.0, max=1.0)

    return np.clip(values, 0.0, 1.0)


def compute_radial_distance_km(
    lon: ArrayLike,
    lat: ArrayLike,
    center_lon: float,
    center_lat: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> ArrayLike:
    """
    Compute great-circle radial distance from grid cells to cyclone center.

    The calculation uses the haversine formula.

    Parameters
    ----------
    lon : numpy.ndarray or xarray.DataArray
        Longitude field in degrees.
    lat : numpy.ndarray or xarray.DataArray
        Latitude field in degrees.
    center_lon : float
        Cyclone center longitude in degrees.
    center_lat : float
        Cyclone center latitude in degrees.
    earth_radius_km : float, default=6371.0088
        Mean Earth radius in kilometers.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Radial distance from the cyclone center in kilometers.
    """
    delta_lon = normalize_longitude_delta(lon - center_lon)
    delta_lat = lat - center_lat

    lon_distance_rad = np.deg2rad(delta_lon)
    lat_distance_rad = np.deg2rad(delta_lat)

    lat_rad = np.deg2rad(lat)
    center_lat_rad = np.deg2rad(center_lat)

    haversine_argument = (
        np.sin(lat_distance_rad / 2.0) ** 2
        + np.cos(center_lat_rad) * np.cos(lat_rad) * np.sin(lon_distance_rad / 2.0) ** 2
    )

    haversine_argument = _clip_unit_interval(haversine_argument)

    angular_distance = 2.0 * np.arcsin(np.sqrt(haversine_argument))

    return earth_radius_km * angular_distance


def compute_relative_quadrant(
    lon: ArrayLike,
    lat: ArrayLike,
    center_lon: float,
    center_lat: float,
) -> ArrayLike:
    """
    Compute the relative quadrant of each grid cell.

    Quadrant convention:

    - NE: north and east of the cyclone center
    - SE: south and east of the cyclone center
    - SW: south and west of the cyclone center
    - NW: north and west of the cyclone center

    Boundary convention:

    - cells exactly on the center latitude are considered northern;
    - cells exactly on the center longitude are considered eastern.

    Parameters
    ----------
    lon : numpy.ndarray or xarray.DataArray
        Longitude field in degrees.
    lat : numpy.ndarray or xarray.DataArray
        Latitude field in degrees.
    center_lon : float
        Cyclone center longitude in degrees.
    center_lat : float
        Cyclone center latitude in degrees.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Quadrant labels: RNE, RSE, RSW or RNW.
    """
    delta_lon = normalize_longitude_delta(lon - center_lon)

    is_east = delta_lon >= 0.0
    is_north = lat >= center_lat

    if _contains_xarray_object(lon, lat):
        return xr.where(
            is_north & is_east,
            "RNE",
            xr.where(
                (~is_north) & is_east,
                "RSE",
                xr.where(
                    (~is_north) & (~is_east),
                    "RSW",
                    "RNW",
                ),
            ),
        )

    return np.where(
        is_north & is_east,
        "RNE",
        np.where(
            (~is_north) & is_east,
            "RSE",
            np.where(
                (~is_north) & (~is_east),
                "RSW",
                "RNW",
            ),
        ),
    )


def _validate_quadrant_radii(
    radii_by_quadrant: dict[str, float],
) -> dict[str, float]:
    """
    Validate and standardize a dictionary of quadrant-specific radii.

    Parameters
    ----------
    radii_by_quadrant : dict[str, float]
        Dictionary with keys RNE, RSE, RSW and RNW.

    Returns
    -------
    dict[str, float]
        Standardized dictionary with uppercase quadrant keys.

    Raises
    ------
    KeyError
        If any required quadrant is missing.
    """
    standardized = {
        str(key).upper(): float(value) for key, value in radii_by_quadrant.items()
    }

    required = set(QUADRANTS)
    missing = required.difference(standardized)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing quadrant radius/radii: {missing_text}")

    return standardized


def assign_quadrant_radius(
    quadrant: ArrayLike,
    radii_by_quadrant: dict[str, float],
) -> ArrayLike:
    """
    Assign a radius value to each cell based on its quadrant.

    This is used to construct fields such as R34_q or ROCLOUD_q.

    Parameters
    ----------
    quadrant : numpy.ndarray or xarray.DataArray
        Quadrant labels for each grid cell.
    radii_by_quadrant : dict[str, float]
        Dictionary with quadrant radii. Required keys are:
        RNE, RSE, RSW and RNW.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Radius field with the same shape as quadrant.
    """
    radii = _validate_quadrant_radii(radii_by_quadrant)

    if _contains_xarray_object(quadrant):
        radius = xr.full_like(quadrant, fill_value=np.nan, dtype=float)

        for quad in QUADRANTS:
            radius = xr.where(quadrant == quad, radii[quad], radius)

        return radius

    quadrant_array = np.asarray(quadrant).astype(str)
    radius = np.full(quadrant_array.shape, np.nan, dtype=float)

    for quad in QUADRANTS:
        radius = np.where(quadrant_array == quad, radii[quad], radius)

    return radius


def build_geometry_fields(
    lon: ArrayLike,
    lat: ArrayLike,
    center_lon: float,
    center_lat: float,
) -> dict[str, ArrayLike]:
    """
    Build the basic cyclone-centered geometry fields.

    Parameters
    ----------
    lon : numpy.ndarray or xarray.DataArray
        Longitude field in degrees.
    lat : numpy.ndarray or xarray.DataArray
        Latitude field in degrees.
    center_lon : float
        Cyclone center longitude in degrees.
    center_lat : float
        Cyclone center latitude in degrees.

    Returns
    -------
    dict[str, numpy.ndarray or xarray.DataArray]
        Dictionary with:

        - radius_km
        - quadrant
    """
    radius_km = compute_radial_distance_km(
        lon=lon,
        lat=lat,
        center_lon=center_lon,
        center_lat=center_lat,
    )

    quadrant = compute_relative_quadrant(
        lon=lon,
        lat=lat,
        center_lon=center_lon,
        center_lat=center_lat,
    )

    return {
        "radius_km": radius_km,
        "quadrant": quadrant,
    }
