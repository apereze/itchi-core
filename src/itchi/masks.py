"""
Spatial masks for ITCHI.

This module defines the direct, indirect and exterior regions used by
ITCHI v0.1.

Definitions
-----------
Direct region:
    r <= R34

Indirect region:
    R34 < r <= ROCLOUD

Exterior region:
    r > ROCLOUD

where r is the radial distance from the tropical cyclone center.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def build_direct_mask(
    radius_km: np.ndarray | xr.DataArray,
    r34_km: float | np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Build the direct-region mask.

    The direct region is defined as:

        r <= R34

    Parameters
    ----------
    radius_km : numpy.ndarray or xarray.DataArray
        Radial distance from the cyclone center in kilometers.
    r34_km : float, numpy.ndarray or xarray.DataArray
        Tropical-storm-force wind radius in kilometers.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Boolean mask for the direct region.
    """
    mask = radius_km <= r34_km

    if _contains_xarray_object(mask):
        return mask.fillna(False).astype(bool)

    return np.asarray(mask, dtype=bool)


def build_indirect_mask(
    radius_km: np.ndarray | xr.DataArray,
    r34_km: float | np.ndarray | xr.DataArray,
    rocloud_km: float | np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Build the indirect-region mask.

    The indirect region is defined as:

        R34 < r <= ROCLOUD

    Parameters
    ----------
    radius_km : numpy.ndarray or xarray.DataArray
        Radial distance from the cyclone center in kilometers.
    r34_km : float, numpy.ndarray or xarray.DataArray
        Tropical-storm-force wind radius in kilometers.
    rocloud_km : float, numpy.ndarray or xarray.DataArray
        External cyclone-attribution radius in kilometers.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Boolean mask for the indirect region.
    """
    mask = (radius_km > r34_km) & (radius_km <= rocloud_km)

    if _contains_xarray_object(mask):
        return mask.fillna(False).astype(bool)

    return np.asarray(mask, dtype=bool)


def build_exterior_mask(
    radius_km: np.ndarray | xr.DataArray,
    rocloud_km: float | np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Build the exterior-region mask.

    The exterior region is defined as:

        r > ROCLOUD

    Parameters
    ----------
    radius_km : numpy.ndarray or xarray.DataArray
        Radial distance from the cyclone center in kilometers.
    rocloud_km : float, numpy.ndarray or xarray.DataArray
        External cyclone-attribution radius in kilometers.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Boolean mask for the exterior region.
    """
    mask = radius_km > rocloud_km

    if _contains_xarray_object(mask):
        return mask.fillna(False).astype(bool)

    return np.asarray(mask, dtype=bool)


def build_region_masks(
    radius_km: np.ndarray | xr.DataArray,
    r34_km: float | np.ndarray | xr.DataArray,
    rocloud_km: float | np.ndarray | xr.DataArray,
) -> dict[str, np.ndarray | xr.DataArray]:
    """
    Build all ITCHI spatial masks.

    Parameters
    ----------
    radius_km : numpy.ndarray or xarray.DataArray
        Radial distance from the cyclone center in kilometers.
    r34_km : float, numpy.ndarray or xarray.DataArray
        Tropical-storm-force wind radius in kilometers.
    rocloud_km : float, numpy.ndarray or xarray.DataArray
        External cyclone-attribution radius in kilometers.

    Returns
    -------
    dict[str, numpy.ndarray or xarray.DataArray]
        Dictionary with direct, indirect and exterior masks.
    """
    direct = build_direct_mask(radius_km=radius_km, r34_km=r34_km)

    indirect = build_indirect_mask(
        radius_km=radius_km,
        r34_km=r34_km,
        rocloud_km=rocloud_km,
    )

    exterior = build_exterior_mask(
        radius_km=radius_km,
        rocloud_km=rocloud_km,
    )

    return {
        "direct": direct,
        "indirect": indirect,
        "exterior": exterior,
    }