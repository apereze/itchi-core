"""
Tests for ITCHI spatial masks.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from itchi.masks import (
    build_direct_mask,
    build_exterior_mask,
    build_indirect_mask,
    build_region_masks,
)


def test_direct_mask_numpy() -> None:
    """
    Test direct mask using NumPy arrays.
    """
    radius = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    r34 = 50.0

    mask = build_direct_mask(radius_km=radius, r34_km=r34)

    expected = np.array([True, True, True, False, False, False])

    np.testing.assert_array_equal(mask, expected)


def test_indirect_mask_numpy() -> None:
    """
    Test indirect mask using NumPy arrays.
    """
    radius = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    r34 = 50.0
    rocloud = 100.0

    mask = build_indirect_mask(
        radius_km=radius,
        r34_km=r34,
        rocloud_km=rocloud,
    )

    expected = np.array([False, False, False, True, True, False])

    np.testing.assert_array_equal(mask, expected)


def test_exterior_mask_numpy() -> None:
    """
    Test exterior mask using NumPy arrays.
    """
    radius = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    rocloud = 100.0

    mask = build_exterior_mask(radius_km=radius, rocloud_km=rocloud)

    expected = np.array([False, False, False, False, False, True])

    np.testing.assert_array_equal(mask, expected)


def test_region_masks_are_mutually_exclusive() -> None:
    """
    Test that direct, indirect and exterior masks do not overlap.
    """
    radius = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    r34 = 50.0
    rocloud = 100.0

    masks = build_region_masks(
        radius_km=radius,
        r34_km=r34,
        rocloud_km=rocloud,
    )

    direct = masks["direct"]
    indirect = masks["indirect"]
    exterior = masks["exterior"]

    assert not np.any(direct & indirect)
    assert not np.any(direct & exterior)
    assert not np.any(indirect & exterior)


def test_region_masks_cover_valid_domain() -> None:
    """
    Test that all valid radius points are assigned to exactly one region.
    """
    radius = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    r34 = 50.0
    rocloud = 100.0

    masks = build_region_masks(
        radius_km=radius,
        r34_km=r34,
        rocloud_km=rocloud,
    )

    total_mask = (
        masks["direct"].astype(int)
        + masks["indirect"].astype(int)
        + masks["exterior"].astype(int)
    )

    expected = np.ones_like(radius, dtype=int)

    np.testing.assert_array_equal(total_mask, expected)


def test_masks_preserve_xarray_dimensions() -> None:
    """
    Test that masks preserve xarray dimensions and coordinates.
    """
    radius = xr.DataArray(
        data=[[10.0, 60.0], [90.0, 130.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="radius_km",
    )

    masks = build_region_masks(
        radius_km=radius,
        r34_km=50.0,
        rocloud_km=100.0,
    )

    for mask in masks.values():
        assert isinstance(mask, xr.DataArray)
        assert mask.dims == radius.dims
        assert mask.shape == radius.shape
        assert mask.dtype == bool