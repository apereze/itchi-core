"""
Tests for cyclone-centered geometry utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itchi.geometry import (
    assign_quadrant_radius,
    build_geometry_fields,
    compute_radial_distance_km,
    compute_relative_quadrant,
    normalize_longitude_delta,
)


def test_normalize_longitude_delta() -> None:
    """
    Test longitude-difference normalization.
    """
    delta = np.array([-190.0, -180.0, -10.0, 0.0, 10.0, 180.0, 190.0])

    normalized = normalize_longitude_delta(delta)

    expected = np.array([170.0, -180.0, -10.0, 0.0, 10.0, -180.0, -170.0])

    np.testing.assert_allclose(normalized, expected)


def test_radial_distance_zero_at_center() -> None:
    """
    Test that distance is zero at the cyclone center.
    """
    lon = np.array([-100.0])
    lat = np.array([15.0])

    distance = compute_radial_distance_km(
        lon=lon,
        lat=lat,
        center_lon=-100.0,
        center_lat=15.0,
    )

    np.testing.assert_allclose(distance, np.array([0.0]), atol=1.0e-8)


def test_radial_distance_one_degree_latitude() -> None:
    """
    Test approximate distance for one degree of latitude.
    """
    lon = np.array([0.0])
    lat = np.array([1.0])

    distance = compute_radial_distance_km(
        lon=lon,
        lat=lat,
        center_lon=0.0,
        center_lat=0.0,
    )

    np.testing.assert_allclose(distance, np.array([111.2]), atol=0.5)


def test_radial_distance_across_dateline() -> None:
    """
    Test that distance across the dateline is handled correctly.
    """
    lon = np.array([-179.0])
    lat = np.array([0.0])

    distance = compute_radial_distance_km(
        lon=lon,
        lat=lat,
        center_lon=179.0,
        center_lat=0.0,
    )

    np.testing.assert_allclose(distance, np.array([222.4]), atol=1.0)


def test_relative_quadrant_numpy() -> None:
    """
    Test quadrant assignment using NumPy arrays.
    """
    lon = np.array([1.0, 1.0, -1.0, -1.0])
    lat = np.array([1.0, -1.0, -1.0, 1.0])

    quadrant = compute_relative_quadrant(
        lon=lon,
        lat=lat,
        center_lon=0.0,
        center_lat=0.0,
    )

    expected = np.array(["RNE", "RSE", "RSW", "RNW"])

    np.testing.assert_array_equal(quadrant, expected)


def test_assign_quadrant_radius_numpy() -> None:
    """
    Test assignment of quadrant-specific radii using NumPy arrays.
    """
    quadrant = np.array(["RNE", "RSE", "RSW", "RNW"])

    radii = {
        "RNE": 100.0,
        "RSE": 200.0,
        "RSW": 300.0,
        "RNW": 400.0,
    }

    radius = assign_quadrant_radius(
        quadrant=quadrant,
        radii_by_quadrant=radii,
    )

    expected = np.array([100.0, 200.0, 300.0, 400.0])

    np.testing.assert_allclose(radius, expected)


def test_assign_quadrant_radius_missing_key() -> None:
    """
    Test that missing quadrant radii raise an error.
    """
    quadrant = np.array(["RNE", "RSE"])

    radii = {
        "RNE": 100.0,
        "RSE": 200.0,
        "RSW": 300.0,
    }

    with pytest.raises(KeyError):
        assign_quadrant_radius(
            quadrant=quadrant,
            radii_by_quadrant=radii,
        )


def test_build_geometry_fields_xarray() -> None:
    """
    Test geometry-field construction with xarray inputs.
    """
    lon = xr.DataArray(
        data=[[-101.0, -99.0], [-101.0, -99.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [14.0, 16.0],
            "lon": [-101.0, -99.0],
        },
        name="lon",
    )

    lat = xr.DataArray(
        data=[[16.0, 16.0], [14.0, 14.0]],
        dims=("lat", "lon"),
        coords=lon.coords,
        name="lat",
    )

    geometry = build_geometry_fields(
        lon=lon,
        lat=lat,
        center_lon=-100.0,
        center_lat=15.0,
    )

    assert set(geometry) == {"radius_km", "quadrant"}

    assert isinstance(geometry["radius_km"], xr.DataArray)
    assert isinstance(geometry["quadrant"], xr.DataArray)

    assert geometry["radius_km"].dims == lon.dims
    assert geometry["quadrant"].dims == lon.dims

    assert geometry["radius_km"].shape == lon.shape
    assert geometry["quadrant"].shape == lon.shape


def test_assign_quadrant_radius_xarray() -> None:
    """
    Test quadrant-specific radius assignment with xarray inputs.
    """
    quadrant = xr.DataArray(
        data=[["RNW", "RNE"], ["RSW", "RSE"]],
        dims=("lat", "lon"),
        coords={
            "lat": [16.0, 14.0],
            "lon": [-101.0, -99.0],
        },
        name="quadrant",
    )

    radii = {
        "RNE": 100.0,
        "RSE": 200.0,
        "RSW": 300.0,
        "RNW": 400.0,
    }

    radius = assign_quadrant_radius(
        quadrant=quadrant,
        radii_by_quadrant=radii,
    )

    expected = xr.DataArray(
        data=[[400.0, 100.0], [300.0, 200.0]],
        dims=quadrant.dims,
        coords=quadrant.coords,
    )

    xr.testing.assert_allclose(radius, expected)