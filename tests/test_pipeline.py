"""
Tests for the high-level ITCHI snapshot pipeline.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from itchi.pipeline import compute_itchi_snapshot, compute_itchi_snapshot_from_grid


def test_compute_itchi_snapshot_returns_expected_keys() -> None:
    """
    Test that the snapshot pipeline returns all expected outputs.
    """
    precipitation = np.array([5.0, 15.0, 25.0, 25.0, 30.0, 40.0])
    radius_km = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    wind_hazard = np.array([1.0, 0.5, 0.3, 0.2, 0.0, 0.0])

    result = compute_itchi_snapshot(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        radius_km=radius_km,
        r34_km=50.0,
        rocloud_km=100.0,
        wind_hazard_normalized=wind_hazard,
    )

    expected_keys = {
        "M_direct",
        "M_indirect",
        "M_exterior",
        "H_P",
        "H_Pdir",
        "H_Pind",
        "H_W",
        "H_dir",
        "H_ind",
        "ITCHI",
    }

    assert set(result) == expected_keys


def test_compute_itchi_snapshot_expected_numpy_values() -> None:
    """
    Test the complete ITCHI calculation with synthetic NumPy inputs.
    """
    precipitation = np.array([5.0, 15.0, 25.0, 25.0, 30.0, 40.0])
    radius_km = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
    wind_hazard = np.array([1.0, 0.5, 0.3, 0.2, 0.0, 0.0])

    result = compute_itchi_snapshot(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        radius_km=radius_km,
        r34_km=50.0,
        rocloud_km=100.0,
        wind_hazard_normalized=wind_hazard,
    )

    expected_hp = np.array([0.0, 0.25, 0.75, 0.75, 1.0, 1.0])
    expected_h_pdir = np.array([0.0, 0.25, 0.75, 0.0, 0.0, 0.0])
    expected_h_pind = np.array([0.0, 0.0, 0.0, 0.75, 1.0, 0.0])
    expected_h_w = np.array([1.0, 0.5, 0.3, 0.0, 0.0, 0.0])
    expected_h_dir = np.array([1.0, 0.625, 0.825, 0.0, 0.0, 0.0])
    expected_h_ind = np.array([0.0, 0.0, 0.0, 0.75, 1.0, 0.0])
    expected_itchi = np.array([1.0, 0.625, 0.825, 0.75, 1.0, 0.0])

    np.testing.assert_allclose(result["H_P"], expected_hp)
    np.testing.assert_allclose(result["H_Pdir"], expected_h_pdir)
    np.testing.assert_allclose(result["H_Pind"], expected_h_pind)
    np.testing.assert_allclose(result["H_W"], expected_h_w)
    np.testing.assert_allclose(result["H_dir"], expected_h_dir)
    np.testing.assert_allclose(result["H_ind"], expected_h_ind)
    np.testing.assert_allclose(result["ITCHI"], expected_itchi)


def test_compute_itchi_snapshot_is_bounded() -> None:
    """
    Test that the final ITCHI output remains within [0, 1].
    """
    precipitation = np.array([-999.0, 15.0, 1000.0])
    radius_km = np.array([0.0, 75.0, 125.0])
    wind_hazard = np.array([-1.0, 0.5, 2.0])

    result = compute_itchi_snapshot(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        radius_km=radius_km,
        r34_km=50.0,
        rocloud_km=100.0,
        wind_hazard_normalized=wind_hazard,
    )

    itchi = result["ITCHI"]

    assert np.nanmin(itchi) >= 0.0
    assert np.nanmax(itchi) <= 1.0


def test_compute_itchi_snapshot_preserves_xarray_dimensions() -> None:
    """
    Test that xarray inputs preserve dimensions and coordinates.
    """
    coords = {
        "lat": [15.0, 16.0],
        "lon": [-100.0, -99.0],
    }

    precipitation = xr.DataArray(
        data=[[5.0, 15.0], [25.0, 40.0]],
        dims=("lat", "lon"),
        coords=coords,
        name="precipitation",
    )

    radius_km = xr.DataArray(
        data=[[10.0, 40.0], [80.0, 130.0]],
        dims=("lat", "lon"),
        coords=coords,
        name="radius_km",
    )

    wind_hazard = xr.DataArray(
        data=[[1.0, 0.5], [0.3, 0.0]],
        dims=("lat", "lon"),
        coords=coords,
        name="V_star",
    )

    result = compute_itchi_snapshot(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        radius_km=radius_km,
        r34_km=50.0,
        rocloud_km=100.0,
        wind_hazard_normalized=wind_hazard,
    )

    for value in result.values():
        assert isinstance(value, xr.DataArray)
        assert value.dims == precipitation.dims
        assert value.shape == precipitation.shape

    assert float(result["ITCHI"].min()) >= 0.0
    assert float(result["ITCHI"].max()) <= 1.0


def test_compute_itchi_snapshot_from_grid_returns_geometry_fields() -> None:
    """
    Test that the grid-based pipeline returns geometry and ITCHI fields.
    """
    precipitation = np.array([5.0, 15.0, 25.0, 25.0, 30.0, 40.0])
    lon = np.array([0.0, 0.2, 0.5, 0.9, 1.2, 1.5])
    lat = np.zeros_like(lon)
    wind_hazard = np.array([1.0, 0.5, 0.3, 0.2, 0.0, 0.0])

    result = compute_itchi_snapshot_from_grid(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        center_lon=0.0,
        center_lat=0.0,
        r34_by_quadrant={
            "NE": 40.0,
            "SE": 40.0,
            "SW": 40.0,
            "NW": 40.0,
        },
        r34_unit="nm",
        rocloud_by_quadrant={
            "NE": 130.0,
            "SE": 130.0,
            "SW": 130.0,
            "NW": 130.0,
        },
        rocloud_unit="km",
        wind_hazard_normalized=wind_hazard,
    )

    expected_keys = {
        "radius_km",
        "quadrant",
        "R34_q",
        "ROCLOUD_q",
        "M_direct",
        "M_indirect",
        "M_exterior",
        "H_P",
        "H_Pdir",
        "H_Pind",
        "H_W",
        "H_dir",
        "H_ind",
        "ITCHI",
    }

    assert set(result) == expected_keys


def test_compute_itchi_snapshot_from_grid_expected_numpy_values() -> None:
    """
    Test grid-based ITCHI calculation with synthetic NumPy inputs.
    """
    precipitation = np.array([5.0, 15.0, 25.0, 25.0, 30.0, 40.0])
    lon = np.array([0.0, 0.2, 0.5, 0.9, 1.2, 1.5])
    lat = np.zeros_like(lon)
    wind_hazard = np.array([1.0, 0.5, 0.3, 0.2, 0.0, 0.0])

    result = compute_itchi_snapshot_from_grid(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        center_lon=0.0,
        center_lat=0.0,
        r34_by_quadrant={
            "RNE": 40.0,
            "RSE": 40.0,
            "RSW": 40.0,
            "RNW": 40.0,
        },
        r34_unit="nm",
        rocloud_by_quadrant={
            "RNE": 130.0,
            "RSE": 130.0,
            "RSW": 130.0,
            "RNW": 130.0,
        },
        rocloud_unit="km",
        wind_hazard_normalized=wind_hazard,
    )

    expected_h_p = np.array([0.0, 0.25, 0.75, 0.75, 1.0, 1.0])
    expected_h_pdir = np.array([0.0, 0.25, 0.75, 0.0, 0.0, 0.0])
    expected_h_pind = np.array([0.0, 0.0, 0.0, 0.75, 0.0, 0.0])
    expected_h_w = np.array([1.0, 0.5, 0.3, 0.0, 0.0, 0.0])
    expected_h_dir = np.array([1.0, 0.625, 0.825, 0.0, 0.0, 0.0])
    expected_h_ind = np.array([0.0, 0.0, 0.0, 0.75, 0.0, 0.0])
    expected_itchi = np.array([1.0, 0.625, 0.825, 0.75, 0.0, 0.0])

    np.testing.assert_allclose(result["H_P"], expected_h_p)
    np.testing.assert_allclose(result["H_Pdir"], expected_h_pdir)
    np.testing.assert_allclose(result["H_Pind"], expected_h_pind)
    np.testing.assert_allclose(result["H_W"], expected_h_w)
    np.testing.assert_allclose(result["H_dir"], expected_h_dir)
    np.testing.assert_allclose(result["H_ind"], expected_h_ind)
    np.testing.assert_allclose(result["ITCHI"], expected_itchi)


def test_compute_itchi_snapshot_from_grid_preserves_xarray_dimensions() -> None:
    """
    Test that the grid-based pipeline preserves xarray dimensions.
    """
    coords = {
        "lat": [0.0, 0.5],
        "lon": [0.0, 0.5],
    }

    precipitation = xr.DataArray(
        data=[[5.0, 15.0], [25.0, 40.0]],
        dims=("lat", "lon"),
        coords=coords,
        name="precipitation",
    )

    lon = xr.DataArray(
        data=[[0.0, 0.5], [0.0, 0.5]],
        dims=("lat", "lon"),
        coords=coords,
        name="lon",
    )

    lat = xr.DataArray(
        data=[[0.0, 0.0], [0.5, 0.5]],
        dims=("lat", "lon"),
        coords=coords,
        name="lat",
    )

    wind_hazard = xr.DataArray(
        data=[[1.0, 0.5], [0.3, 0.0]],
        dims=("lat", "lon"),
        coords=coords,
        name="V_star",
    )

    result = compute_itchi_snapshot_from_grid(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        center_lon=0.0,
        center_lat=0.0,
        r34_by_quadrant={
            "RNE": 40.0,
            "RSE": 40.0,
            "RSW": 40.0,
            "RNW": 40.0,
        },
        r34_unit="nm",
        rocloud_by_quadrant={
            "RNE": 130.0,
            "RSE": 130.0,
            "RSW": 130.0,
            "RNW": 130.0,
        },
        rocloud_unit="km",
        wind_hazard_normalized=wind_hazard,
    )

    for value in result.values():
        assert isinstance(value, xr.DataArray)
        assert value.dims == precipitation.dims
        assert value.shape == precipitation.shape

    assert float(result["ITCHI"].min()) >= 0.0
    assert float(result["ITCHI"].max()) <= 1.0
