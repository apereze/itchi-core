"""
Tests for ITCHI wind-hazard utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itchi.wind import (
    apply_wind_mask,
    compute_radial_wind_profile,
    compute_wind_hazard_from_profile,
    normalize_wind_hazard,
)


def test_compute_radial_wind_profile_numpy() -> None:
    """
    Test simple radial wind profile with NumPy arrays.
    """
    radius = np.array([0.0, 10.0, 20.0, 80.0])

    wind = compute_radial_wind_profile(
        radius_km=radius,
        vmax_kt=80.0,
        rmw_km=20.0,
        inner_exponent=1.0,
        outer_decay_exponent=1.0,
    )

    expected = np.array([0.000004, 40.0, 80.0, 20.0])

    np.testing.assert_allclose(wind, expected, atol=1.0e-5)


def test_compute_radial_wind_profile_outer_radius() -> None:
    """
    Test that wind is zero outside outer_radius_km.
    """
    radius = np.array([10.0, 20.0, 100.0])

    wind = compute_radial_wind_profile(
        radius_km=radius,
        vmax_kt=80.0,
        rmw_km=20.0,
        outer_radius_km=50.0,
    )

    assert wind[2] == pytest.approx(0.0)


def test_normalize_wind_hazard_above_tropical_storm_threshold() -> None:
    """
    Test wind normalization for Vmax > 34 kt.
    """
    wind = np.array([0.0, 34.0, 57.0, 80.0])

    hazard = normalize_wind_hazard(
        wind_kt=wind,
        vmax_kt=80.0,
        reference_wind_kt=34.0,
    )

    expected = np.array([0.0, 0.0, 0.5, 1.0])

    np.testing.assert_allclose(hazard, expected)


def test_normalize_wind_hazard_below_threshold_relative_to_vmax() -> None:
    """
    Test relative normalization for tropical-depression cases.
    """
    wind = np.array([0.0, 15.0, 30.0])

    hazard = normalize_wind_hazard(
        wind_kt=wind,
        vmax_kt=30.0,
        reference_wind_kt=34.0,
        below_threshold_mode="relative_to_vmax",
    )

    expected = np.array([0.0, 0.5, 1.0])

    np.testing.assert_allclose(hazard, expected)


def test_normalize_wind_hazard_below_threshold_zero_mode() -> None:
    """
    Test zero wind hazard mode for systems below 34 kt.
    """
    wind = np.array([0.0, 15.0, 30.0])

    hazard = normalize_wind_hazard(
        wind_kt=wind,
        vmax_kt=30.0,
        reference_wind_kt=34.0,
        below_threshold_mode="zero",
    )

    expected = np.zeros_like(wind, dtype=float)

    np.testing.assert_allclose(hazard, expected)


def test_normalize_wind_hazard_rejects_unknown_mode() -> None:
    """
    Test that unsupported below-threshold modes raise an error.
    """
    wind = np.array([0.0, 15.0, 30.0])

    with pytest.raises(ValueError):
        normalize_wind_hazard(
            wind_kt=wind,
            vmax_kt=30.0,
            reference_wind_kt=34.0,
            below_threshold_mode="unsupported",
        )


def test_apply_wind_mask_numpy() -> None:
    """
    Test applying a Boolean mask to wind hazard.
    """
    wind_hazard = np.array([1.0, 0.5, 0.2])
    mask = np.array([True, False, True])

    result = apply_wind_mask(
        wind_hazard_normalized=wind_hazard,
        mask=mask,
    )

    expected = np.array([1.0, 0.0, 0.2])

    np.testing.assert_allclose(result, expected)


def test_compute_wind_hazard_from_profile_numpy() -> None:
    """
    Test wind-hazard computation from radial profile.
    """
    radius = np.array([0.0, 10.0, 20.0, 80.0])

    hazard = compute_wind_hazard_from_profile(
        radius_km=radius,
        vmax_kt=80.0,
        rmw_km=20.0,
        inner_exponent=1.0,
        outer_decay_exponent=1.0,
    )

    assert np.nanmin(hazard) >= 0.0
    assert np.nanmax(hazard) <= 1.0
    assert hazard[2] == pytest.approx(1.0)


def test_compute_wind_hazard_from_profile_with_mask() -> None:
    """
    Test wind hazard with active mask.
    """
    radius = np.array([10.0, 20.0, 80.0])
    mask = np.array([True, True, False])

    hazard = compute_wind_hazard_from_profile(
        radius_km=radius,
        vmax_kt=80.0,
        rmw_km=20.0,
        inner_exponent=1.0,
        outer_decay_exponent=1.0,
        active_mask=mask,
    )

    assert hazard[2] == pytest.approx(0.0)


def test_wind_hazard_preserves_xarray_dimensions() -> None:
    """
    Test wind hazard with xarray inputs.
    """
    radius = xr.DataArray(
        data=[[10.0, 20.0], [40.0, 80.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="radius_km",
    )

    hazard = compute_wind_hazard_from_profile(
        radius_km=radius,
        vmax_kt=80.0,
        rmw_km=20.0,
        inner_exponent=1.0,
        outer_decay_exponent=1.0,
    )

    assert isinstance(hazard, xr.DataArray)
    assert hazard.dims == radius.dims
    assert hazard.shape == radius.shape
    assert float(hazard.min()) >= 0.0
    assert float(hazard.max()) <= 1.0


def test_compute_radial_wind_profile_rejects_invalid_parameters() -> None:
    """
    Test validation for invalid profile parameters.
    """
    radius = np.array([10.0, 20.0])

    with pytest.raises(ValueError):
        compute_radial_wind_profile(
            radius_km=radius,
            vmax_kt=0.0,
            rmw_km=20.0,
        )

    with pytest.raises(ValueError):
        compute_radial_wind_profile(
            radius_km=radius,
            vmax_kt=80.0,
            rmw_km=0.0,
        )
