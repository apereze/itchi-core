"""
Tests for the final ITCHI index calculation.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itchi.index import compute_itchi, compute_itchi_from_components


def test_compute_itchi_baseline_numpy() -> None:
    """
    Test final ITCHI calculation with neutral weights.
    """
    h_dir = np.array([0.0, 0.5, 1.0])
    h_ind = np.array([0.0, 0.5, 0.0])

    itchi = compute_itchi(
        direct_component=h_dir,
        indirect_component=h_ind,
    )

    expected = np.array([0.0, 0.75, 1.0])

    np.testing.assert_allclose(itchi, expected)


def test_compute_itchi_zero_when_components_are_zero() -> None:
    """
    Test that ITCHI is zero when both components are zero.
    """
    h_dir = np.zeros(5)
    h_ind = np.zeros(5)

    itchi = compute_itchi(
        direct_component=h_dir,
        indirect_component=h_ind,
    )

    np.testing.assert_allclose(itchi, np.zeros(5))


def test_compute_itchi_one_when_any_component_is_one() -> None:
    """
    Test that ITCHI reaches one when any component reaches one.
    """
    h_dir = np.array([1.0, 0.0, 1.0])
    h_ind = np.array([0.0, 1.0, 1.0])

    itchi = compute_itchi(
        direct_component=h_dir,
        indirect_component=h_ind,
    )

    expected = np.array([1.0, 1.0, 1.0])

    np.testing.assert_allclose(itchi, expected)


def test_compute_itchi_is_bounded() -> None:
    """
    Test that ITCHI remains bounded in [0, 1] even if inputs are outside bounds.
    """
    h_dir = np.array([-1.0, 0.5, 2.0])
    h_ind = np.array([-0.5, 0.5, 1.5])

    itchi = compute_itchi(
        direct_component=h_dir,
        indirect_component=h_ind,
    )

    assert np.nanmin(itchi) >= 0.0
    assert np.nanmax(itchi) <= 1.0


def test_compute_itchi_with_weights() -> None:
    """
    Test that weighting exponents are applied correctly.
    """
    h_dir = np.array([0.5])
    h_ind = np.array([0.5])

    itchi = compute_itchi(
        direct_component=h_dir,
        indirect_component=h_ind,
        lambda_direct=2.0,
        mu_indirect=1.0,
    )

    expected = np.array([1.0 - ((1.0 - 0.5) ** 2.0 * (1.0 - 0.5))])

    np.testing.assert_allclose(itchi, expected)


def test_compute_itchi_rejects_negative_weights() -> None:
    """
    Test that negative exponents raise an error.
    """
    h_dir = np.array([0.5])
    h_ind = np.array([0.5])

    with pytest.raises(ValueError):
        compute_itchi(
            direct_component=h_dir,
            indirect_component=h_ind,
            lambda_direct=-1.0,
        )

    with pytest.raises(ValueError):
        compute_itchi(
            direct_component=h_dir,
            indirect_component=h_ind,
            mu_indirect=-1.0,
        )


def test_compute_itchi_from_components_dict() -> None:
    """
    Test ITCHI calculation from a component dictionary.
    """
    components = {
        "H_dir": np.array([0.0, 0.5, 1.0]),
        "H_ind": np.array([0.0, 0.5, 0.0]),
    }

    itchi = compute_itchi_from_components(components)

    expected = np.array([0.0, 0.75, 1.0])

    np.testing.assert_allclose(itchi, expected)


def test_compute_itchi_from_components_missing_keys() -> None:
    """
    Test that missing required component keys raise an error.
    """
    components = {
        "H_dir": np.array([0.5]),
    }

    with pytest.raises(KeyError):
        compute_itchi_from_components(components)


def test_compute_itchi_preserves_xarray_dimensions() -> None:
    """
    Test that xarray input preserves dimensions and coordinates.
    """
    h_dir = xr.DataArray(
        data=[[0.0, 0.5], [1.0, 0.2]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="H_dir",
    )

    h_ind = xr.DataArray(
        data=[[0.0, 0.5], [0.0, 0.8]],
        dims=("lat", "lon"),
        coords=h_dir.coords,
        name="H_ind",
    )

    itchi = compute_itchi(
        direct_component=h_dir,
        indirect_component=h_ind,
    )

    assert isinstance(itchi, xr.DataArray)
    assert itchi.dims == h_dir.dims
    assert itchi.shape == h_dir.shape
    assert float(itchi.min()) >= 0.0
    assert float(itchi.max()) <= 1.0