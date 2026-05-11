"""
Tests for ITCHI hazard components.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from itchi.components import (
    compute_direct_component,
    compute_direct_precipitation_hazard,
    compute_hazard_components,
    compute_indirect_component,
    compute_indirect_precipitation_hazard,
    compute_wind_hazard,
)


def test_direct_precipitation_hazard_numpy() -> None:
    """
    Test direct precipitation hazard using a Boolean direct mask.
    """
    hp = np.array([0.1, 0.4, 0.7, 1.0])
    direct_mask = np.array([True, True, False, False])

    hpdir = compute_direct_precipitation_hazard(
        precipitation_hazard=hp,
        direct_mask=direct_mask,
    )

    expected = np.array([0.1, 0.4, 0.0, 0.0])

    np.testing.assert_allclose(hpdir, expected)


def test_indirect_precipitation_hazard_numpy() -> None:
    """
    Test indirect precipitation hazard using a Boolean indirect mask.
    """
    hp = np.array([0.1, 0.4, 0.7, 1.0])
    indirect_mask = np.array([False, False, True, False])

    hpind = compute_indirect_precipitation_hazard(
        precipitation_hazard=hp,
        indirect_mask=indirect_mask,
    )

    expected = np.array([0.0, 0.0, 0.7, 0.0])

    np.testing.assert_allclose(hpind, expected)


def test_wind_hazard_numpy() -> None:
    """
    Test wind hazard activation only inside the direct region.
    """
    v_star = np.array([1.0, 0.5, 0.3, 0.2])
    direct_mask = np.array([True, True, False, False])

    hw = compute_wind_hazard(
        wind_hazard_normalized=v_star,
        direct_mask=direct_mask,
    )

    expected = np.array([1.0, 0.5, 0.0, 0.0])

    np.testing.assert_allclose(hw, expected)


def test_direct_component_noisy_or_numpy() -> None:
    """
    Test the bounded union formulation for the direct component.
    """
    hw = np.array([0.0, 0.5, 1.0])
    hpdir = np.array([0.0, 0.5, 1.0])

    hdir = compute_direct_component(
        wind_hazard=hw,
        direct_precipitation_hazard=hpdir,
    )

    expected = np.array([0.0, 0.75, 1.0])

    np.testing.assert_allclose(hdir, expected)


def test_indirect_component_equals_indirect_precipitation() -> None:
    """
    Test that H_ind equals H_Pind in ITCHI v0.1.
    """
    hpind = np.array([0.0, 0.2, 0.8, 1.0])

    hind = compute_indirect_component(
        indirect_precipitation_hazard=hpind,
    )

    np.testing.assert_allclose(hind, hpind)


def test_compute_hazard_components_numpy() -> None:
    """
    Test complete hazard component construction.
    """
    hp = np.array([0.2, 0.4, 0.6, 0.8])
    v_star = np.array([1.0, 0.5, 0.3, 0.2])

    direct_mask = np.array([True, True, False, False])
    indirect_mask = np.array([False, False, True, False])

    components = compute_hazard_components(
        precipitation_hazard=hp,
        wind_hazard_normalized=v_star,
        direct_mask=direct_mask,
        indirect_mask=indirect_mask,
    )

    expected_hpdir = np.array([0.2, 0.4, 0.0, 0.0])
    expected_hpind = np.array([0.0, 0.0, 0.6, 0.0])
    expected_hw = np.array([1.0, 0.5, 0.0, 0.0])
    expected_hdir = np.array([1.0, 0.7, 0.0, 0.0])
    expected_hind = np.array([0.0, 0.0, 0.6, 0.0])

    np.testing.assert_allclose(components["H_Pdir"], expected_hpdir)
    np.testing.assert_allclose(components["H_Pind"], expected_hpind)
    np.testing.assert_allclose(components["H_W"], expected_hw)
    np.testing.assert_allclose(components["H_dir"], expected_hdir)
    np.testing.assert_allclose(components["H_ind"], expected_hind)


def test_components_are_bounded() -> None:
    """
    Test that components remain bounded between 0 and 1.
    """
    hp = np.array([-1.0, 0.5, 2.0])
    v_star = np.array([-0.5, 0.5, 1.5])

    direct_mask = np.array([True, True, True])
    indirect_mask = np.array([False, False, False])

    components = compute_hazard_components(
        precipitation_hazard=hp,
        wind_hazard_normalized=v_star,
        direct_mask=direct_mask,
        indirect_mask=indirect_mask,
    )

    for component in components.values():
        assert np.nanmin(component) >= 0.0
        assert np.nanmax(component) <= 1.0


def test_components_preserve_xarray_dimensions() -> None:
    """
    Test that component calculations preserve xarray dimensions.
    """
    hp = xr.DataArray(
        data=[[0.2, 0.4], [0.6, 0.8]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="H_P",
    )

    v_star = xr.DataArray(
        data=[[1.0, 0.5], [0.3, 0.2]],
        dims=("lat", "lon"),
        coords=hp.coords,
        name="V_star",
    )

    direct_mask = xr.DataArray(
        data=[[True, True], [False, False]],
        dims=("lat", "lon"),
        coords=hp.coords,
    )

    indirect_mask = xr.DataArray(
        data=[[False, False], [True, False]],
        dims=("lat", "lon"),
        coords=hp.coords,
    )

    components = compute_hazard_components(
        precipitation_hazard=hp,
        wind_hazard_normalized=v_star,
        direct_mask=direct_mask,
        indirect_mask=indirect_mask,
    )

    for component in components.values():
        assert isinstance(component, xr.DataArray)
        assert component.dims == hp.dims
        assert component.shape == hp.shape
        assert float(component.min()) >= 0.0
        assert float(component.max()) <= 1.0
