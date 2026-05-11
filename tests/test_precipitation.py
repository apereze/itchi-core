"""
Tests for precipitation hazard calculations.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from itchi.precipitation import compute_precipitation_hazard


def test_precipitation_hazard_piecewise_values() -> None:
    """
    Test expected values of the piecewise precipitation hazard function.
    """
    precipitation = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])

    hazard = compute_precipitation_hazard(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
    )

    expected = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])

    np.testing.assert_allclose(hazard, expected)


def test_precipitation_hazard_is_bounded() -> None:
    """
    Test that precipitation hazard remains within [0, 1].
    """
    precipitation = np.array([-999.0, 0.0, 10.0, 1000.0])

    hazard = compute_precipitation_hazard(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
    )

    assert np.nanmin(hazard) >= 0.0
    assert np.nanmax(hazard) <= 1.0


def test_precipitation_hazard_handles_equal_percentiles() -> None:
    """
    Test numerical stability when percentile intervals collapse.
    """
    precipitation = np.array([9.0, 10.0, 15.0, 20.0])

    hazard = compute_precipitation_hazard(
        precipitation=precipitation,
        q90=10.0,
        q95=10.0,
        q99=20.0,
    )

    assert np.nanmin(hazard) >= 0.0
    assert np.nanmax(hazard) <= 1.0


def test_precipitation_hazard_preserves_xarray_dimensions() -> None:
    """
    Test that xarray input returns xarray output with dimensions preserved.
    """
    precipitation = xr.DataArray(
        data=[[5.0, 10.0], [20.0, 35.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="precipitation",
    )

    hazard = compute_precipitation_hazard(
        precipitation=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
    )

    assert isinstance(hazard, xr.DataArray)
    assert hazard.dims == precipitation.dims
    assert hazard.shape == precipitation.shape
    assert float(hazard.min()) >= 0.0
    assert float(hazard.max()) <= 1.0