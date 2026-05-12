"""
Tests for event-level ITCHI aggregation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from itchi.aggregation import (
    compute_event_accumulated,
    compute_event_max,
    compute_event_products,
)


def test_compute_event_max_numpy() -> None:
    """
    Test event maximum with NumPy input.
    """
    itchi = np.array(
        [
            [0.0, 0.2, 0.5],
            [0.3, 0.1, 0.7],
            [0.2, 0.6, 0.4],
        ]
    )

    result = compute_event_max(itchi, axis=0)

    expected = np.array([0.3, 0.6, 0.7])

    np.testing.assert_allclose(result, expected)


def test_compute_event_accumulated_numpy() -> None:
    """
    Test bounded accumulated ITCHI with NumPy input.
    """
    itchi = np.array(
        [
            [0.0, 0.2],
            [0.5, 0.4],
        ]
    )

    result = compute_event_accumulated(itchi, axis=0)

    expected = np.array(
        [
            1.0 - (1.0 - 0.0) * (1.0 - 0.5),
            1.0 - (1.0 - 0.2) * (1.0 - 0.4),
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_compute_event_products_numpy() -> None:
    """
    Test that standard event products are returned.
    """
    itchi = np.array(
        [
            [0.0, 0.2],
            [0.5, 0.4],
        ]
    )

    products = compute_event_products(itchi, axis=0)

    assert set(products) == {"ITCHI_max", "ITCHI_acc"}

    np.testing.assert_allclose(products["ITCHI_max"], np.array([0.5, 0.4]))

    expected_acc = np.array(
        [
            1.0 - (1.0 - 0.0) * (1.0 - 0.5),
            1.0 - (1.0 - 0.2) * (1.0 - 0.4),
        ]
    )

    np.testing.assert_allclose(products["ITCHI_acc"], expected_acc)


def test_event_products_are_bounded() -> None:
    """
    Test that event products remain bounded in [0, 1].
    """
    itchi = np.array(
        [
            [-1.0, 0.2],
            [0.5, 2.0],
        ]
    )

    products = compute_event_products(itchi, axis=0)

    for product in products.values():
        assert np.nanmin(product) >= 0.0
        assert np.nanmax(product) <= 1.0


def test_compute_event_products_skipna_numpy() -> None:
    """
    Test NaN handling for NumPy aggregation.
    """
    itchi = np.array(
        [
            [np.nan, 0.2],
            [0.5, np.nan],
        ]
    )

    products = compute_event_products(itchi, axis=0, skipna=True)

    np.testing.assert_allclose(products["ITCHI_max"], np.array([0.5, 0.2]))
    np.testing.assert_allclose(products["ITCHI_acc"], np.array([0.5, 0.2]))


def test_compute_event_products_xarray() -> None:
    """
    Test event products with xarray input.
    """
    itchi = xr.DataArray(
        data=[
            [[0.0, 0.2], [0.5, 0.1]],
            [[0.3, 0.4], [0.2, 0.7]],
        ],
        dims=("time", "lat", "lon"),
        coords={
            "time": ["2020-01-01T00:00", "2020-01-01T06:00"],
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="ITCHI",
    )

    products = compute_event_products(itchi, dim="time")

    assert set(products) == {"ITCHI_max", "ITCHI_acc"}

    for product in products.values():
        assert isinstance(product, xr.DataArray)
        assert product.dims == ("lat", "lon")
        assert product.shape == (2, 2)
        assert float(product.min()) >= 0.0
        assert float(product.max()) <= 1.0

    expected_max = xr.DataArray(
        data=[[0.3, 0.4], [0.5, 0.7]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
    )

    xr.testing.assert_allclose(products["ITCHI_max"], expected_max)
