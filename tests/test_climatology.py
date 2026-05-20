"""
Tests for ITCHI precipitation climatology utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itchi.climatology import (
    climatology_dict_to_dataset,
    compute_precipitation_quantiles,
    compute_standard_precipitation_climatology,
    extract_standard_percentiles,
    quantile_level_to_name,
    validate_quantile_levels,
)


def test_quantile_level_to_name() -> None:
    """
    Test quantile level to percentile name conversion.
    """
    assert quantile_level_to_name(0.90) == "Q90"
    assert quantile_level_to_name(0.95) == "Q95"
    assert quantile_level_to_name(0.99) == "Q99"


def test_validate_quantile_levels_accepts_valid_values() -> None:
    """
    Test valid quantile levels.
    """
    validate_quantile_levels([0.90, 0.95, 0.99])


def test_validate_quantile_levels_rejects_invalid_values() -> None:
    """
    Test invalid quantile levels.
    """
    with pytest.raises(ValueError):
        validate_quantile_levels([0.90, 1.10])

    with pytest.raises(ValueError):
        validate_quantile_levels([])


def test_compute_precipitation_quantiles_numpy() -> None:
    """
    Test precipitation quantile calculation with NumPy input.
    """
    precipitation = np.array(
        [
            [0.0, 10.0],
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
        ]
    )

    result = compute_precipitation_quantiles(
        precipitation=precipitation,
        quantile_levels=(0.5, 0.9),
        axis=0,
    )

    assert set(result) == {"Q50", "Q90"}

    expected_q50 = np.array([20.0, 30.0])
    expected_q90 = np.array([36.0, 46.0])

    np.testing.assert_allclose(result["Q50"], expected_q50)
    np.testing.assert_allclose(result["Q90"], expected_q90)


def test_compute_standard_precipitation_climatology_numpy() -> None:
    """
    Test standard Q90, Q95 and Q99 climatology with NumPy input.
    """
    precipitation = np.arange(100.0).reshape(100, 1)

    result = compute_standard_precipitation_climatology(
        precipitation=precipitation,
        axis=0,
    )

    assert set(result) == {"Q90", "Q95", "Q99"}
    assert result["Q90"].shape == (1,)
    assert result["Q95"].shape == (1,)
    assert result["Q99"].shape == (1,)


def test_compute_precipitation_quantiles_numpy_skipna() -> None:
    """
    Test NaN-aware quantile calculation.
    """
    precipitation = np.array(
        [
            [0.0, np.nan],
            [10.0, 20.0],
            [20.0, 30.0],
        ]
    )

    result = compute_precipitation_quantiles(
        precipitation=precipitation,
        quantile_levels=(0.5,),
        axis=0,
        skipna=True,
    )

    expected = np.array([10.0, 25.0])

    np.testing.assert_allclose(result["Q50"], expected)


def test_compute_precipitation_quantiles_xarray() -> None:
    """
    Test precipitation quantile calculation with xarray input.
    """
    precipitation = xr.DataArray(
        data=[
            [[0.0, 10.0], [20.0, 30.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[20.0, 30.0], [40.0, 50.0]],
        ],
        dims=("time", "lat", "lon"),
        coords={
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="precipitation",
    )

    result = compute_precipitation_quantiles(
        precipitation=precipitation,
        quantile_levels=(0.5,),
        dim="time",
    )

    assert set(result) == {"Q50"}
    assert isinstance(result["Q50"], xr.DataArray)
    assert result["Q50"].dims == ("lat", "lon")
    assert result["Q50"].shape == (2, 2)

    expected = xr.DataArray(
        data=[[10.0, 20.0], [30.0, 40.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="Q50",
    )

    xr.testing.assert_allclose(result["Q50"], expected)


def test_climatology_dict_to_dataset() -> None:
    """
    Test conversion from xarray climatology dictionary to Dataset.
    """
    q90 = xr.DataArray(
        data=[[10.0, 20.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0],
            "lon": [-100.0, -99.0],
        },
        name="Q90",
    )

    dataset = climatology_dict_to_dataset({"Q90": q90})

    assert isinstance(dataset, xr.Dataset)
    assert "Q90" in dataset
    xr.testing.assert_allclose(dataset["Q90"], q90)


def test_climatology_dict_to_dataset_rejects_numpy() -> None:
    """
    Test that Dataset conversion rejects NumPy arrays.
    """
    with pytest.raises(ValueError):
        climatology_dict_to_dataset({"Q90": np.array([1.0])})


def test_extract_standard_percentiles_from_dict() -> None:
    """
    Test extracting standard percentile fields from a dictionary.
    """
    climatology = {
        "Q90": np.array([10.0]),
        "Q95": np.array([20.0]),
        "Q99": np.array([30.0]),
    }

    q90, q95, q99 = extract_standard_percentiles(climatology)

    np.testing.assert_allclose(q90, np.array([10.0]))
    np.testing.assert_allclose(q95, np.array([20.0]))
    np.testing.assert_allclose(q99, np.array([30.0]))


def test_extract_standard_percentiles_rejects_missing() -> None:
    """
    Test that missing percentile fields raise an error.
    """
    climatology = {
        "Q90": np.array([10.0]),
        "Q95": np.array([20.0]),
    }

    with pytest.raises(KeyError):
        extract_standard_percentiles(climatology)
