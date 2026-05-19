"""
Tests for event-level ITCHI compiler.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itchi.compiler import (
    compile_itchi_event,
    extract_snapshot_metadata,
    stack_snapshot_results,
)


def _make_numpy_snapshot_input(
    precipitation: np.ndarray,
    wind_hazard: np.ndarray | None = None,
) -> dict:
    """
    Create a synthetic NumPy snapshot input for compiler tests.
    """
    if wind_hazard is None:
        wind_hazard = np.array([1.0, 0.5, 0.3, 0.0])

    return {
        "precipitation": precipitation,
        "q90": 10.0,
        "q95": 20.0,
        "q99": 30.0,
        "lon": np.array([0.0, 0.2, 0.5, 0.9]),
        "lat": np.array([0.0, 0.0, 0.0, 0.0]),
        "center_lon": 0.0,
        "center_lat": 0.0,
        "r34_by_quadrant": {
            "RNE": 40.0,
            "RSE": 40.0,
            "RSW": 40.0,
            "RNW": 40.0,
        },
        "r34_unit": "nm",
        "rocloud_by_quadrant": {
            "RNE": 130.0,
            "RSE": 130.0,
            "RSW": 130.0,
            "RNW": 130.0,
        },
        "rocloud_unit": "km",
        "wind_hazard_normalized": wind_hazard,
    }


def _make_xarray_snapshot_input(precipitation: xr.DataArray) -> dict:
    """
    Create a synthetic xarray snapshot input for compiler tests.
    """
    coords = precipitation.coords

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

    return {
        "precipitation": precipitation,
        "q90": 10.0,
        "q95": 20.0,
        "q99": 30.0,
        "lon": lon,
        "lat": lat,
        "center_lon": 0.0,
        "center_lat": 0.0,
        "r34_by_quadrant": {
            "RNE": 40.0,
            "RSE": 40.0,
            "RSW": 40.0,
            "RNW": 40.0,
        },
        "r34_unit": "nm",
        "rocloud_by_quadrant": {
            "RNE": 130.0,
            "RSE": 130.0,
            "RSW": 130.0,
            "RNW": 130.0,
        },
        "rocloud_unit": "km",
        "wind_hazard_normalized": wind_hazard,
    }


def test_stack_snapshot_results_rejects_empty_sequence() -> None:
    """
    Test that empty snapshot results are rejected.
    """
    with pytest.raises(ValueError):
        stack_snapshot_results([])


def test_stack_snapshot_results_rejects_missing_variable() -> None:
    """
    Test that missing requested variables raise an error.
    """
    snapshot_results = [
        {
            "ITCHI": np.array([0.0]),
            "H_P": np.array([0.0]),
        },
        {
            "ITCHI": np.array([0.5]),
        },
    ]

    with pytest.raises(KeyError):
        stack_snapshot_results(
            snapshot_results=snapshot_results,
            variable_names=("ITCHI", "H_P"),
        )


def test_stack_snapshot_results_numpy() -> None:
    """
    Test stacking NumPy snapshot variables.
    """
    snapshot_results = [
        {
            "ITCHI": np.array([0.0, 0.5]),
            "H_P": np.array([0.1, 0.2]),
        },
        {
            "ITCHI": np.array([0.2, 0.8]),
            "H_P": np.array([0.3, 0.4]),
        },
    ]

    stacked = stack_snapshot_results(
        snapshot_results=snapshot_results,
        variable_names=("ITCHI", "H_P"),
    )

    assert stacked["ITCHI"].shape == (2, 2)
    assert stacked["H_P"].shape == (2, 2)

    np.testing.assert_allclose(
        stacked["ITCHI"],
        np.array(
            [
                [0.0, 0.5],
                [0.2, 0.8],
            ]
        ),
    )


def test_extract_snapshot_metadata() -> None:
    """
    Test metadata extraction from snapshot results.
    """
    snapshot_results = [
        {
            "ITCHI": np.array([0.0]),
            "R_direct_source": "R34",
            "wind_hazard_source": "provided",
        },
        {
            "ITCHI": np.array([0.5]),
            "R_direct_source": "RMW_fallback_for_tropical_depression",
            "wind_hazard_source": "radial_profile",
        },
    ]

    metadata = extract_snapshot_metadata(
        snapshot_results=snapshot_results,
        time_values=["2020-01-01T00:00", "2020-01-01T06:00"],
    )

    assert len(metadata) == 2
    assert metadata[0]["time"] == "2020-01-01T00:00"
    assert metadata[0]["R_direct_source"] == "R34"
    assert metadata[1]["wind_hazard_source"] == "radial_profile"


def test_compile_itchi_event_rejects_empty_inputs() -> None:
    """
    Test that empty snapshot inputs are rejected.
    """
    with pytest.raises(ValueError):
        compile_itchi_event([])


def test_compile_itchi_event_rejects_wrong_time_length() -> None:
    """
    Test that time_values length must match number of snapshots.
    """
    snapshot_inputs = [
        _make_numpy_snapshot_input(np.array([5.0, 15.0, 25.0, 25.0])),
        _make_numpy_snapshot_input(np.array([5.0, 5.0, 15.0, 30.0])),
    ]

    with pytest.raises(ValueError):
        compile_itchi_event(
            snapshot_inputs=snapshot_inputs,
            time_values=["2020-01-01T00:00"],
        )


def test_compile_itchi_event_numpy_outputs() -> None:
    """
    Test event compilation using NumPy inputs.
    """
    snapshot_inputs = [
        _make_numpy_snapshot_input(np.array([5.0, 15.0, 25.0, 25.0])),
        _make_numpy_snapshot_input(np.array([5.0, 5.0, 15.0, 30.0])),
    ]

    compiled = compile_itchi_event(
        snapshot_inputs=snapshot_inputs,
        time_values=["2020-01-01T00:00", "2020-01-01T06:00"],
    )

    assert set(compiled) == {"snapshots", "metadata", "event_products"}

    snapshots = compiled["snapshots"]
    metadata = compiled["metadata"]
    event_products = compiled["event_products"]

    assert snapshots["ITCHI"].shape == (2, 4)
    assert len(metadata) == 2
    assert metadata[0]["time"] == "2020-01-01T00:00"
    assert metadata[0]["R_direct_source"] == "R34"
    assert metadata[0]["wind_hazard_source"] == "provided"

    expected_itchi = np.array(
        [
            [1.0, 0.625, 0.825, 0.75],
            [1.0, 0.5, 0.475, 1.0],
        ]
    )

    expected_max = np.array([1.0, 0.625, 0.825, 1.0])
    expected_acc = np.array([1.0, 0.8125, 0.908125, 1.0])

    np.testing.assert_allclose(snapshots["ITCHI"], expected_itchi)
    np.testing.assert_allclose(event_products["ITCHI_max"], expected_max)
    np.testing.assert_allclose(event_products["ITCHI_acc"], expected_acc)


def test_compile_itchi_event_without_event_products() -> None:
    """
    Test compilation without event-level products.
    """
    snapshot_inputs = [
        _make_numpy_snapshot_input(np.array([5.0, 15.0, 25.0, 25.0])),
        _make_numpy_snapshot_input(np.array([5.0, 5.0, 15.0, 30.0])),
    ]

    compiled = compile_itchi_event(
        snapshot_inputs=snapshot_inputs,
        include_event_products=False,
    )

    assert set(compiled) == {"snapshots", "metadata"}
    assert compiled["snapshots"]["ITCHI"].shape == (2, 4)
    assert len(compiled["metadata"]) == 2


def test_compile_itchi_event_with_internal_wind_profile() -> None:
    """
    Test compilation when wind hazard is computed internally.
    """
    snapshot_inputs = [
        {
            **_make_numpy_snapshot_input(np.array([5.0, 15.0, 25.0, 25.0])),
            "wind_hazard_normalized": None,
            "vmax_kt": 80.0,
            "rmw_km": 20.0,
        },
        {
            **_make_numpy_snapshot_input(np.array([5.0, 5.0, 15.0, 30.0])),
            "wind_hazard_normalized": None,
            "vmax_kt": 80.0,
            "rmw_km": 20.0,
        },
    ]

    compiled = compile_itchi_event(
        snapshot_inputs=snapshot_inputs,
        time_values=["2020-01-01T00:00", "2020-01-01T06:00"],
    )

    assert compiled["snapshots"]["ITCHI"].shape == (2, 4)
    assert compiled["metadata"][0]["wind_hazard_source"] == "radial_profile"
    assert compiled["metadata"][1]["wind_hazard_source"] == "radial_profile"


def test_compile_itchi_event_xarray_outputs() -> None:
    """
    Test event compilation using xarray inputs.
    """
    coords = {
        "lat": [0.0, 0.5],
        "lon": [0.0, 0.5],
    }

    precipitation_0 = xr.DataArray(
        data=[[5.0, 15.0], [25.0, 40.0]],
        dims=("lat", "lon"),
        coords=coords,
        name="precipitation",
    )

    precipitation_1 = xr.DataArray(
        data=[[5.0, 5.0], [15.0, 30.0]],
        dims=("lat", "lon"),
        coords=coords,
        name="precipitation",
    )

    snapshot_inputs = [
        _make_xarray_snapshot_input(precipitation_0),
        _make_xarray_snapshot_input(precipitation_1),
    ]

    compiled = compile_itchi_event(
        snapshot_inputs=snapshot_inputs,
        time_values=["2020-01-01T00:00", "2020-01-01T06:00"],
    )

    snapshots = compiled["snapshots"]
    metadata = compiled["metadata"]
    event_products = compiled["event_products"]

    assert isinstance(snapshots["ITCHI"], xr.DataArray)
    assert snapshots["ITCHI"].dims == ("time", "lat", "lon")
    assert snapshots["ITCHI"].shape == (2, 2, 2)

    assert len(metadata) == 2
    assert metadata[0]["time"] == "2020-01-01T00:00"
    assert metadata[0]["wind_hazard_source"] == "provided"

    assert isinstance(event_products["ITCHI_max"], xr.DataArray)
    assert isinstance(event_products["ITCHI_acc"], xr.DataArray)

    assert event_products["ITCHI_max"].dims == ("lat", "lon")
    assert event_products["ITCHI_acc"].dims == ("lat", "lon")

    assert float(event_products["ITCHI_max"].min()) >= 0.0
    assert float(event_products["ITCHI_max"].max()) <= 1.0
    assert float(event_products["ITCHI_acc"].min()) >= 0.0
    assert float(event_products["ITCHI_acc"].max()) <= 1.0
