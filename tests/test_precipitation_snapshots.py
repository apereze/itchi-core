"""
Tests for precipitation snapshot utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itchi.precipitation_snapshots import (
    build_synoptic_time_mask,
    extract_precipitation_variable,
    filter_synoptic_snapshots,
    infer_time_coordinate,
    is_synoptic_time,
    prepare_precipitation_snapshot_for_pipeline,
    select_precipitation_snapshot,
)


def _make_precipitation_dataarray() -> xr.DataArray:
    """
    Build a small 3-hourly precipitation DataArray for testing.
    """
    valid_times = np.array(
        [
            "2020-10-22T00:00:00",
            "2020-10-22T03:00:00",
            "2020-10-22T06:00:00",
            "2020-10-22T09:00:00",
            "2020-10-22T12:00:00",
        ],
        dtype="datetime64[ns]",
    )

    data = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)

    return xr.DataArray(
        data=data,
        dims=("valid_time", "lat", "lon"),
        coords={
            "valid_time": valid_times,
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="precipitation",
    )


def test_infer_time_coordinate_prefers_valid_time() -> None:
    """
    Test automatic detection of valid_time.
    """
    precipitation = _make_precipitation_dataarray()

    assert infer_time_coordinate(precipitation) == "valid_time"


def test_is_synoptic_time() -> None:
    """
    Test synoptic time detection.
    """
    assert is_synoptic_time("2020-10-22T00:00:00")
    assert is_synoptic_time("2020-10-22T06:00:00")
    assert not is_synoptic_time("2020-10-22T03:00:00")
    assert not is_synoptic_time("2020-10-22T06:30:00")


def test_build_synoptic_time_mask() -> None:
    """
    Test Boolean mask construction for synoptic times.
    """
    valid_times = np.array(
        [
            "2020-10-22T00:00:00",
            "2020-10-22T03:00:00",
            "2020-10-22T06:00:00",
        ],
        dtype="datetime64[ns]",
    )

    mask = build_synoptic_time_mask(valid_times)

    expected = np.array([True, False, True])

    np.testing.assert_array_equal(mask, expected)


def test_filter_synoptic_snapshots_does_not_accumulate() -> None:
    """
    Test that synoptic filtering keeps snapshots without summing them.
    """
    precipitation = _make_precipitation_dataarray()

    filtered = filter_synoptic_snapshots(precipitation)

    expected_times = np.array(
        [
            "2020-10-22T00:00:00",
            "2020-10-22T06:00:00",
            "2020-10-22T12:00:00",
        ],
        dtype="datetime64[ns]",
    )

    np.testing.assert_array_equal(filtered["valid_time"].values, expected_times)

    expected_values = precipitation.isel(valid_time=[0, 2, 4]).values

    np.testing.assert_allclose(filtered.values, expected_values)


def test_select_precipitation_snapshot_exact() -> None:
    """
    Test exact snapshot selection.
    """
    precipitation = _make_precipitation_dataarray()

    selected = select_precipitation_snapshot(
        data=precipitation,
        target_time="2020-10-22T06:00:00",
        method="exact",
    )

    expected = precipitation.isel(valid_time=2, drop=True)

    xr.testing.assert_allclose(selected, expected)
    assert "valid_time" not in selected.dims


def test_select_precipitation_snapshot_nearest_within_tolerance() -> None:
    """
    Test nearest snapshot selection within explicit tolerance.
    """
    precipitation = _make_precipitation_dataarray()

    selected = select_precipitation_snapshot(
        data=precipitation,
        target_time="2020-10-22T06:20:00",
        method="nearest",
        tolerance="30min",
    )

    expected = precipitation.isel(valid_time=2, drop=True)

    xr.testing.assert_allclose(selected, expected)


def test_select_precipitation_snapshot_nearest_requires_tolerance() -> None:
    """
    Test that nearest selection requires explicit tolerance.
    """
    precipitation = _make_precipitation_dataarray()

    with pytest.raises(ValueError):
        select_precipitation_snapshot(
            data=precipitation,
            target_time="2020-10-22T06:20:00",
            method="nearest",
        )


def test_select_precipitation_snapshot_nearest_outside_tolerance() -> None:
    """
    Test that nearest selection fails outside tolerance.
    """
    precipitation = _make_precipitation_dataarray()

    with pytest.raises(ValueError):
        select_precipitation_snapshot(
            data=precipitation,
            target_time="2020-10-22T06:45:00",
            method="nearest",
            tolerance="30min",
        )


def test_extract_precipitation_variable_from_dataset() -> None:
    """
    Test precipitation variable extraction from Dataset.
    """
    precipitation = _make_precipitation_dataarray()
    dataset = xr.Dataset({"precipitation": precipitation})

    extracted = extract_precipitation_variable(
        data=dataset,
        precipitation_variable="precipitation",
    )

    xr.testing.assert_allclose(extracted, precipitation)


def test_extract_precipitation_variable_requires_name_for_multivar_dataset() -> None:
    """
    Test that multi-variable Datasets require an explicit variable name.
    """
    precipitation = _make_precipitation_dataarray()
    dataset = xr.Dataset(
        {
            "precipitation": precipitation,
            "other_variable": precipitation + 1.0,
        }
    )

    with pytest.raises(ValueError):
        extract_precipitation_variable(dataset)


def test_prepare_precipitation_snapshot_for_pipeline_from_dataset() -> None:
    """
    Test pipeline-ready snapshot preparation from Dataset.
    """
    precipitation = _make_precipitation_dataarray()
    dataset = xr.Dataset({"precipitation": precipitation})

    selected = prepare_precipitation_snapshot_for_pipeline(
        data=dataset,
        target_time="2020-10-22T06:00:00",
        precipitation_variable="precipitation",
    )

    expected = precipitation.isel(valid_time=2, drop=True)

    xr.testing.assert_allclose(selected, expected)


def test_prepare_precipitation_snapshot_rejects_non_synoptic_target() -> None:
    """
    Test that non-synoptic target times are rejected by default.
    """
    precipitation = _make_precipitation_dataarray()

    with pytest.raises(ValueError):
        prepare_precipitation_snapshot_for_pipeline(
            data=precipitation,
            target_time="2020-10-22T06:20:00",
            method="nearest",
            tolerance="30min",
        )
