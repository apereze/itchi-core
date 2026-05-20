"""
Tests for MSWEP NetCDF adapter utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from itchi.mswep import (
    extract_mswep_precipitation,
    filter_mswep_synoptic_snapshots,
    get_mswep_lon_lat,
    infer_mswep_lat_lon_names,
    prepare_mswep_precipitation,
    read_mswep_dataset,
    read_mswep_precipitation,
    select_mswep_precipitation_snapshot,
    standardize_mswep_time,
    subset_mswep_domain,
)


def _make_mswep_dataset(time_name: str = "time") -> xr.Dataset:
    """
    Build a small synthetic MSWEP-like Dataset.
    """
    times = np.array(
        [
            "2020-10-22T00:00:00",
            "2020-10-22T03:00:00",
            "2020-10-22T06:00:00",
            "2020-10-22T09:00:00",
            "2020-10-22T12:00:00",
        ],
        dtype="datetime64[ns]",
    )

    lat = np.array([17.0, 16.0, 15.0])
    lon = np.array([-101.0, -100.0, -99.0, -98.0])
    values = np.arange(5 * 3 * 4, dtype=float).reshape(5, 3, 4)

    precipitation = xr.DataArray(
        values,
        dims=(time_name, "lat", "lon"),
        coords={
            time_name: times,
            "lat": lat,
            "lon": lon,
        },
        name="precipitation",
        attrs={"units": "mm"},
    )

    return xr.Dataset({"precipitation": precipitation})


def test_read_mswep_dataset_from_netcdf(tmp_path: Path) -> None:
    """
    Test opening a synthetic MSWEP NetCDF file.
    """
    dataset = _make_mswep_dataset()
    path = tmp_path / "mswep_test.nc"
    dataset.to_netcdf(path)

    result = read_mswep_dataset(path)

    assert "precipitation" in result.data_vars
    assert result["precipitation"].shape == (5, 3, 4)


def test_read_mswep_dataset_rejects_missing_file(tmp_path: Path) -> None:
    """
    Test missing MSWEP file handling.
    """
    with pytest.raises(FileNotFoundError):
        read_mswep_dataset(tmp_path / "missing.nc")


def test_extract_mswep_precipitation_by_candidate_name() -> None:
    """
    Test precipitation extraction using default candidate names.
    """
    dataset = _make_mswep_dataset()

    result = extract_mswep_precipitation(dataset)

    assert isinstance(result, xr.DataArray)
    assert result.name == "precipitation"


def test_extract_mswep_precipitation_explicit_variable() -> None:
    """
    Test precipitation extraction with an explicit variable name.
    """
    dataset = xr.Dataset({"precip": _make_mswep_dataset()["precipitation"]})

    result = extract_mswep_precipitation(
        dataset,
        precipitation_variable="precip",
    )

    assert result.name == "precip"


def test_standardize_mswep_time_renames_time_to_valid_time() -> None:
    """
    Test time-coordinate standardization.
    """
    dataset = _make_mswep_dataset(time_name="time")

    result = standardize_mswep_time(dataset)

    assert "valid_time" in result.dims
    assert "time" not in result.dims


def test_standardize_mswep_time_preserves_existing_valid_time() -> None:
    """
    Test no-op behavior when valid_time already exists.
    """
    dataset = _make_mswep_dataset(time_name="valid_time")

    result = standardize_mswep_time(dataset)

    assert result is dataset


def test_infer_mswep_lat_lon_names() -> None:
    """
    Test latitude/longitude coordinate-name inference.
    """
    dataset = _make_mswep_dataset()

    lat_name, lon_name = infer_mswep_lat_lon_names(dataset)

    assert lat_name == "lat"
    assert lon_name == "lon"


def test_get_mswep_lon_lat_returns_2d_grids() -> None:
    """
    Test extraction and broadcasting of lon/lat grids.
    """
    dataset = _make_mswep_dataset()

    lon, lat = get_mswep_lon_lat(dataset)

    assert lon.dims == ("lat", "lon")
    assert lat.dims == ("lat", "lon")
    assert lon.shape == (3, 4)
    assert lat.shape == (3, 4)
    assert float(lon.isel(lat=0, lon=1)) == pytest.approx(-100.0)
    assert float(lat.isel(lat=1, lon=0)) == pytest.approx(16.0)


def test_get_mswep_lon_lat_can_return_1d_coordinates() -> None:
    """
    Test extraction of original 1D coordinates.
    """
    dataset = _make_mswep_dataset()

    lon, lat = get_mswep_lon_lat(dataset, as_2d=False)

    assert lon.dims == ("lon",)
    assert lat.dims == ("lat",)


def test_subset_mswep_domain_handles_descending_latitude() -> None:
    """
    Test spatial subsetting with descending latitude coordinates.
    """
    dataset = _make_mswep_dataset()

    subset = subset_mswep_domain(
        dataset,
        lon_min=-100.5,
        lon_max=-98.5,
        lat_min=15.5,
        lat_max=17.5,
    )

    np.testing.assert_array_equal(subset["lon"].values, np.array([-100.0, -99.0]))
    np.testing.assert_array_equal(subset["lat"].values, np.array([17.0, 16.0]))


def test_subset_mswep_domain_converts_negative_bounds_for_0360_longitude() -> None:
    """
    Test conversion of negative longitude bounds for 0-360 datasets.
    """
    dataset = _make_mswep_dataset().assign_coords(
        lon=np.array([258.0, 259.0, 260.0, 261.0])
    )

    subset = subset_mswep_domain(
        dataset,
        lon_min=-101.5,
        lon_max=-99.5,
        lat_min=15.5,
        lat_max=17.5,
    )

    np.testing.assert_array_equal(subset["lon"].values, np.array([259.0, 260.0]))


def test_filter_mswep_synoptic_snapshots_does_not_accumulate() -> None:
    """
    Test synoptic filtering without temporal accumulation.
    """
    dataset = _make_mswep_dataset()

    result = filter_mswep_synoptic_snapshots(dataset)

    expected_times = np.array(
        [
            "2020-10-22T00:00:00",
            "2020-10-22T06:00:00",
            "2020-10-22T12:00:00",
        ],
        dtype="datetime64[ns]",
    )

    np.testing.assert_array_equal(result["valid_time"].values, expected_times)
    np.testing.assert_allclose(
        result["precipitation"].values,
        dataset["precipitation"].isel(time=[0, 2, 4]).values,
    )


def test_prepare_mswep_precipitation_returns_synoptic_dataarray() -> None:
    """
    Test preparation of an MSWEP precipitation DataArray.
    """
    dataset = _make_mswep_dataset()

    result = prepare_mswep_precipitation(dataset)

    assert isinstance(result, xr.DataArray)
    assert result.name == "precipitation"
    assert result.dims == ("valid_time", "lat", "lon")
    assert result.sizes["valid_time"] == 3


def test_select_mswep_precipitation_snapshot_exact() -> None:
    """
    Test selecting one MSWEP snapshot by exact valid time.
    """
    dataset = _make_mswep_dataset()

    result = select_mswep_precipitation_snapshot(
        dataset,
        target_time="2020-10-22T06:00:00",
    )

    expected = dataset["precipitation"].isel(time=2, drop=True)
    expected = expected.rename({"time": "valid_time"}) if "time" in expected.dims else expected

    xr.testing.assert_allclose(result, expected)
    assert "valid_time" not in result.dims


def test_select_mswep_precipitation_snapshot_rejects_non_synoptic_target() -> None:
    """
    Test rejection of non-synoptic target times by default.
    """
    dataset = _make_mswep_dataset()

    with pytest.raises(ValueError):
        select_mswep_precipitation_snapshot(
            dataset,
            target_time="2020-10-22T06:30:00",
            method="nearest",
            tolerance="45min",
        )


def test_read_mswep_precipitation_convenience_wrapper(tmp_path: Path) -> None:
    """
    Test opening and preparing precipitation through one convenience wrapper.
    """
    dataset = _make_mswep_dataset()
    path = tmp_path / "mswep_test.nc"
    dataset.to_netcdf(path)

    result = read_mswep_precipitation(path)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("valid_time", "lat", "lon")
    assert result.sizes["valid_time"] == 3
