"""
Tests for ITCHI input/output utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from itchi.io import (
    compiled_event_to_dataset,
    ensure_directory,
    ensure_parent_directory,
    infer_xarray_format,
    read_dataset,
    result_to_dataset,
    write_compiled_event,
    write_dataset,
    write_result,
)


def test_ensure_directory(tmp_path: Path) -> None:
    """
    Test directory creation.
    """
    directory = tmp_path / "a" / "b"

    result = ensure_directory(directory)

    assert result.exists()
    assert result.is_dir()


def test_ensure_parent_directory(tmp_path: Path) -> None:
    """
    Test parent-directory creation for a file path.
    """
    file_path = tmp_path / "a" / "b" / "file.nc"

    result = ensure_parent_directory(file_path)

    assert result == file_path
    assert file_path.parent.exists()
    assert file_path.parent.is_dir()


def test_infer_xarray_format_netcdf() -> None:
    """
    Test NetCDF format inference.
    """
    assert infer_xarray_format("output.nc") == "netcdf"
    assert infer_xarray_format("output.nc4") == "netcdf"


def test_infer_xarray_format_zarr() -> None:
    """
    Test Zarr format inference.
    """
    assert infer_xarray_format("output.zarr") == "zarr"


def test_infer_xarray_format_rejects_unknown_extension() -> None:
    """
    Test that unknown extensions fail.
    """
    with pytest.raises(ValueError):
        infer_xarray_format("output.txt")


def test_result_to_dataset_numpy() -> None:
    """
    Test conversion from NumPy-like result dictionary to Dataset.
    """
    result = {
        "ITCHI": np.array([[0.0, 0.5], [0.2, 1.0]]),
        "H_P": np.array([[0.0, 0.5], [0.2, 1.0]]),
    }

    dataset = result_to_dataset(
        result=result,
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        attrs={
            "storm_id": "TEST",
            "title": "Synthetic ITCHI output",
        },
    )

    assert isinstance(dataset, xr.Dataset)
    assert set(dataset.data_vars) == {"ITCHI", "H_P"}
    assert dataset["ITCHI"].dims == ("lat", "lon")
    assert dataset.attrs["storm_id"] == "TEST"


def test_result_to_dataset_requires_dims_for_numpy_arrays() -> None:
    """
    Test that dims are required for non-scalar NumPy-like values.
    """
    result = {
        "ITCHI": np.array([0.0, 0.5]),
    }

    with pytest.raises(ValueError):
        result_to_dataset(result)


def test_result_to_dataset_xarray() -> None:
    """
    Test conversion from xarray DataArray result dictionary to Dataset.
    """
    itchi = xr.DataArray(
        data=[[0.0, 0.5], [0.2, 1.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="ITCHI",
    )

    dataset = result_to_dataset({"ITCHI": itchi})

    assert isinstance(dataset, xr.Dataset)
    assert "ITCHI" in dataset
    assert dataset["ITCHI"].dims == ("lat", "lon")
    xr.testing.assert_allclose(dataset["ITCHI"], itchi)


def test_result_to_dataset_include_exclude() -> None:
    """
    Test include and exclude filters.
    """
    result = {
        "ITCHI": np.array([0.0, 1.0]),
        "H_P": np.array([0.0, 1.0]),
        "H_W": np.array([0.0, 0.5]),
    }

    dataset = result_to_dataset(
        result=result,
        dims=("point",),
        include=["ITCHI", "H_P"],
        exclude=["H_P"],
    )

    assert set(dataset.data_vars) == {"ITCHI"}


def test_write_and_read_dataset_netcdf(tmp_path: Path) -> None:
    """
    Test writing and reading a NetCDF Dataset.
    """
    dataset = xr.Dataset(
        data_vars={
            "ITCHI": (("lat", "lon"), [[0.0, 0.5], [0.2, 1.0]]),
        },
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
    )

    output_path = tmp_path / "itchi.nc"

    written_path = write_dataset(dataset, output_path)

    assert written_path.exists()

    loaded = read_dataset(output_path)

    try:
        xr.testing.assert_allclose(loaded["ITCHI"], dataset["ITCHI"])
    finally:
        loaded.close()


def test_write_dataset_respects_overwrite_false(tmp_path: Path) -> None:
    """
    Test that overwrite=False prevents replacing existing files.
    """
    dataset = xr.Dataset(
        data_vars={
            "ITCHI": ("point", [0.0, 1.0]),
        }
    )

    output_path = tmp_path / "itchi.nc"

    write_dataset(dataset, output_path)

    with pytest.raises(FileExistsError):
        write_dataset(dataset, output_path, overwrite=False)


def test_write_result_netcdf(tmp_path: Path) -> None:
    """
    Test writing a result dictionary directly to NetCDF.
    """
    result = {
        "ITCHI": np.array([[0.0, 0.5], [0.2, 1.0]]),
        "H_P": np.array([[0.0, 0.5], [0.2, 1.0]]),
    }

    output_path = tmp_path / "outputs" / "snapshot.nc"

    written_path = write_result(
        result=result,
        path=output_path,
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        attrs={
            "storm_id": "TEST",
        },
    )

    assert written_path.exists()

    loaded = read_dataset(written_path)

    try:
        assert set(loaded.data_vars) == {"ITCHI", "H_P"}
        assert loaded.attrs["storm_id"] == "TEST"
    finally:
        loaded.close()


def test_compiled_event_to_dataset_xarray() -> None:
    """
    Test conversion from compiled event dictionary to xarray Dataset.
    """
    itchi = xr.DataArray(
        data=[
            [[0.0, 0.5], [0.2, 1.0]],
            [[0.1, 0.7], [0.4, 0.9]],
        ],
        dims=("time", "lat", "lon"),
        coords={
            "time": ["2020-01-01T00:00", "2020-01-01T06:00"],
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="ITCHI",
    )

    itchi_max = itchi.max(dim="time")
    itchi_max.name = "ITCHI_max"

    compiled_event = {
        "snapshots": {
            "ITCHI": itchi,
        },
        "metadata": [
            {
                "time": "2020-01-01T00:00",
                "wind_hazard_source": "provided",
            },
            {
                "time": "2020-01-01T06:00",
                "wind_hazard_source": "provided",
            },
        ],
        "event_products": {
            "ITCHI_max": itchi_max,
        },
    }

    dataset = compiled_event_to_dataset(
        compiled_event=compiled_event,
        attrs={
            "storm_id": "TEST",
            "title": "Synthetic compiled ITCHI event",
        },
    )

    assert isinstance(dataset, xr.Dataset)
    assert set(dataset.data_vars) == {"ITCHI", "ITCHI_max"}
    assert dataset["ITCHI"].dims == ("time", "lat", "lon")
    assert dataset["ITCHI_max"].dims == ("lat", "lon")
    assert dataset.attrs["storm_id"] == "TEST"
    assert "snapshot_metadata_json" in dataset.attrs
    assert dataset.attrs["n_snapshot_metadata_records"] == 2


def test_compiled_event_to_dataset_rejects_missing_snapshots() -> None:
    """
    Test that compiled event conversion requires snapshots.
    """
    with pytest.raises(KeyError):
        compiled_event_to_dataset({"metadata": []})


def test_write_compiled_event_netcdf(tmp_path: Path) -> None:
    """
    Test writing a compiled ITCHI event to NetCDF.
    """
    itchi = xr.DataArray(
        data=[
            [[0.0, 0.5], [0.2, 1.0]],
            [[0.1, 0.7], [0.4, 0.9]],
        ],
        dims=("time", "lat", "lon"),
        coords={
            "time": ["2020-01-01T00:00", "2020-01-01T06:00"],
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
        name="ITCHI",
    )

    itchi_max = itchi.max(dim="time")
    itchi_max.name = "ITCHI_max"

    compiled_event = {
        "snapshots": {
            "ITCHI": itchi,
        },
        "metadata": [
            {
                "time": "2020-01-01T00:00",
                "R_direct_source": "R34",
            },
            {
                "time": "2020-01-01T06:00",
                "R_direct_source": "R34",
            },
        ],
        "event_products": {
            "ITCHI_max": itchi_max,
        },
    }

    output_path = tmp_path / "outputs" / "compiled_event.nc"

    written_path = write_compiled_event(
        compiled_event=compiled_event,
        path=output_path,
        attrs={
            "storm_id": "TEST",
        },
    )

    assert written_path.exists()

    loaded = read_dataset(written_path)

    try:
        assert set(loaded.data_vars) == {"ITCHI", "ITCHI_max"}
        assert loaded["ITCHI"].dims == ("time", "lat", "lon")
        assert loaded["ITCHI_max"].dims == ("lat", "lon")
        assert loaded.attrs["storm_id"] == "TEST"
        assert "snapshot_metadata_json" in loaded.attrs
    finally:
        loaded.close()
