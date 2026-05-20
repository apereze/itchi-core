"""
Tests for IBTrACS NetCDF adapter utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from itchi.ibtracs import (
    get_ibtracs_storm_ids,
    ibtracs_to_track_dataframe,
    read_ibtracs_track_dataframe,
    select_ibtracs_storm,
)


def _make_ibtracs_like_dataset() -> xr.Dataset:
    """
    Build a minimal IBTrACS-like Dataset for testing.
    """
    storm_ids = np.array(["AL012020", "EP182023"], dtype="U8")

    iso_time = np.array(
        [
            ["2020-06-01 00:00:00", "2020-06-01 06:00:00", ""],
            ["2023-10-24 00:00:00", "2023-10-24 06:00:00", "2023-10-24 12:00:00"],
        ],
        dtype="U19",
    )

    lat = np.array(
        [
            [15.0, 15.5, np.nan],
            [12.0, 12.5, 13.0],
        ]
    )

    lon = np.array(
        [
            [-100.0, -100.5, np.nan],
            [-98.0, -98.5, -99.0],
        ]
    )

    usa_wind = np.array(
        [
            [40.0, 45.0, np.nan],
            [80.0, 90.0, 100.0],
        ]
    )

    usa_pres = np.array(
        [
            [1000.0, 998.0, np.nan],
            [970.0, 960.0, 950.0],
        ]
    )

    usa_rmw = np.array(
        [
            [30.0, 25.0, np.nan],
            [20.0, 18.0, 16.0],
        ]
    )

    usa_r34 = np.array(
        [
            [
                [60.0, 50.0, 40.0, 70.0],
                [65.0, 55.0, 45.0, 75.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [45.0, 40.0, 35.0, 50.0],
                [50.0, 45.0, 40.0, 55.0],
                [55.0, 50.0, 45.0, 60.0],
            ],
        ]
    )

    return xr.Dataset(
        data_vars={
            "sid": (("storm",), storm_ids),
            "iso_time": (("storm", "date_time"), iso_time),
            "lat": (("storm", "date_time"), lat),
            "lon": (("storm", "date_time"), lon),
            "usa_wind": (("storm", "date_time"), usa_wind),
            "usa_pres": (("storm", "date_time"), usa_pres),
            "usa_rmw": (("storm", "date_time"), usa_rmw),
            "usa_r34": (("storm", "date_time", "quadrant"), usa_r34),
        },
        coords={
            "storm": [0, 1],
            "date_time": [0, 1, 2],
            "quadrant": [0, 1, 2, 3],
        },
    )


def test_get_ibtracs_storm_ids() -> None:
    """
    Test extraction of storm identifiers.
    """
    dataset = _make_ibtracs_like_dataset()

    assert get_ibtracs_storm_ids(dataset) == ["AL012020", "EP182023"]


def test_select_ibtracs_storm() -> None:
    """
    Test selecting one storm from an IBTrACS-like Dataset.
    """
    dataset = _make_ibtracs_like_dataset()

    selected = select_ibtracs_storm(dataset, storm_id="EP182023")

    assert "storm" not in selected.dims
    assert selected["lat"].dims == ("date_time",)
    np.testing.assert_allclose(selected["lat"].values, [12.0, 12.5, 13.0])


def test_ibtracs_to_track_dataframe() -> None:
    """
    Test conversion from IBTrACS-like Dataset to ITCHI track table.
    """
    dataset = _make_ibtracs_like_dataset()

    result = ibtracs_to_track_dataframe(
        dataset=dataset,
        storm_id="EP182023",
    )

    expected_columns = {
        "storm_id",
        "time",
        "lat",
        "lon",
        "vmax_kt",
        "pmin_hpa",
        "rmw_km",
        "r34_rne",
        "r34_rse",
        "r34_rsw",
        "r34_rnw",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == 3
    assert list(result["storm_id"]) == ["EP182023", "EP182023", "EP182023"]
    assert list(result["time"]) == list(
        pd.to_datetime(
            [
                "2023-10-24 00:00:00",
                "2023-10-24 06:00:00",
                "2023-10-24 12:00:00",
            ]
        )
    )
    assert list(result["lat"]) == [12.0, 12.5, 13.0]
    assert list(result["lon"]) == [-98.0, -98.5, -99.0]
    assert list(result["vmax_kt"]) == [80.0, 90.0, 100.0]
    assert list(result["pmin_hpa"]) == [970.0, 960.0, 950.0]
    assert list(result["rmw_km"]) == [20.0, 18.0, 16.0]
    assert list(result["r34_rne"]) == [45.0, 50.0, 55.0]
    assert list(result["r34_rse"]) == [40.0, 45.0, 50.0]
    assert list(result["r34_rsw"]) == [35.0, 40.0, 45.0]
    assert list(result["r34_rnw"]) == [50.0, 55.0, 60.0]


def test_ibtracs_to_track_dataframe_drops_missing_core_rows() -> None:
    """
    Test that rows with missing core values are removed by default.
    """
    dataset = _make_ibtracs_like_dataset()

    result = ibtracs_to_track_dataframe(
        dataset=dataset,
        storm_id="AL012020",
    )

    assert len(result) == 2
    assert result["time"].notna().all()
    assert result["lat"].notna().all()
    assert result["lon"].notna().all()


def test_ibtracs_to_track_dataframe_supports_custom_variable_map() -> None:
    """
    Test conversion using custom variable names.
    """
    dataset = _make_ibtracs_like_dataset().rename(
        {
            "sid": "storm_code",
            "iso_time": "valid_time",
            "usa_wind": "wind_kt",
        }
    )

    result = ibtracs_to_track_dataframe(
        dataset=dataset,
        storm_id="EP182023",
        variable_map={
            "storm_id": "storm_code",
            "time": "valid_time",
            "vmax_kt": "wind_kt",
        },
    )

    assert len(result) == 3
    assert list(result["vmax_kt"]) == [80.0, 90.0, 100.0]


def test_ibtracs_to_track_dataframe_rejects_missing_storm() -> None:
    """
    Test error when a storm is absent.
    """
    dataset = _make_ibtracs_like_dataset()

    with pytest.raises(ValueError):
        ibtracs_to_track_dataframe(
            dataset=dataset,
            storm_id="MISSING",
        )


def test_read_ibtracs_track_dataframe_from_netcdf(tmp_path: Path) -> None:
    """
    Test reading an IBTrACS-like NetCDF file from disk.
    """
    path = tmp_path / "ibtracs_test.nc"
    dataset = _make_ibtracs_like_dataset()
    dataset.to_netcdf(path)

    result = read_ibtracs_track_dataframe(
        path=path,
        storm_id="EP182023",
    )

    assert len(result) == 3
    assert list(result["storm_id"]) == ["EP182023", "EP182023", "EP182023"]
    assert result["time"].iloc[0] == pd.Timestamp("2023-10-24 00:00:00")
