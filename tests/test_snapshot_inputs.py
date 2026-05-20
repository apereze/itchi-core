"""
Tests for ITCHI snapshot-input builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from itchi.pipeline import compute_itchi_snapshot_from_grid
from itchi.snapshot_inputs import (
    build_event_snapshot_inputs_from_tables,
    build_snapshot_input_from_tables,
    extract_raw_quadrant_values,
)


def _make_precipitation_data() -> xr.DataArray:
    """
    Build synthetic precipitation data with valid_time dimension.
    """
    times = np.array(
        [
            "2020-09-01T00:00:00",
            "2020-09-01T06:00:00",
        ],
        dtype="datetime64[ns]",
    )

    return xr.DataArray(
        data=[
            [[5.0, 15.0], [25.0, 35.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dims=("valid_time", "y", "x"),
        coords={
            "valid_time": times,
            "y": [0, 1],
            "x": [0, 1],
        },
        name="precipitation",
    )


def _make_lon_lat() -> tuple[xr.DataArray, xr.DataArray]:
    """
    Build small synthetic lon/lat grids.
    """
    coords = {
        "y": [0, 1],
        "x": [0, 1],
    }

    lon = xr.DataArray(
        data=[[0.0, 0.5], [0.0, 0.5]],
        dims=("y", "x"),
        coords=coords,
        name="lon",
    )

    lat = xr.DataArray(
        data=[[0.0, 0.0], [0.5, 0.5]],
        dims=("y", "x"),
        coords=coords,
        name="lat",
    )

    return lon, lat


def _make_track_df() -> pd.DataFrame:
    """
    Build synthetic standardized track table.
    """
    return pd.DataFrame(
        {
            "storm_id": ["TEST", "TEST"],
            "time": pd.to_datetime(
                [
                    "2020-09-01T00:00:00",
                    "2020-09-01T06:00:00",
                ]
            ),
            "lat": [0.0, 0.1],
            "lon": [0.0, 0.1],
            "vmax_kt": [80.0, 75.0],
            "rmw_km": [25.0, 30.0],
            "r34_rne": [45.0, 40.0],
            "r34_rse": [40.0, 35.0],
            "r34_rsw": [35.0, 30.0],
            "r34_rnw": [50.0, 45.0],
        }
    )


def _make_rocloud_df() -> pd.DataFrame:
    """
    Build synthetic standardized ROCLOUD table.
    """
    return pd.DataFrame(
        {
            "storm_id": ["TEST", "TEST"],
            "time": pd.to_datetime(
                [
                    "2020-09-01T00:00:00",
                    "2020-09-01T06:00:00",
                ]
            ),
            "rocloud_rne": [250.0, 240.0],
            "rocloud_rse": [230.0, 220.0],
            "rocloud_rsw": [220.0, 210.0],
            "rocloud_rnw": [260.0, 250.0],
        }
    )


def test_extract_raw_quadrant_values_uses_canonical_keys() -> None:
    """
    Test raw quadrant extraction with canonical ITCHI keys.
    """
    record = {
        "ne": 1.0,
        "se": 2.0,
        "sw": 3.0,
        "nw": 4.0,
    }

    values = extract_raw_quadrant_values(
        record=record,
        column_map={
            "NE": "ne",
            "SE": "se",
            "SW": "sw",
            "NW": "nw",
        },
    )

    assert values == {
        "RNE": 1.0,
        "RSE": 2.0,
        "RSW": 3.0,
        "RNW": 4.0,
    }


def test_build_snapshot_input_from_tables() -> None:
    """
    Test building one pipeline-compatible snapshot input.
    """
    precipitation = _make_precipitation_data()
    lon, lat = _make_lon_lat()

    snapshot_input = build_snapshot_input_from_tables(
        precipitation_data=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        track_df=_make_track_df(),
        rocloud_df=_make_rocloud_df(),
        storm_id="TEST",
        target_time="2020-09-01T00:00:00",
    )

    assert snapshot_input["precipitation"].dims == ("y", "x")
    assert snapshot_input["center_lon"] == pytest.approx(0.0)
    assert snapshot_input["center_lat"] == pytest.approx(0.0)
    assert snapshot_input["vmax_kt"] == pytest.approx(80.0)
    assert snapshot_input["rmw_km"] == pytest.approx(25.0)
    assert snapshot_input["r34_unit"] == "nm"
    assert snapshot_input["rocloud_unit"] == "km"

    assert set(snapshot_input["r34_by_quadrant"]) == {
        "RNE",
        "RSE",
        "RSW",
        "RNW",
    }

    assert set(snapshot_input["rocloud_by_quadrant"]) == {
        "RNE",
        "RSE",
        "RSW",
        "RNW",
    }


def test_snapshot_input_is_compatible_with_pipeline() -> None:
    """
    Test that the constructed dictionary can be passed to the pipeline.
    """
    precipitation = _make_precipitation_data()
    lon, lat = _make_lon_lat()

    snapshot_input = build_snapshot_input_from_tables(
        precipitation_data=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        track_df=_make_track_df(),
        rocloud_df=_make_rocloud_df(),
        storm_id="TEST",
        target_time="2020-09-01T00:00:00",
    )

    result = compute_itchi_snapshot_from_grid(**snapshot_input)

    assert "ITCHI" in result
    assert isinstance(result["ITCHI"], xr.DataArray)
    assert result["ITCHI"].dims == ("y", "x")
    assert float(result["ITCHI"].min()) >= 0.0
    assert float(result["ITCHI"].max()) <= 1.0


def test_build_event_snapshot_inputs_from_tables() -> None:
    """
    Test building multiple snapshot inputs for one event.
    """
    precipitation = _make_precipitation_data()
    lon, lat = _make_lon_lat()

    target_times = [
        "2020-09-01T00:00:00",
        "2020-09-01T06:00:00",
    ]

    snapshot_inputs = build_event_snapshot_inputs_from_tables(
        precipitation_data=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        track_df=_make_track_df(),
        rocloud_df=_make_rocloud_df(),
        storm_id="TEST",
        target_times=target_times,
    )

    assert len(snapshot_inputs) == 2
    assert snapshot_inputs[0]["center_lon"] == pytest.approx(0.0)
    assert snapshot_inputs[1]["center_lon"] == pytest.approx(0.1)


def test_build_snapshot_input_requires_wind_metadata_when_no_wind_field() -> None:
    """
    Test that missing wind metadata are rejected if no wind field is provided.
    """
    precipitation = _make_precipitation_data()
    lon, lat = _make_lon_lat()

    track_df = _make_track_df().drop(columns=["vmax_kt"])

    with pytest.raises(ValueError):
        build_snapshot_input_from_tables(
            precipitation_data=precipitation,
            q90=10.0,
            q95=20.0,
            q99=30.0,
            lon=lon,
            lat=lat,
            track_df=track_df,
            rocloud_df=_make_rocloud_df(),
            storm_id="TEST",
            target_time="2020-09-01T00:00:00",
        )


def test_build_snapshot_input_accepts_precomputed_wind_field() -> None:
    """
    Test that precomputed wind hazard allows missing Vmax/RMW metadata.
    """
    precipitation = _make_precipitation_data()
    lon, lat = _make_lon_lat()

    wind = xr.DataArray(
        data=[[1.0, 0.5], [0.2, 0.0]],
        dims=("y", "x"),
        coords=lon.coords,
        name="V_star",
    )

    track_df = _make_track_df().drop(columns=["vmax_kt", "rmw_km"])

    snapshot_input = build_snapshot_input_from_tables(
        precipitation_data=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        track_df=track_df,
        rocloud_df=_make_rocloud_df(),
        storm_id="TEST",
        target_time="2020-09-01T00:00:00",
        wind_hazard_normalized=wind,
    )

    assert snapshot_input["wind_hazard_normalized"] is wind
    assert snapshot_input["vmax_kt"] is None
    assert snapshot_input["rmw_km"] is None


def test_snapshot_input_copies_missing_lon_lat_coords_from_precipitation() -> None:
    """
    Test that lon/lat inherit missing dimension coordinates from precipitation.

    This prevents quality-control failures where ITCHI has y/x coordinates
    but geometry-derived fields such as radius_km do not.
    """
    precipitation = _make_precipitation_data()

    lon = xr.DataArray(
        data=[[0.0, 0.5], [0.0, 0.5]],
        dims=("y", "x"),
        name="lon",
    )

    lat = xr.DataArray(
        data=[[0.0, 0.0], [0.5, 0.5]],
        dims=("y", "x"),
        name="lat",
    )

    snapshot_input = build_snapshot_input_from_tables(
        precipitation_data=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        track_df=_make_track_df(),
        rocloud_df=_make_rocloud_df(),
        storm_id="TEST",
        target_time="2020-09-01T00:00:00",
    )

    assert "y" in snapshot_input["lon"].coords
    assert "x" in snapshot_input["lon"].coords
    assert "y" in snapshot_input["lat"].coords
    assert "x" in snapshot_input["lat"].coords

    result = compute_itchi_snapshot_from_grid(**snapshot_input)

    assert "ITCHI" in result
    assert "y" in result["radius_km"].coords
    assert "x" in result["radius_km"].coords
    assert result["ITCHI"].dims == ("y", "x")
    assert result["radius_km"].dims == ("y", "x")
