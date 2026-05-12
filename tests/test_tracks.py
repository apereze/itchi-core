"""
Tests for tropical cyclone track utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itchi.tracks import (
    extract_quadrant_radii,
    filter_synoptic_times,
    get_track_snapshot,
    has_complete_quadrant_radii,
    read_track_table,
    standardize_column_names,
    standardize_track_dataframe,
    validate_track_columns,
)


def test_standardize_column_names() -> None:
    """
    Test standard column-name normalization.
    """
    df = pd.DataFrame(
        {
            " Storm ID ": ["AL012020"],
            "ISO-Time": ["2020-06-01 00:00"],
            "Latitude": [15.0],
            "Longitude": [-100.0],
        }
    )

    result = standardize_column_names(df)

    assert list(result.columns) == [
        "storm_id",
        "iso_time",
        "latitude",
        "longitude",
    ]


def test_standardize_track_dataframe_with_column_map() -> None:
    """
    Test full track standardization with source-to-canonical column mapping.
    """
    df = pd.DataFrame(
        {
            "SID": ["AL012020", "AL012020"],
            "ISO_TIME": ["2020-06-01 06:00", "2020-06-01 00:00"],
            "LAT": [15.5, 15.0],
            "LON": [-100.5, -100.0],
            "USA_WIND": [45, 40],
            "USA_PRES": [998, 1000],
        }
    )

    result = standardize_track_dataframe(
        df,
        column_map={
            "storm_id": "sid",
            "time": "iso_time",
            "lat": "lat",
            "lon": "lon",
            "vmax_kt": "usa_wind",
            "pmin_hpa": "usa_pres",
        },
    )

    assert list(result["time"]) == [
        pd.Timestamp("2020-06-01 00:00"),
        pd.Timestamp("2020-06-01 06:00"),
    ]

    assert list(result["vmax_kt"]) == [40, 45]
    assert list(result["pmin_hpa"]) == [1000, 998]


def test_validate_track_columns_rejects_missing_columns() -> None:
    """
    Test that required missing columns raise an error.
    """
    df = pd.DataFrame(
        {
            "storm_id": ["AL012020"],
            "time": ["2020-06-01 00:00"],
        }
    )

    with pytest.raises(KeyError):
        validate_track_columns(df)


def test_filter_synoptic_times() -> None:
    """
    Test filtering to 00, 06, 12 and 18 UTC.
    """
    df = pd.DataFrame(
        {
            "storm_id": ["A"] * 5,
            "time": pd.to_datetime(
                [
                    "2020-06-01 00:00",
                    "2020-06-01 03:00",
                    "2020-06-01 06:00",
                    "2020-06-01 09:00",
                    "2020-06-01 12:00",
                ]
            ),
            "lat": [0, 1, 2, 3, 4],
            "lon": [0, 1, 2, 3, 4],
        }
    )

    result = filter_synoptic_times(df)

    assert list(result["time"].dt.hour) == [0, 6, 12]


def test_get_track_snapshot_exact() -> None:
    """
    Test exact track snapshot selection.
    """
    df = pd.DataFrame(
        {
            "storm_id": ["AL012020", "AL012020"],
            "time": pd.to_datetime(["2020-06-01 00:00", "2020-06-01 06:00"]),
            "lat": [15.0, 15.5],
            "lon": [-100.0, -100.5],
        }
    )

    row = get_track_snapshot(
        df,
        storm_id="AL012020",
        time="2020-06-01 06:00",
    )

    assert row["lat"] == pytest.approx(15.5)
    assert row["lon"] == pytest.approx(-100.5)


def test_get_track_snapshot_nearest_with_tolerance() -> None:
    """
    Test nearest track snapshot selection within tolerance.
    """
    df = pd.DataFrame(
        {
            "storm_id": ["AL012020", "AL012020"],
            "time": pd.to_datetime(["2020-06-01 00:00", "2020-06-01 06:00"]),
            "lat": [15.0, 15.5],
            "lon": [-100.0, -100.5],
        }
    )

    row = get_track_snapshot(
        df,
        storm_id="AL012020",
        time="2020-06-01 05:30",
        tolerance_hours=1.0,
    )

    assert row["time"] == pd.Timestamp("2020-06-01 06:00")


def test_get_track_snapshot_rejects_outside_tolerance() -> None:
    """
    Test that nearest selection fails outside tolerance.
    """
    df = pd.DataFrame(
        {
            "storm_id": ["AL012020"],
            "time": pd.to_datetime(["2020-06-01 00:00"]),
            "lat": [15.0],
            "lon": [-100.0],
        }
    )

    with pytest.raises(ValueError):
        get_track_snapshot(
            df,
            storm_id="AL012020",
            time="2020-06-01 03:00",
            tolerance_hours=1.0,
        )


def test_extract_quadrant_radii_from_track_row() -> None:
    """
    Test extraction and conversion of quadrant radii.
    """
    row = pd.Series(
        {
            "r34_ne": 60.0,
            "r34_se": 50.0,
            "r34_sw": 40.0,
            "r34_nw": 70.0,
        }
    )

    radii = extract_quadrant_radii(
        row,
        column_map={
            "NE": "r34_ne",
            "SE": "r34_se",
            "SW": "r34_sw",
            "NW": "r34_nw",
        },
        input_unit="nm",
    )

    assert radii["RNE"] == pytest.approx(111.12)
    assert radii["RSE"] == pytest.approx(92.60)
    assert radii["RSW"] == pytest.approx(74.08)
    assert radii["RNW"] == pytest.approx(129.64)


def test_has_complete_quadrant_radii() -> None:
    """
    Test complete quadrant radius detection.
    """
    row = {
        "r34_rne": 60.0,
        "r34_rse": 50.0,
        "r34_rsw": 40.0,
        "r34_rnw": 70.0,
    }

    assert has_complete_quadrant_radii(row)


def test_has_complete_quadrant_radii_rejects_missing_value() -> None:
    """
    Test incomplete quadrant radius detection.
    """
    row = {
        "r34_rne": 60.0,
        "r34_rse": 50.0,
        "r34_rsw": np.nan,
        "r34_rnw": 70.0,
    }

    assert not has_complete_quadrant_radii(row)


def test_read_track_table_csv(tmp_path: Path) -> None:
    """
    Test reading track table from CSV.
    """
    path = tmp_path / "tracks.csv"

    df = pd.DataFrame(
        {
            "storm_id": ["AL012020"],
            "time": ["2020-06-01 00:00"],
            "lat": [15.0],
            "lon": [-100.0],
        }
    )

    df.to_csv(path, index=False)

    result = read_track_table(path)

    assert list(result.columns) == ["storm_id", "time", "lat", "lon"]
    assert result.loc[0, "storm_id"] == "AL012020"


def test_read_track_table_rejects_unknown_extension(tmp_path: Path) -> None:
    """
    Test unsupported track table format.
    """
    path = tmp_path / "tracks.txt"
    path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        read_track_table(path)
