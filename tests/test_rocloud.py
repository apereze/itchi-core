"""
Tests for ROCLOUD utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itchi.rocloud import (
    extract_rocloud_radii,
    get_rocloud_snapshot,
    has_any_rocloud_radius,
    has_complete_rocloud_radii,
    read_rocloud_table,
    standardize_rocloud_dataframe,
    validate_rocloud_columns,
)


def test_standardize_rocloud_dataframe_with_column_map() -> None:
    """
    Test ROCLOUD standardization using a column map.
    """
    df = pd.DataFrame(
        {
            "SID": ["AL012020", "AL012020"],
            "ISO_TIME": ["2020-06-01 06:00", "2020-06-01 00:00"],
            "RNE": [500.0, 450.0],
            "RSE": [400.0, 350.0],
            "RSW": [600.0, 550.0],
            "RNW": [700.0, 650.0],
        }
    )

    result = standardize_rocloud_dataframe(
        df,
        column_map={
            "storm_id": "sid",
            "time": "iso_time",
            "rocloud_rne": "rne",
            "rocloud_rse": "rse",
            "rocloud_rsw": "rsw",
            "rocloud_rnw": "rnw",
        },
    )

    assert list(result["time"]) == [
        pd.Timestamp("2020-06-01 00:00"),
        pd.Timestamp("2020-06-01 06:00"),
    ]
    assert result.loc[0, "rocloud_rne"] == pytest.approx(450.0)


def test_validate_rocloud_columns_rejects_missing_required_columns() -> None:
    """
    Test that missing required columns raise an error.
    """
    df = pd.DataFrame(
        {
            "storm_id": ["AL012020"],
        }
    )

    with pytest.raises(KeyError):
        validate_rocloud_columns(df)


def test_get_rocloud_snapshot_exact() -> None:
    """
    Test exact ROCLOUD snapshot selection.
    """
    df = pd.DataFrame(
        {
            "storm_id": ["AL012020", "AL012020"],
            "time": pd.to_datetime(["2020-06-01 00:00", "2020-06-01 06:00"]),
            "rocloud_rne": [450.0, 500.0],
            "rocloud_rse": [350.0, 400.0],
            "rocloud_rsw": [550.0, 600.0],
            "rocloud_rnw": [650.0, 700.0],
        }
    )

    row = get_rocloud_snapshot(
        df,
        storm_id="AL012020",
        time="2020-06-01 06:00",
    )

    assert row["rocloud_rne"] == pytest.approx(500.0)


def test_get_rocloud_snapshot_nearest_with_tolerance() -> None:
    """
    Test nearest ROCLOUD snapshot selection within tolerance.
    """
    df = pd.DataFrame(
        {
            "storm_id": ["AL012020", "AL012020"],
            "time": pd.to_datetime(["2020-06-01 00:00", "2020-06-01 06:00"]),
            "rocloud_rne": [450.0, 500.0],
            "rocloud_rse": [350.0, 400.0],
            "rocloud_rsw": [550.0, 600.0],
            "rocloud_rnw": [650.0, 700.0],
        }
    )

    row = get_rocloud_snapshot(
        df,
        storm_id="AL012020",
        time="2020-06-01 05:30",
        tolerance_hours=1.0,
    )

    assert row["time"] == pd.Timestamp("2020-06-01 06:00")


def test_extract_rocloud_radii_complete() -> None:
    """
    Test extraction of complete ROCLOUD radii.
    """
    row = pd.Series(
        {
            "rocloud_rne": 500.0,
            "rocloud_rse": 400.0,
            "rocloud_rsw": 600.0,
            "rocloud_rnw": 700.0,
        }
    )

    result = extract_rocloud_radii(row)

    assert result.radii == {
        "RNE": 500.0,
        "RSE": 400.0,
        "RSW": 600.0,
        "RNW": 700.0,
    }
    assert result.source == "observed"
    assert result.filled_quadrants == ()


def test_extract_rocloud_radii_fills_missing_quadrants() -> None:
    """
    Test filling missing ROCLOUD quadrants with mean available value.
    """
    row = pd.Series(
        {
            "rocloud_rne": 500.0,
            "rocloud_rse": 400.0,
            "rocloud_rsw": np.nan,
            "rocloud_rnw": 600.0,
        }
    )

    result = extract_rocloud_radii(row)

    assert result.radii["RSW"] == pytest.approx(500.0)
    assert result.source == "filled_missing_with_mean_available"
    assert result.filled_quadrants == ("RSW",)


def test_extract_rocloud_radii_rejects_all_missing() -> None:
    """
    Test that all-missing ROCLOUD radii cannot be resolved.
    """
    row = pd.Series(
        {
            "rocloud_rne": np.nan,
            "rocloud_rse": np.nan,
            "rocloud_rsw": np.nan,
            "rocloud_rnw": np.nan,
        }
    )

    with pytest.raises(ValueError):
        extract_rocloud_radii(row)


def test_has_any_rocloud_radius() -> None:
    """
    Test detection of at least one available ROCLOUD radius.
    """
    row = {
        "rocloud_rne": np.nan,
        "rocloud_rse": 400.0,
        "rocloud_rsw": np.nan,
        "rocloud_rnw": np.nan,
    }

    assert has_any_rocloud_radius(row)


def test_has_any_rocloud_radius_rejects_all_missing() -> None:
    """
    Test all-missing ROCLOUD detection.
    """
    row = {
        "rocloud_rne": np.nan,
        "rocloud_rse": np.nan,
        "rocloud_rsw": np.nan,
        "rocloud_rnw": np.nan,
    }

    assert not has_any_rocloud_radius(row)


def test_has_complete_rocloud_radii() -> None:
    """
    Test complete ROCLOUD radius detection.
    """
    row = {
        "rocloud_rne": 500.0,
        "rocloud_rse": 400.0,
        "rocloud_rsw": 600.0,
        "rocloud_rnw": 700.0,
    }

    assert has_complete_rocloud_radii(row)


def test_has_complete_rocloud_radii_rejects_missing_values() -> None:
    """
    Test incomplete ROCLOUD radius detection.
    """
    row = {
        "rocloud_rne": 500.0,
        "rocloud_rse": np.nan,
        "rocloud_rsw": 600.0,
        "rocloud_rnw": 700.0,
    }

    assert not has_complete_rocloud_radii(row)


def test_read_rocloud_table_csv(tmp_path: Path) -> None:
    """
    Test reading ROCLOUD table from CSV.
    """
    path = tmp_path / "rocloud.csv"

    df = pd.DataFrame(
        {
            "storm_id": ["AL012020"],
            "time": ["2020-06-01 00:00"],
            "rocloud_rne": [500.0],
            "rocloud_rse": [400.0],
            "rocloud_rsw": [600.0],
            "rocloud_rnw": [700.0],
        }
    )

    df.to_csv(path, index=False)

    result = read_rocloud_table(path)

    assert list(result.columns) == [
        "storm_id",
        "time",
        "rocloud_rne",
        "rocloud_rse",
        "rocloud_rsw",
        "rocloud_rnw",
    ]


def test_read_rocloud_table_rejects_unknown_extension(tmp_path: Path) -> None:
    """
    Test unsupported ROCLOUD table format.
    """
    path = tmp_path / "rocloud.txt"
    path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        read_rocloud_table(path)
