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
    is_rocloud_database_text_file,
    read_rocloud_database_text_table,
    read_rocloud_table,
    rocloud_text_to_dataframe,
    standardize_rocloud_dataframe,
    standardize_rocloud_text_dataframe,
    validate_rocloud_columns,
    validate_rocloud_database_record_counts,
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


def test_read_rocloud_table_dat_assigns_operational_columns(
    tmp_path: Path,
) -> None:
    """
    Test reading the operational 17-column ROCLOUD .dat format.
    """
    path = tmp_path / "EP880.dat"
    path.write_text(
        (
            "9\t8\t2013\t6\t11.0\t-116.2\t37.0\t1008\t"
            "626.05\t542.21\t679.67\t713.31\t640.31\t"
            "0.24\t0.51\t0.31\tCP032013\n"
        ),
        encoding="utf-8",
    )

    result = read_rocloud_table(path)

    assert list(result.columns) == [
        "dd",
        "mm",
        "yy",
        "hh",
        "lat",
        "lon",
        "mws",
        "cpsl",
        "rne",
        "rno",
        "rso",
        "rse",
        "rp",
        "a",
        "d",
        "s",
        "ct",
    ]
    assert result.loc[0, "ct"] == "CP032013"


def test_rocloud_text_to_dataframe_maps_cardinal_quadrants(
    tmp_path: Path,
) -> None:
    """
    Test mapping from RNE/RNO/RSO/RSE to ITCHI canonical ROCLOUD columns.
    """
    path = tmp_path / "EP880.dat"
    path.write_text(
        (
            "9\t8\t2013\t6\t11.0\t-116.2\t37.0\t1008\t"
            "626.05\t542.21\t679.67\t713.31\t640.31\t"
            "0.24\t0.51\t0.31\tCP032013\n"
        ),
        encoding="utf-8",
    )

    result = rocloud_text_to_dataframe(path)

    assert result.loc[0, "storm_id"] == "CP032013"
    assert result.loc[0, "time"] == pd.Timestamp("2013-08-09 06:00")
    assert result.loc[0, "rocloud_rne"] == pytest.approx(626.05)
    assert result.loc[0, "rocloud_rnw"] == pytest.approx(542.21)
    assert result.loc[0, "rocloud_rsw"] == pytest.approx(679.67)
    assert result.loc[0, "rocloud_rse"] == pytest.approx(713.31)
    assert result.loc[0, "vmax_kt"] == pytest.approx(37.0)
    assert result.loc[0, "pmin_hpa"] == pytest.approx(1008.0)


def test_rocloud_text_to_dataframe_handles_hhmm_hours(
    tmp_path: Path,
) -> None:
    """
    Test parsing of compact HHMM values such as 0600.
    """
    path = tmp_path / "rocloud.txt"
    path.write_text(
        (
            "9 8 2013 0600 11.0 -116.2 37.0 1008 "
            "626.05 542.21 679.67 713.31 640.31 "
            "0.24 0.51 0.31 CP032013\n"
        ),
        encoding="utf-8",
    )

    result = rocloud_text_to_dataframe(path)

    assert result.loc[0, "time"] == pd.Timestamp("2013-08-09 06:00")


def test_standardize_rocloud_text_dataframe_replaces_missing_values(
    tmp_path: Path,
) -> None:
    """
    Test conversion of -9999 missing values into NaN.
    """
    path = tmp_path / "NA880.dat"
    path.write_text(
        (
            "7\t6\t2000\t18\t21.0\t-93.0\t46.25\t1008\t"
            "685.02\t-9999\t892.42\t868.72\t749.96\t"
            "0.38\t0.41\t0.13\tAL012000\n"
        ),
        encoding="utf-8",
    )

    raw = read_rocloud_table(path)
    result = standardize_rocloud_text_dataframe(raw)

    assert pd.isna(result.loc[0, "rocloud_rnw"])
    assert result.loc[0, "rocloud_rne"] == pytest.approx(685.02)


def test_rocloud_text_to_dataframe_can_filter_synoptic_records(
    tmp_path: Path,
) -> None:
    """
    Test optional filtering to 00, 06, 12 and 18 UTC.
    """
    path = tmp_path / "NA880.dat"
    path.write_text(
        (
            "5\t6\t2001\t18\t28.5\t-95.3\t92.5\t1002\t"
            "756.06\t185.62\t458.08\t880.89\t570.16\t"
            "0.79\t0.44\t0.38\tAL012001\n"
            "5\t6\t2001\t21\t28.9\t-95.3\t83.25\t1003\t"
            "799.08\t405.27\t831.07\t1002.29\t759.43\t"
            "0.60\t0.49\t0.23\tAL012001\n"
        ),
        encoding="utf-8",
    )

    result = rocloud_text_to_dataframe(path, synoptic_only=True)

    assert len(result) == 1
    assert result.loc[0, "time"] == pd.Timestamp("2001-06-05 18:00")


def test_detects_rocloud_database_text_file(tmp_path: Path) -> None:
    """
    Test detection of database files with storm header lines.
    """
    path = tmp_path / "EP_TCSize_2000_2024.dat"
    path.write_text(
        (
            "EP022000,                BUD,     2,\n"
            "20000613  1200   13.9 -106.6  64  1000   "
            "1062.89 545.64 862.23 825.34 824.03  "
            "0.49  0.18  0.34   1111.07 1074.93 865.39 959.66 1002.76\n"
        ),
        encoding="utf-8",
    )

    assert is_rocloud_database_text_file(path)


def test_read_rocloud_database_text_table_preserves_header_metadata(
    tmp_path: Path,
) -> None:
    """
    Test expanded database reading with storm name and declared count.
    """
    path = tmp_path / "EP_TCSize_2000_2024.dat"
    path.write_text(
        (
            "EP022000,                BUD,     2,\n"
            "20000613  1200   13.9 -106.6  64  1000   "
            "1062.89 545.64 862.23 825.34 824.03  "
            "0.49  0.18  0.34   1111.07 1074.93 865.39 959.66 1002.76\n"
            "20000613  1800   14.4 -107.4  74   999    "
            "893.30 633.68 944.97 800.73 818.17  "
            "0.33  0.38  0.38    853.13 402.21 1082.33 878.99 804.16\n"
        ),
        encoding="utf-8",
    )

    result = read_rocloud_database_text_table(path)

    assert len(result) == 2
    assert result.loc[0, "storm_id"] == "EP022000"
    assert result.loc[0, "storm_name"] == "BUD"
    assert result.loc[0, "declared_entries"] == 2
    assert result.loc[1, "record_number"] == 2


def test_rocloud_database_text_to_dataframe_maps_header_and_rbp(
    tmp_path: Path,
) -> None:
    """
    Test standardization of database block format into ITCHI columns.
    """
    path = tmp_path / "EP_TCSize_2000_2024.dat"
    path.write_text(
        (
            "EP022000,                BUD,     1,\n"
            "20000613  1200   13.9 -106.6  64  1000   "
            "1062.89 545.64 862.23 825.34 824.03  "
            "0.49  0.18  0.34   1111.07 1074.93 865.39 959.66 1002.76\n"
        ),
        encoding="utf-8",
    )

    result = rocloud_text_to_dataframe(path)

    assert result.loc[0, "storm_id"] == "EP022000"
    assert result.loc[0, "storm_name"] == "BUD"
    assert result.loc[0, "time"] == pd.Timestamp("2000-06-13 12:00")
    assert result.loc[0, "rocloud_rne"] == pytest.approx(1062.89)
    assert result.loc[0, "rocloud_rnw"] == pytest.approx(545.64)
    assert result.loc[0, "rocloud_rsw"] == pytest.approx(862.23)
    assert result.loc[0, "rocloud_rse"] == pytest.approx(825.34)
    assert result.loc[0, "rbp_rne"] == pytest.approx(1111.07)
    assert result.loc[0, "rbp_rnw"] == pytest.approx(1074.93)
    assert result.loc[0, "rbp_rsw"] == pytest.approx(865.39)
    assert result.loc[0, "rbp_rse"] == pytest.approx(959.66)


def test_rocloud_database_missing_values_are_preserved_as_nan(
    tmp_path: Path,
) -> None:
    """
    Test missing-value handling in database block records.
    """
    path = tmp_path / "NA_TCSize_2000_2024.dat"
    path.write_text(
        (
            "AL012000,            UNNAMED,      1,\n"
            "20000607  1800   21.0  -93.0  46  1008    "
            "685.02 553.70 892.42 868.72 749.96  "
            "0.38  0.41  0.13    -9999.00 -9999.00 -9999.00 -9999.00 -9999.00\n"
        ),
        encoding="utf-8",
    )

    result = rocloud_text_to_dataframe(path)

    assert pd.isna(result.loc[0, "rbp_rne"])
    assert pd.isna(result.loc[0, "rbp_mean"])
    assert result.loc[0, "rocloud_rne"] == pytest.approx(685.02)


def test_rocloud_database_record_count_validation(tmp_path: Path) -> None:
    """
    Test optional strict validation of declared record counts.
    """
    path = tmp_path / "NA_TCSize_2000_2024.dat"
    path.write_text(
        (
            "AL012000,            UNNAMED,      2,\n"
            "20000607  1800   21.0  -93.0  46  1008    "
            "685.02 553.70 892.42 868.72 749.96  "
            "0.38  0.41  0.13    262.10 0.00 925.46 716.01 634.52\n"
        ),
        encoding="utf-8",
    )

    raw = read_rocloud_database_text_table(path)

    with pytest.raises(ValueError):
        validate_rocloud_database_record_counts(raw)

    with pytest.raises(ValueError):
        read_rocloud_database_text_table(path, validate_record_counts=True)


def test_rocloud_database_text_to_dataframe_can_filter_synoptic_records(
    tmp_path: Path,
) -> None:
    """
    Test synoptic filtering in database block format.
    """
    path = tmp_path / "EP_TCSize_2000_2024.dat"
    path.write_text(
        (
            "EP022000,                BUD,     2,\n"
            "20000613  1200   13.9 -106.6  64  1000   "
            "1062.89 545.64 862.23 825.34 824.03  "
            "0.49  0.18  0.34   1111.07 1074.93 865.39 959.66 1002.76\n"
            "20000613  2100   14.4 -107.4  74   999    "
            "893.30 633.68 944.97 800.73 818.17  "
            "0.33  0.38  0.38    853.13 402.21 1082.33 878.99 804.16\n"
        ),
        encoding="utf-8",
    )

    result = rocloud_text_to_dataframe(path, synoptic_only=True)

    assert len(result) == 1
    assert result.loc[0, "time"] == pd.Timestamp("2000-06-13 12:00")


def test_read_rocloud_table_rejects_unknown_extension(tmp_path: Path) -> None:
    """
    Test unsupported ROCLOUD table format.
    """
    path = tmp_path / "rocloud.json"
    path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        read_rocloud_table(path)
