"""
Track utilities for ITCHI.

This module provides utilities to read, standardize and query tropical
cyclone track data.

The main goal is to convert raw best-track or IBTrACS-like tables into
a predictable structure for ITCHI.

Canonical minimum columns
-------------------------
storm_id
time
lat
lon

Optional columns
----------------
vmax_kt
pmin_hpa
r34_rne
r34_rse
r34_rsw
r34_rnw
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from itchi.constants import QUADRANTS, SYNOPTIC_HOURS_UTC
from itchi.units import convert_quadrant_radii_to_km, normalize_quadrant_key

REQUIRED_TRACK_COLUMNS: tuple[str, ...] = ("storm_id", "time", "lat", "lon")


DEFAULT_R34_COLUMN_MAP: dict[str, str] = {
    "RNE": "r34_rne",
    "RSE": "r34_rse",
    "RSW": "r34_rsw",
    "RNW": "r34_rnw",
}


def read_track_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a track table from CSV or Parquet.

    Parameters
    ----------
    path : str or pathlib.Path
        Input track table path.
    **kwargs : Any
        Additional keyword arguments passed to pandas.

    Returns
    -------
    pandas.DataFrame
        Loaded track table.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the file extension is unsupported.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Track table not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path, **kwargs)

    if suffix == ".parquet":
        return pd.read_parquet(input_path, **kwargs)

    raise ValueError("Unsupported track table format. " "Expected .csv or .parquet.")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names.

    Transformations:

    - strip leading/trailing spaces;
    - lowercase;
    - replace spaces, hyphens and slashes with underscores;
    - collapse repeated underscores.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Copy of the DataFrame with standardized column names.
    """
    result = df.copy()

    standardized_columns = (
        result.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    result.columns = standardized_columns

    return result


def rename_track_columns(
    df: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Rename raw track columns to ITCHI canonical names.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column_map : Mapping[str, str] or None, default=None
        Mapping from canonical ITCHI column names to source column names.

        Example:
        {
            "storm_id": "sid",
            "time": "iso_time",
            "lat": "latitude",
            "lon": "longitude",
            "vmax_kt": "usa_wind",
            "pmin_hpa": "usa_pres",
        }

    Returns
    -------
    pandas.DataFrame
        DataFrame with renamed columns.
    """
    result = df.copy()

    if column_map is None:
        return result

    rename_map = {
        str(source).lower().strip(): str(target).lower().strip()
        for target, source in column_map.items()
    }

    return result.rename(columns=rename_map)


def validate_track_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_TRACK_COLUMNS,
) -> None:
    """
    Validate that required track columns are present.

    Parameters
    ----------
    df : pandas.DataFrame
        Track DataFrame.
    required_columns : sequence of str
        Required column names.

    Raises
    ------
    KeyError
        If any required column is missing.
    """
    missing = set(required_columns).difference(df.columns)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing required track column(s): {missing_text}")


def standardize_track_dataframe(
    df: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
    required_columns: Sequence[str] = REQUIRED_TRACK_COLUMNS,
    parse_time: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Standardize a tropical cyclone track DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw track DataFrame.
    column_map : Mapping[str, str] or None, default=None
        Mapping from canonical ITCHI names to source column names.
    required_columns : sequence of str
        Required columns after standardization.
    parse_time : bool, default=True
        Whether to parse the time column with pandas.to_datetime.
    sort : bool, default=True
        Whether to sort by storm_id and time.

    Returns
    -------
    pandas.DataFrame
        Standardized track DataFrame.
    """
    result = standardize_column_names(df)
    result = rename_track_columns(result, column_map=column_map)

    validate_track_columns(result, required_columns=required_columns)

    result = result.copy()

    if parse_time:
        result["time"] = pd.to_datetime(result["time"], errors="raise")

    result["storm_id"] = result["storm_id"].astype(str)

    for column in ("lat", "lon", "vmax_kt", "pmin_hpa"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if sort:
        result = result.sort_values(["storm_id", "time"]).reset_index(drop=True)

    return result


def filter_synoptic_times(
    df: pd.DataFrame,
    time_col: str = "time",
    synoptic_hours: Sequence[int] = SYNOPTIC_HOURS_UTC,
) -> pd.DataFrame:
    """
    Filter a track DataFrame to synoptic times.

    Parameters
    ----------
    df : pandas.DataFrame
        Track DataFrame.
    time_col : str, default="time"
        Name of the time column.
    synoptic_hours : sequence of int
        Allowed UTC hours.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    if time_col not in df.columns:
        raise KeyError(f"Missing time column: {time_col}")

    times = pd.to_datetime(df[time_col], errors="raise")
    allowed_hours = set(int(hour) for hour in synoptic_hours)

    mask = times.dt.hour.isin(allowed_hours)

    return df.loc[mask].copy().reset_index(drop=True)


def get_track_snapshot(
    df: pd.DataFrame,
    storm_id: str,
    time: str | pd.Timestamp,
    storm_id_col: str = "storm_id",
    time_col: str = "time",
    tolerance_hours: float | None = None,
) -> pd.Series:
    """
    Get a single track row for a storm and target time.

    If tolerance_hours is None, the match must be exact.
    If tolerance_hours is provided, the nearest row within the tolerance
    is returned.

    Parameters
    ----------
    df : pandas.DataFrame
        Standardized track DataFrame.
    storm_id : str
        Storm identifier.
    time : str or pandas.Timestamp
        Target time.
    storm_id_col : str, default="storm_id"
        Storm identifier column.
    time_col : str, default="time"
        Time column.
    tolerance_hours : float or None, default=None
        Optional nearest-time tolerance in hours.

    Returns
    -------
    pandas.Series
        Track row.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If no matching row is found.
    """
    for column in (storm_id_col, time_col):
        if column not in df.columns:
            raise KeyError(f"Missing required column: {column}")

    target_time = pd.Timestamp(time)
    storm_mask = df[storm_id_col].astype(str) == str(storm_id)
    storm_df = df.loc[storm_mask].copy()

    if storm_df.empty:
        raise ValueError(f"No track rows found for storm_id={storm_id!r}.")

    storm_df[time_col] = pd.to_datetime(storm_df[time_col], errors="raise")

    if tolerance_hours is None:
        match = storm_df.loc[storm_df[time_col] == target_time]

        if match.empty:
            raise ValueError(
                f"No exact track row found for storm_id={storm_id!r} "
                f"at time={target_time}."
            )

        if len(match) > 1:
            raise ValueError(
                f"Multiple track rows found for storm_id={storm_id!r} "
                f"at time={target_time}."
            )

        return match.iloc[0]

    tolerance = pd.Timedelta(hours=float(tolerance_hours))
    time_delta = (storm_df[time_col] - target_time).abs()
    nearest_idx = time_delta.idxmin()
    nearest_delta = time_delta.loc[nearest_idx]

    if nearest_delta > tolerance:
        raise ValueError(
            f"No track row found within {tolerance_hours} hours for "
            f"storm_id={storm_id!r} at time={target_time}."
        )

    return storm_df.loc[nearest_idx]


def _get_record_value(record: Mapping[str, Any] | pd.Series, column: str) -> Any:
    """
    Extract a value from a mapping-like record.
    """
    if column not in record:
        raise KeyError(f"Missing radius column: {column}")

    return record[column]


def extract_quadrant_radii(
    record: Mapping[str, Any] | pd.Series,
    column_map: Mapping[str, str] = DEFAULT_R34_COLUMN_MAP,
    input_unit: str = "nm",
) -> dict[str, float]:
    """
    Extract and convert quadrant-specific radii from a track row.

    Parameters
    ----------
    record : Mapping or pandas.Series
        Track row or dictionary containing radius values.
    column_map : Mapping[str, str], default=DEFAULT_R34_COLUMN_MAP
        Mapping from quadrant labels to column names.
        Accepted quadrant keys include RNE/RSE/RSW/RNW or NE/SE/SW/NW.
    input_unit : str, default="nm"
        Unit of the input radii.

    Returns
    -------
    dict[str, float]
        Quadrant-specific radii in kilometers with keys:
        RNE, RSE, RSW, RNW.
    """
    raw_radii: dict[str, Any] = {}

    for quadrant_key, column in column_map.items():
        quadrant = normalize_quadrant_key(quadrant_key)
        raw_radii[quadrant] = _get_record_value(record, column)

    return convert_quadrant_radii_to_km(
        radii_by_quadrant=raw_radii,
        input_unit=input_unit,
    )


def has_complete_quadrant_radii(
    record: Mapping[str, Any] | pd.Series,
    column_map: Mapping[str, str] = DEFAULT_R34_COLUMN_MAP,
) -> bool:
    """
    Check whether all quadrant radius columns are present and finite.

    Parameters
    ----------
    record : Mapping or pandas.Series
        Track row or dictionary containing radius values.
    column_map : Mapping[str, str], default=DEFAULT_R34_COLUMN_MAP
        Mapping from quadrant labels to column names.

    Returns
    -------
    bool
        True if all radius values are present and finite.
    """
    try:
        radii = {
            normalize_quadrant_key(quadrant_key): _get_record_value(record, column)
            for quadrant_key, column in column_map.items()
        }
    except KeyError:
        return False

    required = set(QUADRANTS)

    if set(radii) != required:
        return False

    return all(
        np.isfinite(float(value)) and float(value) >= 0.0 for value in radii.values()
    )
