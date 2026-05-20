"""
ROCLOUD utilities for ITCHI.

This module provides utilities to read, standardize and query ROCLOUD
quadrant radii.

ROCLOUD is used in ITCHI as the external attribution radius for
cyclone-related precipitation.

Canonical minimum columns
-------------------------
storm_id
time

Optional but expected radius columns
------------------------------------
rocloud_rne
rocloud_rse
rocloud_rsw
rocloud_rnw
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from itchi.constants import SYNOPTIC_HOURS_UTC
from itchi.radii import ResolvedRadii, resolve_attribution_radius
from itchi.units import normalize_quadrant_key

REQUIRED_ROCLOUD_COLUMNS: tuple[str, ...] = ("storm_id", "time")


DEFAULT_ROCLOUD_COLUMN_MAP: dict[str, str] = {
    "RNE": "rocloud_rne",
    "RSE": "rocloud_rse",
    "RSW": "rocloud_rsw",
    "RNW": "rocloud_rnw",
}


ROCLOUD_TEXT_COLUMNS: tuple[str, ...] = (
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
)

ROCLOUD_TEXT_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column for column in ROCLOUD_TEXT_COLUMNS if column != "ct"
)

ROCLOUD_TEXT_EXTENSIONS: tuple[str, ...] = (".dat", ".txt", ".dot")

ROCLOUD_TEXT_CANONICAL_RENAME_MAP: dict[str, str] = {
    "ct": "storm_id",
    "mws": "vmax_kt",
    "cpsl": "pmin_hpa",
    "rne": "rocloud_rne",
    "rno": "rocloud_rnw",
    "rso": "rocloud_rsw",
    "rse": "rocloud_rse",
    "rp": "rocloud_mean_km",
    "a": "asymmetry",
    "d": "dispersion",
    "s": "solidity",
}


def read_rocloud_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a ROCLOUD table from CSV, Parquet or ROCLOUD text format.

    ROCLOUD text files are expected to follow the 17-column format used by
    ``NA880.dat`` and ``EP880.dat``:

    ``dd mm yy hh lat lon MWS CPSL RNE RNO RSO RSE Rp A D S CT``

    Parameters
    ----------
    path : str or pathlib.Path
        Input ROCLOUD table path.
    **kwargs : Any
        Additional keyword arguments passed to pandas. For text files,
        keyword arguments are passed to :func:`read_rocloud_text_table`.

    Returns
    -------
    pandas.DataFrame
        Loaded ROCLOUD table.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the file extension is unsupported.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"ROCLOUD table not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path, **kwargs)

    if suffix == ".parquet":
        return pd.read_parquet(input_path, **kwargs)

    if suffix in ROCLOUD_TEXT_EXTENSIONS:
        return read_rocloud_text_table(input_path, **kwargs)

    raise ValueError(
        "Unsupported ROCLOUD table format. Expected .csv, .parquet, "
        ".dat, .txt or .dot."
    )


def read_rocloud_text_table(
    path: str | Path,
    names: Sequence[str] = ROCLOUD_TEXT_COLUMNS,
    missing_values: Sequence[float | int] = (-9999,),
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a ROCLOUD text file without header.

    The operational ROCLOUD files used by ITCHI contain tabulated records
    without column names. The expected default columns are:

    ``dd mm yy hh lat lon MWS CPSL RNE RNO RSO RSE Rp A D S CT``

    where ``RNO`` is the northwestern quadrant and ``RSO`` is the
    southwestern quadrant.

    Parameters
    ----------
    path : str or pathlib.Path
        Input ROCLOUD text file.
    names : sequence of str, default=ROCLOUD_TEXT_COLUMNS
        Column names assigned to the input file.
    missing_values : sequence of int or float, default=(-9999,)
        Values used to denote missing data.
    **kwargs : Any
        Additional keyword arguments passed to :func:`pandas.read_table`.

    Returns
    -------
    pandas.DataFrame
        Raw ROCLOUD text table with standardized lowercase column names.
    """
    input_path = Path(path)

    read_kwargs: dict[str, Any] = {
        "names": list(names),
        "index_col": False,
        "sep": r"\s+",
        "engine": "python",
    }
    read_kwargs.update(kwargs)

    result = pd.read_table(input_path, **read_kwargs)
    result = standardize_column_names(result)

    numeric_columns = [
        column
        for column in ROCLOUD_TEXT_NUMERIC_COLUMNS
        if column in result.columns
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if missing_values:
        for column in numeric_columns:
            result[column] = result[column].mask(
                result[column].isin(missing_values),
                other=pd.NA,
            )

    if "ct" in result.columns:
        result["ct"] = result["ct"].astype(str).str.strip()

    return result


def _build_rocloud_text_time(
    df: pd.DataFrame,
    day_col: str = "dd",
    month_col: str = "mm",
    year_col: str = "yy",
    hour_col: str = "hh",
) -> pd.Series:
    """
    Build a datetime Series from ROCLOUD text date and hour columns.

    The hour column accepts both integer hours, for example ``6`` and ``18``,
    and compact HHMM values, for example ``0600`` and ``1800``.
    """
    for column in (day_col, month_col, year_col, hour_col):
        if column not in df.columns:
            raise KeyError(f"Missing ROCLOUD text date/time column: {column}")

    year = pd.to_numeric(df[year_col], errors="raise").astype(int)
    month = pd.to_numeric(df[month_col], errors="raise").astype(int)
    day = pd.to_numeric(df[day_col], errors="raise").astype(int)
    hour_raw = pd.to_numeric(df[hour_col], errors="raise").astype(int)

    hour = hour_raw.where(hour_raw < 100, hour_raw // 100)
    minute = pd.Series(0, index=df.index).where(hour_raw < 100, hour_raw % 100)

    invalid_time = (
        (hour < 0)
        | (hour > 23)
        | (minute < 0)
        | (minute > 59)
    )

    if bool(invalid_time.any()):
        raise ValueError("Invalid ROCLOUD hour/minute values were found.")

    date = pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": day,
        },
        errors="raise",
    )

    return date + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(
        minute,
        unit="m",
    )


def rocloud_text_to_dataframe(
    path: str | Path,
    names: Sequence[str] = ROCLOUD_TEXT_COLUMNS,
    missing_values: Sequence[float | int] = (-9999,),
    synoptic_only: bool = False,
    sort: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read and standardize a ROCLOUD text file for ITCHI.

    The output follows the internal ROCLOUD contract:

    - ``storm_id``
    - ``time``
    - ``rocloud_rne``
    - ``rocloud_rse``
    - ``rocloud_rsw``
    - ``rocloud_rnw``

    Additional metadata columns are preserved when present, including
    cyclone center coordinates, maximum wind, central pressure, mean
    ROCLOUD radius and shape metrics.

    Parameters
    ----------
    path : str or pathlib.Path
        Input ROCLOUD text file.
    names : sequence of str, default=ROCLOUD_TEXT_COLUMNS
        Column names assigned to the input file.
    missing_values : sequence of int or float, default=(-9999,)
        Values used to denote missing data.
    synoptic_only : bool, default=False
        If True, keep only 00, 06, 12 and 18 UTC records.
    sort : bool, default=True
        Whether to sort by storm_id and time.
    **kwargs : Any
        Additional keyword arguments passed to :func:`read_rocloud_text_table`.

    Returns
    -------
    pandas.DataFrame
        Standardized ROCLOUD DataFrame.
    """
    raw = read_rocloud_text_table(
        path=path,
        names=names,
        missing_values=missing_values,
        **kwargs,
    )

    return standardize_rocloud_text_dataframe(
        raw,
        synoptic_only=synoptic_only,
        sort=sort,
    )


def standardize_rocloud_text_dataframe(
    df: pd.DataFrame,
    synoptic_only: bool = False,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Standardize a raw ROCLOUD 17-column text DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw ROCLOUD text DataFrame with columns equivalent to
        ``ROCLOUD_TEXT_COLUMNS``.
    synoptic_only : bool, default=False
        If True, keep only 00, 06, 12 and 18 UTC records.
    sort : bool, default=True
        Whether to sort by storm_id and time.

    Returns
    -------
    pandas.DataFrame
        DataFrame standardized to the ITCHI ROCLOUD contract.
    """
    result = standardize_column_names(df)
    missing = set(ROCLOUD_TEXT_COLUMNS).difference(result.columns)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing ROCLOUD text column(s): {missing_text}")

    result = result.copy()
    result["time"] = _build_rocloud_text_time(result)

    result = result.rename(columns=ROCLOUD_TEXT_CANONICAL_RENAME_MAP)

    canonical = standardize_rocloud_dataframe(
        result,
        parse_time=True,
        sort=sort,
    )

    if synoptic_only:
        canonical = canonical.loc[
            canonical["time"].dt.hour.isin(SYNOPTIC_HOURS_UTC)
            & canonical["time"].dt.minute.eq(0)
        ].reset_index(drop=True)

    return canonical


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
        Copy of DataFrame with standardized column names.
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


def rename_rocloud_columns(
    df: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Rename raw ROCLOUD columns to ITCHI canonical names.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column_map : Mapping[str, str] or None, default=None
        Mapping from canonical ITCHI names to source column names.

        Example:
        {
            "storm_id": "sid",
            "time": "iso_time",
            "rocloud_rne": "rne",
            "rocloud_rse": "rse",
            "rocloud_rsw": "rsw",
            "rocloud_rnw": "rnw",
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


def validate_rocloud_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_ROCLOUD_COLUMNS,
) -> None:
    """
    Validate that required ROCLOUD columns are present.

    Parameters
    ----------
    df : pandas.DataFrame
        ROCLOUD DataFrame.
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
        raise KeyError(f"Missing required ROCLOUD column(s): {missing_text}")


def standardize_rocloud_dataframe(
    df: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
    required_columns: Sequence[str] = REQUIRED_ROCLOUD_COLUMNS,
    parse_time: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Standardize a ROCLOUD DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw ROCLOUD DataFrame.
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
        Standardized ROCLOUD DataFrame.
    """
    result = standardize_column_names(df)
    result = rename_rocloud_columns(result, column_map=column_map)

    validate_rocloud_columns(result, required_columns=required_columns)

    result = result.copy()

    if parse_time:
        result["time"] = pd.to_datetime(result["time"], errors="raise")

    result["storm_id"] = result["storm_id"].astype(str)

    for column in DEFAULT_ROCLOUD_COLUMN_MAP.values():
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if sort:
        result = result.sort_values(["storm_id", "time"]).reset_index(drop=True)

    return result


def get_rocloud_snapshot(
    df: pd.DataFrame,
    storm_id: str,
    time: str | pd.Timestamp,
    storm_id_col: str = "storm_id",
    time_col: str = "time",
    tolerance_hours: float | None = None,
) -> pd.Series:
    """
    Get a single ROCLOUD row for a storm and target time.

    If tolerance_hours is None, the match must be exact.
    If tolerance_hours is provided, the nearest row within tolerance is returned.

    Parameters
    ----------
    df : pandas.DataFrame
        Standardized ROCLOUD DataFrame.
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
        ROCLOUD row.

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
        raise ValueError(f"No ROCLOUD rows found for storm_id={storm_id!r}.")

    storm_df[time_col] = pd.to_datetime(storm_df[time_col], errors="raise")

    if tolerance_hours is None:
        match = storm_df.loc[storm_df[time_col] == target_time]

        if match.empty:
            raise ValueError(
                f"No exact ROCLOUD row found for storm_id={storm_id!r} "
                f"at time={target_time}."
            )

        if len(match) > 1:
            raise ValueError(
                f"Multiple ROCLOUD rows found for storm_id={storm_id!r} "
                f"at time={target_time}."
            )

        return match.iloc[0]

    tolerance = pd.Timedelta(hours=float(tolerance_hours))
    time_delta = (storm_df[time_col] - target_time).abs()
    nearest_idx = time_delta.idxmin()
    nearest_delta = time_delta.loc[nearest_idx]

    if nearest_delta > tolerance:
        raise ValueError(
            f"No ROCLOUD row found within {tolerance_hours} hours for "
            f"storm_id={storm_id!r} at time={target_time}."
        )

    return storm_df.loc[nearest_idx]


def _get_record_value(
    record: Mapping[str, Any] | pd.Series,
    column: str,
    missing_value: Any = None,
) -> Any:
    """
    Extract a value from a mapping-like record.

    Missing columns are returned as missing_value instead of raising.
    This allows partial ROCLOUD quadrant information to be resolved later.
    """
    if column not in record:
        return missing_value

    return record[column]


def extract_rocloud_radii(
    record: Mapping[str, Any] | pd.Series,
    column_map: Mapping[str, str] = DEFAULT_ROCLOUD_COLUMN_MAP,
    fill_strategy: str = "mean_available",
) -> ResolvedRadii:
    """
    Extract and resolve ROCLOUD quadrant radii from a row.

    Missing or invalid quadrant values are filled using the selected strategy.

    Parameters
    ----------
    record : Mapping or pandas.Series
        ROCLOUD row or dictionary containing radius values.
    column_map : Mapping[str, str], default=DEFAULT_ROCLOUD_COLUMN_MAP
        Mapping from quadrant labels to column names.
    fill_strategy : str, default="mean_available"
        Strategy used to fill missing quadrant values.

    Returns
    -------
    ResolvedRadii
        Resolved ROCLOUD radii and metadata.
    """
    raw_radii: dict[str, Any] = {}

    for quadrant_key, column in column_map.items():
        quadrant = normalize_quadrant_key(quadrant_key)
        raw_radii[quadrant] = _get_record_value(record, column)

    return resolve_attribution_radius(
        rocloud_by_quadrant=raw_radii,
        fill_strategy=fill_strategy,
    )


def has_any_rocloud_radius(
    record: Mapping[str, Any] | pd.Series,
    column_map: Mapping[str, str] = DEFAULT_ROCLOUD_COLUMN_MAP,
) -> bool:
    """
    Check whether at least one ROCLOUD quadrant radius is available.

    Parameters
    ----------
    record : Mapping or pandas.Series
        ROCLOUD row or dictionary containing radius values.
    column_map : Mapping[str, str], default=DEFAULT_ROCLOUD_COLUMN_MAP
        Mapping from quadrant labels to column names.

    Returns
    -------
    bool
        True if at least one radius value is finite and non-negative.
    """
    for column in column_map.values():
        value = _get_record_value(record, column)

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue

        if pd.notna(value_float) and value_float >= 0.0:
            return True

    return False


def has_complete_rocloud_radii(
    record: Mapping[str, Any] | pd.Series,
    column_map: Mapping[str, str] = DEFAULT_ROCLOUD_COLUMN_MAP,
) -> bool:
    """
    Check whether all ROCLOUD quadrant radii are available.

    Parameters
    ----------
    record : Mapping or pandas.Series
        ROCLOUD row or dictionary containing radius values.
    column_map : Mapping[str, str], default=DEFAULT_ROCLOUD_COLUMN_MAP
        Mapping from quadrant labels to column names.

    Returns
    -------
    bool
        True if all four quadrant radii are finite and non-negative.
    """
    for column in column_map.values():
        value = _get_record_value(record, column)

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return False

        if pd.isna(value_float) or value_float < 0.0:
            return False

    return True
