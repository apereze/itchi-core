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

ROCLOUD_DATABASE_RECORD_COLUMNS: tuple[str, ...] = (
    "date",
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
    "rbp_rne",
    "rbp_rno",
    "rbp_rso",
    "rbp_rse",
    "rbp_mean",
)

ROCLOUD_TEXT_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column for column in ROCLOUD_TEXT_COLUMNS if column != "ct"
)

ROCLOUD_DATABASE_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column for column in ROCLOUD_DATABASE_RECORD_COLUMNS if column != "date"
) + ("declared_entries", "record_number")

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

ROCLOUD_DATABASE_CANONICAL_RENAME_MAP: dict[str, str] = {
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
    "rbp_rno": "rbp_rnw",
    "rbp_rso": "rbp_rsw",
}

ROCLOUD_STANDARD_NUMERIC_COLUMNS: tuple[str, ...] = (
    "lat",
    "lon",
    "vmax_kt",
    "pmin_hpa",
    "rocloud_rne",
    "rocloud_rnw",
    "rocloud_rsw",
    "rocloud_rse",
    "rocloud_mean_km",
    "asymmetry",
    "dispersion",
    "solidity",
    "rbp_rne",
    "rbp_rnw",
    "rbp_rsw",
    "rbp_rse",
    "rbp_mean",
    "declared_entries",
    "record_number",
)


def read_rocloud_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a ROCLOUD table from CSV, Parquet or ROCLOUD text format.

    Supported text formats are:

    1. Flat 17-column files:
       ``dd mm yy hh lat lon MWS CPSL RNE RNO RSO RSE Rp A D S CT``

    2. Database block files with storm headers:
       ``storm_id, storm_name, declared_entries,`` followed by records with
       ``date hh lat lon MWS CPSL RNE RNO RSO RSE Rp A D S RBP...``.

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
    validate_record_counts: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a ROCLOUD text file.

    The reader automatically detects whether the file is a flat table or a
    database block file with one header per tropical cyclone.

    Parameters
    ----------
    path : str or pathlib.Path
        Input ROCLOUD text file.
    names : sequence of str, default=ROCLOUD_TEXT_COLUMNS
        Column names assigned to flat 17-column input files.
    missing_values : sequence of int or float, default=(-9999,)
        Values used to denote missing data.
    validate_record_counts : bool, default=False
        If True, validate that each block has the declared number of records.
        This is disabled by default because some released files may keep the
        best-track declared count while only retaining available ROCLOUD rows.
    **kwargs : Any
        Additional keyword arguments passed to :func:`pandas.read_table` for
        flat files.

    Returns
    -------
    pandas.DataFrame
        Raw ROCLOUD text table with standardized lowercase column names.
    """
    input_path = Path(path)

    if is_rocloud_database_text_file(input_path):
        return read_rocloud_database_text_table(
            input_path,
            missing_values=missing_values,
            validate_record_counts=validate_record_counts,
        )

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
        column for column in ROCLOUD_TEXT_NUMERIC_COLUMNS if column in result.columns
    ]

    result = _coerce_numeric_columns(result, numeric_columns)
    result = _replace_missing_numeric_values(result, numeric_columns, missing_values)

    if "ct" in result.columns:
        result["ct"] = result["ct"].astype(str).str.strip()

    return result


def is_rocloud_database_text_file(path: str | Path) -> bool:
    """
    Return True if a ROCLOUD text file starts with a storm header line.
    """
    input_path = Path(path)

    with input_path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue

            return _parse_rocloud_database_header(line) is not None

    return False


def _parse_rocloud_database_header(line: str) -> dict[str, str | int] | None:
    """
    Parse a ROCLOUD database block header.

    Expected header example:

    ``EP132006, LANE, 17,``
    """
    if "," not in line:
        return None

    parts = [part.strip() for part in line.strip().split(",")]
    parts = [part for part in parts if part != ""]

    if len(parts) < 3:
        return None

    storm_id, storm_name, declared_entries = parts[:3]

    if not storm_id:
        return None

    try:
        declared_entries_int = int(declared_entries)
    except ValueError:
        return None

    return {
        "storm_id": storm_id,
        "storm_name": storm_name,
        "declared_entries": declared_entries_int,
    }


def read_rocloud_database_text_table(
    path: str | Path,
    missing_values: Sequence[float | int] = (-9999,),
    validate_record_counts: bool = False,
) -> pd.DataFrame:
    """
    Read the block-based ROCLOUD database text format.

    The format contains one header per storm:

    ``storm_id, storm_name, declared_entries,``

    followed by records with 19 fields:

    ``date hh lat lon MWS CPSL RNE RNO RSO RSE Rp A D S``
    ``RBP_RNE RBP_RNO RBP_RSO RBP_RSE RBP_mean``

    Parameters
    ----------
    path : str or pathlib.Path
        Input ROCLOUD database text file.
    missing_values : sequence of int or float, default=(-9999,)
        Values used to denote missing data.
    validate_record_counts : bool, default=False
        If True, validate declared record counts by storm.

    Returns
    -------
    pandas.DataFrame
        Expanded row-level table with storm metadata repeated per record.
    """
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    current_storm: dict[str, str | int] | None = None
    current_record_number = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            header = _parse_rocloud_database_header(stripped)

            if header is not None:
                current_storm = header
                current_record_number = 0
                continue

            if current_storm is None:
                raise ValueError(
                    "ROCLOUD database record found before any storm header "
                    f"at line {line_number}."
                )

            values = stripped.split()

            if len(values) != len(ROCLOUD_DATABASE_RECORD_COLUMNS):
                raise ValueError(
                    "Invalid ROCLOUD database record length at line "
                    f"{line_number}. Expected "
                    f"{len(ROCLOUD_DATABASE_RECORD_COLUMNS)} fields, "
                    f"got {len(values)}."
                )

            current_record_number += 1
            record = dict(zip(ROCLOUD_DATABASE_RECORD_COLUMNS, values, strict=True))
            record.update(current_storm)
            record["record_number"] = current_record_number
            records.append(record)

    if not records:
        raise ValueError(f"No ROCLOUD records were found in {input_path}.")

    result = pd.DataFrame.from_records(records)
    result = standardize_column_names(result)

    numeric_columns = [
        column
        for column in ROCLOUD_DATABASE_NUMERIC_COLUMNS
        if column in result.columns
    ]

    result = _coerce_numeric_columns(result, numeric_columns)
    result = _replace_missing_numeric_values(result, numeric_columns, missing_values)

    for column in ("storm_id", "storm_name", "date"):
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip()

    if validate_record_counts:
        validate_rocloud_database_record_counts(result)

    return result


def validate_rocloud_database_record_counts(df: pd.DataFrame) -> None:
    """
    Validate database block record counts against declared header counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Expanded ROCLOUD database table.

    Raises
    ------
    KeyError
        If required metadata columns are missing.
    ValueError
        If any storm has a count mismatch.
    """
    required = {"storm_id", "declared_entries"}
    missing = required.difference(df.columns)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing record-count column(s): {missing_text}")

    mismatches: list[str] = []

    for storm_id, storm_df in df.groupby("storm_id", sort=False):
        declared = int(storm_df["declared_entries"].iloc[0])
        actual = int(len(storm_df))

        if declared != actual:
            mismatches.append(f"{storm_id}: declared={declared}, parsed={actual}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"ROCLOUD record-count mismatch: {mismatch_text}")


def _coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """
    Convert selected columns to numeric values.
    """
    result = df.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def _replace_missing_numeric_values(
    df: pd.DataFrame,
    columns: Sequence[str],
    missing_values: Sequence[float | int],
) -> pd.DataFrame:
    """
    Replace explicit numeric missing-value codes with pandas NA.
    """
    result = df.copy()

    if not missing_values:
        return result

    for column in columns:
        if column in result.columns:
            result[column] = result[column].mask(
                result[column].isin(missing_values),
                other=pd.NA,
            )

    return result


def _split_compact_hour(hour_raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Split integer hour or compact HHMM values into hour and minute Series.
    """
    hour = hour_raw.where(hour_raw < 100, hour_raw // 100)
    minute = pd.Series(0, index=hour_raw.index).where(hour_raw < 100, hour_raw % 100)

    invalid_time = (hour < 0) | (hour > 23) | (minute < 0) | (minute > 59)

    if bool(invalid_time.any()):
        raise ValueError("Invalid ROCLOUD hour/minute values were found.")

    return hour, minute


def _build_rocloud_text_time(
    df: pd.DataFrame,
    day_col: str = "dd",
    month_col: str = "mm",
    year_col: str = "yy",
    hour_col: str = "hh",
) -> pd.Series:
    """
    Build a datetime Series from flat ROCLOUD date and hour columns.

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
    hour, minute = _split_compact_hour(hour_raw)

    date = pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": day,
        },
        errors="raise",
    )

    return (
        date
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(
            minute,
            unit="m",
        )
    )


def _build_rocloud_database_time(
    df: pd.DataFrame,
    date_col: str = "date",
    hour_col: str = "hh",
) -> pd.Series:
    """
    Build a datetime Series from database date and hour columns.

    The database date column is expected as YYYYMMDD.
    """
    for column in (date_col, hour_col):
        if column not in df.columns:
            raise KeyError(f"Missing ROCLOUD database date/time column: {column}")

    date_text = (
        df[date_col]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )
    date_text = date_text.str.zfill(8)
    hour_raw = pd.to_numeric(df[hour_col], errors="raise").astype(int)
    hour, minute = _split_compact_hour(hour_raw)

    date = pd.to_datetime(date_text, format="%Y%m%d", errors="raise")

    return (
        date
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(
            minute,
            unit="m",
        )
    )


def rocloud_text_to_dataframe(
    path: str | Path,
    names: Sequence[str] = ROCLOUD_TEXT_COLUMNS,
    missing_values: Sequence[float | int] = (-9999,),
    synoptic_only: bool = False,
    sort: bool = True,
    validate_record_counts: bool = False,
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
    ROCLOUD radius, shape metrics, RBP radii and storm names.

    Parameters
    ----------
    path : str or pathlib.Path
        Input ROCLOUD text file.
    names : sequence of str, default=ROCLOUD_TEXT_COLUMNS
        Column names assigned to flat 17-column input files.
    missing_values : sequence of int or float, default=(-9999,)
        Values used to denote missing data.
    synoptic_only : bool, default=False
        If True, keep only 00, 06, 12 and 18 UTC records.
    sort : bool, default=True
        Whether to sort by storm_id and time.
    validate_record_counts : bool, default=False
        If True, validate database-block declared record counts.
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
        validate_record_counts=validate_record_counts,
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
    Standardize a raw ROCLOUD text DataFrame.

    Supported raw layouts are:

    - flat 17-column rows with ``CT`` as the storm identifier;
    - database block rows already expanded with ``storm_id`` and
      ``storm_name`` metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw ROCLOUD text DataFrame.
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
    flat_columns = set(ROCLOUD_TEXT_COLUMNS)
    database_columns = {"storm_id", "storm_name", "date", "hh"}

    if flat_columns.issubset(result.columns):
        result = result.copy()
        result["time"] = _build_rocloud_text_time(result)
        result = result.rename(columns=ROCLOUD_TEXT_CANONICAL_RENAME_MAP)
    elif database_columns.issubset(result.columns):
        result = result.copy()
        result["time"] = _build_rocloud_database_time(result)
        result = result.rename(columns=ROCLOUD_DATABASE_CANONICAL_RENAME_MAP)
    else:
        expected_text = ", ".join(ROCLOUD_TEXT_COLUMNS)
        expected_database = ", ".join(sorted(database_columns))
        raise KeyError(
            "Unrecognized ROCLOUD text layout. Expected either flat columns "
            f"({expected_text}) or database metadata columns "
            f"({expected_database})."
        )

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

    for column in ROCLOUD_STANDARD_NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "storm_name" in result.columns:
        result["storm_name"] = result["storm_name"].astype(str).str.strip()

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
