"""
Precipitation snapshot utilities for ITCHI.

ITCHI v0.1 treats precipitation as a snapshot field aligned with tropical
cyclone synoptic times. This module provides utilities to select, filter and
prepare precipitation snapshots without assuming temporal accumulation.

Important
---------
This module must not sum precipitation fields across time. It only selects
valid snapshots from xarray DataArray or Dataset objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from itchi.constants import SYNOPTIC_HOURS_UTC

XarrayObject = xr.DataArray | xr.Dataset
TimeLike = str | np.datetime64 | pd.Timestamp
TimedeltaLike = str | np.timedelta64 | pd.Timedelta | None


def infer_time_coordinate(
    data: XarrayObject,
    candidates: Sequence[str] = ("valid_time", "time"),
) -> str:
    """
    Infer the time coordinate used by an xarray object.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Object containing a time coordinate or time dimension.
    candidates : sequence of str, default=("valid_time", "time")
        Candidate names checked in order.

    Returns
    -------
    str
        Name of the detected time coordinate.

    Raises
    ------
    ValueError
        If no candidate time coordinate is found.
    """
    for candidate in candidates:
        if candidate in data.coords or candidate in data.dims:
            return candidate

    candidates_text = ", ".join(candidates)
    raise ValueError(
        "Could not infer precipitation time coordinate. "
        f"Expected one of: {candidates_text}."
    )


def _require_time_dimension(data: XarrayObject, time_coord: str) -> None:
    """
    Require the time coordinate to also be a dimension.

    Snapshot selection is intentionally dimension-based to avoid ambiguous
    indexing over auxiliary coordinates.
    """
    if time_coord not in data.dims:
        raise ValueError(
            f"Time coordinate {time_coord!r} must be a dimension for "
            "snapshot selection."
        )


def _to_timestamp(value: TimeLike) -> pd.Timestamp:
    """
    Convert a time-like value to pandas.Timestamp.
    """
    return pd.Timestamp(value)


def _to_timedelta(value: TimedeltaLike) -> pd.Timedelta | None:
    """
    Convert a tolerance-like value to pandas.Timedelta.
    """
    if value is None:
        return None

    return pd.Timedelta(value)


def is_synoptic_time(
    valid_time: TimeLike,
    synoptic_hours: Sequence[int] = SYNOPTIC_HOURS_UTC,
) -> bool:
    """
    Check whether a timestamp corresponds to a canonical synoptic hour.

    A valid synoptic timestamp must occur exactly at one of the configured
    UTC hours and must have minute, second and sub-second components equal
    to zero.

    Parameters
    ----------
    valid_time : str, numpy.datetime64 or pandas.Timestamp
        Timestamp to evaluate.
    synoptic_hours : sequence of int, default=SYNOPTIC_HOURS_UTC
        Accepted synoptic hours.

    Returns
    -------
    bool
        True if the timestamp is synoptic; otherwise False.
    """
    timestamp = _to_timestamp(valid_time)

    return (
        timestamp.hour in synoptic_hours
        and timestamp.minute == 0
        and timestamp.second == 0
        and timestamp.microsecond == 0
        and timestamp.nanosecond == 0
    )


def build_synoptic_time_mask(
    valid_times: Sequence[Any],
    synoptic_hours: Sequence[int] = SYNOPTIC_HOURS_UTC,
) -> np.ndarray:
    """
    Build a Boolean mask identifying synoptic timestamps.

    Parameters
    ----------
    valid_times : sequence
        Time values from an xarray coordinate.
    synoptic_hours : sequence of int, default=SYNOPTIC_HOURS_UTC
        Accepted synoptic hours.

    Returns
    -------
    numpy.ndarray
        Boolean mask where True indicates a synoptic timestamp.
    """
    return np.asarray(
        [
            is_synoptic_time(
                valid_time=valid_time,
                synoptic_hours=synoptic_hours,
            )
            for valid_time in valid_times
        ],
        dtype=bool,
    )


def filter_synoptic_snapshots(
    data: XarrayObject,
    time_coord: str | None = None,
    synoptic_hours: Sequence[int] = SYNOPTIC_HOURS_UTC,
) -> XarrayObject:
    """
    Keep only precipitation snapshots at canonical synoptic hours.

    This function does not aggregate or accumulate precipitation. It only
    filters existing snapshots.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Precipitation object with a time dimension.
    time_coord : str or None, default=None
        Name of the time coordinate. If None, it is inferred.
    synoptic_hours : sequence of int, default=SYNOPTIC_HOURS_UTC
        Accepted synoptic hours.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Object containing only synoptic snapshots.
    """
    resolved_time_coord = time_coord or infer_time_coordinate(data)
    _require_time_dimension(data, resolved_time_coord)

    valid_times = data[resolved_time_coord].values
    mask = build_synoptic_time_mask(
        valid_times=valid_times,
        synoptic_hours=synoptic_hours,
    )

    return data.isel({resolved_time_coord: mask})


def select_precipitation_snapshot(
    data: xr.DataArray,
    target_time: TimeLike,
    time_coord: str | None = None,
    method: str = "exact",
    tolerance: TimedeltaLike = None,
) -> xr.DataArray:
    """
    Select a single precipitation snapshot.

    This function selects one snapshot and never performs temporal
    accumulation.

    Parameters
    ----------
    data : xarray.DataArray
        Precipitation field with a time dimension.
    target_time : str, numpy.datetime64 or pandas.Timestamp
        Desired valid time.
    time_coord : str or None, default=None
        Name of the time coordinate. If None, it is inferred.
    method : {"exact", "nearest"}, default="exact"
        Snapshot selection method.
    tolerance : str, numpy.timedelta64, pandas.Timedelta or None, default=None
        Maximum allowed distance from target_time when method="nearest".
        Required for nearest-neighbor selection.

    Returns
    -------
    xarray.DataArray
        Selected precipitation snapshot with the time dimension removed.

    Raises
    ------
    ValueError
        If selection fails, if method is invalid, or if nearest selection is
        requested without tolerance.
    """
    if method not in {"exact", "nearest"}:
        raise ValueError("method must be either 'exact' or 'nearest'.")

    resolved_time_coord = time_coord or infer_time_coordinate(data)
    _require_time_dimension(data, resolved_time_coord)

    target_timestamp = _to_timestamp(target_time)
    target_value = np.datetime64(target_timestamp.to_datetime64())

    if method == "exact":
        try:
            return data.sel({resolved_time_coord: target_value}, drop=True)
        except KeyError as exc:
            raise ValueError(
                "No exact precipitation snapshot found for "
                f"target_time={target_timestamp}."
            ) from exc

    tolerance_value = _to_timedelta(tolerance)

    if tolerance_value is None:
        raise ValueError(
            "tolerance is required when method='nearest' to avoid "
            "uncontrolled temporal matching."
        )

    try:
        return data.sel(
            {resolved_time_coord: target_value},
            method="nearest",
            tolerance=tolerance_value,
            drop=True,
        )
    except KeyError as exc:
        raise ValueError(
            "No precipitation snapshot found within tolerance for "
            f"target_time={target_timestamp} and tolerance={tolerance_value}."
        ) from exc


def extract_precipitation_variable(
    data: xr.DataArray | xr.Dataset,
    precipitation_variable: str | None = None,
) -> xr.DataArray:
    """
    Extract a precipitation DataArray from a DataArray or Dataset.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input precipitation object.
    precipitation_variable : str or None, default=None
        Variable name to extract when data is a Dataset.

    Returns
    -------
    xarray.DataArray
        Precipitation field.

    Raises
    ------
    KeyError
        If the requested variable is not present.
    ValueError
        If data is a Dataset with multiple variables and no variable name
        is provided.
    """
    if isinstance(data, xr.DataArray):
        return data

    if precipitation_variable is not None:
        if precipitation_variable not in data.data_vars:
            raise KeyError(f"Variable {precipitation_variable!r} not found in Dataset.")

        return data[precipitation_variable]

    data_variables = list(data.data_vars)

    if len(data_variables) != 1:
        raise ValueError(
            "precipitation_variable must be provided when Dataset contains "
            "zero or multiple data variables."
        )

    return data[data_variables[0]]


def prepare_precipitation_snapshot_for_pipeline(
    data: xr.DataArray | xr.Dataset,
    target_time: TimeLike,
    precipitation_variable: str | None = None,
    time_coord: str | None = None,
    method: str = "exact",
    tolerance: TimedeltaLike = None,
    require_synoptic_target: bool = True,
) -> xr.DataArray:
    """
    Prepare a precipitation snapshot for ITCHI pipeline execution.

    This function extracts a precipitation variable if needed and selects a
    single valid-time snapshot compatible with pipeline.py.

    It does not accumulate precipitation.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Precipitation object.
    target_time : str, numpy.datetime64 or pandas.Timestamp
        Desired cyclone/snapshot valid time.
    precipitation_variable : str or None, default=None
        Variable name to extract when data is a Dataset.
    time_coord : str or None, default=None
        Name of the time coordinate. If None, it is inferred.
    method : {"exact", "nearest"}, default="exact"
        Snapshot selection method.
    tolerance : str, numpy.timedelta64, pandas.Timedelta or None, default=None
        Maximum allowed temporal distance for method="nearest".
    require_synoptic_target : bool, default=True
        If True, target_time must be exactly 00, 06, 12 or 18 UTC.

    Returns
    -------
    xarray.DataArray
        Selected precipitation snapshot.
    """
    if require_synoptic_target and not is_synoptic_time(target_time):
        raise ValueError(
            "target_time must be an exact synoptic time when "
            "require_synoptic_target=True."
        )

    precipitation = extract_precipitation_variable(
        data=data,
        precipitation_variable=precipitation_variable,
    )

    return select_precipitation_snapshot(
        data=precipitation,
        target_time=target_time,
        time_coord=time_coord,
        method=method,
        tolerance=tolerance,
    )
