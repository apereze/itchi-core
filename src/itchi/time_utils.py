"""
Temporal utility functions for ITCHI workflows.

The core package already handles synoptic selection for precipitation and
ROCLOUD-specific readers. This module provides small generic helpers for
notebooks, examples and table-level validation workflows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from itchi.constants import SYNOPTIC_HOURS_UTC


def normalize_hour_value(value: Any) -> int:
    """
    Normalize an hour value to integer UTC hour.

    The function accepts integer-like hours such as ``6`` and compact HHMM
    values such as ``0600`` or ``1800``.

    Parameters
    ----------
    value : Any
        Input hour value.

    Returns
    -------
    int
        Hour in the range 0-23.

    Raises
    ------
    ValueError
        If the value cannot be interpreted as a valid hour.
    """
    if pd.isna(value):
        raise ValueError("Hour value is missing.")

    hour_raw = int(value)
    hour = hour_raw if hour_raw < 100 else hour_raw // 100
    minute = 0 if hour_raw < 100 else hour_raw % 100

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid hour value: {value!r}")

    return hour


def filter_synoptic_dataframe(
    df: pd.DataFrame,
    hour_column: str = "hh",
    synoptic_hours: Sequence[int] = SYNOPTIC_HOURS_UTC,
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Filter a DataFrame to canonical synoptic hours.

    This is the standardized replacement for notebook-level helpers such as
    ``each6h``. It does not infer or build timestamps; it only filters rows
    using an existing hour column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing an hour column.
    hour_column : str, default="hh"
        Column with integer hour values or compact HHMM values.
    synoptic_hours : sequence of int, default=(0, 6, 12, 18)
        Accepted UTC synoptic hours.
    reset_index : bool, default=True
        Whether to reset the output index.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.

    Raises
    ------
    KeyError
        If ``hour_column`` is absent.
    ValueError
        If hour values cannot be normalized.
    """
    if hour_column not in df.columns:
        raise KeyError(f"Hour column {hour_column!r} not found.")

    valid_hours = {int(hour) for hour in synoptic_hours}
    normalized_hours = df[hour_column].map(normalize_hour_value)
    result = df.loc[normalized_hours.isin(valid_hours)].copy()

    if reset_index:
        result = result.reset_index(drop=True)

    return result


def each6h(
    df: pd.DataFrame,
    hour_column: str = "hh",
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Backward-compatible alias for filtering rows at 00, 06, 12 and 18 UTC.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    hour_column : str, default="hh"
        Hour column.
    reset_index : bool, default=True
        Whether to reset the output index.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only 6-hourly synoptic records.
    """
    return filter_synoptic_dataframe(
        df=df,
        hour_column=hour_column,
        reset_index=reset_index,
    )
