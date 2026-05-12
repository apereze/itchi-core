"""
Event-level aggregation utilities for ITCHI.

This module aggregates time-dependent ITCHI snapshots into event-level
products.

Main products
-------------
ITCHI_max:
    Maximum ITCHI value experienced by each grid cell during the event.

ITCHI_acc:
    Bounded accumulated/persistent ITCHI hazard during the event.

The accumulated product uses the same bounded-union logic used elsewhere
in ITCHI:

    ITCHI_acc = 1 - product(1 - ITCHI_t)

This formulation remains bounded in [0, 1].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import ITCHI_MAX_VALUE, ITCHI_MIN_VALUE

ArrayLike = np.ndarray | xr.DataArray


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def _clip_index(
    values: ArrayLike,
    clip_min: float = ITCHI_MIN_VALUE,
    clip_max: float = ITCHI_MAX_VALUE,
) -> ArrayLike:
    """
    Clip ITCHI values to the valid interval [0, 1].
    """
    if _contains_xarray_object(values):
        return values.clip(min=clip_min, max=clip_max)

    return np.clip(np.asarray(values, dtype=float), clip_min, clip_max)


def compute_event_max(
    itchi: ArrayLike,
    dim: str = "time",
    axis: int = 0,
    skipna: bool = True,
) -> ArrayLike:
    """
    Compute maximum ITCHI by event.

    Parameters
    ----------
    itchi : numpy.ndarray or xarray.DataArray
        Time-dependent ITCHI field.
    dim : str, default="time"
        Time dimension name for xarray inputs.
    axis : int, default=0
        Time axis for NumPy inputs.
    skipna : bool, default=True
        Whether to ignore NaN values.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Maximum ITCHI value across time.
    """
    values = _clip_index(itchi)

    if _contains_xarray_object(values):
        return values.max(dim=dim, skipna=skipna)

    if skipna:
        return np.nanmax(values, axis=axis)

    return np.max(values, axis=axis)


def compute_event_accumulated(
    itchi: ArrayLike,
    dim: str = "time",
    axis: int = 0,
    skipna: bool = True,
) -> ArrayLike:
    """
    Compute accumulated ITCHI by event using bounded union.

    The accumulated hazard is defined as:

        ITCHI_acc = 1 - product(1 - ITCHI_t)

    Parameters
    ----------
    itchi : numpy.ndarray or xarray.DataArray
        Time-dependent ITCHI field.
    dim : str, default="time"
        Time dimension name for xarray inputs.
    axis : int, default=0
        Time axis for NumPy inputs.
    skipna : bool, default=True
        Whether to ignore NaN values. When skipna=True, NaNs are treated
        as missing snapshots and do not increase accumulated hazard.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Accumulated ITCHI value across time.
    """
    values = _clip_index(itchi)

    if _contains_xarray_object(values):
        if skipna:
            valid_count = values.notnull().sum(dim=dim)
            filled = values.fillna(0.0)
            accumulated = 1.0 - (1.0 - filled).prod(dim=dim, skipna=False)
            return accumulated.where(valid_count > 0)

        return 1.0 - (1.0 - values).prod(dim=dim, skipna=False)

    values_np = np.asarray(values, dtype=float)

    if skipna:
        valid_count = np.sum(~np.isnan(values_np), axis=axis)
        filled = np.nan_to_num(values_np, nan=0.0)
        accumulated = 1.0 - np.prod(1.0 - filled, axis=axis)
        return np.where(valid_count > 0, accumulated, np.nan)

    return 1.0 - np.prod(1.0 - values_np, axis=axis)


def compute_event_products(
    itchi: ArrayLike,
    dim: str = "time",
    axis: int = 0,
    skipna: bool = True,
) -> dict[str, ArrayLike]:
    """
    Compute standard event-level ITCHI products.

    Parameters
    ----------
    itchi : numpy.ndarray or xarray.DataArray
        Time-dependent ITCHI field.
    dim : str, default="time"
        Time dimension name for xarray inputs.
    axis : int, default=0
        Time axis for NumPy inputs.
    skipna : bool, default=True
        Whether to ignore NaN values.

    Returns
    -------
    dict[str, numpy.ndarray or xarray.DataArray]
        Dictionary with:

        - ITCHI_max
        - ITCHI_acc
    """
    return {
        "ITCHI_max": compute_event_max(
            itchi=itchi,
            dim=dim,
            axis=axis,
            skipna=skipna,
        ),
        "ITCHI_acc": compute_event_accumulated(
            itchi=itchi,
            dim=dim,
            axis=axis,
            skipna=skipna,
        ),
    }
