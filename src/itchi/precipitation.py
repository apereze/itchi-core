"""
Precipitation hazard utilities for ITCHI.

This module implements the normalized precipitation hazard component H_P
using local climatological percentiles Q90, Q95 and Q99.

ITCHI v0.1 treats precipitation as a snapshot field aligned with the
cyclone synoptic time, not as a temporal accumulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import DEFAULT_EPSILON


def _is_xarray_object(obj: Any) -> bool:
    """
    Check whether an object is an xarray DataArray or Dataset.
    """
    return isinstance(obj, xr.DataArray | xr.Dataset)


def compute_precipitation_hazard(
    precipitation: np.ndarray | xr.DataArray,
    q90: float | np.ndarray | xr.DataArray,
    q95: float | np.ndarray | xr.DataArray,
    q99: float | np.ndarray | xr.DataArray,
    epsilon: float = DEFAULT_EPSILON,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray | xr.DataArray:
    """
    Compute normalized precipitation hazard H_P.

    The hazard function is defined by a piecewise linear scaling:

    - P < Q90       -> H_P = 0
    - Q90 to Q95   -> H_P increases from 0 to 0.5
    - Q95 to Q99   -> H_P increases from 0.5 to 1
    - P >= Q99     -> H_P = 1

    Parameters
    ----------
    precipitation : numpy.ndarray or xarray.DataArray
        Precipitation snapshot field.
    q90 : float, numpy.ndarray or xarray.DataArray
        Local 90th percentile.
    q95 : float, numpy.ndarray or xarray.DataArray
        Local 95th percentile.
    q99 : float, numpy.ndarray or xarray.DataArray
        Local 99th percentile.
    epsilon : float, default=1e-6
        Small value used to avoid division by zero.
    clip_min : float, default=0.0
        Minimum allowed hazard value.
    clip_max : float, default=1.0
        Maximum allowed hazard value.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Normalized precipitation hazard in the interval [0, 1].
    """
    if any(_is_xarray_object(obj) for obj in (precipitation, q90, q95, q99)):
        return _compute_precipitation_hazard_xarray(
            precipitation=precipitation,
            q90=q90,
            q95=q95,
            q99=q99,
            epsilon=epsilon,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    return _compute_precipitation_hazard_numpy(
        precipitation=precipitation,
        q90=q90,
        q95=q95,
        q99=q99,
        epsilon=epsilon,
        clip_min=clip_min,
        clip_max=clip_max,
    )


def _compute_precipitation_hazard_numpy(
    precipitation: np.ndarray,
    q90: float | np.ndarray,
    q95: float | np.ndarray,
    q99: float | np.ndarray,
    epsilon: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """
    NumPy implementation of the precipitation hazard function.
    """
    p = np.asarray(precipitation, dtype=float)
    q90_arr = np.asarray(q90, dtype=float)
    q95_arr = np.asarray(q95, dtype=float)
    q99_arr = np.asarray(q99, dtype=float)

    denominator_low = np.maximum(q95_arr - q90_arr, epsilon)
    denominator_high = np.maximum(q99_arr - q95_arr, epsilon)

    hazard_low = 0.5 * ((p - q90_arr) / denominator_low)
    hazard_high = 0.5 + 0.5 * ((p - q95_arr) / denominator_high)

    hazard = np.select(
        condlist=[
            p < q90_arr,
            (p >= q90_arr) & (p < q95_arr),
            (p >= q95_arr) & (p < q99_arr),
            p >= q99_arr,
        ],
        choicelist=[
            0.0,
            hazard_low,
            hazard_high,
            1.0,
        ],
        default=np.nan,
    )

    return np.clip(hazard, clip_min, clip_max)


def _compute_precipitation_hazard_xarray(
    precipitation: xr.DataArray,
    q90: float | xr.DataArray,
    q95: float | xr.DataArray,
    q99: float | xr.DataArray,
    epsilon: float,
    clip_min: float,
    clip_max: float,
) -> xr.DataArray:
    """
    xarray implementation of the precipitation hazard function.
    """
    denominator_low = xr.where((q95 - q90) > epsilon, q95 - q90, epsilon)
    denominator_high = xr.where((q99 - q95) > epsilon, q99 - q95, epsilon)

    hazard_low = 0.5 * ((precipitation - q90) / denominator_low)
    hazard_high = 0.5 + 0.5 * ((precipitation - q95) / denominator_high)

    hazard = xr.where(
        precipitation < q90,
        0.0,
        xr.where(
            precipitation < q95,
            hazard_low,
            xr.where(
                precipitation < q99,
                hazard_high,
                1.0,
            ),
        ),
    )

    return hazard.clip(min=clip_min, max=clip_max)