"""
Climatology utilities for ITCHI.

This module provides utilities to compute precipitation percentile
climatologies used by the ITCHI precipitation hazard component.

The core percentiles are:

- Q90: onset of low or occasional hazard
- Q95: high precipitation hazard
- Q99: maximum precipitation hazard threshold

The functions are intentionally general and support both NumPy arrays and
xarray DataArrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

ArrayLike = np.ndarray | xr.DataArray

DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.90, 0.95, 0.99)
DEFAULT_QUANTILE_NAMES: tuple[str, ...] = ("Q90", "Q95", "Q99")


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def validate_quantile_levels(
    quantile_levels: Sequence[float],
) -> None:
    """
    Validate quantile levels.

    Parameters
    ----------
    quantile_levels : sequence of float
        Quantile levels in the interval [0, 1].

    Raises
    ------
    ValueError
        If any quantile is outside [0, 1] or if the sequence is empty.
    """
    if len(quantile_levels) == 0:
        raise ValueError("At least one quantile level is required.")

    for quantile in quantile_levels:
        quantile_float = float(quantile)

        if quantile_float < 0.0 or quantile_float > 1.0:
            raise ValueError(
                "Quantile levels must be within [0, 1]. " f"Received: {quantile}"
            )


def quantile_level_to_name(quantile_level: float) -> str:
    """
    Convert a quantile level to a canonical percentile name.

    Examples
    --------
    0.90 -> Q90
    0.95 -> Q95
    0.99 -> Q99
    """
    percentile = int(round(float(quantile_level) * 100.0))

    return f"Q{percentile}"


def compute_precipitation_quantiles(
    precipitation: ArrayLike,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    dim: str = "time",
    axis: int = 0,
    skipna: bool = True,
) -> dict[str, ArrayLike]:
    """
    Compute precipitation quantiles.

    Parameters
    ----------
    precipitation : numpy.ndarray or xarray.DataArray
        Historical or reference precipitation samples.
    quantile_levels : sequence of float, default=(0.90, 0.95, 0.99)
        Quantile levels to compute.
    dim : str, default="time"
        Dimension over which to compute quantiles for xarray inputs.
    axis : int, default=0
        Axis over which to compute quantiles for NumPy inputs.
    skipna : bool, default=True
        Whether to ignore NaN values.

    Returns
    -------
    dict[str, numpy.ndarray or xarray.DataArray]
        Dictionary with percentile fields, for example:
        Q90, Q95 and Q99.
    """
    validate_quantile_levels(quantile_levels)

    if _contains_xarray_object(precipitation):
        quantiles = precipitation.quantile(
            q=list(quantile_levels),
            dim=dim,
            skipna=skipna,
        )

        result: dict[str, xr.DataArray] = {}

        for quantile in quantile_levels:
            name = quantile_level_to_name(quantile)
            selected = quantiles.sel(quantile=quantile, drop=True)
            selected.name = name
            result[name] = selected

        return result

    array = np.asarray(precipitation, dtype=float)

    if skipna:
        values = np.nanquantile(
            array,
            q=list(quantile_levels),
            axis=axis,
        )
    else:
        values = np.quantile(
            array,
            q=list(quantile_levels),
            axis=axis,
        )

    result_np: dict[str, np.ndarray] = {}

    for idx, quantile in enumerate(quantile_levels):
        name = quantile_level_to_name(quantile)
        result_np[name] = values[idx]

    return result_np


def compute_standard_precipitation_climatology(
    precipitation: ArrayLike,
    dim: str = "time",
    axis: int = 0,
    skipna: bool = True,
) -> dict[str, ArrayLike]:
    """
    Compute the standard ITCHI precipitation climatology.

    This returns Q90, Q95 and Q99.
    """
    return compute_precipitation_quantiles(
        precipitation=precipitation,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        dim=dim,
        axis=axis,
        skipna=skipna,
    )


def climatology_dict_to_dataset(
    climatology: dict[str, ArrayLike],
) -> xr.Dataset:
    """
    Convert a climatology dictionary to an xarray Dataset.

    Parameters
    ----------
    climatology : dict[str, numpy.ndarray or xarray.DataArray]
        Dictionary with percentile fields.

    Returns
    -------
    xarray.Dataset
        Dataset containing climatological percentile fields.

    Raises
    ------
    ValueError
        If NumPy arrays are provided without xarray metadata.
    """
    dataset = xr.Dataset()

    for name, value in climatology.items():
        if not isinstance(value, xr.DataArray):
            raise ValueError(
                "climatology_dict_to_dataset requires xarray DataArray values. "
                f"Variable {name!r} is not an xarray DataArray."
            )

        dataset[name] = value

    return dataset


def extract_standard_percentiles(
    climatology: xr.Dataset | dict[str, ArrayLike],
    q90_name: str = "Q90",
    q95_name: str = "Q95",
    q99_name: str = "Q99",
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Extract Q90, Q95 and Q99 from a climatology object.

    Parameters
    ----------
    climatology : xarray.Dataset or dict
        Climatology containing Q90, Q95 and Q99.
    q90_name : str, default="Q90"
        Variable name for the 90th percentile.
    q95_name : str, default="Q95"
        Variable name for the 95th percentile.
    q99_name : str, default="Q99"
        Variable name for the 99th percentile.

    Returns
    -------
    tuple
        Q90, Q95 and Q99 fields.
    """
    required = (q90_name, q95_name, q99_name)

    missing = [name for name in required if name not in climatology]

    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"Missing climatology percentile(s): {missing_text}")

    return (
        climatology[q90_name],
        climatology[q95_name],
        climatology[q99_name],
    )
