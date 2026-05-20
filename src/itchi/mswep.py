"""
MSWEP NetCDF adapter for ITCHI.

This module provides source-specific utilities for reading MSWEP
precipitation files and adapting them to the internal ITCHI precipitation
snapshot contract.

Important
---------
MSWEP precipitation is treated here as a valid-time snapshot field.
This module must not sum, accumulate or resample precipitation through time.
It only opens files, standardizes coordinates, subsets space and selects
existing snapshots.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from itchi.precipitation_snapshots import (
    TimedeltaLike,
    TimeLike,
    extract_precipitation_variable,
    filter_synoptic_snapshots,
    infer_time_coordinate,
    prepare_precipitation_snapshot_for_pipeline,
)

XarrayObject = xr.DataArray | xr.Dataset

DEFAULT_MSWEP_PRECIPITATION_VARIABLE_CANDIDATES: tuple[str, ...] = (
    "precipitation",
    "precip",
    "precipitation_amount",
    "precipitation_flux",
    "pr",
)

DEFAULT_MSWEP_TIME_CANDIDATES: tuple[str, ...] = ("valid_time", "time")
DEFAULT_MSWEP_LATITUDE_CANDIDATES: tuple[str, ...] = ("lat", "latitude", "y")
DEFAULT_MSWEP_LONGITUDE_CANDIDATES: tuple[str, ...] = ("lon", "longitude", "x")


def read_mswep_dataset(
    path: str | Path,
    chunks: dict[str, int] | str | None = None,
    engine: str | None = None,
    decode_times: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Open an MSWEP NetCDF file as an xarray Dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Input MSWEP NetCDF path.
    chunks : dict, str or None, default=None
        Optional dask chunking passed to :func:`xarray.open_dataset`.
    engine : str or None, default=None
        Optional xarray backend engine.
    decode_times : bool, default=True
        Whether to decode CF-compliant time coordinates.
    **kwargs : Any
        Additional keyword arguments passed to :func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset
        Opened MSWEP dataset.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"MSWEP NetCDF file not found: {input_path}")

    open_kwargs: dict[str, Any] = {
        "decode_times": decode_times,
    }

    if chunks is not None:
        open_kwargs["chunks"] = chunks

    if engine is not None:
        open_kwargs["engine"] = engine

    open_kwargs.update(kwargs)

    return xr.open_dataset(input_path, **open_kwargs)


def extract_mswep_precipitation(
    data: xr.DataArray | xr.Dataset,
    precipitation_variable: str | None = None,
    variable_candidates: Sequence[str] = (
        DEFAULT_MSWEP_PRECIPITATION_VARIABLE_CANDIDATES
    ),
) -> xr.DataArray:
    """
    Extract the MSWEP precipitation variable.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input MSWEP object.
    precipitation_variable : str or None, default=None
        Explicit precipitation variable name. If provided, it takes priority.
    variable_candidates : sequence of str
        Candidate variable names checked when no explicit name is provided.

    Returns
    -------
    xarray.DataArray
        MSWEP precipitation field.

    Raises
    ------
    KeyError
        If the explicit variable is not present.
    ValueError
        If no unambiguous precipitation variable can be inferred.
    """
    if isinstance(data, xr.DataArray):
        return data

    if precipitation_variable is not None:
        return extract_precipitation_variable(
            data=data,
            precipitation_variable=precipitation_variable,
        )

    for candidate in variable_candidates:
        if candidate in data.data_vars:
            return data[candidate]

    return extract_precipitation_variable(data=data)


def standardize_mswep_time(
    data: XarrayObject,
    time_coord: str | None = None,
    target_time_coord: str = "valid_time",
    candidates: Sequence[str] = DEFAULT_MSWEP_TIME_CANDIDATES,
) -> XarrayObject:
    """
    Standardize MSWEP time coordinate naming.

    This function renames the detected time dimension/coordinate to
    ``valid_time`` by default. It does not resample, aggregate or accumulate
    precipitation.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input object with a time dimension.
    time_coord : str or None, default=None
        Existing time coordinate. If None, it is inferred.
    target_time_coord : str, default="valid_time"
        Desired output time coordinate name.
    candidates : sequence of str
        Candidate time-coordinate names used for inference.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Object with standardized time coordinate naming.
    """
    resolved_time_coord = time_coord or infer_time_coordinate(
        data,
        candidates=candidates,
    )

    if resolved_time_coord == target_time_coord:
        return data

    if target_time_coord in data.coords or target_time_coord in data.dims:
        raise ValueError(
            f"Cannot rename {resolved_time_coord!r} to {target_time_coord!r}: "
            "target coordinate already exists."
        )

    return data.rename({resolved_time_coord: target_time_coord})


def _find_coordinate_name(
    data: XarrayObject,
    candidates: Sequence[str],
    label: str,
) -> str:
    """
    Find a coordinate or dimension name from ordered candidates.
    """
    for candidate in candidates:
        if candidate in data.coords or candidate in data.dims:
            return candidate

    candidates_text = ", ".join(candidates)
    raise ValueError(
        f"Could not infer MSWEP {label}. Expected one of: {candidates_text}."
    )


def infer_mswep_lat_lon_names(
    data: XarrayObject,
    latitude_candidates: Sequence[str] = DEFAULT_MSWEP_LATITUDE_CANDIDATES,
    longitude_candidates: Sequence[str] = DEFAULT_MSWEP_LONGITUDE_CANDIDATES,
) -> tuple[str, str]:
    """
    Infer MSWEP latitude and longitude coordinate names.

    Returns
    -------
    tuple[str, str]
        ``(lat_name, lon_name)``.
    """
    lat_name = _find_coordinate_name(
        data=data,
        candidates=latitude_candidates,
        label="latitude coordinate",
    )
    lon_name = _find_coordinate_name(
        data=data,
        candidates=longitude_candidates,
        label="longitude coordinate",
    )

    return lat_name, lon_name


def _get_coordinate(data: XarrayObject, name: str) -> xr.DataArray:
    """
    Return a coordinate DataArray and fail if only a bare dimension exists.
    """
    if name in data.coords:
        return data.coords[name]

    raise ValueError(
        f"MSWEP coordinate {name!r} is a dimension but has no coordinate "
        "values. Explicit latitude/longitude coordinates are required."
    )


def get_mswep_lon_lat(
    data: xr.DataArray | xr.Dataset,
    lat_name: str | None = None,
    lon_name: str | None = None,
    as_2d: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Extract longitude and latitude coordinates from MSWEP data.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input MSWEP object.
    lat_name, lon_name : str or None, default=None
        Coordinate names. If not provided, they are inferred.
    as_2d : bool, default=True
        If True and coordinates are one-dimensional, broadcast them into
        two-dimensional longitude/latitude grids.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        ``(lon, lat)`` coordinate fields.
    """
    resolved_lat_name, resolved_lon_name = infer_mswep_lat_lon_names(data)
    resolved_lat_name = lat_name or resolved_lat_name
    resolved_lon_name = lon_name or resolved_lon_name

    lat = _get_coordinate(data, resolved_lat_name)
    lon = _get_coordinate(data, resolved_lon_name)

    if not as_2d:
        return lon, lat

    if lat.ndim == 1 and lon.ndim == 1:
        lat_grid, lon_grid = xr.broadcast(lat, lon)
        return lon_grid.rename("lon"), lat_grid.rename("lat")

    if lat.shape != lon.shape:
        raise ValueError(
            "Latitude and longitude coordinates must have matching shapes "
            "when they are not one-dimensional."
        )

    return lon.rename("lon"), lat.rename("lat")


def _normalize_longitude_bounds_for_data(
    lon_values: np.ndarray,
    lon_min: float,
    lon_max: float,
) -> tuple[float, float]:
    """
    Convert negative longitude bounds to 0-360 when needed by the data.
    """
    finite_lon = lon_values[np.isfinite(lon_values)]

    if finite_lon.size == 0:
        raise ValueError("Longitude coordinate contains no finite values.")

    uses_0360 = float(finite_lon.min()) >= 0.0 and float(finite_lon.max()) > 180.0

    if not uses_0360:
        return lon_min, lon_max

    lon_min_norm = lon_min % 360.0
    lon_max_norm = lon_max % 360.0

    if lon_min_norm > lon_max_norm:
        raise ValueError(
            "Longitude bounds cross the dateline after conversion to 0-360. "
            "Dateline-crossing subsets are not currently supported."
        )

    return lon_min_norm, lon_max_norm


def _slice_for_coordinate(
    coord: xr.DataArray,
    lower: float,
    upper: float,
) -> slice:
    """
    Build an orientation-aware slice for a one-dimensional coordinate.
    """
    values = np.asarray(coord.values, dtype=float)

    if values.ndim != 1:
        raise ValueError("Spatial subsetting currently requires 1D coordinates.")

    if values.size == 0:
        raise ValueError("Cannot subset an empty coordinate.")

    if values[0] <= values[-1]:
        return slice(lower, upper)

    return slice(upper, lower)


def subset_mswep_domain(
    data: XarrayObject,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    lat_name: str | None = None,
    lon_name: str | None = None,
    pad_degrees: float = 0.0,
) -> XarrayObject:
    """
    Spatially subset MSWEP data by a latitude/longitude bounding box.

    This operation only subsets existing grid cells. It does not interpolate
    or regrid.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input MSWEP object.
    lon_min, lon_max, lat_min, lat_max : float
        Bounding-box limits in degrees.
    lat_name, lon_name : str or None, default=None
        Coordinate names. If not provided, they are inferred.
    pad_degrees : float, default=0.0
        Optional padding applied to each side of the bounding box.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Spatial subset.
    """
    resolved_lat_name, resolved_lon_name = infer_mswep_lat_lon_names(data)
    resolved_lat_name = lat_name or resolved_lat_name
    resolved_lon_name = lon_name or resolved_lon_name

    lat = _get_coordinate(data, resolved_lat_name)
    lon = _get_coordinate(data, resolved_lon_name)

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("subset_mswep_domain currently supports only 1D lat/lon.")

    lon_lower = float(lon_min) - float(pad_degrees)
    lon_upper = float(lon_max) + float(pad_degrees)
    lat_lower = float(lat_min) - float(pad_degrees)
    lat_upper = float(lat_max) + float(pad_degrees)

    lon_lower, lon_upper = _normalize_longitude_bounds_for_data(
        lon_values=np.asarray(lon.values, dtype=float),
        lon_min=lon_lower,
        lon_max=lon_upper,
    )

    return data.sel(
        {
            resolved_lon_name: _slice_for_coordinate(lon, lon_lower, lon_upper),
            resolved_lat_name: _slice_for_coordinate(lat, lat_lower, lat_upper),
        }
    )


def filter_mswep_synoptic_snapshots(
    data: XarrayObject,
    time_coord: str | None = None,
    target_time_coord: str = "valid_time",
) -> XarrayObject:
    """
    Keep only MSWEP snapshots at 00, 06, 12 and 18 UTC.

    The function standardizes the time coordinate and filters existing
    snapshots. It never accumulates precipitation.
    """
    standardized = standardize_mswep_time(
        data=data,
        time_coord=time_coord,
        target_time_coord=target_time_coord,
    )

    return filter_synoptic_snapshots(
        data=standardized,
        time_coord=target_time_coord,
    )


def prepare_mswep_precipitation(
    data: xr.DataArray | xr.Dataset,
    precipitation_variable: str | None = None,
    synoptic_only: bool = True,
    time_coord: str | None = None,
    target_time_coord: str = "valid_time",
) -> xr.DataArray:
    """
    Prepare an MSWEP precipitation DataArray for ITCHI workflows.

    This extracts the precipitation variable, standardizes time naming and
    optionally filters to synoptic snapshots. No temporal accumulation is
    performed.
    """
    precipitation = extract_mswep_precipitation(
        data=data,
        precipitation_variable=precipitation_variable,
    )
    precipitation = standardize_mswep_time(
        data=precipitation,
        time_coord=time_coord,
        target_time_coord=target_time_coord,
    )

    if synoptic_only:
        precipitation = filter_synoptic_snapshots(
            data=precipitation,
            time_coord=target_time_coord,
        )

    return precipitation


def select_mswep_precipitation_snapshot(
    data: xr.DataArray | xr.Dataset,
    target_time: TimeLike,
    precipitation_variable: str | None = None,
    time_coord: str | None = None,
    method: str = "exact",
    tolerance: TimedeltaLike = None,
    require_synoptic_target: bool = True,
) -> xr.DataArray:
    """
    Select one MSWEP precipitation snapshot for ITCHI.

    This function does not accumulate neighboring MSWEP files or time steps.
    It selects one existing valid-time field.
    """
    precipitation = extract_mswep_precipitation(
        data=data,
        precipitation_variable=precipitation_variable,
    )
    precipitation = standardize_mswep_time(
        data=precipitation,
        time_coord=time_coord,
    )

    return prepare_precipitation_snapshot_for_pipeline(
        data=precipitation,
        target_time=target_time,
        precipitation_variable=None,
        time_coord="valid_time",
        method=method,
        tolerance=tolerance,
        require_synoptic_target=require_synoptic_target,
    )


def read_mswep_precipitation(
    path: str | Path,
    precipitation_variable: str | None = None,
    synoptic_only: bool = True,
    chunks: dict[str, int] | str | None = None,
    engine: str | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Open an MSWEP NetCDF file and return a prepared precipitation DataArray.

    This is a convenience wrapper around :func:`read_mswep_dataset` and
    :func:`prepare_mswep_precipitation`.
    """
    dataset = read_mswep_dataset(
        path=path,
        chunks=chunks,
        engine=engine,
        **kwargs,
    )

    return prepare_mswep_precipitation(
        data=dataset,
        precipitation_variable=precipitation_variable,
        synoptic_only=synoptic_only,
    )
