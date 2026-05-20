"""
Snapshot-input builders for ITCHI.

This module connects tabular cyclone metadata, ROCLOUD radii and
precipitation snapshots into dictionaries compatible with:

    itchi.pipeline.compute_itchi_snapshot_from_grid

and, by extension:

    itchi.compiler.compile_itchi_event

Important design choice
-----------------------
This module does not resolve or impute quadrant radii. It passes raw
quadrant values to pipeline.py so that the pipeline remains responsible for
radius resolution and metadata tracking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from itchi.precipitation_snapshots import (
    TimedeltaLike,
    TimeLike,
    prepare_precipitation_snapshot_for_pipeline,
)
from itchi.rocloud import DEFAULT_ROCLOUD_COLUMN_MAP, get_rocloud_snapshot
from itchi.tracks import DEFAULT_R34_COLUMN_MAP, get_track_snapshot
from itchi.units import normalize_quadrant_key

ArrayLike = np.ndarray | xr.DataArray


def _get_record_value(
    record: Mapping[str, Any] | pd.Series,
    column: str,
    default: Any = None,
) -> Any:
    """
    Extract a value from a mapping-like record.

    Parameters
    ----------
    record : Mapping or pandas.Series
        Input record.
    column : str
        Column name.
    default : Any, default=None
        Value returned when the column is absent.

    Returns
    -------
    Any
        Extracted value or default.
    """
    if isinstance(record, pd.Series):
        return record.get(column, default)

    return record.get(column, default)


def _get_required_float(
    record: Mapping[str, Any] | pd.Series,
    column: str,
    label: str,
) -> float:
    """
    Extract a required finite float from a record.
    """
    value = _get_record_value(record, column)

    if value is None:
        raise KeyError(f"Missing required {label} column: {column}")

    value_float = float(value)

    if not np.isfinite(value_float):
        raise ValueError(f"Invalid {label} value in column {column!r}: {value}")

    return value_float


def _get_optional_float(
    record: Mapping[str, Any] | pd.Series,
    column: str,
) -> float | None:
    """
    Extract an optional finite float from a record.
    """
    value = _get_record_value(record, column)

    if value is None or pd.isna(value):
        return None

    value_float = float(value)

    if not np.isfinite(value_float):
        return None

    return value_float


def _copy_missing_dimension_coords(
    value: ArrayLike | None,
    reference: xr.DataArray,
) -> ArrayLike | None:
    """
    Copy missing dimension coordinates from a reference DataArray.

    This is useful when lon/lat grids are provided as xarray DataArrays with
    the correct dimensions and shape but without explicit dimension
    coordinates. Preserving coordinates avoids downstream quality-control
    failures during field-alignment checks.
    """
    if value is None:
        return None

    if not isinstance(value, xr.DataArray):
        return value

    if value.dims != reference.dims:
        return value

    if value.shape != reference.shape:
        return value

    coords_to_add = {
        dim: reference.coords[dim]
        for dim in reference.dims
        if dim in reference.coords and dim not in value.coords
    }

    if not coords_to_add:
        return value

    return value.assign_coords(coords_to_add)


def _align_spatial_inputs_to_precipitation(
    precipitation: xr.DataArray,
    lon: ArrayLike,
    lat: ArrayLike,
    wind_hazard_normalized: ArrayLike | None = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike | None]:
    """
    Align xarray spatial inputs to precipitation coordinates when possible.

    This does not interpolate or regrid. It only copies missing dimension
    coordinates when dims and shape already match.
    """
    lon_aligned = _copy_missing_dimension_coords(
        value=lon,
        reference=precipitation,
    )

    lat_aligned = _copy_missing_dimension_coords(
        value=lat,
        reference=precipitation,
    )

    wind_aligned = _copy_missing_dimension_coords(
        value=wind_hazard_normalized,
        reference=precipitation,
    )

    return lon_aligned, lat_aligned, wind_aligned


def extract_raw_quadrant_values(
    record: Mapping[str, Any] | pd.Series,
    column_map: Mapping[str, str],
) -> dict[str, Any]:
    """
    Extract raw quadrant values from a tabular record.

    Missing columns are represented as None so that downstream radius
    resolution can decide whether to fill, fallback or fail.

    Parameters
    ----------
    record : Mapping or pandas.Series
        Track or ROCLOUD row.
    column_map : Mapping[str, str]
        Mapping from quadrant labels to source column names.

    Returns
    -------
    dict[str, Any]
        Dictionary with canonical ITCHI quadrant keys: RNE, RSE, RSW, RNW.
    """
    values: dict[str, Any] = {}

    for quadrant_key, column in column_map.items():
        quadrant = normalize_quadrant_key(quadrant_key)
        values[quadrant] = _get_record_value(record, column, default=None)

    return values


def build_snapshot_input_from_records(
    precipitation: xr.DataArray,
    q90: float | ArrayLike,
    q95: float | ArrayLike,
    q99: float | ArrayLike,
    lon: ArrayLike,
    lat: ArrayLike,
    track_record: Mapping[str, Any] | pd.Series,
    rocloud_record: Mapping[str, Any] | pd.Series,
    r34_column_map: Mapping[str, str] = DEFAULT_R34_COLUMN_MAP,
    rocloud_column_map: Mapping[str, str] = DEFAULT_ROCLOUD_COLUMN_MAP,
    r34_unit: str = "nm",
    rocloud_unit: str = "km",
    center_lon_column: str = "lon",
    center_lat_column: str = "lat",
    vmax_column: str = "vmax_kt",
    rmw_column: str = "rmw_km",
    wind_hazard_normalized: ArrayLike | None = None,
    fallback_direct_radius_km: float | None = None,
    radius_fill_strategy: str = "mean_available",
    wind_below_threshold_mode: str = "relative_to_vmax",
    run_quality_control: bool = True,
) -> dict[str, Any]:
    """
    Build one pipeline-compatible snapshot input from selected records.

    Parameters
    ----------
    precipitation : xarray.DataArray
        Already selected precipitation snapshot.
    q90, q95, q99 : float, numpy.ndarray or xarray.DataArray
        Precipitation climatological percentile thresholds.
    lon, lat : numpy.ndarray or xarray.DataArray
        Longitude and latitude grids.
    track_record : Mapping or pandas.Series
        Selected track row.
    rocloud_record : Mapping or pandas.Series
        Selected ROCLOUD row.
    r34_column_map : Mapping[str, str]
        Mapping from quadrant labels to track R34 columns.
    rocloud_column_map : Mapping[str, str]
        Mapping from quadrant labels to ROCLOUD columns.
    r34_unit : str, default="nm"
        Unit of R34 values extracted from track_record.
    rocloud_unit : str, default="km"
        Unit of ROCLOUD values extracted from rocloud_record.
    center_lon_column, center_lat_column : str
        Track columns containing cyclone center coordinates.
    vmax_column, rmw_column : str
        Track columns containing Vmax and RMW.
    wind_hazard_normalized : numpy.ndarray, xarray.DataArray or None
        Optional precomputed wind hazard field.
    fallback_direct_radius_km : float or None
        Optional fallback direct radius.
    radius_fill_strategy : str
        Radius fill strategy used later by pipeline.py.
    wind_below_threshold_mode : str
        Wind-hazard normalization mode for weak systems.
    run_quality_control : bool
        Whether pipeline-level quality control should run.

    Returns
    -------
    dict[str, Any]
        Dictionary compatible with compute_itchi_snapshot_from_grid.
    """
    center_lon = _get_required_float(
        record=track_record,
        column=center_lon_column,
        label="center longitude",
    )

    center_lat = _get_required_float(
        record=track_record,
        column=center_lat_column,
        label="center latitude",
    )

    vmax_kt = _get_optional_float(track_record, vmax_column)
    rmw_km = _get_optional_float(track_record, rmw_column)

    if wind_hazard_normalized is None and (vmax_kt is None or rmw_km is None):
        raise ValueError(
            "wind_hazard_normalized was not provided, so both vmax_kt and "
            "rmw_km must be available in the track record."
        )

    r34_by_quadrant = extract_raw_quadrant_values(
        record=track_record,
        column_map=r34_column_map,
    )

    rocloud_by_quadrant = extract_raw_quadrant_values(
        record=rocloud_record,
        column_map=rocloud_column_map,
    )

    lon, lat, wind_hazard_normalized = _align_spatial_inputs_to_precipitation(
        precipitation=precipitation,
        lon=lon,
        lat=lat,
        wind_hazard_normalized=wind_hazard_normalized,
    )

    return {
        "precipitation": precipitation,
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "lon": lon,
        "lat": lat,
        "center_lon": center_lon,
        "center_lat": center_lat,
        "r34_by_quadrant": r34_by_quadrant,
        "rocloud_by_quadrant": rocloud_by_quadrant,
        "wind_hazard_normalized": wind_hazard_normalized,
        "r34_unit": r34_unit,
        "rocloud_unit": rocloud_unit,
        "vmax_kt": vmax_kt,
        "rmw_km": rmw_km,
        "fallback_direct_radius_km": fallback_direct_radius_km,
        "radius_fill_strategy": radius_fill_strategy,
        "wind_below_threshold_mode": wind_below_threshold_mode,
        "run_quality_control": run_quality_control,
    }


def build_snapshot_input_from_tables(
    precipitation_data: xr.DataArray | xr.Dataset,
    q90: float | ArrayLike,
    q95: float | ArrayLike,
    q99: float | ArrayLike,
    lon: ArrayLike,
    lat: ArrayLike,
    track_df: pd.DataFrame,
    rocloud_df: pd.DataFrame,
    storm_id: str,
    target_time: TimeLike,
    precipitation_variable: str | None = None,
    precipitation_time_coord: str | None = None,
    precipitation_method: str = "exact",
    precipitation_tolerance: TimedeltaLike = None,
    track_tolerance_hours: float | None = None,
    rocloud_tolerance_hours: float | None = None,
    r34_column_map: Mapping[str, str] = DEFAULT_R34_COLUMN_MAP,
    rocloud_column_map: Mapping[str, str] = DEFAULT_ROCLOUD_COLUMN_MAP,
    r34_unit: str = "nm",
    rocloud_unit: str = "km",
    wind_hazard_normalized: ArrayLike | None = None,
    fallback_direct_radius_km: float | None = None,
    radius_fill_strategy: str = "mean_available",
    wind_below_threshold_mode: str = "relative_to_vmax",
    require_synoptic_target: bool = True,
    run_quality_control: bool = True,
) -> dict[str, Any]:
    """
    Build one pipeline-compatible snapshot input from tables.

    This function selects:

    - one precipitation snapshot;
    - one track row;
    - one ROCLOUD row;

    and combines them into a dictionary accepted by
    compute_itchi_snapshot_from_grid.

    Returns
    -------
    dict[str, Any]
        Pipeline-compatible snapshot input.
    """
    precipitation_snapshot = prepare_precipitation_snapshot_for_pipeline(
        data=precipitation_data,
        target_time=target_time,
        precipitation_variable=precipitation_variable,
        time_coord=precipitation_time_coord,
        method=precipitation_method,
        tolerance=precipitation_tolerance,
        require_synoptic_target=require_synoptic_target,
    )

    track_record = get_track_snapshot(
        df=track_df,
        storm_id=storm_id,
        time=target_time,
        tolerance_hours=track_tolerance_hours,
    )

    rocloud_record = get_rocloud_snapshot(
        df=rocloud_df,
        storm_id=storm_id,
        time=target_time,
        tolerance_hours=rocloud_tolerance_hours,
    )

    return build_snapshot_input_from_records(
        precipitation=precipitation_snapshot,
        q90=q90,
        q95=q95,
        q99=q99,
        lon=lon,
        lat=lat,
        track_record=track_record,
        rocloud_record=rocloud_record,
        r34_column_map=r34_column_map,
        rocloud_column_map=rocloud_column_map,
        r34_unit=r34_unit,
        rocloud_unit=rocloud_unit,
        wind_hazard_normalized=wind_hazard_normalized,
        fallback_direct_radius_km=fallback_direct_radius_km,
        radius_fill_strategy=radius_fill_strategy,
        wind_below_threshold_mode=wind_below_threshold_mode,
        run_quality_control=run_quality_control,
    )


def build_event_snapshot_inputs_from_tables(
    precipitation_data: xr.DataArray | xr.Dataset,
    q90: float | ArrayLike,
    q95: float | ArrayLike,
    q99: float | ArrayLike,
    lon: ArrayLike,
    lat: ArrayLike,
    track_df: pd.DataFrame,
    rocloud_df: pd.DataFrame,
    storm_id: str,
    target_times: Sequence[TimeLike],
    precipitation_variable: str | None = None,
    precipitation_time_coord: str | None = None,
    precipitation_method: str = "exact",
    precipitation_tolerance: TimedeltaLike = None,
    track_tolerance_hours: float | None = None,
    rocloud_tolerance_hours: float | None = None,
    r34_column_map: Mapping[str, str] = DEFAULT_R34_COLUMN_MAP,
    rocloud_column_map: Mapping[str, str] = DEFAULT_ROCLOUD_COLUMN_MAP,
    r34_unit: str = "nm",
    rocloud_unit: str = "km",
    wind_hazard_normalized_by_time: Sequence[ArrayLike | None] | None = None,
    fallback_direct_radius_km: float | None = None,
    radius_fill_strategy: str = "mean_available",
    wind_below_threshold_mode: str = "relative_to_vmax",
    require_synoptic_target: bool = True,
    run_quality_control: bool = True,
) -> list[dict[str, Any]]:
    """
    Build multiple pipeline-compatible snapshot inputs for one event.

    Parameters
    ----------
    target_times : sequence of time-like
        Times to select from precipitation, track and ROCLOUD tables.
    wind_hazard_normalized_by_time : sequence or None
        Optional sequence of precomputed wind hazard fields. If provided, its
        length must match target_times.

    Returns
    -------
    list[dict[str, Any]]
        Snapshot inputs ready for compile_itchi_event.
    """
    if wind_hazard_normalized_by_time is not None and len(
        wind_hazard_normalized_by_time
    ) != len(target_times):
        raise ValueError(
            "wind_hazard_normalized_by_time length must match target_times."
        )

    snapshot_inputs: list[dict[str, Any]] = []

    for idx, target_time in enumerate(target_times):
        wind_hazard_normalized = (
            None
            if wind_hazard_normalized_by_time is None
            else wind_hazard_normalized_by_time[idx]
        )

        snapshot_input = build_snapshot_input_from_tables(
            precipitation_data=precipitation_data,
            q90=q90,
            q95=q95,
            q99=q99,
            lon=lon,
            lat=lat,
            track_df=track_df,
            rocloud_df=rocloud_df,
            storm_id=storm_id,
            target_time=target_time,
            precipitation_variable=precipitation_variable,
            precipitation_time_coord=precipitation_time_coord,
            precipitation_method=precipitation_method,
            precipitation_tolerance=precipitation_tolerance,
            track_tolerance_hours=track_tolerance_hours,
            rocloud_tolerance_hours=rocloud_tolerance_hours,
            r34_column_map=r34_column_map,
            rocloud_column_map=rocloud_column_map,
            r34_unit=r34_unit,
            rocloud_unit=rocloud_unit,
            wind_hazard_normalized=wind_hazard_normalized,
            fallback_direct_radius_km=fallback_direct_radius_km,
            radius_fill_strategy=radius_fill_strategy,
            wind_below_threshold_mode=wind_below_threshold_mode,
            require_synoptic_target=require_synoptic_target,
            run_quality_control=run_quality_control,
        )

        snapshot_inputs.append(snapshot_input)

    return snapshot_inputs
