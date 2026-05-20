"""
Real-data ITCHI workflow template.

This example connects the real-data adapters implemented in itchi-core:

- IBTrACS NetCDF   -> track_df
- ROCLOUD .dat/.txt/.dot -> rocloud_df
- MSWEP NetCDF     -> precipitation snapshots

and compiles an ITCHI event product.

Important
---------
This script treats MSWEP precipitation as valid-time snapshots. It does not
sum, accumulate or resample precipitation through time.

Example
-------
Run from repository root:

    python examples/real_data_template.py \
      --ibtracs-path data/ibtracs/ibtracs.nc \
      --rocloud-path data/rocloud/EP_TCSize_2000_2024.dat \
      --mswep-path data/mswep/event_precip.nc \
      --precipitation-variable precipitation \
      --storm-id EP182023 \
      --start-time 2023-10-24T00:00:00 \
      --end-time 2023-10-25T06:00:00 \
      --q90 10 \
      --q95 20 \
      --q99 30 \
      --output-path outputs/events/EP182023_itchi.nc

For production experiments, q90/q95/q99 should normally be gridded
climatological percentile fields rather than scalars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from itchi.compiler import compile_itchi_event
from itchi.ibtracs import read_ibtracs_track_dataframe
from itchi.io import write_compiled_event
from itchi.mswep import (
    get_mswep_lon_lat,
    prepare_mswep_precipitation,
    read_mswep_dataset,
    subset_mswep_domain,
)
from itchi.precipitation_snapshots import is_synoptic_time
from itchi.rocloud import rocloud_text_to_dataframe
from itchi.snapshot_inputs import build_event_snapshot_inputs_from_tables


def parse_scalar_or_path(value: str) -> float | Path:
    """
    Parse a CLI value as a float if possible, otherwise as a path.
    """
    try:
        return float(value)
    except ValueError:
        return Path(value)


def load_threshold_field(
    value: str,
    variable_name: str | None = None,
) -> float | xr.DataArray:
    """
    Load a precipitation percentile threshold.

    Thresholds may be provided either as scalar CLI values or as NetCDF paths.
    When a NetCDF path is used, variable_name is required if the dataset has
    multiple variables.
    """
    parsed = parse_scalar_or_path(value)

    if isinstance(parsed, float):
        return parsed

    if not parsed.exists():
        raise FileNotFoundError(f"Threshold file not found: {parsed}")

    dataset = xr.open_dataset(parsed)

    if variable_name is not None:
        if variable_name not in dataset.data_vars:
            raise KeyError(f"Variable {variable_name!r} not found in {parsed}.")

        return dataset[variable_name]

    data_variables = list(dataset.data_vars)

    if len(data_variables) != 1:
        raise ValueError(
            "Threshold NetCDF files with multiple variables require an "
            "explicit --q90-variable, --q95-variable or --q99-variable."
        )

    return dataset[data_variables[0]]


def build_target_times(
    start_time: str,
    end_time: str,
    freq: str = "6h",
) -> list[pd.Timestamp]:
    """
    Build inclusive synoptic target times.
    """
    times = list(pd.date_range(start=start_time, end=end_time, freq=freq))

    if not times:
        raise ValueError("No target times were generated.")

    non_synoptic = [time for time in times if not is_synoptic_time(time)]

    if non_synoptic:
        text = ", ".join(str(time) for time in non_synoptic)
        raise ValueError(f"Generated non-synoptic target times: {text}")

    return times


def subset_if_requested(
    data: xr.DataArray | xr.Dataset,
    lon_min: float | None,
    lon_max: float | None,
    lat_min: float | None,
    lat_max: float | None,
    pad_degrees: float = 0.0,
) -> xr.DataArray | xr.Dataset:
    """
    Apply optional spatial subsetting.
    """
    bounds = [lon_min, lon_max, lat_min, lat_max]

    if all(value is None for value in bounds):
        return data

    if any(value is None for value in bounds):
        raise ValueError(
            "Spatial subsetting requires all bounds: lon_min, lon_max, "
            "lat_min and lat_max."
        )

    assert lon_min is not None
    assert lon_max is not None
    assert lat_min is not None
    assert lat_max is not None

    return subset_mswep_domain(
        data=data,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        pad_degrees=pad_degrees,
    )


def load_real_inputs(args: argparse.Namespace) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    float | xr.DataArray,
    float | xr.DataArray,
    float | xr.DataArray,
]:
    """
    Load and prepare all real-data inputs.
    """
    track_df = read_ibtracs_track_dataframe(
        path=args.ibtracs_path,
        storm_id=args.storm_id,
    )

    rocloud_df = rocloud_text_to_dataframe(
        path=args.rocloud_path,
        synoptic_only=args.rocloud_synoptic_only,
        validate_record_counts=args.validate_rocloud_record_counts,
    )

    mswep_dataset = read_mswep_dataset(
        path=args.mswep_path,
        chunks=args.mswep_chunks,
    )

    mswep_dataset = subset_if_requested(
        data=mswep_dataset,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        pad_degrees=args.pad_degrees,
    )

    precipitation = prepare_mswep_precipitation(
        data=mswep_dataset,
        precipitation_variable=args.precipitation_variable,
        synoptic_only=True,
    )

    lon, lat = get_mswep_lon_lat(precipitation)

    q90 = load_threshold_field(args.q90, variable_name=args.q90_variable)
    q95 = load_threshold_field(args.q95, variable_name=args.q95_variable)
    q99 = load_threshold_field(args.q99, variable_name=args.q99_variable)

    q90 = subset_threshold_if_requested(q90, args)
    q95 = subset_threshold_if_requested(q95, args)
    q99 = subset_threshold_if_requested(q99, args)

    return track_df, rocloud_df, precipitation, lon, lat, q90, q95, q99


def subset_threshold_if_requested(
    threshold: float | xr.DataArray,
    args: argparse.Namespace,
) -> float | xr.DataArray:
    """
    Apply the same optional spatial subset to gridded threshold fields.
    """
    if isinstance(threshold, float):
        return threshold

    return subset_if_requested(
        data=threshold,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        pad_degrees=args.pad_degrees,
    )


def run_real_data_template(args: argparse.Namespace) -> Path:
    """
    Execute the real-data ITCHI workflow.
    """
    target_times = build_target_times(
        start_time=args.start_time,
        end_time=args.end_time,
    )

    (
        track_df,
        rocloud_df,
        precipitation,
        lon,
        lat,
        q90,
        q95,
        q99,
    ) = load_real_inputs(args)

    snapshot_inputs = build_event_snapshot_inputs_from_tables(
        precipitation_data=precipitation,
        q90=q90,
        q95=q95,
        q99=q99,
        lon=lon,
        lat=lat,
        track_df=track_df,
        rocloud_df=rocloud_df,
        storm_id=args.storm_id,
        target_times=target_times,
        precipitation_variable=None,
        precipitation_time_coord="valid_time",
        precipitation_method=args.precipitation_method,
        precipitation_tolerance=args.precipitation_tolerance,
        track_tolerance_hours=args.track_tolerance_hours,
        rocloud_tolerance_hours=args.rocloud_tolerance_hours,
        r34_unit=args.r34_unit,
        rocloud_unit="km",
        fallback_direct_radius_km=args.fallback_direct_radius_km,
        radius_fill_strategy=args.radius_fill_strategy,
        wind_below_threshold_mode=args.wind_below_threshold_mode,
        run_quality_control=not args.skip_quality_control,
    )

    compiled = compile_itchi_event(
        snapshot_inputs=snapshot_inputs,
        time_values=target_times,
        run_snapshot_quality_control=not args.skip_quality_control,
    )

    output_path = write_compiled_event(
        compiled_event=compiled,
        path=args.output_path,
        attrs={
            "title": "Real-data ITCHI event template output",
            "storm_id": args.storm_id,
            "itchi_version": "0.1",
            "precipitation_source": "MSWEP NetCDF",
            "track_source": "IBTrACS NetCDF",
            "rocloud_source": "ROCLOUD text database",
            "precipitation_treatment": "snapshot_not_accumulation",
            "start_time": str(target_times[0]),
            "end_time": str(target_times[-1]),
        },
        overwrite=not args.no_overwrite,
    )

    print("Real-data ITCHI template completed.")
    print(f"Storm ID: {args.storm_id}")
    print(f"Snapshots: {len(target_times)}")
    print(f"Output: {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compile an ITCHI event from IBTrACS, ROCLOUD and MSWEP.",
    )

    parser.add_argument("--ibtracs-path", required=True, help="IBTrACS NetCDF path.")
    parser.add_argument("--rocloud-path", required=True, help="ROCLOUD .dat/.txt/.dot path.")
    parser.add_argument("--mswep-path", required=True, help="MSWEP NetCDF path.")
    parser.add_argument("--precipitation-variable", default=None)
    parser.add_argument("--storm-id", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument(
        "--q90",
        required=True,
        help="Q90 scalar value or NetCDF path.",
    )
    parser.add_argument(
        "--q95",
        required=True,
        help="Q95 scalar value or NetCDF path.",
    )
    parser.add_argument(
        "--q99",
        required=True,
        help="Q99 scalar value or NetCDF path.",
    )
    parser.add_argument("--q90-variable", default=None)
    parser.add_argument("--q95-variable", default=None)
    parser.add_argument("--q99-variable", default=None)

    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--pad-degrees", type=float, default=0.0)

    parser.add_argument(
        "--precipitation-method",
        choices=("exact", "nearest"),
        default="exact",
    )
    parser.add_argument("--precipitation-tolerance", default=None)
    parser.add_argument("--track-tolerance-hours", type=float, default=None)
    parser.add_argument("--rocloud-tolerance-hours", type=float, default=None)

    parser.add_argument(
        "--r34-unit",
        default="nm",
        help="Unit of R34 radii extracted from IBTrACS. Default: nm.",
    )
    parser.add_argument("--fallback-direct-radius-km", type=float, default=None)
    parser.add_argument("--radius-fill-strategy", default="mean_available")
    parser.add_argument("--wind-below-threshold-mode", default="relative_to_vmax")

    parser.add_argument(
        "--rocloud-synoptic-only",
        action="store_true",
        help="Filter ROCLOUD rows to 00/06/12/18 UTC before matching.",
    )
    parser.add_argument(
        "--validate-rocloud-record-counts",
        action="store_true",
        help="Strictly validate ROCLOUD declared entries against parsed rows.",
    )
    parser.add_argument(
        "--skip-quality-control",
        action="store_true",
        help="Disable snapshot quality-control checks.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if output path already exists.",
    )
    parser.add_argument(
        "--mswep-chunks",
        default=None,
        help="Optional xarray chunks argument. Use cautiously from CLI.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_real_data_template(parse_args())
