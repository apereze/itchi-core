"""
Synthetic table-driven smoke test for ITCHI.

This example validates the workflow that connects tabular cyclone metadata,
ROCLOUD radii and precipitation snapshots into the ITCHI compiler.

It is closer to the expected real-data workflow than
examples/smoke_test_synthetic.py because it builds snapshot inputs from
track and ROCLOUD tables.

Run from repository root:

    python examples/smoke_test_from_tables.py

Optionally export the compiled event:

    python examples/smoke_test_from_tables.py \
      --output-path outputs/events/smoke_test_from_tables.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from itchi.compiler import compile_itchi_event
from itchi.io import write_compiled_event
from itchi.snapshot_inputs import build_event_snapshot_inputs_from_tables


def build_grid(n_points: int = 81) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Build a synthetic lon/lat grid with explicit xarray coordinates.
    """
    y = np.arange(n_points)
    x = np.arange(n_points)

    lon_1d = np.linspace(-2.0, 2.0, n_points)
    lat_1d = np.linspace(-2.0, 2.0, n_points)

    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    lon = xr.DataArray(
        data=lon_2d,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="lon",
    )

    lat = xr.DataArray(
        data=lat_2d,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="lat",
    )

    return lon, lat


def synthetic_precipitation_snapshot(
    lon: xr.DataArray,
    lat: xr.DataArray,
    center_lon: float,
    center_lat: float,
    amplitude: float = 35.0,
    sigma: float = 0.55,
) -> xr.DataArray:
    """
    Build one synthetic precipitation snapshot.

    The field represents an instantaneous/snapshot precipitation intensity,
    not an accumulated precipitation field.
    """
    distance2 = (lon - center_lon) ** 2 + (lat - center_lat) ** 2
    core = amplitude * np.exp(-distance2 / (2.0 * sigma**2))
    background = 5.0

    precipitation = background + core
    precipitation.name = "precipitation"

    return precipitation


def build_precipitation_data(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """
    Build a two-snapshot precipitation DataArray with valid_time dimension.
    """
    precip_00 = synthetic_precipitation_snapshot(
        lon=lon,
        lat=lat,
        center_lon=-0.3,
        center_lat=0.0,
    )

    precip_06 = synthetic_precipitation_snapshot(
        lon=lon,
        lat=lat,
        center_lon=0.3,
        center_lat=0.2,
    )

    valid_times = np.array(
        [
            "2020-09-01T00:00:00",
            "2020-09-01T06:00:00",
        ],
        dtype="datetime64[ns]",
    )

    precipitation = xr.concat(
        [precip_00, precip_06],
        dim=xr.DataArray(
            data=valid_times,
            dims=("valid_time",),
            name="valid_time",
        ),
    )

    precipitation.name = "precipitation"

    return precipitation


def build_track_table() -> pd.DataFrame:
    """
    Build a synthetic standardized track table.

    R34 values are intentionally provided in nautical miles, matching the
    common best-track convention.
    """
    return pd.DataFrame(
        {
            "storm_id": ["SYNTHETIC", "SYNTHETIC"],
            "time": pd.to_datetime(
                [
                    "2020-09-01T00:00:00",
                    "2020-09-01T06:00:00",
                ]
            ),
            "lat": [0.0, 0.2],
            "lon": [-0.3, 0.3],
            "vmax_kt": [80.0, 75.0],
            "rmw_km": [25.0, 30.0],
            "r34_rne": [45.0, 40.0],
            "r34_rse": [40.0, 35.0],
            "r34_rsw": [35.0, 30.0],
            "r34_rnw": [50.0, 45.0],
        }
    )


def build_rocloud_table() -> pd.DataFrame:
    """
    Build a synthetic standardized ROCLOUD table.

    ROCLOUD values are provided in kilometers.
    """
    return pd.DataFrame(
        {
            "storm_id": ["SYNTHETIC", "SYNTHETIC"],
            "time": pd.to_datetime(
                [
                    "2020-09-01T00:00:00",
                    "2020-09-01T06:00:00",
                ]
            ),
            "rocloud_rne": [250.0, 240.0],
            "rocloud_rse": [230.0, 220.0],
            "rocloud_rsw": [220.0, 210.0],
            "rocloud_rnw": [260.0, 250.0],
        }
    )


def assert_unit_interval(name: str, values: xr.DataArray) -> None:
    """
    Validate that a hazard field remains within [0, 1].
    """
    min_value = float(values.min(skipna=True))
    max_value = float(values.max(skipna=True))

    if min_value < 0.0 or max_value > 1.0:
        raise ValueError(
            f"{name} is outside [0, 1]. " f"Observed min={min_value}, max={max_value}."
        )


def plot_table_smoke_test(
    compiled: dict,
    output_path: Path | None = None,
) -> None:
    """
    Plot diagnostic event-level outputs.
    """
    itchi = compiled["snapshots"]["ITCHI"]
    itchi_max = compiled["event_products"]["ITCHI_max"]
    itchi_acc = compiled["event_products"]["ITCHI_acc"]

    fields = [
        (itchi.isel(time=0), "ITCHI 00 UTC"),
        (itchi.isel(time=1), "ITCHI 06 UTC"),
        (itchi_max, "ITCHI_max"),
        (itchi_acc, "ITCHI_acc"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        constrained_layout=True,
    )

    for ax, (field, title) in zip(axes.ravel(), fields, strict=False):
        image = ax.pcolormesh(field.values, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved figure: {output_path}")

    plt.show()


def run_table_smoke_test(
    output_path: str | None = None,
    make_plot: bool = False,
    figure_path: str | None = None,
) -> None:
    """
    Run the table-driven synthetic smoke test.
    """
    lon, lat = build_grid()
    precipitation = build_precipitation_data(lon=lon, lat=lat)
    track_df = build_track_table()
    rocloud_df = build_rocloud_table()

    target_times = [
        "2020-09-01T00:00:00",
        "2020-09-01T06:00:00",
    ]

    snapshot_inputs = build_event_snapshot_inputs_from_tables(
        precipitation_data=precipitation,
        q90=10.0,
        q95=20.0,
        q99=30.0,
        lon=lon,
        lat=lat,
        track_df=track_df,
        rocloud_df=rocloud_df,
        storm_id="SYNTHETIC",
        target_times=target_times,
        r34_unit="nm",
        rocloud_unit="km",
    )

    compiled = compile_itchi_event(
        snapshot_inputs=snapshot_inputs,
        time_values=target_times,
    )

    itchi = compiled["snapshots"]["ITCHI"]
    itchi_max = compiled["event_products"]["ITCHI_max"]
    itchi_acc = compiled["event_products"]["ITCHI_acc"]

    assert_unit_interval("ITCHI", itchi)
    assert_unit_interval("ITCHI_max", itchi_max)
    assert_unit_interval("ITCHI_acc", itchi_acc)

    print("Table-driven synthetic ITCHI smoke test passed.")
    print(f"ITCHI dims: {itchi.dims}")
    print(f"ITCHI shape: {itchi.shape}")
    print(f"ITCHI min/max: {float(itchi.min()):.3f}, {float(itchi.max()):.3f}")
    print(
        "ITCHI_max min/max: "
        f"{float(itchi_max.min()):.3f}, {float(itchi_max.max()):.3f}"
    )
    print(
        "ITCHI_acc min/max: "
        f"{float(itchi_acc.min()):.3f}, {float(itchi_acc.max()):.3f}"
    )

    print("\nMetadata:")
    for item in compiled["metadata"]:
        print(item)

    if output_path is not None:
        written_path = write_compiled_event(
            compiled_event=compiled,
            path=output_path,
            attrs={
                "title": "Synthetic table-driven ITCHI smoke test",
                "storm_id": "SYNTHETIC",
                "itchi_version": "0.1",
                "precipitation_treatment": "snapshot_not_accumulation",
                "workflow": "tables_to_snapshot_inputs_to_compiler",
            },
            overwrite=True,
        )

        print(f"\nSaved compiled ITCHI event: {written_path}")

    if make_plot:
        plot_table_smoke_test(
            compiled=compiled,
            output_path=Path(figure_path) if figure_path is not None else None,
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run table-driven synthetic ITCHI smoke test.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to save compiled ITCHI event as NetCDF or Zarr.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show diagnostic plots.",
    )

    parser.add_argument(
        "--figure-path",
        type=str,
        default=None,
        help="Optional path to save diagnostic figure.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_table_smoke_test(
        output_path=args.output_path,
        make_plot=args.plot,
        figure_path=args.figure_path,
    )
