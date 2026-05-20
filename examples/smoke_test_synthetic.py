"""
Synthetic smoke test for ITCHI.

This script validates the end-to-end ITCHI workflow using synthetic data.

It performs:

1. synthetic lon/lat grid creation;
2. synthetic precipitation snapshot generation;
3. ITCHI event compilation;
4. event-level product calculation;
5. basic numerical checks.

Run from repository root:

    python examples/smoke_test_synthetic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from itchi.compiler import compile_itchi_event
from itchi.io import write_compiled_event


def synthetic_precipitation(
    lon: np.ndarray,
    lat: np.ndarray,
    center_lon: float,
    center_lat: float,
    amplitude: float = 35.0,
    sigma: float = 0.55,
) -> np.ndarray:
    """
    Build a synthetic precipitation snapshot.

    Parameters
    ----------
    lon : numpy.ndarray
        Longitude grid.
    lat : numpy.ndarray
        Latitude grid.
    center_lon : float
        Synthetic precipitation-center longitude.
    center_lat : float
        Synthetic precipitation-center latitude.
    amplitude : float, default=35.0
        Core precipitation amplitude.
    sigma : float, default=0.55
        Spatial spread of synthetic precipitation.

    Returns
    -------
    numpy.ndarray
        Synthetic precipitation snapshot.
    """
    distance2 = (lon - center_lon) ** 2 + (lat - center_lat) ** 2
    core = amplitude * np.exp(-distance2 / (2.0 * sigma**2))
    background = 5.0

    return background + core


def build_grid(
    lon_min: float = -2.0,
    lon_max: float = 2.0,
    lat_min: float = -2.0,
    lat_max: float = 2.0,
    n_points: int = 81,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic regular lon/lat grid.
    """
    lon_1d = np.linspace(lon_min, lon_max, n_points)
    lat_1d = np.linspace(lat_min, lat_max, n_points)

    return np.meshgrid(lon_1d, lat_1d)


def make_snapshot_input(
    precipitation: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    center_lon: float,
    center_lat: float,
    vmax_kt: float = 80.0,
    rmw_km: float = 25.0,
) -> dict:
    """
    Build one ITCHI-compatible synthetic snapshot input.
    """
    return {
        "precipitation": precipitation,
        "q90": 10.0,
        "q95": 20.0,
        "q99": 30.0,
        "lon": lon,
        "lat": lat,
        "center_lon": center_lon,
        "center_lat": center_lat,
        "r34_by_quadrant": {
            "RNE": 45.0,
            "RSE": 40.0,
            "RSW": 35.0,
            "RNW": 50.0,
        },
        "r34_unit": "nm",
        "rocloud_by_quadrant": {
            "RNE": 250.0,
            "RSE": 230.0,
            "RSW": 220.0,
            "RNW": 260.0,
        },
        "rocloud_unit": "km",
        "wind_hazard_normalized": None,
        "vmax_kt": vmax_kt,
        "rmw_km": rmw_km,
        "run_quality_control": True,
    }


def assert_unit_interval(name: str, values: np.ndarray) -> None:
    """
    Validate that a field remains within [0, 1].
    """
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))

    if min_value < 0.0 or max_value > 1.0:
        raise ValueError(
            f"{name} is outside [0, 1]. " f"Observed min={min_value}, max={max_value}."
        )


def plot_smoke_test(
    lon: np.ndarray,
    lat: np.ndarray,
    precipitation: np.ndarray,
    itchi_snapshot: np.ndarray,
    itchi_max: np.ndarray,
    itchi_acc: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """
    Plot synthetic smoke-test fields.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        constrained_layout=True,
    )

    fields = [
        (precipitation, "Precipitation snapshot 00 UTC"),
        (itchi_snapshot, "ITCHI snapshot 00 UTC"),
        (itchi_max, "ITCHI_max"),
        (itchi_acc, "ITCHI_acc"),
    ]

    for ax, (field, title) in zip(axes.ravel(), fields, strict=False):
        im = ax.pcolormesh(lon, lat, field, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved figure: {output_path}")

    plt.show()


def run_smoke_test(
    make_plot: bool = False,
    figure_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """
    Run the synthetic ITCHI smoke test.
    """
    lon, lat = build_grid()

    precip_00 = synthetic_precipitation(
        lon=lon,
        lat=lat,
        center_lon=-0.3,
        center_lat=0.0,
    )

    precip_06 = synthetic_precipitation(
        lon=lon,
        lat=lat,
        center_lon=0.3,
        center_lat=0.2,
    )

    snapshot_inputs = [
        make_snapshot_input(
            precipitation=precip_00,
            lon=lon,
            lat=lat,
            center_lon=-0.3,
            center_lat=0.0,
        ),
        make_snapshot_input(
            precipitation=precip_06,
            lon=lon,
            lat=lat,
            center_lon=0.3,
            center_lat=0.2,
        ),
    ]

    compiled = compile_itchi_event(
        snapshot_inputs=snapshot_inputs,
        time_values=["2020-09-01T00:00", "2020-09-01T06:00"],
    )

    itchi = compiled["snapshots"]["ITCHI"]
    itchi_max = compiled["event_products"]["ITCHI_max"]
    itchi_acc = compiled["event_products"]["ITCHI_acc"]

    assert_unit_interval("ITCHI", itchi)
    assert_unit_interval("ITCHI_max", itchi_max)
    assert_unit_interval("ITCHI_acc", itchi_acc)

    print("Synthetic ITCHI smoke test passed.")
    print(f"ITCHI shape: {itchi.shape}")
    print(f"ITCHI min/max: {np.nanmin(itchi):.3f}, {np.nanmax(itchi):.3f}")
    print(
        "ITCHI_max min/max: " f"{np.nanmin(itchi_max):.3f}, {np.nanmax(itchi_max):.3f}"
    )
    print(
        "ITCHI_acc min/max: " f"{np.nanmin(itchi_acc):.3f}, {np.nanmax(itchi_acc):.3f}"
    )

    if output_path is not None:
        written_path = write_compiled_event(
            compiled_event=compiled,
            path=output_path,
            attrs={
                "title": "Synthetic ITCHI smoke test",
                "storm_id": "SYNTHETIC",
                "itchi_version": "0.1",
                "precipitation_treatment": "snapshot_not_accumulation",
                "description": (
                    "Synthetic end-to-end ITCHI event generated for "
                    "workflow validation."
                ),
            },
            time_dim="time",
            snapshot_spatial_dims=("y", "x"),
            event_spatial_dims=("y", "x"),
            overwrite=True,
        )
        print(f"\nSaved compiled ITCHI event: {written_path}")

    if make_plot:
        output_path = Path(figure_path) if figure_path is not None else None

        plot_smoke_test(
            lon=lon,
            lat=lat,
            precipitation=precip_00,
            itchi_snapshot=itchi[0],
            itchi_max=itchi_max,
            itchi_acc=itchi_acc,
            output_path=output_path,
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run synthetic ITCHI smoke test.",
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

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to save compiled ITCHI event as NetCDF or Zarr.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_smoke_test(
        make_plot=args.plot,
        figure_path=args.figure_path,
        output_path=args.output_path,
    )
