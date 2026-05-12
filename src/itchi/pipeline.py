"""
High-level ITCHI computational pipeline.

This module integrates the core ITCHI components for cyclone-centered
precipitation snapshots.

Two levels of calculation are provided:

1. compute_itchi_snapshot
   Computes ITCHI when radius_km, R34_q and ROCLOUD_q are already prepared.

2. compute_itchi_snapshot_from_grid
   Computes ITCHI directly from lon/lat grids, cyclone center coordinates,
   quadrant-specific R34 and quadrant-specific ROCLOUD radii.

The second function is closer to the expected operational use with real
tropical cyclone data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from itchi.components import compute_hazard_components
from itchi.geometry import (
    assign_quadrant_radius,
    build_geometry_fields,
)
from itchi.index import compute_itchi_from_components
from itchi.masks import build_region_masks
from itchi.precipitation import compute_precipitation_hazard
from itchi.quality_control import run_snapshot_quality_control
from itchi.units import convert_quadrant_radii_to_km

ArrayLike = np.ndarray | xr.DataArray


def compute_itchi_snapshot(
    precipitation: ArrayLike,
    q90: float | ArrayLike,
    q95: float | ArrayLike,
    q99: float | ArrayLike,
    radius_km: ArrayLike,
    r34_km: float | ArrayLike,
    rocloud_km: float | ArrayLike,
    wind_hazard_normalized: ArrayLike,
    alpha_wind: float = 1.0,
    beta_precip_direct: float = 1.0,
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
    run_quality_control: bool = False,
) -> dict[str, Any]:
    """
    Compute ITCHI for a single cyclone-centered precipitation snapshot.

    This lower-level function assumes that the radial geometry is already
    available.

    Parameters
    ----------
    precipitation : numpy.ndarray or xarray.DataArray
        Precipitation snapshot field.
    q90 : float, numpy.ndarray or xarray.DataArray
        Local 90th percentile of comparable precipitation snapshots.
    q95 : float, numpy.ndarray or xarray.DataArray
        Local 95th percentile of comparable precipitation snapshots.
    q99 : float, numpy.ndarray or xarray.DataArray
        Local 99th percentile of comparable precipitation snapshots.
    radius_km : numpy.ndarray or xarray.DataArray
        Radial distance from the cyclone center in kilometers.
    r34_km : float, numpy.ndarray or xarray.DataArray
        Tropical-storm-force wind radius in kilometers.
        This can be either a scalar or a grid-cell-specific R34_q field.
    rocloud_km : float, numpy.ndarray or xarray.DataArray
        External cyclone-attribution radius in kilometers.
        This can be either a scalar or a grid-cell-specific ROCLOUD_q field.
    wind_hazard_normalized : numpy.ndarray or xarray.DataArray
        Normalized wind hazard field V* in [0, 1].
    alpha_wind : float, default=1.0
        Weight/exponent for wind hazard in the direct component.
    beta_precip_direct : float, default=1.0
        Weight/exponent for direct precipitation hazard.
    lambda_direct : float, default=1.0
        Weight/exponent for the direct component in final ITCHI.
    mu_indirect : float, default=1.0
        Weight/exponent for the indirect component in final ITCHI.
    run_quality_control : bool, default=False
        Whether to run standard snapshot quality-control checks.

    Returns
    -------
    dict[str, Any]
        Dictionary containing masks, intermediate components and final ITCHI.
    """
    precipitation_hazard = compute_precipitation_hazard(
        precipitation=precipitation,
        q90=q90,
        q95=q95,
        q99=q99,
    )

    masks = build_region_masks(
        radius_km=radius_km,
        r34_km=r34_km,
        rocloud_km=rocloud_km,
    )

    components = compute_hazard_components(
        precipitation_hazard=precipitation_hazard,
        wind_hazard_normalized=wind_hazard_normalized,
        direct_mask=masks["direct"],
        indirect_mask=masks["indirect"],
        alpha=alpha_wind,
        beta=beta_precip_direct,
    )

    itchi = compute_itchi_from_components(
        components=components,
        lambda_direct=lambda_direct,
        mu_indirect=mu_indirect,
    )

    result = {
        "M_direct": masks["direct"],
        "M_indirect": masks["indirect"],
        "M_exterior": masks["exterior"],
        "H_P": precipitation_hazard,
        "H_Pdir": components["H_Pdir"],
        "H_Pind": components["H_Pind"],
        "H_W": components["H_W"],
        "H_dir": components["H_dir"],
        "H_ind": components["H_ind"],
        "ITCHI": itchi,
    }

    if run_quality_control:
        run_snapshot_quality_control(result)

    return result


def compute_itchi_snapshot_from_grid(
    precipitation: ArrayLike,
    q90: float | ArrayLike,
    q95: float | ArrayLike,
    q99: float | ArrayLike,
    lon: ArrayLike,
    lat: ArrayLike,
    center_lon: float,
    center_lat: float,
    r34_by_quadrant: Mapping[str, float | int | None],
    rocloud_by_quadrant: Mapping[str, float | int | None],
    wind_hazard_normalized: ArrayLike,
    r34_unit: str = "nm",
    rocloud_unit: str = "km",
    alpha_wind: float = 1.0,
    beta_precip_direct: float = 1.0,
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
    run_quality_control: bool = False,
) -> dict[str, Any]:
    """
    Compute ITCHI directly from a lon/lat grid and cyclone-center metadata.

    This function is the preferred high-level interface for a real cyclone
    snapshot. It performs the following steps:

    1. Convert R34 and ROCLOUD radii to kilometers.
    2. Compute radial distance from each grid cell to the cyclone center.
    3. Compute the relative quadrant of each grid cell.
    4. Assign quadrant-specific R34_q and ROCLOUD_q to each grid cell.
    5. Compute precipitation hazard, masks, components and final ITCHI.

    Parameters
    ----------
    precipitation : numpy.ndarray or xarray.DataArray
        Precipitation snapshot field.
    q90 : float, numpy.ndarray or xarray.DataArray
        Local 90th percentile of comparable precipitation snapshots.
    q95 : float, numpy.ndarray or xarray.DataArray
        Local 95th percentile of comparable precipitation snapshots.
    q99 : float, numpy.ndarray or xarray.DataArray
        Local 99th percentile of comparable precipitation snapshots.
    lon : numpy.ndarray or xarray.DataArray
        Longitude field in degrees.
    lat : numpy.ndarray or xarray.DataArray
        Latitude field in degrees.
    center_lon : float
        Cyclone center longitude in degrees.
    center_lat : float
        Cyclone center latitude in degrees.
    r34_by_quadrant : Mapping[str, float, int or None]
        R34 radii by quadrant. Accepted keys are RNE, RSE, RSW, RNW
        or aliases NE, SE, SW, NW.
    rocloud_by_quadrant : Mapping[str, float, int or None]
        ROCLOUD radii by quadrant. Accepted keys are RNE, RSE, RSW, RNW
        or aliases NE, SE, SW, NW.
    wind_hazard_normalized : numpy.ndarray or xarray.DataArray
        Normalized wind hazard field V* in [0, 1].
    r34_unit : str, default="nm"
        Unit of R34 radii. IBTrACS commonly reports wind radii in nautical miles.
    rocloud_unit : str, default="km"
        Unit of ROCLOUD radii.
    alpha_wind : float, default=1.0
        Weight/exponent for wind hazard in the direct component.
    beta_precip_direct : float, default=1.0
        Weight/exponent for direct precipitation hazard.
    lambda_direct : float, default=1.0
        Weight/exponent for the direct component in final ITCHI.
    mu_indirect : float, default=1.0
        Weight/exponent for the indirect component in final ITCHI.
    run_quality_control : bool, default=False
        Whether to run standard snapshot quality-control checks.
    Returns
    -------
    dict[str, Any]
        Dictionary containing geometry fields, masks, intermediate components
        and final ITCHI.
    """
    r34_km_by_quadrant = convert_quadrant_radii_to_km(
        radii_by_quadrant=r34_by_quadrant,
        input_unit=r34_unit,
    )

    rocloud_km_by_quadrant = convert_quadrant_radii_to_km(
        radii_by_quadrant=rocloud_by_quadrant,
        input_unit=rocloud_unit,
    )

    geometry = build_geometry_fields(
        lon=lon,
        lat=lat,
        center_lon=center_lon,
        center_lat=center_lat,
    )

    radius_km = geometry["radius_km"]
    quadrant = geometry["quadrant"]

    r34_q = assign_quadrant_radius(
        quadrant=quadrant,
        radii_by_quadrant=r34_km_by_quadrant,
    )

    rocloud_q = assign_quadrant_radius(
        quadrant=quadrant,
        radii_by_quadrant=rocloud_km_by_quadrant,
    )

    result = compute_itchi_snapshot(
        precipitation=precipitation,
        q90=q90,
        q95=q95,
        q99=q99,
        radius_km=radius_km,
        r34_km=r34_q,
        rocloud_km=rocloud_q,
        wind_hazard_normalized=wind_hazard_normalized,
        alpha_wind=alpha_wind,
        beta_precip_direct=beta_precip_direct,
        lambda_direct=lambda_direct,
        mu_indirect=mu_indirect,
        run_quality_control=False,
    )

    full_result = {
        "radius_km": radius_km,
        "quadrant": quadrant,
        "R34_q": r34_q,
        "ROCLOUD_q": rocloud_q,
        **result,
    }

    if run_quality_control:
        run_snapshot_quality_control(full_result)

    return full_result
