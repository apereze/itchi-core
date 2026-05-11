"""
High-level ITCHI computational pipeline.

This module integrates the core ITCHI components for a single
cyclone-centered precipitation snapshot.

The current pipeline assumes that the following fields are already prepared:

- precipitation snapshot aligned with the cyclone synoptic time;
- local precipitation percentiles Q90, Q95 and Q99;
- radial distance from the cyclone center;
- R34 radius;
- ROCLOUD radius;
- normalized wind hazard field V*.

This module does not yet compute geometry, temporal alignment or wind profiles.
Those steps will be implemented in separate modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from itchi.components import compute_hazard_components
from itchi.index import compute_itchi_from_components
from itchi.masks import build_region_masks
from itchi.precipitation import compute_precipitation_hazard

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
) -> dict[str, Any]:
    """
    Compute ITCHI for a single cyclone-centered precipitation snapshot.

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
    rocloud_km : float, numpy.ndarray or xarray.DataArray
        External cyclone-attribution radius in kilometers.
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

    return {
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
