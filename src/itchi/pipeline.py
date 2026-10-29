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

3. compute_itchi_snapshot_v02
   ITCHI v0.2 entry point: takes P6, R24, A72 and their percentile
   climatologies, computes the precipitation matrix, the W06 wind with an
   outer tail that ends at R_W06, and the attribution radius ROCLOUD with
   R_W06 as a floor by quadrant. Its keyword defaults equal
   configs/default.yaml (see config.itchi_v02_settings_from_config).

ITCHI v0.2 options of compute_itchi_snapshot_from_grid (its defaults
reproduce v0.1 and are kept for backward compatibility):

- ``precipitation_hazard``: precomputed H_P, e.g. the P6 x (R24, A72) matrix of
  precipitation.compute_precipitation_matrix_hazard; q90/q95/q99 are then
  not used.
- ``wind_model="w06_tail"``: W06 wind profile calibrated to the observed
  mean R34, normalized by quadrant, with a tail that decays to zero at the
  attribution radius; the tail enters the indirect component.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from itchi.components import compute_hazard_components
from itchi.constants import ITCHI_VERSION, QUADRANTS
from itchi.geometry import (
    assign_quadrant_radius,
    build_geometry_fields,
)
from itchi.index import compute_itchi_from_components
from itchi.masks import build_region_masks
from itchi.parametric_radii import (
    R_W06_DEFAULTS,
    compute_w06_reference_radii,
    compute_w06_wind_hazard_with_tail,
)
from itchi.precipitation import (
    PRECIP_MATRIX_V02,
    compute_precipitation_hazard,
    compute_precipitation_matrix_hazard,
)
from itchi.quality_control import run_snapshot_quality_control
from itchi.radii import (
    resolve_attribution_radius,
    resolve_attribution_radius_with_w06,
    resolve_direct_radius,
    standardize_quadrant_radii,
)
from itchi.units import convert_quadrant_radii_to_km
from itchi.wind import compute_wind_hazard_from_profile

ArrayLike = np.ndarray | xr.DataArray


def compute_itchi_snapshot(
    precipitation: ArrayLike,
    q90: float | ArrayLike | None,
    q95: float | ArrayLike | None,
    q99: float | ArrayLike | None,
    radius_km: ArrayLike,
    r34_km: float | ArrayLike,
    rocloud_km: float | ArrayLike,
    wind_hazard_normalized: ArrayLike,
    alpha_wind: float = 1.0,
    beta_precip_direct: float = 1.0,
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
    run_quality_control: bool = False,
    precipitation_hazard: ArrayLike | None = None,
    wind_hazard_indirect: ArrayLike | None = None,
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
    precipitation_hazard : numpy.ndarray, xarray.DataArray or None
        ITCHI v0.2: precomputed H_P (e.g. P6 x (R24, A72) matrix). If given,
        q90/q95/q99 are ignored.
    wind_hazard_indirect : numpy.ndarray, xarray.DataArray or None
        ITCHI v0.2: wind hazard with outer tail for the indirect region.

    Returns
    -------
    dict[str, Any]
        Dictionary containing masks, intermediate components and final ITCHI.
    """
    if precipitation_hazard is None:
        if q90 is None or q95 is None or q99 is None:
            raise ValueError("q90, q95 and q99 are required when precipitation_hazard is None.")
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
        wind_hazard_indirect=wind_hazard_indirect,
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

    for key in ("H_Wdir", "H_Wind"):
        if key in components:
            result[key] = components[key]

    if run_quality_control:
        run_snapshot_quality_control(result)

    return result


def compute_itchi_snapshot_from_grid(
    precipitation: ArrayLike,
    q90: float | ArrayLike | None,
    q95: float | ArrayLike | None,
    q99: float | ArrayLike | None,
    lon: ArrayLike,
    lat: ArrayLike,
    center_lon: float,
    center_lat: float,
    r34_by_quadrant: Mapping[str, float | int | None],
    rocloud_by_quadrant: Mapping[str, float | int | None],
    wind_hazard_normalized: ArrayLike | None = None,
    r34_unit: str = "nm",
    rocloud_unit: str = "km",
    vmax_kt: float | None = None,
    rmw_km: float | None = None,
    fallback_direct_radius_km: float | None = None,
    radius_fill_strategy: str = "mean_available",
    wind_below_threshold_mode: str = "relative_to_vmax",
    wind_inner_exponent: float = 1.0,
    wind_outer_decay_exponent: float = 0.5,
    wind_outer_radius_km: float | None = None,
    alpha_wind: float = 1.0,
    beta_precip_direct: float = 1.0,
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
    run_quality_control: bool = False,
    precipitation_hazard: ArrayLike | None = None,
    wind_model: str = "v01",
    wind_tail_h34: float = 0.2,
    r34_mean_km: float | None = None,
    w06_profile_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute ITCHI directly from a lon/lat grid and cyclone-center metadata.

    This function is the preferred high-level interface for a real cyclone
    snapshot. It performs the following steps:

    1. Convert R34 and ROCLOUD radii to kilometers.
    2. Resolve the effective direct-region radius R_direct_q.
    3. Resolve the external attribution radius ROCLOUD_q.
    4. Compute radial distance from each grid cell to the cyclone center.
    5. Compute the relative quadrant of each grid cell.
    6. Assign quadrant-specific R_direct_q and ROCLOUD_q to each grid cell.
    7. Use or compute normalized wind hazard V*.
    8. Compute precipitation hazard, masks, components and final ITCHI.

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
        or aliases NE, SE, SW, NW. These may be partially missing.
    rocloud_by_quadrant : Mapping[str, float, int or None]
        ROCLOUD radii by quadrant. Accepted keys are RNE, RSE, RSW, RNW
        or aliases NE, SE, SW, NW. These may be partially missing.
    wind_hazard_normalized : numpy.ndarray, xarray.DataArray or None, default=None
        Precomputed normalized wind hazard field V* in [0, 1].
        If None, it is computed from vmax_kt, rmw_km and radius_km.
    r34_unit : str, default="nm"
        Unit of R34 radii. IBTrACS commonly reports wind radii in nautical miles.
    rocloud_unit : str, default="km"
        Unit of ROCLOUD radii.
    vmax_kt : float or None, default=None
        Maximum sustained wind in knots. Used both for direct-radius fallback
        and wind-hazard profile computation.
    rmw_km : float or None, default=None
        Radius of maximum wind in kilometers. Used both as a possible direct
        radius fallback and for wind-hazard profile computation.
    fallback_direct_radius_km : float or None, default=None
        Optional configured fallback radius in kilometers.
    radius_fill_strategy : str, default="mean_available"
        Strategy used to fill missing R34 or ROCLOUD quadrants.
    wind_below_threshold_mode : {"relative_to_vmax", "zero"}, default="relative_to_vmax"
        Normalization rule when vmax_kt <= 34 kt.
    wind_inner_exponent : float, default=1.0
        Inner radial wind profile exponent.
    wind_outer_decay_exponent : float, default=0.5
        Outer radial wind decay exponent.
    wind_outer_radius_km : float or None, default=None
        Optional radius beyond which parametric wind is set to zero.
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
    precipitation_hazard : numpy.ndarray, xarray.DataArray or None
        ITCHI v0.2: precomputed H_P (e.g. P6 x (R24, A72) matrix); q90/q95/q99 are
        then ignored and may be None.
    wind_model : {"v01", "w06_tail"}, default="v01"
        "v01": wind hazard only inside R_direct (provided or modified Rankine).
        "w06_tail": ITCHI v0.2 W06 wind with outer tail down to the
        attribution radius; requires vmax_kt and center_lat.
    wind_tail_h34 : float, default=0.2
        Hazard at the 34-kt radius for wind_model="w06_tail".
    r34_mean_km : float or None, default=None
        Observed mean R34 (km) used to calibrate W06; None takes the mean of
        the valid observed R34 quadrants (no calibration if there are none).
    w06_profile_kwargs : Mapping or None, default=None
        Extra arguments for Willoughby2006Profile (e.g. rmax_variant).

    Returns
    -------
    dict[str, Any]
        Dictionary containing geometry fields, masks, intermediate components,
        radius-resolution metadata, wind metadata and final ITCHI.
    """
    if wind_model not in {"v01", "w06_tail"}:
        raise ValueError(f"Unsupported wind_model: {wind_model!r}")
    r34_km_by_quadrant = convert_quadrant_radii_to_km(
        radii_by_quadrant=r34_by_quadrant,
        input_unit=r34_unit,
    )

    rocloud_km_by_quadrant = convert_quadrant_radii_to_km(
        radii_by_quadrant=rocloud_by_quadrant,
        input_unit=rocloud_unit,
    )

    resolved_direct_radius = resolve_direct_radius(
        r34_by_quadrant=r34_km_by_quadrant,
        vmax_kt=vmax_kt,
        rmw_km=rmw_km,
        fallback_radius_km=fallback_direct_radius_km,
        fill_strategy=radius_fill_strategy,
    )

    resolved_rocloud_radius = resolve_attribution_radius(
        rocloud_by_quadrant=rocloud_km_by_quadrant,
        fill_strategy=radius_fill_strategy,
    )

    geometry = build_geometry_fields(
        lon=lon,
        lat=lat,
        center_lon=center_lon,
        center_lat=center_lat,
    )

    radius_km = geometry["radius_km"]
    quadrant = geometry["quadrant"]

    direct_radius_q = assign_quadrant_radius(
        quadrant=quadrant,
        radii_by_quadrant=resolved_direct_radius.radii,
    )

    rocloud_q = assign_quadrant_radius(
        quadrant=quadrant,
        radii_by_quadrant=resolved_rocloud_radius.radii,
    )

    wind_hazard_indirect = None
    wind_extra: dict[str, Any] = {}

    if wind_model == "w06_tail":
        if wind_hazard_normalized is not None:
            raise ValueError("wind_hazard_normalized cannot be combined with wind_model='w06_tail'.")
        if vmax_kt is None:
            raise ValueError("wind_model='w06_tail' requires vmax_kt.")

        if r34_mean_km is None:
            observed = [
                float(v) for v in r34_km_by_quadrant.values()
                if v is not None and np.isfinite(float(v)) and float(v) > 0.0
            ]
            r34_mean_km = float(np.mean(observed)) if observed else None

        w06 = compute_w06_wind_hazard_with_tail(
            radius_km=radius_km,
            direct_radius_km=direct_radius_q,
            attribution_radius_km=rocloud_q,
            vmax_kt=float(vmax_kt),
            lat=float(center_lat),
            rmw_km=rmw_km,
            r34_mean_km=r34_mean_km,
            h34=wind_tail_h34,
            **dict(w06_profile_kwargs or {}),
        )
        wind_hazard_for_pipeline = w06["H_W"]
        wind_hazard_indirect = w06["H_W"]
        wind_hazard_source = "w06_tail"
        wind_extra = {
            "wind_kt": w06["wind_kt"],
            "w06_x1_km": w06["x1_km"],
            "w06_x1_calibrated": w06["x1_calibrated"],
            "wind_tail_h34": wind_tail_h34,
        }

    elif wind_hazard_normalized is None:
        if vmax_kt is None or rmw_km is None:
            raise ValueError(
                "wind_hazard_normalized was not provided. "
                "To compute wind hazard internally, both vmax_kt and rmw_km "
                "must be provided."
            )

        wind_hazard_for_pipeline = compute_wind_hazard_from_profile(
            radius_km=radius_km,
            vmax_kt=float(vmax_kt),
            rmw_km=float(rmw_km),
            below_threshold_mode=wind_below_threshold_mode,
            inner_exponent=wind_inner_exponent,
            outer_decay_exponent=wind_outer_decay_exponent,
            outer_radius_km=wind_outer_radius_km,
        )

        wind_hazard_source = "radial_profile"

    else:
        wind_hazard_for_pipeline = wind_hazard_normalized
        wind_hazard_source = "provided"

    result = compute_itchi_snapshot(
        precipitation=precipitation,
        q90=q90,
        q95=q95,
        q99=q99,
        radius_km=radius_km,
        r34_km=direct_radius_q,
        rocloud_km=rocloud_q,
        wind_hazard_normalized=wind_hazard_for_pipeline,
        alpha_wind=alpha_wind,
        beta_precip_direct=beta_precip_direct,
        lambda_direct=lambda_direct,
        mu_indirect=mu_indirect,
        run_quality_control=False,
        precipitation_hazard=precipitation_hazard,
        wind_hazard_indirect=wind_hazard_indirect,
    )

    full_result = {
        "radius_km": radius_km,
        "quadrant": quadrant,
        "R_direct_q": direct_radius_q,
        "ROCLOUD_q": rocloud_q,
        "R_direct_source": resolved_direct_radius.source,
        "R_direct_filled_quadrants": resolved_direct_radius.filled_quadrants,
        "R_direct_used_fallback": resolved_direct_radius.used_fallback,
        "ROCLOUD_source": resolved_rocloud_radius.source,
        "ROCLOUD_filled_quadrants": resolved_rocloud_radius.filled_quadrants,
        "wind_hazard_source": wind_hazard_source,
        **result,
    }

    # v0.2 metadata only when a v0.2 option is used (v0.1 output keys unchanged)
    if precipitation_hazard is not None or wind_model != "v01":
        full_result.update(
            itchi_version=ITCHI_VERSION,
            precipitation_hazard_source=(
                "provided" if precipitation_hazard is not None else "q90_q95_q99"
            ),
            **wind_extra,
        )

    if run_quality_control:
        run_snapshot_quality_control(full_result)

    return full_result


# ============================================================
# ITCHI v0.2 entry point
# ============================================================


def _positive_or_none(value: float | None) -> float | None:
    """
    Float value if finite and > 0, otherwise None.
    """
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > 0.0 else None


def _mean_positive(radii: Mapping[str, float]) -> float | None:
    """
    Mean of the finite, positive quadrant radii or None.
    """
    values = [_positive_or_none(v) for v in radii.values()]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _missing(values: ArrayLike) -> ArrayLike:
    """
    Boolean field of non-finite values (NaN).
    """
    if isinstance(values, xr.DataArray):
        return values.isnull()
    return ~np.isfinite(np.asarray(values, dtype=float))


def _fill_missing(values: ArrayLike, fill_value: float) -> ArrayLike:
    """
    Replace non-finite values by fill_value.
    """
    if isinstance(values, xr.DataArray):
        return values.fillna(fill_value)
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, fill_value)


def compute_itchi_snapshot_v02(
    p6: ArrayLike,
    r24: ArrayLike,
    a72: ArrayLike,
    *,
    quantiles_6h: Mapping[str, float | ArrayLike],
    quantiles_24h: Mapping[str, float | ArrayLike],
    quantiles_72h: Mapping[str, float | ArrayLike],
    lon: ArrayLike,
    lat: ArrayLike,
    center_lon: float,
    center_lat: float,
    vmax_kt: float,
    r34_by_quadrant: Mapping[str, float | int | None],
    rocloud_by_quadrant: Mapping[str, float | int | None] | None = None,
    rmw_km: float | None = None,
    r_w06_km: float | Mapping[str, float] | None = None,
    r34_unit: str = "nm",
    rocloud_unit: str = "km",
    fallback_direct_radius_km: float | None = None,
    w06_direct_fallback: bool = True,
    radius_fill_strategy: str = "mean_available",
    w06_attribution_mode: str = "floor",
    r_w06_kwargs: Mapping[str, Any] | None = None,
    precip_matrix: Mapping[str, Any] | None = None,
    precipitation_nan_value: float | None = 0.0,
    wind_tail_h34: float = 0.2,
    wind_tail_end_kt: float | None = None,
    r34_mean_km: float | None = None,
    w06_profile_kwargs: Mapping[str, Any] | None = None,
    alpha_wind: float = 1.0,
    beta_precip_direct: float = 1.0,
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
    run_quality_control: bool = False,
) -> dict[str, Any]:
    """
    Compute ITCHI v0.2 for one synoptic time on a lon/lat grid.

    Steps
    -----
    1. Precipitation: H_P = P6 x (R24, A72) matrix against the local
       percentiles of each duration (precipitation.
       compute_precipitation_matrix_hazard). The 72-h climatology applies to
       A72 because A72(T) = R72(T - 6 h).
    2. Direct radius R_direct_q: observed R34 (missing quadrants filled);
       without R34, RMW for depressions (observed, else the parametric Rmax)
       or the parametric R34 (w06_direct_fallback).
    3. R_W06: radius where the W06 profile falls to 2 m s-1 with the PA21
       configuration (r_w06_kwargs), symmetric unless r_w06_km gives it by
       quadrant; never below R_direct_q.
    4. Attribution radius R_attr_q: ROCLOUD with R_W06 as floor
       (w06_attribution_mode="floor"): R_W06_q where ROCLOUD_q is missing,
       0 (no detection) or smaller than R_W06_q. Never below R_direct_q.
    5. Wind: W06 calibrated to the observed mean R34, with
       H_W = h34 + (1 - h34)(V - 34)/(Vmax - 34) inside R_direct_q and a tail
       h34 (V - V_out)/(34 - V_out) that vanishes at R_W06_q (or where
       V = wind_tail_end_kt). R_W06 bounds only the wind; the precipitation
       of the indirect region extends to R_attr_q.
    6. Masks, components and ITCHI as in compute_itchi_snapshot, with the
       wind tail in the indirect component:
       H_ind = 1 - (1 - H_Wind)(1 - H_Pind).

    The keyword defaults equal configs/default.yaml; an experiment can pass
    ``**itchi.config.itchi_v02_settings_from_config(cfg)`` built from a
    modified copy of the YAML file.

    Inputs must be all NumPy arrays or all xarray DataArrays on the same
    grid (P6, R24, A72, percentile fields, lon, lat).

    Parameters
    ----------
    p6, r24, a72 : numpy.ndarray or xarray.DataArray
        P6 = precipitation in (T-6h, T]; R24 = (T-24h, T]; A72 = (T-78h,
        T-6h] (precipitation.compute_antecedent_precipitation).
    quantiles_6h, quantiles_24h, quantiles_72h : Mapping
        Local percentiles by name ("Q75", "Q90", ...), e.g. from
        climatology.load_percentile_climatology. 6 h needs the matrix
        p6_nodes; 24 h and 72 h need the percentiles named in the matrix
        escalation (R24: Q90, Q95; A72: Q75).
    lon, lat : numpy.ndarray or xarray.DataArray
        Grid coordinates in degrees, same shape as P6.
    center_lon, center_lat : float
        Cyclone center in degrees.
    vmax_kt : float
        Maximum sustained wind (kt).
    r34_by_quadrant : Mapping
        R34 by quadrant (RNE, RSE, RSW, RNW or NE, SE, SW, NW); may be
        partially or totally missing.
    rocloud_by_quadrant : Mapping or None, default=None
        ROCLOUD by quadrant; None = no ROCLOUD record (R_attr_q = R_W06_q
        except with mode "never").
    rmw_km : float or None, default=None
        Observed RMW (km); None or NaN uses the empirical regression.
    r_w06_km : float, Mapping or None, default=None
        R_W06 in km (scalar or by quadrant); None computes it.
    r34_unit, rocloud_unit : str
        Units of R34 (default "nm", IBTrACS) and ROCLOUD (default "km").
    fallback_direct_radius_km : float or None, default=None
        Explicit direct-radius fallback (km); takes priority over the
        parametric R34.
    w06_direct_fallback : bool, default=True
        Use the parametric R34 as direct fallback when R34 is missing.
    radius_fill_strategy : str, default="mean_available"
        Filling of missing R34 quadrants (and ROCLOUD with mode "never").
    w06_attribution_mode : {"floor", "fallback", "always", "never"}, default="floor"
        Role of R_W06 in the attribution radius
        (radii.resolve_attribution_radius_with_w06).
    r_w06_kwargs : Mapping or None, default=None
        Definition of R_W06 (parametric_radii.compute_w06_reference_radii);
        None uses parametric_radii.R_W06_DEFAULTS (PA21, 2 m s-1).
    precip_matrix : Mapping or None, default=None
        Matrix definition; None uses precipitation.PRECIP_MATRIX_V02.
    precipitation_nan_value : float or None, default=0.0
        H_P where it is NaN (missing P6 or 6-h percentile); None keeps NaN.
        The cells are reported in "H_P_missing".
    wind_tail_h34 : float, default=0.2
        Wind hazard at the 34-kt radius.
    wind_tail_end_kt : float or None, default=None
        Optional cut of the tail at a wind speed (kt, < 34); the tail never
        extends beyond R_W06_q.
    r34_mean_km : float or None, default=None
        Observed mean R34 (km) to calibrate W06; None takes the mean of the
        valid observed quadrants (no calibration if there are none).
    w06_profile_kwargs : Mapping or None, default=None
        Extra arguments for the wind profile (Willoughby2006Profile).
    alpha_wind, beta_precip_direct, lambda_direct, mu_indirect : float
        ITCHI weights (default 1.0).
    run_quality_control : bool, default=False
        Run the snapshot quality-control checks.

    Returns
    -------
    dict[str, Any]
        Fields: radius_km, quadrant, R_direct_q, R_attr_q, R_W06_q, M_direct,
        M_indirect, M_exterior, H_P, H_P_missing, H_Pdir, H_Pind, H_W,
        H_Wdir, H_Wind, H_dir, H_ind, ITCHI, wind_kt. Metadata: radii by
        quadrant and their sources, attribution status by quadrant, W06
        parameters, itchi_version and itchi_formulation = "v0.2".
    """
    vmax = _positive_or_none(vmax_kt)
    if vmax is None:
        raise ValueError(f"vmax_kt must be finite and > 0. Received: {vmax_kt}")

    # 1. Precipitation matrix
    matrix = PRECIP_MATRIX_V02 if precip_matrix is None else precip_matrix
    hazard_precip = compute_precipitation_matrix_hazard(
        precipitation=p6,
        quantiles_6h=quantiles_6h,
        escalation_values={"R24": r24, "A72": a72},
        escalation_quantiles={"R24": quantiles_24h, "A72": quantiles_72h},
        matrix=matrix,
    )
    precip_missing = _missing(hazard_precip)
    if precipitation_nan_value is not None:
        hazard_precip = _fill_missing(hazard_precip, float(precipitation_nan_value))

    # 2. Direct radius and W06 reference radii
    rmw_observed = _positive_or_none(rmw_km)
    r34_km = convert_quadrant_radii_to_km(radii_by_quadrant=r34_by_quadrant, input_unit=r34_unit)
    if r34_mean_km is None:
        r34_mean_km = _mean_positive(r34_km)

    w06_settings = {**R_W06_DEFAULTS, **dict(r_w06_kwargs or {})}
    reference = compute_w06_reference_radii(
        vmax_kt=vmax,
        lat=float(center_lat),
        rmw_km=rmw_observed,
        r34_mean_km=r34_mean_km,
        **w06_settings,
    )

    fallback = fallback_direct_radius_km
    if fallback is None and w06_direct_fallback and np.isfinite(reference["r34_km"]):
        fallback = reference["r34_km"]

    rmw_for_direct = rmw_observed
    if rmw_for_direct is None and np.isfinite(reference["rmax_km"]):
        rmw_for_direct = reference["rmax_km"]

    resolved_direct = resolve_direct_radius(
        r34_by_quadrant=r34_km,
        vmax_kt=vmax,
        rmw_km=rmw_for_direct,
        fallback_radius_km=fallback,
        fill_strategy=radius_fill_strategy,
    )
    direct_radii = resolved_direct.radii

    # 3. R_W06 by quadrant, never below R_direct_q
    if r_w06_km is None:
        if not np.isfinite(reference["r_outer_km"]):
            raise ValueError("R_W06 is undefined for this record (check vmax_kt and the threshold).")
        r_w06_raw = {q: reference["r_outer_km"] for q in QUADRANTS}
        r_w06_source = "w06_parametric"
    else:
        if isinstance(r_w06_km, Mapping):
            r_w06_raw = standardize_quadrant_radii(dict(r_w06_km))
        else:
            r_w06_raw = {q: float(r_w06_km) for q in QUADRANTS}
        r_w06_source = "provided"

    invalid = [q for q in QUADRANTS if _positive_or_none(r_w06_raw[q]) is None]
    if invalid:
        raise ValueError(f"R_W06 must be finite and > 0 in quadrant(s): {', '.join(invalid)}")

    r_w06 = {q: max(float(r_w06_raw[q]), float(direct_radii[q])) for q in QUADRANTS}
    r_w06_clipped = tuple(q for q in QUADRANTS if float(r_w06_raw[q]) < float(direct_radii[q]))

    # 4. Attribution radius: ROCLOUD with R_W06 floor
    rocloud_km = None
    if rocloud_by_quadrant is not None:
        rocloud_km = convert_quadrant_radii_to_km(
            radii_by_quadrant=rocloud_by_quadrant,
            input_unit=rocloud_unit,
        )

    attribution = resolve_attribution_radius_with_w06(
        rocloud_by_quadrant=rocloud_km,
        r_w06_by_quadrant=r_w06,
        direct_by_quadrant=direct_radii,
        mode=w06_attribution_mode,
        fill_strategy=radius_fill_strategy,
    )

    # Geometry and per-cell radii
    geometry = build_geometry_fields(lon=lon, lat=lat, center_lon=center_lon, center_lat=center_lat)
    radius_km = geometry["radius_km"]
    quadrant = geometry["quadrant"]

    direct_q = assign_quadrant_radius(quadrant=quadrant, radii_by_quadrant=direct_radii)
    attribution_q = assign_quadrant_radius(quadrant=quadrant, radii_by_quadrant=attribution.radii)
    r_w06_q = assign_quadrant_radius(quadrant=quadrant, radii_by_quadrant=r_w06)

    # 5. W06 wind with a tail that ends at R_W06
    wind = compute_w06_wind_hazard_with_tail(
        radius_km=radius_km,
        direct_radius_km=direct_q,
        attribution_radius_km=r_w06_q,
        vmax_kt=vmax,
        lat=float(center_lat),
        rmw_km=rmw_observed,
        r34_mean_km=r34_mean_km,
        h34=wind_tail_h34,
        tail_end_wind_kt=wind_tail_end_kt,
        **dict(w06_profile_kwargs or {}),
    )

    # 6. Masks, components and ITCHI
    core = compute_itchi_snapshot(
        precipitation=p6,
        q90=None,
        q95=None,
        q99=None,
        radius_km=radius_km,
        r34_km=direct_q,
        rocloud_km=attribution_q,
        wind_hazard_normalized=wind["H_W"],
        alpha_wind=alpha_wind,
        beta_precip_direct=beta_precip_direct,
        lambda_direct=lambda_direct,
        mu_indirect=mu_indirect,
        run_quality_control=False,
        precipitation_hazard=hazard_precip,
        wind_hazard_indirect=wind["H_W"],
    )

    result: dict[str, Any] = {
        "radius_km": radius_km,
        "quadrant": quadrant,
        "R_direct_q": direct_q,
        "R_attr_q": attribution_q,
        "R_W06_q": r_w06_q,
        **core,
        "H_P_missing": precip_missing,
        "wind_kt": wind["wind_kt"],
        # radii metadata (km, by quadrant)
        "R_direct_by_quadrant": dict(direct_radii),
        "R_direct_source": resolved_direct.source,
        "R_direct_filled_quadrants": resolved_direct.filled_quadrants,
        "R_direct_used_fallback": resolved_direct.used_fallback,
        "R_W06_by_quadrant": r_w06,
        "R_W06_source": r_w06_source,
        "R_W06_clipped_to_direct": r_w06_clipped,
        "R_attr_by_quadrant": attribution.radii,
        "R_attr_status": attribution.status,
        "R_attr_clipped_to_direct": attribution.clipped_to_direct,
        "w06_attribution_mode": attribution.mode,
        # wind and precipitation metadata
        "wind_hazard_source": "w06_tail",
        "w06_x1_km": wind["x1_km"],
        "w06_x1_calibrated": wind["x1_calibrated"],
        "wind_tail_h34": float(wind_tail_h34),
        "wind_tail_end_kt": wind_tail_end_kt,
        "precipitation_hazard_source": "matrix_v02",
        "itchi_version": ITCHI_VERSION,
        "itchi_formulation": "v0.2",
    }

    if run_quality_control:
        # run_snapshot_quality_control checks R_direct_q <= ROCLOUD_q
        run_snapshot_quality_control({**result, "ROCLOUD_q": attribution_q})

    return result
