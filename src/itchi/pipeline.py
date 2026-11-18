"""
ITCHI for one synoptic time on a lon/lat grid.

``compute_itchi_snapshot`` takes P6, R24, A72 and their percentile
climatologies and computes:

1. the precipitation hazard H_P (P6 x (R24, A72) matrix, maximum at the
   6-h P95; precipitation.PRECIP_MATRIX);
2. the direct radius (observed R34, parametric fallback) and R_W06 (W06
   profile, PA21, 2 m s-1);
3. the attribution radius: ROCLOUD with R_W06 as a floor by quadrant;
4. the absolute wind hazard H_W homologated with Saffir-Simpson, with a tail
   below 34 kt that ends at 12 m s-1 or at R_W06_q;
5. masks, direct and indirect components and ITCHI.

Its keyword defaults equal configs/default.yaml
(config.itchi_settings_from_config; checked in tests/test_pipeline.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from itchi.components import compute_hazard_components
from itchi.constants import ITCHI_VERSION, QUADRANTS, SAFFIR_SIMPSON_HAZARD_NODES, TAIL_END_WIND_MS
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
from itchi.precipitation import PRECIP_MATRIX, compute_precipitation_matrix_hazard
from itchi.quality_control import run_snapshot_quality_control
from itchi.radii import (
    resolve_attribution_radius_with_w06,
    resolve_direct_radius,
    standardize_quadrant_radii,
)
from itchi.units import convert_quadrant_radii_to_km
from itchi.wind_profiles import KNOT_TO_MS

ArrayLike = np.ndarray | xr.DataArray


def _combine_hazards(
    precipitation_hazard: ArrayLike,
    wind_hazard: ArrayLike,
    radius_km: ArrayLike,
    direct_radius_km: ArrayLike,
    attribution_radius_km: ArrayLike,
    alpha_wind: float = 1.0,
    beta_precip_direct: float = 1.0,
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
) -> dict[str, Any]:
    """
    Region masks, direct and indirect components and ITCHI.

    The wind hazard enters the direct component and, through its tail, the
    indirect one: H_ind = 1 - (1 - H_Wind)(1 - H_Pind).
    """
    masks = build_region_masks(
        radius_km=radius_km,
        r34_km=direct_radius_km,
        rocloud_km=attribution_radius_km,
    )
    components = compute_hazard_components(
        precipitation_hazard=precipitation_hazard,
        wind_hazard_normalized=wind_hazard,
        direct_mask=masks["direct"],
        indirect_mask=masks["indirect"],
        alpha=alpha_wind,
        beta=beta_precip_direct,
        wind_hazard_indirect=wind_hazard,
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
    return result


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


def compute_itchi_snapshot(
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
    wind_tail_end_kt: float | None = TAIL_END_WIND_MS / KNOT_TO_MS,
    wind_absolute_nodes: tuple[tuple[float, float], ...] = SAFFIR_SIMPSON_HAZARD_NODES,
    r34_mean_km: float | None = None,
    w06_profile_kwargs: Mapping[str, Any] | None = None,
    alpha_wind: float = 1.0,
    beta_precip_direct: float = 1.0,
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
    run_quality_control: bool = False,
) -> dict[str, Any]:
    """
    Compute ITCHI for one synoptic time on a lon/lat grid.

    Steps
    -----
    1. Precipitation: H_P = P6 x (R24, A72) matrix (precipitation.
       PRECIP_MATRIX, maximum at the 6-h P95) against the local
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
    5. Wind: W06 calibrated to the observed mean R34; absolute H_W
       homologated with Saffir-Simpson (wind_absolute_nodes: tail -> class 1,
       TS -> 2, H1 -> 3, H2 -> 4, >= H3 -> 5), with a tail below 34 kt that
       vanishes where V = wind_tail_end_kt (12 m s-1) or at R_W06_q. R_W06
       bounds only the wind; the precipitation of the indirect region
       extends to R_attr_q.
    6. Masks, components and ITCHI, with the wind tail in the indirect component:
       H_ind = 1 - (1 - H_Wind)(1 - H_Pind).

    The keyword defaults equal configs/default.yaml; an experiment passes
    ``**itchi.config.itchi_settings_from_config(cfg)`` built from an overlay
    (config.load_experiment_config).

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
        P6 x (R24, A72) matrix; None uses precipitation.PRECIP_MATRIX.
    precipitation_nan_value : float or None, default=0.0
        H_P where it is NaN (missing P6 or 6-h percentile); None keeps NaN.
        The cells are reported in "H_P_missing".
    wind_tail_end_kt : float or None, default=12 m s-1 (23.3 kt)
        End of the tail at a wind speed (kt, < 34); the tail never extends
        beyond R_W06_q. None keeps the tail down to R_W06_q.
    wind_absolute_nodes : tuple of (wind_kt, hazard)
        Nodes of the absolute scale, default
        constants.SAFFIR_SIMPSON_HAZARD_NODES (tail -> class 1, TS -> 2,
        H1 -> 3, H2 -> 4, H3-H5 -> 5).
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
        parameters and itchi_version.
    """
    vmax = _positive_or_none(vmax_kt)
    if vmax is None:
        raise ValueError(f"vmax_kt must be finite and > 0. Received: {vmax_kt}")

    # 1. Precipitation: P6 x (R24, A72) matrix
    matrix = PRECIP_MATRIX if precip_matrix is None else precip_matrix
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

    # 5. Absolute W06 wind with a tail that ends at 12 m s-1 or R_W06
    wind = compute_w06_wind_hazard_with_tail(
        radius_km=radius_km,
        direct_radius_km=direct_q,
        attribution_radius_km=r_w06_q,
        vmax_kt=vmax,
        lat=float(center_lat),
        rmw_km=rmw_observed,
        r34_mean_km=r34_mean_km,
        tail_end_wind_kt=wind_tail_end_kt,
        absolute_nodes=tuple(tuple(map(float, node)) for node in wind_absolute_nodes),
        **dict(w06_profile_kwargs or {}),
    )

    # 6. Masks, components and ITCHI
    core = _combine_hazards(
        precipitation_hazard=hazard_precip,
        wind_hazard=wind["H_W"],
        radius_km=radius_km,
        direct_radius_km=direct_q,
        attribution_radius_km=attribution_q,
        alpha_wind=alpha_wind,
        beta_precip_direct=beta_precip_direct,
        lambda_direct=lambda_direct,
        mu_indirect=mu_indirect,
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
        "wind_tail_end_kt": wind_tail_end_kt,
        "wind_absolute_nodes": tuple(tuple(map(float, node)) for node in wind_absolute_nodes),
        "outer_wind_kt": wind["outer_wind_kt"],
        "itchi_version": ITCHI_VERSION,
    }

    if run_quality_control:
        # run_snapshot_quality_control checks R_direct_q <= ROCLOUD_q
        run_snapshot_quality_control({**result, "ROCLOUD_q": attribution_q})

    return result
