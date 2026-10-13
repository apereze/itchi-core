"""
Parametric cyclone radii for ITCHI.

This module estimates R34 and an outer cyclone radius from best-track data
with the W06 profile, following PA21, so that ITCHI can run without an
external outer-radius database.

Integration with the ITCHI pipeline (no changes to existing modules)
----------------------------------------------------------------------
- Outer radius -> attribution radius: ``build_parametric_rocloud_record``
  returns a ROCLOUD-compatible record (symmetric, km) accepted by
  ``snapshot_inputs.build_snapshot_input_from_records``.
- Parametric R34 -> direct-radius fallback: pass it as
  ``fallback_direct_radius_km``; observed R34 keeps priority in
  ``radii.resolve_direct_radius``.
- Optional W06 wind field -> ``compute_w06_wind_hazard``, passed to the
  pipeline as ``wind_hazard_normalized``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from itchi.constants import QUADRANTS, TROPICAL_STORM_WIND_KT
from itchi.rocloud import DEFAULT_ROCLOUD_COLUMN_MAP
from itchi.tracks import DEFAULT_R34_COLUMN_MAP
from itchi.units import convert_radius_to_km
from itchi.wind import normalize_wind_hazard
from itchi.wind_profiles import (
    KNOT_TO_MS,
    V_OUTER_PA21_MS,
    V_R34_MS,
    Willoughby2006Profile,
    bisect_decreasing,
    radius_of_wind_km,
)

DEFAULT_THRESHOLDS_MS: dict[str, float] = {
    "r_outer_km": V_OUTER_PA21_MS,
    "r34_km": V_R34_MS,
}


def fit_x1_to_radius(
    vmax_ms: np.ndarray,
    lat: np.ndarray,
    rmax_km: np.ndarray | None,
    radius_obs_km: np.ndarray,
    wind_obs_ms: float = V_R34_MS,
    x1_bounds_km: tuple[float, float] = (10.0, 2000.0),
    n_iter: int = 50,
    **profile_kwargs: Any,
) -> np.ndarray:
    """
    Fit X1 so that the W06 profile passes through (radius_obs_km, wind_obs_ms).

    V(r_obs) increases monotonically with X1 for r_obs > R2. Returns NaN where
    no solution exists within x1_bounds_km, r_obs <= Rmax or r_obs is missing.
    """
    vmax = np.asarray(vmax_ms, dtype=float)
    latitude = np.asarray(lat, dtype=float)
    r_obs = np.asarray(radius_obs_km, dtype=float)
    rmax = Willoughby2006Profile(vmax, latitude, rmax_km, **profile_kwargs).rmax

    def residual(x1: np.ndarray) -> np.ndarray:
        profile = Willoughby2006Profile(
            vmax, latitude, rmax, x1_km=x1, **profile_kwargs
        )
        return wind_obs_ms - profile(r_obs)

    lower = np.full_like(vmax, x1_bounds_km[0])
    upper = np.full_like(vmax, x1_bounds_km[1])

    with np.errstate(invalid="ignore"):
        x1 = bisect_decreasing(residual, lower, upper, n_iter=n_iter)
        valid = (
            np.isfinite(r_obs)
            & (r_obs > rmax)
            & (vmax > wind_obs_ms)
            & (residual(lower) > 0)
            & (residual(upper) <= 0)
        )

    return np.where(valid, x1, np.nan)


def compute_parametric_radii(
    vmax_kt: np.ndarray,
    lat: np.ndarray,
    rmw_km: np.ndarray | None = None,
    r34_obs_km: np.ndarray | None = None,
    thresholds_ms: Mapping[str, float] | None = None,
    **profile_kwargs: Any,
) -> dict[str, np.ndarray]:
    """
    Compute parametric radii (km) with the W06 profile.

    Parameters
    ----------
    vmax_kt, lat : numpy.ndarray
        Maximum sustained wind (kt) and latitude (degrees).
    rmw_km : numpy.ndarray or None
        Observed RMW in km. None -> empirical regression (also fills NaN).
    r34_obs_km : numpy.ndarray or None
        Observed azimuthal-mean R34 in km. If given, X1 is calibrated to it
        where possible; the W06 regression is used elsewhere.
    thresholds_ms : Mapping[str, float] or None
        Output name -> wind threshold in m s-1.
    **profile_kwargs : Any
        Passed to Willoughby2006Profile.

    Returns
    -------
    dict[str, numpy.ndarray]
        Radii for each threshold plus rmax_km, x1_km, n, a and x1_calibrated.
    """
    thresholds = DEFAULT_THRESHOLDS_MS if thresholds_ms is None else thresholds_ms
    vmax_ms = np.asarray(vmax_kt, dtype=float) * KNOT_TO_MS
    latitude = np.asarray(lat, dtype=float)

    rmax = rmw_km
    x1 = None
    calibrated = np.zeros_like(vmax_ms, dtype=bool)

    if r34_obs_km is not None:
        first_guess = Willoughby2006Profile(vmax_ms, latitude, rmw_km, **profile_kwargs)
        x1_fit = fit_x1_to_radius(
            vmax_ms, latitude, first_guess.rmax, r34_obs_km, **profile_kwargs
        )
        calibrated = np.isfinite(x1_fit)
        x1 = np.where(calibrated, x1_fit, first_guess.x1)
        rmax = first_guess.rmax

    profile = Willoughby2006Profile(vmax_ms, latitude, rmax, x1_km=x1, **profile_kwargs)

    result = {
        name: radius_of_wind_km(profile, threshold)
        for name, threshold in thresholds.items()
    }
    result.update(
        rmax_km=profile.rmax,
        x1_km=profile.x1,
        n=profile.n,
        a=profile.a,
        x1_calibrated=calibrated,
    )

    return result


def _mean_positive_quadrant_radius_km(
    track_df: pd.DataFrame,
    column_map: Mapping[str, str],
    input_unit: str,
) -> np.ndarray:
    """
    Azimuthal-mean radius from non-zero, finite quadrant radii (km).
    """
    columns = [column_map[q] for q in QUADRANTS if column_map[q] in track_df]

    if not columns:
        return np.full(len(track_df), np.nan)

    values = track_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values = np.where(values > 0.0, values, np.nan)  # 0 = no 34-kt winds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_value = np.nanmean(values, axis=1)

    return np.asarray(convert_radius_to_km(mean_value, input_unit), dtype=float)


def add_parametric_radii(
    track_df: pd.DataFrame,
    use_observed_rmw: bool = False,
    calibrate_to_r34: bool = False,
    rmw_column: str = "rmw_km",
    rmw_unit: str = "km",
    r34_column_map: Mapping[str, str] = DEFAULT_R34_COLUMN_MAP,
    r34_unit: str = "nm",
    thresholds_ms: Mapping[str, float] | None = None,
    prefix: str = "w06_",
    **profile_kwargs: Any,
) -> pd.DataFrame:
    """
    Add W06 parametric radii to a standardized ITCHI track DataFrame.

    The defaults (use_observed_rmw=False, calibrate_to_r34=False) reproduce
    the PA21 configuration. Output columns are prefixed with ``prefix``.
    """
    result = track_df.copy()
    vmax_kt = pd.to_numeric(result["vmax_kt"], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(result["lat"], errors="coerce").to_numpy(float)

    rmw_km = None
    if use_observed_rmw and rmw_column in result:
        rmw_values = pd.to_numeric(result[rmw_column], errors="coerce").to_numpy(float)
        rmw_km = np.asarray(convert_radius_to_km(rmw_values, rmw_unit), dtype=float)

    r34_obs_km = None
    if calibrate_to_r34:
        r34_obs_km = _mean_positive_quadrant_radius_km(result, r34_column_map, r34_unit)

    radii = compute_parametric_radii(
        vmax_kt=vmax_kt,
        lat=lat,
        rmw_km=rmw_km,
        r34_obs_km=r34_obs_km,
        thresholds_ms=thresholds_ms,
        **profile_kwargs,
    )

    for name, values in radii.items():
        result[f"{prefix}{name}"] = values

    return result


def build_parametric_rocloud_record(
    track_record: Mapping[str, Any] | pd.Series,
    outer_radius_column: str = "w06_r_outer_km",
    column_map: Mapping[str, str] = DEFAULT_ROCLOUD_COLUMN_MAP,
) -> dict[str, Any]:
    """
    Build a symmetric ROCLOUD-compatible record (km) from the W06 outer radius.

    The record is accepted as ``rocloud_record`` by
    ``snapshot_inputs.build_snapshot_input_from_records`` with
    ``rocloud_unit="km"``.
    """
    radius = float(track_record[outer_radius_column])
    record: dict[str, Any] = {column_map[q]: radius for q in QUADRANTS}

    for key in ("storm_id", "time"):
        if key in track_record:
            record[key] = track_record[key]

    return record


def parametric_direct_fallback_km(
    track_record: Mapping[str, Any] | pd.Series,
    r34_column: str = "w06_r34_km",
) -> float | None:
    """
    Return the parametric R34 (km) for ``fallback_direct_radius_km`` or None.
    """
    value = float(track_record.get(r34_column, np.nan))
    return value if np.isfinite(value) else None


def compute_w06_wind_hazard(
    radius_km: np.ndarray,
    vmax_kt: float,
    lat: float,
    rmw_km: float | None = None,
    x1_km: float | None = None,
    reference_wind_kt: float = TROPICAL_STORM_WIND_KT,
    below_threshold_mode: str = "relative_to_vmax",
    **profile_kwargs: Any,
) -> np.ndarray:
    """
    Normalized wind hazard V* from the W06 profile for one track record.

    Unlike the modified Rankine profile in ``wind.py``, a W06 profile
    calibrated to observed R34 satisfies V(R34) = 34 kt, so V* vanishes
    exactly at the direct-region boundary.
    """
    profile = Willoughby2006Profile(
        float(vmax_kt) * KNOT_TO_MS,
        float(lat),
        rmw_km,
        x1_km=x1_km,
        **profile_kwargs,
    )
    wind_kt = profile(np.asarray(radius_km, dtype=float)) / KNOT_TO_MS

    return normalize_wind_hazard(
        wind_kt=wind_kt,
        vmax_kt=float(vmax_kt),
        reference_wind_kt=reference_wind_kt,
        below_threshold_mode=below_threshold_mode,
    )
