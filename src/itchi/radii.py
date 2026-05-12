"""
Radius utilities for ITCHI.

This module handles quadrant-specific radius cleaning, missing-value
imputation and effective direct-radius resolution.

Important distinction
---------------------
R34_q:
    Observed or reported radius of 34-kt winds by quadrant.

R_direct_q:
    Effective radius used by ITCHI to define the direct region.

For tropical storms and stronger systems, R_direct_q is normally R34_q.
For tropical depressions without 34-kt winds, R_direct_q can be based on
the radius of maximum winds, RMW, or another explicitly documented proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from itchi.constants import QUADRANTS
from itchi.units import normalize_quadrant_key


@dataclass(frozen=True)
class ResolvedRadii:
    """
    Container for resolved quadrant radii.

    Attributes
    ----------
    radii : dict[str, float]
        Quadrant-specific radii with keys RNE, RSE, RSW and RNW.
    source : str
        Description of the source or rule used.
    filled_quadrants : tuple[str, ...]
        Quadrants that were filled by imputation or fallback.
    used_fallback : bool
        Whether a fallback rule was used.
    """

    radii: dict[str, float]
    source: str
    filled_quadrants: tuple[str, ...]
    used_fallback: bool = False


def _clean_radius_value(value: Any) -> float:
    """
    Convert a radius value to float and mark invalid values as NaN.

    Negative values and non-finite values are considered missing.
    """
    if value is None:
        return np.nan

    value_float = float(value)

    if not np.isfinite(value_float):
        return np.nan

    if value_float < 0:
        return np.nan

    return value_float


def standardize_quadrant_radii(
    radii_by_quadrant: dict[str, Any],
) -> dict[str, float]:
    """
    Standardize quadrant radius keys and clean invalid values.

    Accepted input keys include:

    - RNE, RSE, RSW, RNW
    - NE, SE, SW, NW

    Parameters
    ----------
    radii_by_quadrant : dict[str, Any]
        Input radii by quadrant.

    Returns
    -------
    dict[str, float]
        Standardized radii with keys RNE, RSE, RSW and RNW.

    Raises
    ------
    KeyError
        If any required quadrant is missing.
    """
    standardized: dict[str, float] = {}

    for key, value in radii_by_quadrant.items():
        quadrant = normalize_quadrant_key(key)
        standardized[quadrant] = _clean_radius_value(value)

    missing = set(QUADRANTS).difference(standardized)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing quadrant radius/radii: {missing_text}")

    return {quadrant: standardized[quadrant] for quadrant in QUADRANTS}


def get_valid_radius_values(
    radii_by_quadrant: dict[str, float],
) -> dict[str, float]:
    """
    Return only valid finite quadrant radii.
    """
    return {
        quadrant: float(value)
        for quadrant, value in radii_by_quadrant.items()
        if np.isfinite(float(value)) and float(value) >= 0.0
    }


def fill_missing_quadrant_radii(
    radii_by_quadrant: dict[str, Any],
    strategy: str = "mean_available",
) -> ResolvedRadii:
    """
    Fill missing quadrant radii using available values.

    Parameters
    ----------
    radii_by_quadrant : dict[str, Any]
        Quadrant-specific radii. Keys can use RNE/RSE/RSW/RNW
        or NE/SE/SW/NW.
    strategy : str, default="mean_available"
        Missing-value strategy. Currently supported:

        - "mean_available"

    Returns
    -------
    ResolvedRadii
        Resolved radii and metadata.

    Raises
    ------
    ValueError
        If all quadrant radii are missing or if the strategy is unsupported.
    """
    if strategy != "mean_available":
        raise ValueError(f"Unsupported radius-filling strategy: {strategy}")

    standardized = standardize_quadrant_radii(radii_by_quadrant)
    valid_values = get_valid_radius_values(standardized)

    if len(valid_values) == 0:
        raise ValueError("Cannot fill quadrant radii: all values are missing.")

    fill_value = float(np.mean(list(valid_values.values())))

    filled_quadrants: list[str] = []
    resolved: dict[str, float] = {}

    for quadrant in QUADRANTS:
        value = standardized[quadrant]

        if np.isfinite(value):
            resolved[quadrant] = float(value)
        else:
            resolved[quadrant] = fill_value
            filled_quadrants.append(quadrant)

    source = (
        "observed" if len(filled_quadrants) == 0 else f"filled_missing_with_{strategy}"
    )

    return ResolvedRadii(
        radii=resolved,
        source=source,
        filled_quadrants=tuple(filled_quadrants),
        used_fallback=False,
    )


def build_uniform_quadrant_radius(
    radius_km: float,
    source: str,
) -> ResolvedRadii:
    """
    Build a uniform quadrant-radius dictionary.

    This is useful when only a scalar fallback radius is available,
    for example RMW for a tropical depression without R34.

    Parameters
    ----------
    radius_km : float
        Radius in kilometers.
    source : str
        Source description.

    Returns
    -------
    ResolvedRadii
        Uniform quadrant radii and metadata.
    """
    radius = _clean_radius_value(radius_km)

    if not np.isfinite(radius):
        raise ValueError("Fallback radius must be finite and non-negative.")

    return ResolvedRadii(
        radii={quadrant: float(radius) for quadrant in QUADRANTS},
        source=source,
        filled_quadrants=tuple(QUADRANTS),
        used_fallback=True,
    )


def resolve_direct_radius(
    r34_by_quadrant: dict[str, Any],
    vmax_kt: float | None = None,
    rmw_km: float | None = None,
    fallback_radius_km: float | None = None,
    tropical_storm_threshold_kt: float = 34.0,
    fill_strategy: str = "mean_available",
) -> ResolvedRadii:
    """
    Resolve the effective direct-region radius R_direct_q.

    Logic
    -----
    1. If at least one valid R34 quadrant exists, fill missing R34 quadrants
       using the selected filling strategy and use the result as R_direct_q.

    2. If no valid R34 exists and the system is below 34 kt, use RMW if
       available.

    3. If RMW is not available, use fallback_radius_km if provided.

    4. If none of the above is possible, raise an error.

    Parameters
    ----------
    r34_by_quadrant : dict[str, Any]
        R34 radii by quadrant in kilometers.
    vmax_kt : float or None, default=None
        Maximum sustained wind in knots.
    rmw_km : float or None, default=None
        Radius of maximum wind in kilometers.
    fallback_radius_km : float or None, default=None
        Optional fallback radius in kilometers.
    tropical_storm_threshold_kt : float, default=34.0
        Threshold for tropical-storm-force winds.
    fill_strategy : str, default="mean_available"
        Strategy for filling missing quadrant radii.

    Returns
    -------
    ResolvedRadii
        Effective direct-region radius by quadrant.

    Raises
    ------
    ValueError
        If the direct radius cannot be resolved.
    """
    standardized = standardize_quadrant_radii(r34_by_quadrant)
    valid_r34 = get_valid_radius_values(standardized)

    if len(valid_r34) > 0:
        resolved = fill_missing_quadrant_radii(
            radii_by_quadrant=standardized,
            strategy=fill_strategy,
        )

        return ResolvedRadii(
            radii=resolved.radii,
            source="R34" if not resolved.filled_quadrants else resolved.source,
            filled_quadrants=resolved.filled_quadrants,
            used_fallback=False,
        )

    is_below_tropical_storm = (
        vmax_kt is not None
        and np.isfinite(float(vmax_kt))
        and float(vmax_kt) < tropical_storm_threshold_kt
    )

    if is_below_tropical_storm and rmw_km is not None:
        return build_uniform_quadrant_radius(
            radius_km=float(rmw_km),
            source="RMW_fallback_for_tropical_depression",
        )

    if fallback_radius_km is not None:
        return build_uniform_quadrant_radius(
            radius_km=float(fallback_radius_km),
            source="configured_fallback_radius",
        )

    raise ValueError(
        "Cannot resolve direct radius. No valid R34 quadrants were found, "
        "and no valid RMW or fallback radius was provided."
    )


def resolve_attribution_radius(
    rocloud_by_quadrant: dict[str, Any],
    fill_strategy: str = "mean_available",
) -> ResolvedRadii:
    """
    Resolve the external attribution radius ROCLOUD_q.

    This function fills missing ROCLOUD quadrants using available quadrants.

    Parameters
    ----------
    rocloud_by_quadrant : dict[str, Any]
        ROCLOUD radii by quadrant in kilometers.
    fill_strategy : str, default="mean_available"
        Strategy for filling missing values.

    Returns
    -------
    ResolvedRadii
        Resolved ROCLOUD radii and metadata.
    """
    return fill_missing_quadrant_radii(
        radii_by_quadrant=rocloud_by_quadrant,
        strategy=fill_strategy,
    )
