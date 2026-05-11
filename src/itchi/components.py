"""
Hazard components for ITCHI.

This module builds the direct and indirect hazard components used by
ITCHI v0.1.

The direct component combines wind hazard and direct precipitation hazard
inside R34 using a bounded union formulation.

The indirect component is defined as precipitation hazard outside R34 but
inside ROCLOUD.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import ITCHI_MAX_VALUE, ITCHI_MIN_VALUE


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def _clip_hazard(
    hazard: np.ndarray | xr.DataArray,
    clip_min: float = ITCHI_MIN_VALUE,
    clip_max: float = ITCHI_MAX_VALUE,
) -> np.ndarray | xr.DataArray:
    """
    Clip a hazard field to the valid ITCHI interval.
    """
    if _contains_xarray_object(hazard):
        return hazard.clip(min=clip_min, max=clip_max)

    return np.clip(np.asarray(hazard, dtype=float), clip_min, clip_max)


def apply_mask_to_hazard(
    hazard: np.ndarray | xr.DataArray,
    mask: np.ndarray | xr.DataArray,
    fill_value: float = 0.0,
) -> np.ndarray | xr.DataArray:
    """
    Apply a Boolean spatial mask to a hazard field.

    Values outside the mask are set to fill_value.

    Parameters
    ----------
    hazard : numpy.ndarray or xarray.DataArray
        Hazard field.
    mask : numpy.ndarray or xarray.DataArray
        Boolean mask where True indicates active cells.
    fill_value : float, default=0.0
        Value assigned outside the active mask.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Masked hazard field.
    """
    if _contains_xarray_object(hazard, mask):
        return xr.where(mask, hazard, fill_value)

    return np.where(np.asarray(mask, dtype=bool), hazard, fill_value)


def compute_direct_precipitation_hazard(
    precipitation_hazard: np.ndarray | xr.DataArray,
    direct_mask: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Compute direct precipitation hazard.

    Direct precipitation hazard is active only inside R34.

    Parameters
    ----------
    precipitation_hazard : numpy.ndarray or xarray.DataArray
        Normalized precipitation hazard H_P.
    direct_mask : numpy.ndarray or xarray.DataArray
        Boolean mask for the direct region.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Direct precipitation hazard H_Pdir.
    """
    precipitation_hazard = _clip_hazard(precipitation_hazard)

    return apply_mask_to_hazard(
        hazard=precipitation_hazard,
        mask=direct_mask,
        fill_value=0.0,
    )


def compute_indirect_precipitation_hazard(
    precipitation_hazard: np.ndarray | xr.DataArray,
    indirect_mask: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Compute indirect precipitation hazard.

    Indirect precipitation hazard is active outside R34 but inside ROCLOUD.

    Parameters
    ----------
    precipitation_hazard : numpy.ndarray or xarray.DataArray
        Normalized precipitation hazard H_P.
    indirect_mask : numpy.ndarray or xarray.DataArray
        Boolean mask for the indirect region.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Indirect precipitation hazard H_Pind.
    """
    precipitation_hazard = _clip_hazard(precipitation_hazard)

    return apply_mask_to_hazard(
        hazard=precipitation_hazard,
        mask=indirect_mask,
        fill_value=0.0,
    )


def compute_wind_hazard(
    wind_hazard_normalized: np.ndarray | xr.DataArray,
    direct_mask: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Compute wind hazard inside the direct region.

    Wind hazard is active only inside R34 in ITCHI v0.1.

    Parameters
    ----------
    wind_hazard_normalized : numpy.ndarray or xarray.DataArray
        Normalized radial wind hazard V* in [0, 1].
    direct_mask : numpy.ndarray or xarray.DataArray
        Boolean mask for the direct region.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Wind hazard H_W.
    """
    wind_hazard_normalized = _clip_hazard(wind_hazard_normalized)

    return apply_mask_to_hazard(
        hazard=wind_hazard_normalized,
        mask=direct_mask,
        fill_value=0.0,
    )


def compute_direct_component(
    wind_hazard: np.ndarray | xr.DataArray,
    direct_precipitation_hazard: np.ndarray | xr.DataArray,
    alpha: float = 1.0,
    beta: float = 1.0,
    clip_min: float = ITCHI_MIN_VALUE,
    clip_max: float = ITCHI_MAX_VALUE,
) -> np.ndarray | xr.DataArray:
    """
    Compute the direct hazard component H_dir.

    The direct component combines wind hazard and direct precipitation hazard
    using a bounded union formulation:

        H_dir = 1 - (1 - H_W)^alpha * (1 - H_Pdir)^beta

    Parameters
    ----------
    wind_hazard : numpy.ndarray or xarray.DataArray
        Wind hazard H_W.
    direct_precipitation_hazard : numpy.ndarray or xarray.DataArray
        Direct precipitation hazard H_Pdir.
    alpha : float, default=1.0
        Weight/exponent for wind hazard.
    beta : float, default=1.0
        Weight/exponent for direct precipitation hazard.
    clip_min : float, default=0.0
        Minimum allowed value.
    clip_max : float, default=1.0
        Maximum allowed value.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Direct hazard component H_dir.
    """
    wind_hazard = _clip_hazard(wind_hazard, clip_min, clip_max)
    direct_precipitation_hazard = _clip_hazard(
        direct_precipitation_hazard,
        clip_min,
        clip_max,
    )

    direct_component = 1.0 - (
        (1.0 - wind_hazard) ** alpha * (1.0 - direct_precipitation_hazard) ** beta
    )

    return _clip_hazard(direct_component, clip_min, clip_max)


def compute_indirect_component(
    indirect_precipitation_hazard: np.ndarray | xr.DataArray,
    clip_min: float = ITCHI_MIN_VALUE,
    clip_max: float = ITCHI_MAX_VALUE,
) -> np.ndarray | xr.DataArray:
    """
    Compute the indirect hazard component H_ind.

    In ITCHI v0.1, the indirect component is defined as:

        H_ind = H_Pind

    Parameters
    ----------
    indirect_precipitation_hazard : numpy.ndarray or xarray.DataArray
        Indirect precipitation hazard H_Pind.
    clip_min : float, default=0.0
        Minimum allowed value.
    clip_max : float, default=1.0
        Maximum allowed value.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Indirect hazard component H_ind.
    """
    return _clip_hazard(indirect_precipitation_hazard, clip_min, clip_max)


def compute_hazard_components(
    precipitation_hazard: np.ndarray | xr.DataArray,
    wind_hazard_normalized: np.ndarray | xr.DataArray,
    direct_mask: np.ndarray | xr.DataArray,
    indirect_mask: np.ndarray | xr.DataArray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[str, np.ndarray | xr.DataArray]:
    """
    Compute all main ITCHI hazard components.

    Parameters
    ----------
    precipitation_hazard : numpy.ndarray or xarray.DataArray
        Normalized precipitation hazard H_P.
    wind_hazard_normalized : numpy.ndarray or xarray.DataArray
        Normalized wind hazard V*.
    direct_mask : numpy.ndarray or xarray.DataArray
        Boolean mask for the direct region.
    indirect_mask : numpy.ndarray or xarray.DataArray
        Boolean mask for the indirect region.
    alpha : float, default=1.0
        Weight/exponent for wind hazard in H_dir.
    beta : float, default=1.0
        Weight/exponent for direct precipitation hazard in H_dir.

    Returns
    -------
    dict[str, numpy.ndarray or xarray.DataArray]
        Dictionary containing H_Pdir, H_Pind, H_W, H_dir and H_ind.
    """
    direct_precipitation_hazard = compute_direct_precipitation_hazard(
        precipitation_hazard=precipitation_hazard,
        direct_mask=direct_mask,
    )

    indirect_precipitation_hazard = compute_indirect_precipitation_hazard(
        precipitation_hazard=precipitation_hazard,
        indirect_mask=indirect_mask,
    )

    wind_hazard = compute_wind_hazard(
        wind_hazard_normalized=wind_hazard_normalized,
        direct_mask=direct_mask,
    )

    direct_component = compute_direct_component(
        wind_hazard=wind_hazard,
        direct_precipitation_hazard=direct_precipitation_hazard,
        alpha=alpha,
        beta=beta,
    )

    indirect_component = compute_indirect_component(
        indirect_precipitation_hazard=indirect_precipitation_hazard,
    )

    return {
        "H_Pdir": direct_precipitation_hazard,
        "H_Pind": indirect_precipitation_hazard,
        "H_W": wind_hazard,
        "H_dir": direct_component,
        "H_ind": indirect_component,
    }
