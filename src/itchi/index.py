"""
Final ITCHI index calculation.

This module implements the final bounded integration of the direct and
indirect hazard components.

For ITCHI v0.1, the final index is defined as:

    ITCHI = 1 - (1 - H_dir)^lambda * (1 - H_ind)^mu

where:

- H_dir is the direct hazard component.
- H_ind is the indirect hazard component.
- lambda and mu control the sensitivity to direct and indirect hazards.

For the baseline v0.1 configuration:

    lambda = 1
    mu = 1
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


def _clip_index(
    values: np.ndarray | xr.DataArray,
    clip_min: float = ITCHI_MIN_VALUE,
    clip_max: float = ITCHI_MAX_VALUE,
) -> np.ndarray | xr.DataArray:
    """
    Clip values to the valid ITCHI interval.
    """
    if _contains_xarray_object(values):
        return values.clip(min=clip_min, max=clip_max)

    return np.clip(np.asarray(values, dtype=float), clip_min, clip_max)


def _validate_weight(value: float, name: str) -> None:
    """
    Validate that a weighting exponent is non-negative.

    Negative exponents are not allowed because they can produce unstable
    behavior when a hazard component approaches 1.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Received: {value}")


def compute_itchi(
    direct_component: np.ndarray | xr.DataArray,
    indirect_component: np.ndarray | xr.DataArray,
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
    clip_min: float = ITCHI_MIN_VALUE,
    clip_max: float = ITCHI_MAX_VALUE,
) -> np.ndarray | xr.DataArray:
    """
    Compute the final ITCHI index.

    Parameters
    ----------
    direct_component : numpy.ndarray or xarray.DataArray
        Direct hazard component H_dir.
    indirect_component : numpy.ndarray or xarray.DataArray
        Indirect hazard component H_ind.
    lambda_direct : float, default=1.0
        Weighting exponent for the direct component.
    mu_indirect : float, default=1.0
        Weighting exponent for the indirect component.
    clip_min : float, default=0.0
        Minimum allowed ITCHI value.
    clip_max : float, default=1.0
        Maximum allowed ITCHI value.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Final ITCHI index in the interval [0, 1].
    """
    _validate_weight(lambda_direct, "lambda_direct")
    _validate_weight(mu_indirect, "mu_indirect")

    h_dir = _clip_index(direct_component, clip_min, clip_max)
    h_ind = _clip_index(indirect_component, clip_min, clip_max)

    itchi = 1.0 - ((1.0 - h_dir) ** lambda_direct * (1.0 - h_ind) ** mu_indirect)

    return _clip_index(itchi, clip_min, clip_max)


def compute_itchi_from_components(
    components: dict[str, np.ndarray | xr.DataArray],
    lambda_direct: float = 1.0,
    mu_indirect: float = 1.0,
) -> np.ndarray | xr.DataArray:
    """
    Compute ITCHI from a dictionary of hazard components.

    The dictionary must contain:

    - "H_dir"
    - "H_ind"

    Parameters
    ----------
    components : dict
        Dictionary containing hazard components.
    lambda_direct : float, default=1.0
        Weighting exponent for the direct component.
    mu_indirect : float, default=1.0
        Weighting exponent for the indirect component.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Final ITCHI index.
    """
    required_keys = {"H_dir", "H_ind"}
    missing_keys = required_keys.difference(components)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing required component(s): {missing}")

    return compute_itchi(
        direct_component=components["H_dir"],
        indirect_component=components["H_ind"],
        lambda_direct=lambda_direct,
        mu_indirect=mu_indirect,
    )
