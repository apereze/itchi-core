"""
Wind hazard of ITCHI.

The wind hazard is absolute and homologated with the Saffir-Simpson scale
(``normalize_wind_hazard_absolute``): the same local wind gives the same H_W
in every storm. The local wind comes from the W06 profile calibrated to R34
(parametric_radii.compute_w06_wind_hazard_with_tail); below 34 kt a tail
decays to zero at the tail-end wind (12 m s-1 or V(R_W06_q)).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import DEFAULT_EPSILON

ArrayLike = np.ndarray | xr.DataArray


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def _clip_unit_interval(values: ArrayLike) -> ArrayLike:
    """
    Clip values to [0, 1] preserving NumPy or xarray type.
    """
    if _contains_xarray_object(values):
        return values.clip(min=0.0, max=1.0)

    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def normalize_wind_hazard_absolute(
    wind_kt: ArrayLike,
    outer_wind_kt: float | ArrayLike,
    nodes: tuple[tuple[float, float], ...] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> ArrayLike:
    """
    Absolute wind hazard homologated with the Saffir-Simpson scale.

    H_W does not depend on the Vmax of the storm: the same local wind gives
    the same hazard in every storm.
    With nodes (V_1, h_1), ..., (V_n, h_n), V_1 = 34 kt:

        V <  V_1 :  H_W = h_1 * clip((V - V_out) / (V_1 - V_out), 0, 1)
        V >= V_1 :  H_W = linear interpolation between nodes, h_n above V_n

    V_out is the tail-end wind of each cell (V(R_W06_q), or the cut speed,
    12 m s-1, whichever is larger). The default nodes (constants.SAFFIR_SIMPSON_HAZARD_NODES)
    put the Saffir-Simpson categories in the SIAT-style classes of 0.2:
    tail -> 1, TS -> 2, H1 -> 3, H2 -> 4, H3-H5 -> 5. The classes are
    (lower, upper], so a wind exactly on a category bound (e.g. 64 kt, H =
    0.4) falls in the lower class; this only happens on a contour line.

    Parameters
    ----------
    wind_kt : numpy.ndarray or xarray.DataArray
        Local sustained wind (kt).
    outer_wind_kt : float, numpy.ndarray or xarray.DataArray
        Tail-end wind (kt) of each cell.
    nodes : tuple of (wind_kt, hazard) or None, default=None
        Increasing winds and non-decreasing hazards in [0, 1]; None uses
        constants.SAFFIR_SIMPSON_HAZARD_NODES.
    epsilon : float, default=1e-6
        Small value for numerical stability.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Wind hazard in [0, 1].
    """
    from itchi.constants import SAFFIR_SIMPSON_HAZARD_NODES

    table = SAFFIR_SIMPSON_HAZARD_NODES if nodes is None else tuple(nodes)
    v_nodes = np.asarray([float(v) for v, _ in table])
    h_nodes = np.asarray([float(h) for _, h in table])
    if v_nodes.size < 1:
        raise ValueError("nodes must contain at least one (wind_kt, hazard) pair.")
    if np.any(np.diff(v_nodes) <= 0):
        raise ValueError(f"node winds must be strictly increasing: {v_nodes.tolist()}")
    if np.any(np.diff(h_nodes) < 0) or h_nodes.min() < 0.0 or h_nodes.max() > 1.0:
        raise ValueError(f"node hazards must be non-decreasing and in [0, 1]: {h_nodes.tolist()}")

    is_xr = _contains_xarray_object(wind_kt, outer_wind_kt)
    where = xr.where if is_xr else np.where
    v1, h1 = float(v_nodes[0]), float(h_nodes[0])

    tail = h1 * _clip_unit_interval((wind_kt - outer_wind_kt) / np.maximum(v1 - outer_wind_kt, epsilon))
    v = np.asarray(wind_kt, dtype=float)
    upper = np.interp(v, v_nodes, h_nodes)          # constant h_n above the last node
    if is_xr and isinstance(wind_kt, xr.DataArray):
        upper = wind_kt.copy(data=upper)

    return _clip_unit_interval(where(np.isfinite(v) & (v >= v1), upper, tail))
