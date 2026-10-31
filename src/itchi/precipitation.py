"""
Precipitation hazard utilities for ITCHI.

This module implements the normalized precipitation hazard component H_P
using local climatological percentiles.

ITCHI v0.1 treats precipitation as a snapshot field aligned with the
cyclone synoptic time, not as a temporal accumulation (Q90/Q95/Q99 ramp,
``compute_precipitation_hazard``).

ITCHI v0.2 adds (``compute_precipitation_matrix_hazard``) a continuous
matrix: the 6-h snapshot P6 follows the R4 ramp (Q75 -> 0, Q90 -> 0.25,
Q95 -> 0.5, Q99 -> 1) and is escalated towards an upper ramp (Q95 -> 0.8)
when the rain persists (R24, (T-24h, T]) on wet ground (exclusive 72-h
antecedent A72, (T-78h, T-6h]), each against its own local percentiles.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import DEFAULT_EPSILON


def _is_xarray_object(obj: Any) -> bool:
    """
    Check whether an object is an xarray DataArray or Dataset.
    """
    return isinstance(obj, xr.DataArray | xr.Dataset)


def compute_precipitation_hazard(
    precipitation: np.ndarray | xr.DataArray,
    q90: float | np.ndarray | xr.DataArray,
    q95: float | np.ndarray | xr.DataArray,
    q99: float | np.ndarray | xr.DataArray,
    epsilon: float = DEFAULT_EPSILON,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray | xr.DataArray:
    """
    Compute normalized precipitation hazard H_P.

    The hazard function is defined by a piecewise linear scaling:

    - P < Q90       -> H_P = 0
    - Q90 to Q95   -> H_P increases from 0 to 0.5
    - Q95 to Q99   -> H_P increases from 0.5 to 1
    - P >= Q99     -> H_P = 1

    Parameters
    ----------
    precipitation : numpy.ndarray or xarray.DataArray
        Precipitation snapshot field.
    q90 : float, numpy.ndarray or xarray.DataArray
        Local 90th percentile.
    q95 : float, numpy.ndarray or xarray.DataArray
        Local 95th percentile.
    q99 : float, numpy.ndarray or xarray.DataArray
        Local 99th percentile.
    epsilon : float, default=1e-6
        Small value used to avoid division by zero.
    clip_min : float, default=0.0
        Minimum allowed hazard value.
    clip_max : float, default=1.0
        Maximum allowed hazard value.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Normalized precipitation hazard in the interval [0, 1].
    """
    if any(_is_xarray_object(obj) for obj in (precipitation, q90, q95, q99)):
        return _compute_precipitation_hazard_xarray(
            precipitation=precipitation,
            q90=q90,
            q95=q95,
            q99=q99,
            epsilon=epsilon,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    return _compute_precipitation_hazard_numpy(
        precipitation=precipitation,
        q90=q90,
        q95=q95,
        q99=q99,
        epsilon=epsilon,
        clip_min=clip_min,
        clip_max=clip_max,
    )


def _compute_precipitation_hazard_numpy(
    precipitation: np.ndarray,
    q90: float | np.ndarray,
    q95: float | np.ndarray,
    q99: float | np.ndarray,
    epsilon: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """
    NumPy implementation of the precipitation hazard function.
    """
    p = np.asarray(precipitation, dtype=float)
    q90_arr = np.asarray(q90, dtype=float)
    q95_arr = np.asarray(q95, dtype=float)
    q99_arr = np.asarray(q99, dtype=float)

    denominator_low = np.maximum(q95_arr - q90_arr, epsilon)
    denominator_high = np.maximum(q99_arr - q95_arr, epsilon)

    hazard_low = 0.5 * ((p - q90_arr) / denominator_low)
    hazard_high = 0.5 + 0.5 * ((p - q95_arr) / denominator_high)

    hazard = np.select(
        condlist=[
            p < q90_arr,
            (p >= q90_arr) & (p < q95_arr),
            (p >= q95_arr) & (p < q99_arr),
            p >= q99_arr,
        ],
        choicelist=[
            0.0,
            hazard_low,
            hazard_high,
            1.0,
        ],
        default=np.nan,
    )

    return np.clip(hazard, clip_min, clip_max)


def _compute_precipitation_hazard_xarray(
    precipitation: xr.DataArray,
    q90: float | xr.DataArray,
    q95: float | xr.DataArray,
    q99: float | xr.DataArray,
    epsilon: float,
    clip_min: float,
    clip_max: float,
) -> xr.DataArray:
    """
    xarray implementation of the precipitation hazard function.
    """
    denominator_low = xr.where((q95 - q90) > epsilon, q95 - q90, epsilon)
    denominator_high = xr.where((q99 - q95) > epsilon, q99 - q95, epsilon)

    hazard_low = 0.5 * ((precipitation - q90) / denominator_low)
    hazard_high = 0.5 + 0.5 * ((precipitation - q95) / denominator_high)

    hazard = xr.where(
        precipitation < q90,
        0.0,
        xr.where(
            precipitation < q95,
            hazard_low,
            xr.where(
                precipitation < q99,
                hazard_high,
                1.0,
            ),
        ),
    )

    return hazard.clip(min=clip_min, max=clip_max)


# ============================================================
# ITCHI v0.2: node ramps and P6 x (R24, A72) matrix
# ============================================================

# R4 ramp: the no-antecedent column of the v0.2 matrix.
RAMP_R4_NODES: dict[str, float] = {"Q75": 0.0, "Q90": 0.25, "Q95": 0.5, "Q99": 1.0}

# Continuous P6 x (R24, A72) matrix, calibrated with municipality-cyclone pairs
# inside the W06 radius (scripts/negativos_w06.py). Discrete rule it reproduces:
# P6 in P95-P99 with R24 >= P95 and A72 >= P75 -> maximum level.
#   p6_nodes   : 6-h percentiles used as ramp nodes
#   base       : H_P at (0 mm, *p6_nodes) without escalation (R4)
#   escalated  : H_P at the same nodes with full escalation (P95 -> 0.8, top
#                SIAT-style class)
#   escalation : variable -> (start, end) percentiles of that variable's own
#                climatology; its weight goes from 0 at start to 1 at end
#                ("0" = 0 mm)
#   combine    : "all" -> product of weights (all conditions), "any" -> maximum
# H_P = (1 - f) * ramp(base) + f * ramp(escalated), f in [0, 1].
PRECIP_MATRIX_V02: dict[str, Any] = {
    "p6_nodes": ("Q75", "Q90", "Q95", "Q99"),
    "base": (0.0, 0.0, 0.25, 0.5, 1.0),
    "escalated": (0.0, 0.0, 0.25, 0.8, 1.0),
    "escalation": {
        "R24": ("Q90", "Q95"),
        "A72": ("0", "Q75"),
    },
    "combine": "all",
}


def _where(cond: Any, x: Any, y: Any) -> Any:
    """
    np.where or xr.where depending on the input types.
    """
    if any(_is_xarray_object(obj) for obj in (cond, x, y)):
        return xr.where(cond, x, y)

    return np.where(cond, x, y)


def _clip_unit(values: Any, clip_min: float = 0.0, clip_max: float = 1.0) -> Any:
    if _is_xarray_object(values):
        return values.clip(min=clip_min, max=clip_max)

    return np.clip(np.asarray(values, dtype=float), clip_min, clip_max)


def compute_precipitation_hazard_ramp(
    precipitation: np.ndarray | xr.DataArray,
    quantiles: Mapping[str, float | np.ndarray | xr.DataArray],
    nodes: Mapping[str, float],
    zero_value: float = 0.0,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray | xr.DataArray:
    """
    Piecewise-linear precipitation hazard through ramp nodes.

    Nodes are (0 mm, zero_value), (Q_1, h_1), ..., (Q_K, h_K), with Q_k the
    local percentiles named in ``nodes`` (keys of ``quantiles``) and
    h_K the value above the last node. With zero_value = 0 and h_1 = 0 the
    hazard is 0 below the first percentile, as in the R3/R4 ramps:

        R3 = {"Q90": 0, "Q95": 0.5, "Q99": 1}  (same as compute_precipitation_hazard)
        R4 = {"Q75": 0, "Q90": 0.25, "Q95": 0.5, "Q99": 1}

    NaN in precipitation or in a used percentile propagates to the output.

    Parameters
    ----------
    precipitation : numpy.ndarray or xarray.DataArray
        Precipitation field (same support as the percentiles).
    quantiles : Mapping[str, float, numpy.ndarray or xarray.DataArray]
        Local percentiles by name, e.g. {"Q75": q75, "Q90": q90, ...}.
    nodes : Mapping[str, float]
        Ordered percentile name -> hazard value at that node.
    zero_value : float, default=0.0
        Hazard value at 0 mm.
    epsilon : float, default=1e-6
        Minimum percentile spacing used to avoid division by zero.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Hazard in [0, 1].
    """
    keys = list(nodes)
    missing = [k for k in keys if k not in quantiles]
    if missing:
        raise KeyError(f"Missing percentile(s) for the ramp: {missing}")

    h = [float(zero_value)] + [float(nodes[k]) for k in keys]
    if any(b < a for a, b in zip(h[:-1], h[1:], strict=False)):
        raise ValueError(f"Ramp values must be non-decreasing: {h}")

    p = precipitation
    q = [0.0] + [quantiles[k] for k in keys]

    hazard = p * 0.0 + h[-1]
    for qk in q[1:]:
        hazard = hazard + qk * 0.0          # broadcast and propagate NaN percentiles

    for k in range(len(q) - 2, -1, -1):
        frac = (p - q[k]) / np.maximum(q[k + 1] - q[k], epsilon)
        segment = h[k] + (h[k + 1] - h[k]) * frac
        hazard = _where(p < q[k + 1], segment, hazard)

    hazard = _where(p < 0.0, h[0], hazard)

    return _clip_unit(hazard)


def compute_precipitation_matrix_hazard(
    precipitation: np.ndarray | xr.DataArray,
    quantiles_6h: Mapping[str, float | np.ndarray | xr.DataArray],
    escalation_values: Mapping[str, float | np.ndarray | xr.DataArray],
    escalation_quantiles: Mapping[str, Mapping[str, float | np.ndarray | xr.DataArray]],
    matrix: Mapping[str, Any] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray | xr.DataArray:
    """
    ITCHI v0.2 precipitation hazard: P6 ramp escalated by persistence and
    antecedent rain.

        H_P = (1 - f) * H_base(P6) + f * H_esc(P6)

    H_base and H_esc are node ramps over the local 6-h percentiles
    (compute_precipitation_hazard_ramp). Each escalation variable v
    (default R24 and A72) gets a weight

        f_v = clip((v - q_start) / (q_end - q_start), 0, 1)

    with q_start, q_end its own local percentiles ("0" = 0 mm), and
    f = prod(f_v) for combine="all" or max(f_v) for combine="any". A missing
    value or percentile gives f_v = 0 (no escalation), e.g. the first 72 h of
    a season. Because H_esc >= H_base node by node, H_P is non-decreasing in
    P6 and in every escalation variable.

    Parameters
    ----------
    precipitation : numpy.ndarray or xarray.DataArray
        6-h precipitation snapshot P6 (T-6h, T].
    quantiles_6h : Mapping
        Local 6-h percentiles by name (Q75, Q90, Q95, Q99 for the default).
    escalation_values : Mapping
        Escalation fields by name, e.g. {"R24": r24, "A72": a72} with
        R24 = precipitation in (T-24h, T] and A72 = precipitation in
        (T-78h, T-6h]; see compute_antecedent_precipitation.
    escalation_quantiles : Mapping
        Local percentiles of each escalation variable, e.g.
        {"R24": {"Q90": ..., "Q95": ...}, "A72": {"Q75": ...}}. A72(T) =
        R72(T - 6 h), so the R72 climatology applies to A72.
    matrix : Mapping or None, default=None
        Matrix definition; None uses PRECIP_MATRIX_V02.
    epsilon : float, default=1e-6
        Minimum percentile spacing.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Hazard H_P in [0, 1].
    """
    m = PRECIP_MATRIX_V02 if matrix is None else matrix
    p6_nodes = tuple(m["p6_nodes"])
    base, escalated = tuple(m["base"]), tuple(m["escalated"])
    combine = m.get("combine", "all")

    for name, values in (("base", base), ("escalated", escalated)):
        if len(values) != len(p6_nodes) + 1:
            raise ValueError(f"{name} needs {len(p6_nodes) + 1} values (0 mm and {', '.join(p6_nodes)}).")
    if np.any(np.asarray(escalated) < np.asarray(base)):
        warnings.warn("escalated ramp is below the base ramp at some node.", stacklevel=2)
    if combine not in {"all", "any"}:
        raise ValueError(f"combine must be 'all' or 'any': {combine!r}")

    h_base = compute_precipitation_hazard_ramp(
        precipitation, quantiles_6h, dict(zip(p6_nodes, base[1:], strict=True)),
        zero_value=base[0], epsilon=epsilon,
    )
    h_esc = compute_precipitation_hazard_ramp(
        precipitation, quantiles_6h, dict(zip(p6_nodes, escalated[1:], strict=True)),
        zero_value=escalated[0], epsilon=epsilon,
    )

    weights = []
    for name, (start, end) in dict(m["escalation"]).items():
        if name not in escalation_values:
            raise KeyError(f"Missing escalation variable: {name}")
        qv = escalation_quantiles.get(name, {})
        bounds = []
        for key in (start, end):
            if key == "0":
                bounds.append(0.0)
            elif key in qv:
                bounds.append(qv[key])
            else:
                raise KeyError(f"Missing percentile {key} for escalation variable {name}")
        x = escalation_values[name]
        w = _clip_unit((x - bounds[0]) / np.maximum(bounds[1] - bounds[0], epsilon))
        valid = np.isfinite(x) & np.isfinite(bounds[0]) & np.isfinite(bounds[1])
        weights.append(_where(valid, w, 0.0))

    f = weights[0]
    for w in weights[1:]:
        f = f * w if combine == "all" else np.maximum(f, w)

    return _clip_unit((1.0 - f) * h_base + f * h_esc)


def compute_antecedent_precipitation(
    precipitation_6h: xr.DataArray,
    hours: int = 72,
    time_dim: str = "time",
    step_hours: int = 6,
    exclude_current: bool = True,
) -> xr.DataArray:
    """
    Running accumulation of 6-h precipitation.

    exclude_current=True  -> A_d(T) = sum over (T - d - 6 h, T - 6 h]
                             (exclusive antecedent, A72 in ITCHI v0.2).
    exclude_current=False -> R_d(T) = sum over (T - d, T].

    A window with a missing step, or that is not contiguous in time (for
    example across the December -> May gap of seasonal files), is NaN. The
    first d/6 - 1 (or d/6) steps of each contiguous series are NaN.

    Parameters
    ----------
    precipitation_6h : xarray.DataArray
        6-h accumulations with a time dimension (window end T).
    hours : int, default=72
        Accumulation length; multiple of step_hours.
    time_dim : str, default="time"
        Name of the time dimension.
    step_hours : int, default=6
        Length of each input accumulation.
    exclude_current : bool, default=True
        Exclude the current 6-h step (antecedent) or include it.

    Returns
    -------
    xarray.DataArray
        Accumulated precipitation in the same units as the input.
    """
    if hours % step_hours or hours < step_hours:
        raise ValueError(f"hours must be a multiple of {step_hours}: {hours}")

    n = hours // step_hours
    times = precipitation_6h[time_dim].values

    def _contiguous(lag: int) -> xr.DataArray:
        ok = np.zeros(times.size, dtype=bool)
        if lag == 0:
            ok[:] = True
        elif times.size > lag:
            ok[lag:] = (times[lag:] - times[:-lag]) == np.timedelta64(step_hours * lag, "h")
        return xr.DataArray(ok, dims=time_dim, coords={time_dim: precipitation_6h[time_dim]})

    acc = precipitation_6h.rolling({time_dim: n}, min_periods=n).sum()
    acc = acc.where(_contiguous(n - 1))

    if exclude_current:
        acc = acc.shift({time_dim: 1}).where(_contiguous(1))

    window = (f"(T-{hours + step_hours}h, T-{step_hours}h]" if exclude_current
              else f"(T-{hours}h, T]")
    return acc.assign_attrs(
        long_name=f"{hours}-h {'antecedent ' if exclude_current else ''}precipitation",
        accumulation_window=window,
    )


# ============================================================
# ITCHI v0.2.1: class table P6 x R24 x A72 (Block 1 experiments)
# ============================================================
#
# Each variable gets a class coordinate u in its own local percentiles:
# u = 0 at 0 mm, u = k at the k-th node percentile (Q75 -> 1, Q90 -> 2, ...),
# linear in between and constant (= K) above the last node. The hazard is a
# table of values at the integer nodes, interpolated multilinearly in
# (u_P6, u_R24, u_A72). A table that is non-decreasing along every axis gives
# a hazard that is non-decreasing in P6, R24 and A72.
#
# Spec (mapping, YAML friendly):
#   kind       : "class_table"
#   axes       : {"P6": nodes, "R24": nodes, "A72": nodes}; first axis = P6,
#                the others are escalation variables (any names and number)
#   p6_values  : H at (0 mm, *P6 nodes) when no rule applies
#   rules      : [{"when": {var: node, ...}, "p6_values": [...]}, ...]; a rule
#                applies at the nodes where every listed variable is at or
#                above its node; the table value is the maximum of p6_values
#                and of the rules that apply
#   table      : alternative to p6_values/rules: explicit nested list with one
#                level per axis (len(nodes) + 1 values per axis)
#   gate       : optional {"P6": node}; below that P6 node the escalation
#                variables are ignored (H = table at their 0-mm node)
#
# PRECIP_TABLE_V02 reproduces compute_precipitation_matrix_hazard with
# PRECIP_MATRIX_V02 (tests/test_class_table.py).

PRECIP_TABLE_V02: dict[str, Any] = {
    "kind": "class_table",
    "axes": {"P6": ("Q75", "Q90", "Q95", "Q99"), "R24": ("Q90", "Q95"), "A72": ("Q75",)},
    "p6_values": (0.0, 0.0, 0.25, 0.5, 1.0),
    "rules": ({"when": {"R24": "Q95", "A72": "Q75"}, "p6_values": (0.0, 0.0, 0.25, 0.8, 1.0)},),
}

_B1_AXES: dict[str, tuple[str, ...]] = {
    "P6": ("Q75", "Q90", "Q95"),
    "R24": ("Q75", "Q90", "Q95"),
    "A72": ("Q75", "Q90", "Q95"),
}

# Block 1 candidates: P95 of 6 h is always the maximum (H = 1); P75 is the
# onset of the alert. They differ in how the R24 and A72 thresholds act.
PRECIP_TABLES_B1: dict[str, dict[str, Any]] = {
    # M0: 6 h only (P75 -> 0, P90 -> 0.5, P95 -> 1); reference without R24/A72
    "M0": {"kind": "class_table", "axes": _B1_AXES, "p6_values": (0.0, 0.0, 0.5, 1.0)},
    # M1: v0.2 escalation moved one class earlier: with R24 >= P95 and
    # A72 >= P75, P90 is already the maximum (P75-P90 ramps 0 -> 1)
    "M1": {
        "kind": "class_table", "axes": _B1_AXES, "p6_values": (0.0, 0.0, 0.5, 1.0),
        "rules": ({"when": {"R24": "Q95", "A72": "Q75"}, "p6_values": (0.0, 0.0, 1.0, 1.0)},),
    },
    # M2: each duration with its own ladder (P90 -> 0.5, P95 -> 1), maximum of
    # the three, only with current rain P6 >= P75 (gate)
    "M2": {
        "kind": "class_table", "axes": _B1_AXES, "p6_values": (0.0, 0.0, 0.5, 1.0),
        "rules": (
            {"when": {"R24": "Q90"}, "p6_values": (0.5, 0.5, 0.5, 1.0)},
            {"when": {"R24": "Q95"}, "p6_values": (1.0, 1.0, 1.0, 1.0)},
            {"when": {"A72": "Q90"}, "p6_values": (0.5, 0.5, 0.5, 1.0)},
            {"when": {"A72": "Q95"}, "p6_values": (1.0, 1.0, 1.0, 1.0)},
        ),
        "gate": {"P6": "Q75"},
    },
}


def compute_percentile_coordinate(
    values: np.ndarray | xr.DataArray,
    quantiles: Mapping[str, float | np.ndarray | xr.DataArray],
    nodes: Sequence[str],
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Class coordinate of a value in its own local percentiles.

    u = 0 at 0 mm (and below), u = k at the k-th node percentile, linear in
    between, u = len(nodes) at and above the last node. NaN where the value or
    a node percentile is NaN.

    Parameters
    ----------
    values : numpy.ndarray or xarray.DataArray
        Precipitation (same support as the percentiles).
    quantiles : Mapping
        Local percentiles by name ("Q75", "Q90", ...).
    nodes : sequence of str
        Ordered percentile names used as nodes.
    epsilon : float, default=1e-6
        Minimum spacing between consecutive nodes.

    Returns
    -------
    numpy.ndarray
        Coordinate in [0, len(nodes)] (NumPy array, broadcast shape).
    """
    nodes = list(nodes)
    if not nodes:
        raise ValueError("At least one percentile node is required.")
    missing = [k for k in nodes if k not in quantiles]
    if missing:
        raise KeyError(f"Missing percentile(s): {missing}")

    x = np.asarray(values, dtype=float)
    q = [np.asarray(quantiles[k], dtype=float) for k in nodes]
    shape = np.broadcast_shapes(x.shape, *(qk.shape for qk in q))
    x = np.broadcast_to(x, shape)

    with np.errstate(invalid="ignore"):
        u = np.full(shape, float(len(nodes)))
        for k in range(len(nodes) - 1, -1, -1):
            lower = 0.0 if k == 0 else q[k - 1]
            frac = (x - lower) / np.maximum(q[k] - lower, epsilon)
            u = np.where(x < q[k], k + frac, u)
        u = np.clip(np.where(x <= 0.0, 0.0, u), 0.0, float(len(nodes)))

    valid = np.isfinite(x)
    for qk in q:
        valid = valid & np.isfinite(qk)
    return np.where(valid, u, np.nan)


def _node_index(axis: str, nodes: Sequence[str], node: str) -> int:
    """
    Table index of a node percentile (0 = 0 mm).
    """
    if node not in nodes:
        raise KeyError(f"{node!r} is not a node of axis {axis!r}: {tuple(nodes)}")
    return list(nodes).index(node) + 1


def build_precipitation_class_table(spec: Mapping[str, Any]) -> np.ndarray:
    """
    Node table of a class-table spec.

    Parameters
    ----------
    spec : Mapping
        Class-table spec (see PRECIP_TABLE_V02 and PRECIP_TABLES_B1).

    Returns
    -------
    numpy.ndarray
        Table with one dimension per axis, len(nodes) + 1 values each (index 0
        = 0 mm), values in [0, 1] and non-decreasing along every axis.

    Raises
    ------
    ValueError
        If the spec is inconsistent, out of [0, 1] or not monotone.
    """
    axes = {str(k): tuple(str(n) for n in v) for k, v in dict(spec["axes"]).items()}
    names = list(axes)
    if not names or names[0] != "P6":
        raise ValueError(f"The first axis must be 'P6': {names}")
    shape = tuple(len(axes[a]) + 1 for a in names)

    if "table" in spec:
        if "p6_values" in spec or "rules" in spec:
            raise ValueError("Give either 'table' or 'p6_values'/'rules', not both.")
        table = np.asarray(spec["table"], dtype=float)
        if table.shape != shape:
            raise ValueError(f"table shape {table.shape} does not match the axes {shape}")
    else:
        base = np.asarray(spec["p6_values"], dtype=float)
        if base.shape != (shape[0],):
            raise ValueError(f"p6_values needs {shape[0]} values (0 mm and {', '.join(axes['P6'])}).")
        table = np.broadcast_to(base.reshape((-1,) + (1,) * (len(names) - 1)), shape).copy()

        for rule in spec.get("rules", ()) or ():
            values = np.asarray(rule["p6_values"], dtype=float)
            if values.shape != (shape[0],):
                raise ValueError(f"rule p6_values needs {shape[0]} values: {rule}")
            if np.any(np.diff(values) < 0.0):
                raise ValueError(f"rule p6_values must be non-decreasing: {rule}")
            when = dict(rule["when"])
            if not when:
                raise ValueError("A rule needs at least one condition.")
            index: list[Any] = [slice(None)] * len(names)
            for var, node in when.items():
                if var == "P6" or var not in axes:
                    raise KeyError(f"Rule condition on an unknown escalation axis: {var!r}")
                index[names.index(var)] = slice(_node_index(var, axes[var], str(node)), None)
            sub = table[tuple(index)]
            table[tuple(index)] = np.maximum(sub, values.reshape((-1,) + (1,) * (len(names) - 1)))

    if np.any(~np.isfinite(table)) or table.min() < 0.0 or table.max() > 1.0:
        raise ValueError("Table values must be finite and in [0, 1].")
    for axis in range(table.ndim):
        if np.any(np.diff(table, axis=axis) < -1e-12):
            raise ValueError(f"Table is not non-decreasing along axis {names[axis]!r}.")

    return table


def compute_precipitation_class_table_hazard(
    precipitation: np.ndarray | xr.DataArray,
    quantiles_6h: Mapping[str, float | np.ndarray | xr.DataArray],
    escalation_values: Mapping[str, float | np.ndarray | xr.DataArray],
    escalation_quantiles: Mapping[str, Mapping[str, float | np.ndarray | xr.DataArray]],
    spec: Mapping[str, Any] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray | xr.DataArray:
    """
    Precipitation hazard from a class table P6 x (escalation variables).

    H = multilinear interpolation of the node table in the class coordinates
    (u_P6, u_R24, u_A72, ...), each against its own local percentiles
    (compute_percentile_coordinate). A missing escalation value or
    percentile counts as its 0-mm node (no escalation); a missing P6 or 6-h
    percentile gives NaN, as in compute_precipitation_matrix_hazard.

    Parameters
    ----------
    precipitation : numpy.ndarray or xarray.DataArray
        P6, precipitation in (T-6h, T].
    quantiles_6h : Mapping
        Local 6-h percentiles by name.
    escalation_values : Mapping
        Escalation fields by axis name, e.g. {"R24": r24, "A72": a72}.
    escalation_quantiles : Mapping
        Local percentiles of each escalation variable, e.g. {"R24": {...}}.
        A72(T) = R72(T - 6 h), so the 72-h climatology applies to A72.
    spec : Mapping or None, default=None
        Class-table spec; None uses PRECIP_TABLES_B1["M1"].
    epsilon : float, default=1e-6
        Minimum percentile spacing.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Hazard H_P in [0, 1] (xarray if any input is xarray).
    """
    s = PRECIP_TABLES_B1["M1"] if spec is None else spec
    table = build_precipitation_class_table(s)
    axes = {str(k): tuple(str(n) for n in v) for k, v in dict(s["axes"]).items()}
    names = list(axes)

    for name in names[1:]:
        if name not in escalation_values:
            raise KeyError(f"Missing escalation variable: {name}")
        if name not in escalation_quantiles:
            raise KeyError(f"Missing percentiles for escalation variable: {name}")

    fields = [precipitation] + [escalation_values[n] for n in names[1:]]
    quantile_sets = [quantiles_6h] + [escalation_quantiles[n] for n in names[1:]]

    template = None
    if any(_is_xarray_object(f) for f in fields) or any(
        _is_xarray_object(v) for qs in quantile_sets for v in qs.values()
    ):
        objects = [f for f in fields if _is_xarray_object(f)] + [
            v for qs in quantile_sets for v in qs.values() if _is_xarray_object(v)
        ]
        template = xr.broadcast(*objects)[0]
        template = xr.zeros_like(template, dtype=float)
        fields = [f + template if _is_xarray_object(f) else f for f in fields]
        quantile_sets = [
            {k: (v + template if _is_xarray_object(v) else v) for k, v in qs.items()}
            for qs in quantile_sets
        ]
        fields = [f.values if _is_xarray_object(f) else f for f in fields]
        quantile_sets = [
            {k: (v.values if _is_xarray_object(v) else v) for k, v in qs.items()}
            for qs in quantile_sets
        ]

    coords = [
        compute_percentile_coordinate(f, qs, axes[n], epsilon=epsilon)
        for f, qs, n in zip(fields, quantile_sets, names, strict=True)
    ]
    shape = np.broadcast_shapes(*(c.shape for c in coords))
    coords = [np.broadcast_to(c, shape) for c in coords]

    p6_missing = ~np.isfinite(coords[0])
    coords = [np.where(np.isfinite(c), c, 0.0) for c in coords]

    gate = dict(s.get("gate") or {})
    if gate:
        if set(gate) != {"P6"}:
            raise ValueError(f"Only a P6 gate is supported: {gate}")
        gate_u = float(_node_index("P6", axes["P6"], str(gate["P6"])))
        closed = coords[0] < gate_u
        coords = [coords[0]] + [np.where(closed, 0.0, c) for c in coords[1:]]

    def _interpolate(us: list[np.ndarray]) -> np.ndarray:
        # corner values V[c_1, ..., c_N, ...], then one linear interpolation per
        # axis (v0 + t (v1 - v0)): equal corners give that value exactly
        lower, weight = [], []
        for u, n in zip(us, table.shape, strict=True):
            i0 = np.clip(np.floor(u), 0, n - 2).astype(int)
            lower.append(i0)
            weight.append(u - i0)
        corners = np.empty((2,) * len(us) + shape)
        for corner in np.ndindex(*(2,) * len(us)):
            corners[corner] = table[tuple(i0 + c for c, i0 in zip(corner, lower, strict=True))]
        values = corners
        for axis in range(len(us) - 1, -1, -1):
            head = (slice(None),) * axis
            v0, v1 = values[head + (0,)], values[head + (1,)]
            values = v0 + weight[axis] * (v1 - v0)
        return values

    hazard = np.clip(_interpolate(coords), 0.0, 1.0)
    hazard = np.where(p6_missing, np.nan, hazard)

    if template is not None:
        return template.copy(data=hazard)
    return hazard


def compute_precipitation_hazard_from_spec(
    precipitation: np.ndarray | xr.DataArray,
    quantiles_6h: Mapping[str, float | np.ndarray | xr.DataArray],
    escalation_values: Mapping[str, float | np.ndarray | xr.DataArray],
    escalation_quantiles: Mapping[str, Mapping[str, float | np.ndarray | xr.DataArray]],
    spec: Mapping[str, Any] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray | xr.DataArray:
    """
    Dispatch on the precipitation spec: "class_table" or the v0.2 matrix.

    A spec with kind == "class_table" uses
    compute_precipitation_class_table_hazard; any other spec (or None) is a
    matrix for compute_precipitation_matrix_hazard.
    """
    if spec is not None and spec.get("kind") == "class_table":
        return compute_precipitation_class_table_hazard(
            precipitation, quantiles_6h, escalation_values, escalation_quantiles,
            spec=spec, epsilon=epsilon,
        )
    return compute_precipitation_matrix_hazard(
        precipitation, quantiles_6h, escalation_values, escalation_quantiles,
        matrix=spec, epsilon=epsilon,
    )
