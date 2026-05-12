"""
Quality-control utilities for ITCHI.

This module validates numerical, geometric and structural consistency
of ITCHI snapshot outputs.

Main checks
-----------
1. Hazard/index fields remain within [0, 1].
2. R34_q <= ROCLOUD_q when both radii are available.
3. The exterior region has zero ITCHI contribution.
4. Direct, indirect and exterior masks do not overlap.
5. Main fields preserve dimensions and coordinates.
6. Geometric radius units are explicitly kilometers before mask construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from itchi.constants import ITCHI_MAX_VALUE, ITCHI_MIN_VALUE

ArrayLike = np.ndarray | xr.DataArray


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def _scalar_bool(value: xr.DataArray) -> bool:
    """
    Convert a scalar xarray object to a Python bool.
    """
    if hasattr(value, "compute"):
        value = value.compute()

    return bool(value.item())


def _has_any_true(values: np.ndarray | xr.DataArray) -> bool:
    """
    Return True if any value is True.
    """
    if _contains_xarray_object(values):
        return _scalar_bool(values.any())

    return bool(np.any(np.asarray(values)))


def validate_bounds(
    values: ArrayLike,
    name: str,
    lower: float = ITCHI_MIN_VALUE,
    upper: float = ITCHI_MAX_VALUE,
    allow_nan: bool = True,
) -> None:
    """
    Validate that a numerical field is bounded within [lower, upper].

    Parameters
    ----------
    values : numpy.ndarray or xarray.DataArray
        Field to validate.
    name : str
        Field name used in error messages.
    lower : float, default=0.0
        Lower valid bound.
    upper : float, default=1.0
        Upper valid bound.
    allow_nan : bool, default=True
        Whether NaN values are allowed.

    Raises
    ------
    ValueError
        If values are outside bounds or if NaNs are found when allow_nan=False.
    """
    if _contains_xarray_object(values):
        if not allow_nan and _has_any_true(values.isnull()):
            raise ValueError(f"{name} contains NaN values.")

        valid = values.notnull() if allow_nan else xr.ones_like(values, dtype=bool)
        violations = ((values < lower) | (values > upper)) & valid

        if _has_any_true(violations):
            raise ValueError(
                f"{name} must be within [{lower}, {upper}]. "
                "At least one invalid value was found."
            )

        return

    arr = np.asarray(values, dtype=float)

    if not allow_nan and np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values.")

    valid = np.isfinite(arr) if allow_nan else np.ones_like(arr, dtype=bool)
    violations = ((arr < lower) | (arr > upper)) & valid

    if np.any(violations):
        raise ValueError(
            f"{name} must be within [{lower}, {upper}]. "
            "At least one invalid value was found."
        )


def validate_hazard_bounds(
    fields: Mapping[str, ArrayLike],
    field_names: Sequence[str] = (
        "ITCHI",
        "H_P",
        "H_Pdir",
        "H_Pind",
        "H_W",
        "H_dir",
        "H_ind",
    ),
    lower: float = ITCHI_MIN_VALUE,
    upper: float = ITCHI_MAX_VALUE,
    allow_nan: bool = True,
) -> None:
    """
    Validate that standard ITCHI hazard fields are bounded in [0, 1].

    Parameters
    ----------
    fields : Mapping[str, ArrayLike]
        Dictionary containing ITCHI fields.
    field_names : sequence of str
        Names of fields to validate if present.
    lower : float, default=0.0
        Lower valid bound.
    upper : float, default=1.0
        Upper valid bound.
    allow_nan : bool, default=True
        Whether NaNs are allowed.
    """
    for name in field_names:
        if name in fields:
            validate_bounds(
                values=fields[name],
                name=name,
                lower=lower,
                upper=upper,
                allow_nan=allow_nan,
            )


def validate_r34_le_rocloud(
    r34_q: ArrayLike,
    rocloud_q: ArrayLike,
    allow_nan: bool = True,
) -> None:
    """
    Validate that R34_q <= ROCLOUD_q where both radii exist.

    Parameters
    ----------
    r34_q : numpy.ndarray or xarray.DataArray
        Grid-cell-specific R34 radius in kilometers.
    rocloud_q : numpy.ndarray or xarray.DataArray
        Grid-cell-specific ROCLOUD radius in kilometers.
    allow_nan : bool, default=True
        Whether NaNs are allowed.

    Raises
    ------
    ValueError
        If R34_q > ROCLOUD_q for any valid cell.
    """
    if _contains_xarray_object(r34_q, rocloud_q):
        if not isinstance(r34_q, xr.DataArray) or not isinstance(
            rocloud_q,
            xr.DataArray,
        ):
            raise TypeError("r34_q and rocloud_q must both be xarray DataArrays.")

        if not allow_nan and (
            _has_any_true(r34_q.isnull()) or _has_any_true(rocloud_q.isnull())
        ):
            raise ValueError("R34_q or ROCLOUD_q contains NaN values.")

        valid = (
            r34_q.notnull() & rocloud_q.notnull()
            if allow_nan
            else xr.ones_like(r34_q, dtype=bool)
        )

        violations = (r34_q > rocloud_q) & valid

        if _has_any_true(violations):
            raise ValueError("R34_q must be less than or equal to ROCLOUD_q.")

        return

    r34_arr = np.asarray(r34_q, dtype=float)
    rocloud_arr = np.asarray(rocloud_q, dtype=float)

    if not allow_nan and (np.any(np.isnan(r34_arr)) or np.any(np.isnan(rocloud_arr))):
        raise ValueError("R34_q or ROCLOUD_q contains NaN values.")

    valid = np.isfinite(r34_arr) & np.isfinite(rocloud_arr) if allow_nan else True
    violations = (r34_arr > rocloud_arr) & valid

    if np.any(violations):
        raise ValueError("R34_q must be less than or equal to ROCLOUD_q.")


def validate_exterior_zero(
    values: ArrayLike,
    exterior_mask: np.ndarray | xr.DataArray,
    name: str = "ITCHI",
    tolerance: float = 1.0e-12,
    allow_nan: bool = True,
) -> None:
    """
    Validate that a field is zero in the exterior region.

    Parameters
    ----------
    values : numpy.ndarray or xarray.DataArray
        Field to validate.
    exterior_mask : numpy.ndarray or xarray.DataArray
        Boolean mask where True indicates exterior region.
    name : str, default="ITCHI"
        Field name used in error messages.
    tolerance : float, default=1e-12
        Numerical tolerance for zero checks.
    allow_nan : bool, default=True
        Whether NaN values are allowed and ignored.

    Raises
    ------
    ValueError
        If values are non-zero in the exterior region.
    """
    if _contains_xarray_object(values, exterior_mask):
        if not isinstance(values, xr.DataArray) or not isinstance(
            exterior_mask,
            xr.DataArray,
        ):
            raise TypeError("values and exterior_mask must both be xarray DataArrays.")

        mask = exterior_mask.fillna(False).astype(bool)
        valid = values.notnull() if allow_nan else xr.ones_like(values, dtype=bool)

        if not allow_nan and _has_any_true(values.isnull()):
            raise ValueError(f"{name} contains NaN values.")

        violations = (np.abs(values) > tolerance) & mask & valid

        if _has_any_true(violations):
            raise ValueError(f"{name} must be zero in the exterior region.")

        return

    arr = np.asarray(values, dtype=float)
    mask = np.asarray(exterior_mask, dtype=bool)

    if not allow_nan and np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values.")

    valid = np.isfinite(arr) if allow_nan else np.ones_like(arr, dtype=bool)
    violations = (np.abs(arr) > tolerance) & mask & valid

    if np.any(violations):
        raise ValueError(f"{name} must be zero in the exterior region.")


def validate_masks_exclusive(
    direct_mask: np.ndarray | xr.DataArray,
    indirect_mask: np.ndarray | xr.DataArray,
    exterior_mask: np.ndarray | xr.DataArray,
    require_complete: bool = True,
) -> None:
    """
    Validate that direct, indirect and exterior masks do not overlap.

    Parameters
    ----------
    direct_mask : numpy.ndarray or xarray.DataArray
        Direct-region mask.
    indirect_mask : numpy.ndarray or xarray.DataArray
        Indirect-region mask.
    exterior_mask : numpy.ndarray or xarray.DataArray
        Exterior-region mask.
    require_complete : bool, default=True
        If True, every cell must belong to exactly one mask.

    Raises
    ------
    ValueError
        If masks overlap or if require_complete=True and some cells are unassigned.
    """
    if _contains_xarray_object(direct_mask, indirect_mask, exterior_mask):
        if not all(
            isinstance(mask, xr.DataArray)
            for mask in (direct_mask, indirect_mask, exterior_mask)
        ):
            raise TypeError("All masks must be xarray DataArrays.")

        direct = direct_mask.fillna(False).astype(bool)
        indirect = indirect_mask.fillna(False).astype(bool)
        exterior = exterior_mask.fillna(False).astype(bool)

        total = direct.astype(int) + indirect.astype(int) + exterior.astype(int)

        if _has_any_true(total > 1):
            raise ValueError("Direct, indirect and exterior masks must not overlap.")

        if require_complete and _has_any_true(total != 1):
            raise ValueError("Every cell must belong to exactly one mask.")

        return

    direct = np.asarray(direct_mask, dtype=bool)
    indirect = np.asarray(indirect_mask, dtype=bool)
    exterior = np.asarray(exterior_mask, dtype=bool)

    total = direct.astype(int) + indirect.astype(int) + exterior.astype(int)

    if np.any(total > 1):
        raise ValueError("Direct, indirect and exterior masks must not overlap.")

    if require_complete and np.any(total != 1):
        raise ValueError("Every cell must belong to exactly one mask.")


def validate_field_alignment(
    reference: Any,
    fields: Mapping[str, Any],
    check_coords: bool = True,
) -> None:
    """
    Validate that fields preserve shape, dimensions and coordinates.

    For xarray inputs, dimensions must match the reference dimensions.
    If check_coords=True, dimension coordinates must also match.

    For NumPy inputs, shapes must match the reference shape.

    Parameters
    ----------
    reference : numpy.ndarray or xarray.DataArray
        Reference field.
    fields : Mapping[str, Any]
        Fields to compare against the reference.
    check_coords : bool, default=True
        Whether to check xarray coordinates.

    Raises
    ------
    TypeError
        If xarray and non-xarray objects are mixed.
    ValueError
        If shapes, dimensions or coordinates are inconsistent.
    """
    if _contains_xarray_object(reference):
        if not isinstance(reference, xr.DataArray):
            raise TypeError("reference must be an xarray DataArray.")

        for name, field in fields.items():
            if not isinstance(field, xr.DataArray):
                raise TypeError(
                    f"{name} must be an xarray DataArray to match the reference."
                )

            if field.dims != reference.dims:
                raise ValueError(
                    f"{name} dimensions {field.dims} do not match "
                    f"reference dimensions {reference.dims}."
                )

            if field.shape != reference.shape:
                raise ValueError(
                    f"{name} shape {field.shape} does not match "
                    f"reference shape {reference.shape}."
                )

            if check_coords:
                for dim in reference.dims:
                    if dim in reference.coords:
                        if dim not in field.coords:
                            raise ValueError(f"{name} is missing coordinate {dim}.")

                        if not field.coords[dim].equals(reference.coords[dim]):
                            raise ValueError(
                                f"{name} coordinate {dim} does not match reference."
                            )

        return

    reference_shape = np.shape(reference)

    for name, field in fields.items():
        if _contains_xarray_object(field):
            raise TypeError(f"{name} cannot be xarray if reference is not xarray.")

        if np.shape(field) != reference_shape:
            raise ValueError(
                f"{name} shape {np.shape(field)} does not match "
                f"reference shape {reference_shape}."
            )


def validate_geometry_units(
    radius_unit: str = "km",
    r34_unit: str = "km",
    rocloud_unit: str = "km",
) -> None:
    """
    Validate that geometric fields are explicitly in kilometers.

    Parameters
    ----------
    radius_unit : str, default="km"
        Unit of radial distance.
    r34_unit : str, default="km"
        Unit of R34_q.
    rocloud_unit : str, default="km"
        Unit of ROCLOUD_q.

    Raises
    ------
    ValueError
        If any geometric unit is not kilometers.
    """
    valid_km_units = {"km", "kilometer", "kilometers", "kilometre", "kilometres"}

    units = {
        "radius_km": radius_unit,
        "R34_q": r34_unit,
        "ROCLOUD_q": rocloud_unit,
    }

    for name, unit in units.items():
        normalized = str(unit).lower().strip()

        if normalized not in valid_km_units:
            raise ValueError(
                f"{name} must be in kilometers before mask construction. "
                f"Received unit: {unit}"
            )


def run_snapshot_quality_control(
    snapshot_result: Mapping[str, Any],
    radius_unit: str = "km",
    r34_unit: str = "km",
    rocloud_unit: str = "km",
    require_complete_masks: bool = True,
    check_coords: bool = True,
) -> None:
    """
    Run standard quality-control checks for one ITCHI snapshot result.

    Parameters
    ----------
    snapshot_result : Mapping[str, Any]
        Output dictionary from the ITCHI snapshot pipeline.
    radius_unit : str, default="km"
        Unit of radius_km.
    r34_unit : str, default="km"
        Unit of R34_q.
    rocloud_unit : str, default="km"
        Unit of ROCLOUD_q.
    require_complete_masks : bool, default=True
        Whether every cell must belong to exactly one region.
    check_coords : bool, default=True
        Whether to check xarray dimension coordinates.

    Raises
    ------
    KeyError
        If required fields are missing.
    ValueError
        If any quality-control rule fails.
    """
    required_fields = {
        "M_direct",
        "M_indirect",
        "M_exterior",
        "H_P",
        "H_Pdir",
        "H_Pind",
        "H_W",
        "H_dir",
        "H_ind",
        "ITCHI",
    }

    missing = required_fields.difference(snapshot_result)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing required snapshot field(s): {missing_text}")

    validate_geometry_units(
        radius_unit=radius_unit,
        r34_unit=r34_unit,
        rocloud_unit=rocloud_unit,
    )

    validate_hazard_bounds(snapshot_result)

    validate_masks_exclusive(
        direct_mask=snapshot_result["M_direct"],
        indirect_mask=snapshot_result["M_indirect"],
        exterior_mask=snapshot_result["M_exterior"],
        require_complete=require_complete_masks,
    )

    validate_exterior_zero(
        values=snapshot_result["ITCHI"],
        exterior_mask=snapshot_result["M_exterior"],
        name="ITCHI",
    )

    if "R_direct_q" in snapshot_result and "ROCLOUD_q" in snapshot_result:
        validate_r34_le_rocloud(
            r34_q=snapshot_result["R_direct_q"],
            rocloud_q=snapshot_result["ROCLOUD_q"],
        )
    elif "R34_q" in snapshot_result and "ROCLOUD_q" in snapshot_result:
        validate_r34_le_rocloud(
            r34_q=snapshot_result["R34_q"],
            rocloud_q=snapshot_result["ROCLOUD_q"],
        )

    reference = snapshot_result["ITCHI"]

    fields_to_check = {
        name: value
        for name, value in snapshot_result.items()
        if name
        in {
            "radius_km",
            "quadrant",
            "R_direct_q",
            "R34_q",
            "ROCLOUD_q",
            "M_direct",
            "M_indirect",
            "M_exterior",
            "H_P",
            "H_Pdir",
            "H_Pind",
            "H_W",
            "H_dir",
            "H_ind",
        }
    }

    validate_field_alignment(
        reference=reference,
        fields=fields_to_check,
        check_coords=check_coords,
    )
