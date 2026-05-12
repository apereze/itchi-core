"""
Tests for ITCHI quality-control utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itchi.quality_control import (
    run_snapshot_quality_control,
    validate_bounds,
    validate_exterior_zero,
    validate_field_alignment,
    validate_geometry_units,
    validate_masks_exclusive,
    validate_r34_le_rocloud,
)


def test_validate_bounds_accepts_valid_values() -> None:
    """
    Test that valid bounded fields pass.
    """
    values = np.array([0.0, 0.5, 1.0])

    validate_bounds(values, name="ITCHI")


def test_validate_bounds_rejects_invalid_values() -> None:
    """
    Test that values outside [0, 1] fail.
    """
    values = np.array([0.0, 1.2])

    with pytest.raises(ValueError):
        validate_bounds(values, name="ITCHI")


def test_validate_r34_le_rocloud_accepts_valid_radii() -> None:
    """
    Test that R34_q <= ROCLOUD_q passes.
    """
    r34 = np.array([50.0, 100.0, np.nan])
    rocloud = np.array([100.0, 100.0, 200.0])

    validate_r34_le_rocloud(r34, rocloud)


def test_validate_r34_le_rocloud_rejects_invalid_radii() -> None:
    """
    Test that R34_q > ROCLOUD_q fails.
    """
    r34 = np.array([50.0, 150.0])
    rocloud = np.array([100.0, 100.0])

    with pytest.raises(ValueError):
        validate_r34_le_rocloud(r34, rocloud)


def test_validate_exterior_zero_accepts_zero_exterior() -> None:
    """
    Test that ITCHI equal to zero in the exterior region passes.
    """
    itchi = np.array([0.5, 0.0, 0.0])
    exterior = np.array([False, True, True])

    validate_exterior_zero(itchi, exterior)


def test_validate_exterior_zero_rejects_nonzero_exterior() -> None:
    """
    Test that non-zero ITCHI in the exterior region fails.
    """
    itchi = np.array([0.5, 0.1, 0.0])
    exterior = np.array([False, True, True])

    with pytest.raises(ValueError):
        validate_exterior_zero(itchi, exterior)


def test_validate_masks_exclusive_accepts_valid_masks() -> None:
    """
    Test mutually exclusive and complete masks.
    """
    direct = np.array([True, False, False])
    indirect = np.array([False, True, False])
    exterior = np.array([False, False, True])

    validate_masks_exclusive(direct, indirect, exterior)


def test_validate_masks_exclusive_rejects_overlap() -> None:
    """
    Test overlapping masks fail.
    """
    direct = np.array([True, True, False])
    indirect = np.array([False, True, False])
    exterior = np.array([False, False, True])

    with pytest.raises(ValueError):
        validate_masks_exclusive(direct, indirect, exterior)


def test_validate_masks_exclusive_rejects_incomplete_masks() -> None:
    """
    Test incomplete masks fail when require_complete=True.
    """
    direct = np.array([True, False, False])
    indirect = np.array([False, True, False])
    exterior = np.array([False, False, False])

    with pytest.raises(ValueError):
        validate_masks_exclusive(direct, indirect, exterior)


def test_validate_field_alignment_numpy() -> None:
    """
    Test field alignment with NumPy arrays.
    """
    reference = np.array([0.0, 0.5, 1.0])
    fields = {
        "H_P": np.array([0.0, 0.5, 1.0]),
        "H_W": np.array([0.0, 0.2, 0.3]),
    }

    validate_field_alignment(reference, fields)


def test_validate_field_alignment_rejects_numpy_shape_mismatch() -> None:
    """
    Test field alignment fails when NumPy shapes differ.
    """
    reference = np.array([0.0, 0.5, 1.0])
    fields = {
        "H_P": np.array([[0.0, 0.5, 1.0]]),
    }

    with pytest.raises(ValueError):
        validate_field_alignment(reference, fields)


def test_validate_field_alignment_xarray() -> None:
    """
    Test field alignment with xarray objects.
    """
    reference = xr.DataArray(
        data=[[0.0, 0.5], [0.2, 1.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [15.0, 16.0],
            "lon": [-100.0, -99.0],
        },
    )

    fields = {
        "H_P": xr.DataArray(
            data=[[0.0, 0.5], [0.2, 1.0]],
            dims=("lat", "lon"),
            coords=reference.coords,
        ),
    }

    validate_field_alignment(reference, fields)


def test_validate_geometry_units_accepts_km() -> None:
    """
    Test that kilometer units pass.
    """
    validate_geometry_units(
        radius_unit="km",
        r34_unit="kilometers",
        rocloud_unit="kilometres",
    )


def test_validate_geometry_units_rejects_non_km() -> None:
    """
    Test that non-kilometer units fail.
    """
    with pytest.raises(ValueError):
        validate_geometry_units(
            radius_unit="km",
            r34_unit="nm",
            rocloud_unit="km",
        )


def test_run_snapshot_quality_control_accepts_valid_numpy_snapshot() -> None:
    """
    Test complete quality control on a valid synthetic NumPy snapshot.
    """
    result = {
        "radius_km": np.array([0.0, 75.0, 150.0]),
        "quadrant": np.array(["RNE", "RNE", "RNE"]),
        "R34_q": np.array([100.0, 100.0, 100.0]),
        "ROCLOUD_q": np.array([200.0, 200.0, 200.0]),
        "M_direct": np.array([True, True, False]),
        "M_indirect": np.array([False, False, True]),
        "M_exterior": np.array([False, False, False]),
        "H_P": np.array([0.2, 0.5, 0.7]),
        "H_Pdir": np.array([0.2, 0.5, 0.0]),
        "H_Pind": np.array([0.0, 0.0, 0.7]),
        "H_W": np.array([0.9, 0.4, 0.0]),
        "H_dir": np.array([0.92, 0.7, 0.0]),
        "H_ind": np.array([0.0, 0.0, 0.7]),
        "ITCHI": np.array([0.92, 0.7, 0.7]),
    }

    run_snapshot_quality_control(result)


def test_run_snapshot_quality_control_rejects_nonzero_exterior() -> None:
    """
    Test complete quality control fails for non-zero exterior ITCHI.
    """
    result = {
        "radius_km": np.array([0.0, 75.0, 250.0]),
        "quadrant": np.array(["RNE", "RNE", "RNE"]),
        "R34_q": np.array([100.0, 100.0, 100.0]),
        "ROCLOUD_q": np.array([200.0, 200.0, 200.0]),
        "M_direct": np.array([True, True, False]),
        "M_indirect": np.array([False, False, False]),
        "M_exterior": np.array([False, False, True]),
        "H_P": np.array([0.2, 0.5, 0.7]),
        "H_Pdir": np.array([0.2, 0.5, 0.0]),
        "H_Pind": np.array([0.0, 0.0, 0.0]),
        "H_W": np.array([0.9, 0.4, 0.0]),
        "H_dir": np.array([0.92, 0.7, 0.0]),
        "H_ind": np.array([0.0, 0.0, 0.0]),
        "ITCHI": np.array([0.92, 0.7, 0.1]),
    }

    with pytest.raises(ValueError):
        run_snapshot_quality_control(result)
