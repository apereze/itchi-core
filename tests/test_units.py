"""
Tests for ITCHI unit-conversion utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itchi.units import (
    convert_quadrant_radii_to_km,
    convert_radius_to_km,
    km_to_nautical_miles,
    nautical_miles_to_km,
    normalize_quadrant_key,
)


def test_nautical_miles_to_km_scalar() -> None:
    """
    Test scalar conversion from nautical miles to kilometers.
    """
    value = nautical_miles_to_km(10.0)

    assert value == pytest.approx(18.52)


def test_km_to_nautical_miles_scalar() -> None:
    """
    Test scalar conversion from kilometers to nautical miles.
    """
    value = km_to_nautical_miles(18.52)

    assert value == pytest.approx(10.0)


def test_convert_radius_to_km_from_km() -> None:
    """
    Test that kilometer inputs remain unchanged.
    """
    value = convert_radius_to_km(100.0, input_unit="km")

    assert value == pytest.approx(100.0)


def test_convert_radius_to_km_from_nautical_miles() -> None:
    """
    Test radius conversion from nautical miles to kilometers.
    """
    value = convert_radius_to_km(60.0, input_unit="nm")

    assert value == pytest.approx(111.12)


def test_convert_radius_to_km_xarray() -> None:
    """
    Test unit conversion preserving xarray objects.
    """
    radius_nm = xr.DataArray(
        data=[10.0, 20.0],
        dims=("point",),
        coords={"point": [0, 1]},
        name="radius_nm",
    )

    radius_km = convert_radius_to_km(radius_nm, input_unit="nm")

    expected = xr.DataArray(
        data=[18.52, 37.04],
        dims=("point",),
        coords={"point": [0, 1]},
        name="radius_nm",
    )

    xr.testing.assert_allclose(radius_km, expected)


def test_normalize_quadrant_key() -> None:
    """
    Test quadrant-key normalization to ITCHI convention.
    """
    assert normalize_quadrant_key("NE") == "RNE"
    assert normalize_quadrant_key("SE") == "RSE"
    assert normalize_quadrant_key("SW") == "RSW"
    assert normalize_quadrant_key("NW") == "RNW"

    assert normalize_quadrant_key("RNE") == "RNE"
    assert normalize_quadrant_key("RSE") == "RSE"
    assert normalize_quadrant_key("RSW") == "RSW"
    assert normalize_quadrant_key("RNW") == "RNW"


def test_normalize_quadrant_key_rejects_unknown_key() -> None:
    """
    Test that unknown quadrant keys raise an error.
    """
    with pytest.raises(KeyError):
        normalize_quadrant_key("north")


def test_convert_quadrant_radii_to_km_from_nm() -> None:
    """
    Test quadrant-specific conversion from nautical miles to kilometers.
    """
    radii_nm = {
        "RNE": 60.0,
        "RSE": 50.0,
        "RSW": 40.0,
        "RNW": 70.0,
    }

    radii_km = convert_quadrant_radii_to_km(
        radii_by_quadrant=radii_nm,
        input_unit="nm",
    )

    expected = {
        "RNE": 111.12,
        "RSE": 92.60,
        "RSW": 74.08,
        "RNW": 129.64,
    }

    for quadrant, expected_value in expected.items():
        assert radii_km[quadrant] == pytest.approx(expected_value)


def test_convert_quadrant_radii_accepts_short_aliases() -> None:
    """
    Test conversion when input keys use NE, SE, SW and NW.
    """
    radii_nm = {
        "NE": 60.0,
        "SE": 50.0,
        "SW": 40.0,
        "NW": 70.0,
    }

    radii_km = convert_quadrant_radii_to_km(
        radii_by_quadrant=radii_nm,
        input_unit="nm",
    )

    assert set(radii_km) == {"RNE", "RSE", "RSW", "RNW"}
    assert radii_km["RNE"] == pytest.approx(111.12)
    assert radii_km["RSE"] == pytest.approx(92.60)
    assert radii_km["RSW"] == pytest.approx(74.08)
    assert radii_km["RNW"] == pytest.approx(129.64)


def test_convert_quadrant_radii_missing_values_become_nan() -> None:
    """
    Test that missing or negative radii are converted to NaN.
    """
    radii_nm = {
        "RNE": -999.0,
        "RSE": 50.0,
        "RSW": None,
        "RNW": 70.0,
    }

    radii_km = convert_quadrant_radii_to_km(
        radii_by_quadrant=radii_nm,
        input_unit="nm",
    )

    assert np.isnan(radii_km["RNE"])
    assert radii_km["RSE"] == pytest.approx(92.60)
    assert np.isnan(radii_km["RSW"])
    assert radii_km["RNW"] == pytest.approx(129.64)


def test_convert_quadrant_radii_missing_quadrant_raises_error() -> None:
    """
    Test that missing required quadrants raise an error.
    """
    radii_nm = {
        "RNE": 60.0,
        "RSE": 50.0,
        "RSW": 40.0,
    }

    with pytest.raises(KeyError):
        convert_quadrant_radii_to_km(
            radii_by_quadrant=radii_nm,
            input_unit="nm",
        )


def test_convert_radius_to_km_rejects_unknown_unit() -> None:
    """
    Test that unsupported radius units raise an error.
    """
    with pytest.raises(ValueError):
        convert_radius_to_km(100.0, input_unit="meters")
