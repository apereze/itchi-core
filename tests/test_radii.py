"""
Tests for quadrant-radius utilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from itchi.radii import (
    build_uniform_quadrant_radius,
    fill_missing_quadrant_radii,
    resolve_attribution_radius,
    resolve_direct_radius,
    standardize_quadrant_radii,
)


def test_standardize_quadrant_radii_accepts_aliases() -> None:
    """
    Test that NE/SE/SW/NW aliases are standardized.
    """
    radii = {
        "NE": 100.0,
        "SE": 90.0,
        "SW": 80.0,
        "NW": 110.0,
    }

    result = standardize_quadrant_radii(radii)

    assert set(result) == {"RNE", "RSE", "RSW", "RNW"}
    assert result["RNE"] == pytest.approx(100.0)
    assert result["RSE"] == pytest.approx(90.0)
    assert result["RSW"] == pytest.approx(80.0)
    assert result["RNW"] == pytest.approx(110.0)


def test_fill_missing_quadrant_radii_with_mean() -> None:
    """
    Test filling missing quadrant values with mean of available values.
    """
    radii = {
        "RNE": 120.0,
        "RSE": 100.0,
        "RSW": np.nan,
        "RNW": 140.0,
    }

    result = fill_missing_quadrant_radii(radii)

    assert result.radii["RNE"] == pytest.approx(120.0)
    assert result.radii["RSE"] == pytest.approx(100.0)
    assert result.radii["RSW"] == pytest.approx(120.0)
    assert result.radii["RNW"] == pytest.approx(140.0)
    assert result.filled_quadrants == ("RSW",)
    assert result.used_fallback is False


def test_fill_missing_quadrant_radii_rejects_all_missing() -> None:
    """
    Test that all-missing quadrant radii cannot be mean-filled.
    """
    radii = {
        "RNE": np.nan,
        "RSE": None,
        "RSW": -999.0,
        "RNW": np.nan,
    }

    with pytest.raises(ValueError):
        fill_missing_quadrant_radii(radii)


def test_build_uniform_quadrant_radius() -> None:
    """
    Test uniform fallback radius construction.
    """
    result = build_uniform_quadrant_radius(
        radius_km=25.0,
        source="RMW_fallback",
    )

    assert result.radii == {
        "RNE": 25.0,
        "RSE": 25.0,
        "RSW": 25.0,
        "RNW": 25.0,
    }
    assert result.used_fallback is True
    assert result.filled_quadrants == ("RNE", "RSE", "RSW", "RNW")


def test_resolve_direct_radius_uses_available_r34_and_fills_missing() -> None:
    """
    Test direct-radius resolution from partially available R34.
    """
    r34 = {
        "RNE": 120.0,
        "RSE": 100.0,
        "RSW": np.nan,
        "RNW": 140.0,
    }

    result = resolve_direct_radius(
        r34_by_quadrant=r34,
        vmax_kt=45.0,
        rmw_km=30.0,
    )

    assert result.radii["RSW"] == pytest.approx(120.0)
    assert result.used_fallback is False
    assert result.filled_quadrants == ("RSW",)


def test_resolve_direct_radius_uses_rmw_for_tropical_depression() -> None:
    """
    Test that RMW is used when no R34 exists and Vmax < 34 kt.
    """
    r34 = {
        "RNE": np.nan,
        "RSE": np.nan,
        "RSW": np.nan,
        "RNW": np.nan,
    }

    result = resolve_direct_radius(
        r34_by_quadrant=r34,
        vmax_kt=30.0,
        rmw_km=22.0,
    )

    assert result.radii == {
        "RNE": 22.0,
        "RSE": 22.0,
        "RSW": 22.0,
        "RNW": 22.0,
    }
    assert result.used_fallback is True
    assert result.source == "RMW_fallback_for_tropical_depression"


def test_resolve_direct_radius_uses_configured_fallback() -> None:
    """
    Test fallback radius when no R34 and no RMW are available.
    """
    r34 = {
        "RNE": np.nan,
        "RSE": np.nan,
        "RSW": np.nan,
        "RNW": np.nan,
    }

    result = resolve_direct_radius(
        r34_by_quadrant=r34,
        vmax_kt=30.0,
        rmw_km=None,
        fallback_radius_km=15.0,
    )

    assert result.radii["RNE"] == pytest.approx(15.0)
    assert result.used_fallback is True
    assert result.source == "configured_fallback_radius"


def test_resolve_direct_radius_rejects_unresolvable_case() -> None:
    """
    Test that unresolved direct radius raises an error.
    """
    r34 = {
        "RNE": np.nan,
        "RSE": np.nan,
        "RSW": np.nan,
        "RNW": np.nan,
    }

    with pytest.raises(ValueError):
        resolve_direct_radius(
            r34_by_quadrant=r34,
            vmax_kt=30.0,
            rmw_km=None,
            fallback_radius_km=None,
        )


def test_resolve_attribution_radius_fills_missing_rocloud() -> None:
    """
    Test ROCLOUD radius resolution with missing quadrants.
    """
    rocloud = {
        "RNE": 500.0,
        "RSE": 400.0,
        "RSW": np.nan,
        "RNW": 600.0,
    }

    result = resolve_attribution_radius(rocloud)

    assert result.radii["RSW"] == pytest.approx(500.0)
    assert result.filled_quadrants == ("RSW",)
