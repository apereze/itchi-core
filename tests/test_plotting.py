"""
Tests for plotting helper utilities.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.colors as mcolors
import pytest

from itchi.plotting import (
    SIAT_DANGER_BOUNDS,
    SIAT_DANGER_COLORS,
    MapStyle,
    classify_itchi_to_siat_levels,
    crear_colormap,
    create_segmented_colormap,
    get_crameri_colormap,
    get_default_itchi_colormap,
    get_siat_hazard_cmap_norm,
    get_siat_hazard_colormap,
    get_siat_hazard_norm,
)


def test_map_style_defaults() -> None:
    """
    Test default map style values.
    """
    style = MapStyle()

    assert style.lon_extent == (-120.0, -80.0)
    assert style.lat_extent == (10.0, 35.0)
    assert style.grid_spacing == pytest.approx(4.0)


def test_create_segmented_colormap() -> None:
    """
    Test custom segmented colormap creation.
    """
    cmap = create_segmented_colormap(
        [
            (0.0, 0.0, 1.0),
            0.5,
            (1.0, 1.0, 1.0),
            0.8,
            (1.0, 0.0, 0.0),
        ],
        name="TestMap",
    )

    assert isinstance(cmap, mcolors.LinearSegmentedColormap)
    assert cmap.name == "TestMap"


def test_crear_colormap_alias() -> None:
    """
    Test backward-compatible Spanish colormap alias.
    """
    cmap = crear_colormap(
        [
            (0.0, 0.0, 1.0),
            0.5,
            (1.0, 1.0, 1.0),
            0.8,
            (1.0, 0.0, 0.0),
        ]
    )

    assert isinstance(cmap, mcolors.LinearSegmentedColormap)


def test_create_segmented_colormap_rejects_invalid_sequence() -> None:
    """
    Test malformed colormap sequence validation.
    """
    with pytest.raises(ValueError):
        create_segmented_colormap([0.5, (1.0, 0.0, 0.0)])


def test_get_crameri_colormap() -> None:
    """
    Test Crameri colormap retrieval.
    """
    cmap = get_crameri_colormap("batlow")

    assert isinstance(cmap, mcolors.Colormap)


def test_get_default_itchi_colormap_uses_crameri() -> None:
    """
    Test default ITCHI colormap retrieval.
    """
    cmap = get_default_itchi_colormap()

    assert isinstance(cmap, mcolors.Colormap)
    assert cmap.name == "batlow"


def test_get_siat_hazard_colormap() -> None:
    """
    Test SIAT-style discrete hazard colormap.
    """
    cmap = get_siat_hazard_colormap()

    assert isinstance(cmap, mcolors.ListedColormap)
    assert cmap.N == 6
    assert tuple(cmap.colors) == SIAT_DANGER_COLORS


def test_get_siat_hazard_norm() -> None:
    """
    Test SIAT-style hazard norm.
    """
    norm = get_siat_hazard_norm()

    assert isinstance(norm, mcolors.BoundaryNorm)
    np.testing.assert_array_equal(norm.boundaries, np.asarray(SIAT_DANGER_BOUNDS))


def test_get_siat_hazard_cmap_norm() -> None:
    """
    Test paired SIAT-style colormap and norm.
    """
    cmap, norm = get_siat_hazard_cmap_norm()

    assert isinstance(cmap, mcolors.ListedColormap)
    assert isinstance(norm, mcolors.BoundaryNorm)


def test_classify_itchi_to_siat_levels_numpy() -> None:
    """
    Test conversion from continuous ITCHI to SIAT-style classes.
    """
    values = np.array([0.0, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0, np.nan])

    result = classify_itchi_to_siat_levels(values)

    expected = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_classify_itchi_to_siat_levels_xarray() -> None:
    """
    Test xarray-preserving ITCHI classification.
    """
    field = xr.DataArray(
        data=np.array([[0.0, 0.3], [0.7, 1.0]]),
        dims=("y", "x"),
        name="ITCHI",
    )

    result = classify_itchi_to_siat_levels(field)

    assert isinstance(result, xr.DataArray)
    assert result.name == "ITCHI_siat_level"
    np.testing.assert_allclose(result.values, np.array([[0.0, 2.0], [4.0, 5.0]]))


def test_classify_itchi_to_siat_rejects_invalid_thresholds() -> None:
    """
    Test SIAT classification threshold validation.
    """
    with pytest.raises(ValueError):
        classify_itchi_to_siat_levels(np.array([0.1]), thresholds=[0.0, 0.5, 1.0])

    with pytest.raises(ValueError):
        classify_itchi_to_siat_levels(
            np.array([0.1]),
            thresholds=[0.0, 0.2, 0.2, 0.6, 0.8, 1.0],
        )
