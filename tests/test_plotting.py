"""
Tests for plotting helper utilities.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import pytest

from itchi.plotting import (
    MapStyle,
    crear_colormap,
    create_segmented_colormap,
    get_default_itchi_colormap,
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


def test_get_default_itchi_colormap() -> None:
    """
    Test default ITCHI colormap retrieval.
    """
    cmap = get_default_itchi_colormap()

    assert isinstance(cmap, mcolors.Colormap)
