"""
Plotting and cartographic helpers for ITCHI notebooks.

These functions are intentionally separated from the scientific core. They
provide reusable map conventions for exploratory validation, diagnostics and
figures without changing ITCHI calculations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import numpy as np


@dataclass(frozen=True)
class MapStyle:
    """
    Styling options for ITCHI map axes.

    The defaults are tuned for tropical cyclone maps over Mexico, Central
    America, the Gulf of Mexico, the Caribbean and the eastern Pacific.
    """

    lon_extent: tuple[float, float] = (-120.0, -80.0)
    lat_extent: tuple[float, float] = (10.0, 35.0)
    coastline_linewidth: float = 1.2
    border_linewidth: float = 1.0
    spine_linewidth: float = 1.5
    tick_width: float = 1.5
    tick_size: float = 6.0
    label_size: float = 11.0
    grid_spacing: float = 4.0
    grid_color: str = "0.5"
    grid_alpha: float = 0.35
    grid_linestyle: str = "--"
    land_color: str = "lightgray"
    land_alpha: float = 0.8
    lon_tick_spacing: float = 5.0
    lat_tick_spacing: float = 5.0
    feature_scale: str = "50m"


def configure_map_axis(
    ax: Any,
    crs: Any | None = None,
    style: MapStyle | None = None,
    add_land: bool = True,
    add_coastline: bool = True,
    add_borders: bool = True,
    add_gridlines: bool = True,
) -> Any:
    """
    Configure a Cartopy GeoAxes with ITCHI map conventions.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axis to configure.
    crs : cartopy.crs projection or None, default=None
        Coordinate reference system used by longitude/latitude data. If None,
        PlateCarree is created internally.
    style : MapStyle or None, default=None
        Map styling options.
    add_land, add_coastline, add_borders, add_gridlines : bool
        Whether to add common geographic elements.

    Returns
    -------
    axis
        The configured axis.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    resolved_crs = crs or ccrs.PlateCarree()
    resolved_style = style or MapStyle()

    if add_land:
        ax.add_feature(
            cfeature.LAND.with_scale(resolved_style.feature_scale),
            facecolor=resolved_style.land_color,
            alpha=resolved_style.land_alpha,
            zorder=0,
        )

    if add_coastline:
        ax.add_feature(
            cfeature.COASTLINE.with_scale(resolved_style.feature_scale),
            linewidth=resolved_style.coastline_linewidth,
            zorder=4,
        )

    if add_borders:
        ax.add_feature(
            cfeature.BORDERS.with_scale(resolved_style.feature_scale),
            linewidth=resolved_style.border_linewidth,
            zorder=4,
        )

    if "geo" in ax.spines:
        ax.spines["geo"].set_linewidth(resolved_style.spine_linewidth)

    ax.tick_params(
        axis="both",
        width=resolved_style.tick_width,
        size=resolved_style.tick_size,
        which="both",
        labelsize=resolved_style.label_size,
    )

    lon_min, lon_max = resolved_style.lon_extent
    lat_min, lat_max = resolved_style.lat_extent

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=resolved_crs)

    ax.set_xticks(
        np.arange(
            lon_min,
            lon_max + resolved_style.lon_tick_spacing,
            resolved_style.lon_tick_spacing,
        ),
        crs=resolved_crs,
    )
    ax.set_yticks(
        np.arange(
            lat_min,
            lat_max + resolved_style.lat_tick_spacing,
            resolved_style.lat_tick_spacing,
        ),
        crs=resolved_crs,
    )
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    if add_gridlines:
        ax.gridlines(
            draw_labels=False,
            linewidth=0.8,
            color=resolved_style.grid_color,
            alpha=resolved_style.grid_alpha,
            linestyle=resolved_style.grid_linestyle,
            zorder=3,
            xlocs=np.arange(lon_min, lon_max + 1, resolved_style.grid_spacing),
            ylocs=np.arange(lat_min, lat_max + 1, resolved_style.grid_spacing),
        )

    return ax


def configurar_mapa(
    ax: Any,
    crs: Any,
    lon_extent: tuple[float, float] = (-120.0, -80.0),
    lat_extent: tuple[float, float] = (10.0, 35.0),
    linewidth: float = 1.2,
    tick_width: float = 1.5,
    tick_size: float = 6.0,
    label_size: float = 11.0,
    grid_spacing: float = 4.0,
    grid_color: str = "0.5",
    grid_alpha: float = 0.35,
    grid_linestyle: str = "--",
    land_color: str = "lightgray",
    land_alpha: float = 0.8,
    lon_tick_spacing: float = 5.0,
    lat_tick_spacing: float = 5.0,
) -> Any:
    """
    Backward-compatible Spanish wrapper for :func:`configure_map_axis`.

    The previous notebook helper is preserved as an alias with improved
    defaults and cleaner internals. New code should prefer
    ``configure_map_axis`` and ``MapStyle``.
    """
    style = MapStyle(
        lon_extent=lon_extent,
        lat_extent=lat_extent,
        coastline_linewidth=linewidth,
        border_linewidth=linewidth,
        spine_linewidth=tick_width,
        tick_width=tick_width,
        tick_size=tick_size,
        label_size=label_size,
        grid_spacing=grid_spacing,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        grid_linestyle=grid_linestyle,
        land_color=land_color,
        land_alpha=land_alpha,
        lon_tick_spacing=lon_tick_spacing,
        lat_tick_spacing=lat_tick_spacing,
    )

    return configure_map_axis(ax=ax, crs=crs, style=style)


def create_segmented_colormap(
    sequence: Sequence[float | tuple[float, float, float]],
    name: str = "ITCHICustomMap",
) -> mcolors.LinearSegmentedColormap:
    """
    Create a LinearSegmentedColormap from color-transition sequence.

    Parameters
    ----------
    sequence : sequence
        Alternating RGB tuples and floats. Floats must be increasing and in
        the interval [0, 1]. Example:

        ``[(0.0, 0.0, 1.0), 0.5, (1.0, 1.0, 1.0), 0.8, (1.0, 0.0, 0.0)]``

    name : str, default="ITCHICustomMap"
        Colormap name.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap.
    """
    seq = [(None, None, None), 0.0, *list(sequence), 1.0, (None, None, None)]
    color_dict: dict[str, list[list[float | None]]] = {
        "red": [],
        "green": [],
        "blue": [],
    }

    for idx, item in enumerate(seq):
        if isinstance(item, float):
            previous_color = seq[idx - 1]
            next_color = seq[idx + 1]

            if not isinstance(previous_color, tuple) or not isinstance(
                next_color, tuple
            ):
                raise ValueError("Floats in sequence must be surrounded by RGB tuples.")

            r1, g1, b1 = previous_color
            r2, g2, b2 = next_color
            color_dict["red"].append([item, r1, r2])
            color_dict["green"].append([item, g1, g2])
            color_dict["blue"].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap(name, color_dict)


def crear_colormap(
    seq: Sequence[float | tuple[float, float, float]],
) -> mcolors.LinearSegmentedColormap:
    """
    Backward-compatible Spanish wrapper for custom colormap creation.
    """
    return create_segmented_colormap(seq)


def get_default_itchi_colormap() -> mcolors.Colormap:
    """
    Return a perceptually simple default colormap for bounded ITCHI fields.

    The returned colormap is suitable for fields in [0, 1]. It uses a
    built-in Matplotlib colormap to avoid adding external dependencies or
    hard-coded color tables.
    """
    import matplotlib.pyplot as plt

    return plt.get_cmap("viridis")


def plot_itchi_field(
    field: Any,
    ax: Any,
    lon: Any | None = None,
    lat: Any | None = None,
    crs: Any | None = None,
    cmap: Any | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    add_colorbar: bool = True,
    colorbar_label: str = "ITCHI",
    **pcolormesh_kwargs: Any,
) -> Any:
    """
    Plot one ITCHI-like bounded field on a Matplotlib/Cartopy axis.

    Parameters
    ----------
    field : array-like or xarray.DataArray
        Field to plot.
    ax : matplotlib axis
        Target axis.
    lon, lat : array-like or None
        Optional longitude and latitude grids. If omitted, the field is
        plotted using xarray's default plotting or array indices.
    crs : cartopy.crs projection or None
        Transform used for lon/lat plotting.
    cmap : colormap or None
        Colormap. If None, a default ITCHI colormap is used.
    vmin, vmax : float
        Color normalization bounds.
    add_colorbar : bool
        Whether to add a colorbar.
    colorbar_label : str
        Colorbar label.
    **pcolormesh_kwargs : Any
        Additional keyword arguments passed to pcolormesh.

    Returns
    -------
    matplotlib artist
        The generated mappable artist.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import xarray as xr

    resolved_crs = crs or ccrs.PlateCarree()
    resolved_cmap = cmap or get_default_itchi_colormap()

    if lon is not None and lat is not None:
        artist = ax.pcolormesh(
            lon,
            lat,
            field,
            transform=resolved_crs,
            cmap=resolved_cmap,
            vmin=vmin,
            vmax=vmax,
            shading=pcolormesh_kwargs.pop("shading", "auto"),
            **pcolormesh_kwargs,
        )
    elif isinstance(field, xr.DataArray):
        artist = field.plot(
            ax=ax,
            cmap=resolved_cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            **pcolormesh_kwargs,
        )
    else:
        artist = ax.pcolormesh(
            field,
            cmap=resolved_cmap,
            vmin=vmin,
            vmax=vmax,
            shading=pcolormesh_kwargs.pop("shading", "auto"),
            **pcolormesh_kwargs,
        )

    if add_colorbar:
        colorbar = plt.colorbar(artist, ax=ax)
        colorbar.set_label(colorbar_label)

    return artist
