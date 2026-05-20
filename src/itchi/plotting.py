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

SIAT_DANGER_COLORS: tuple[str, ...] = (
    "lightgrey",
    "dodgerblue",
    "limegreen",
    "yellow",
    "darkorange",
    "red",
)

SIAT_DANGER_BOUNDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

SIAT_DANGER_LABELS: tuple[str, ...] = (
    "0 No hazard / exterior",
    "1 Very low",
    "2 Low",
    "3 Moderate",
    "4 High",
    "5 Extreme",
)

DEFAULT_ITCHI_SIAT_THRESHOLDS: tuple[float, ...] = (
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
)

DEFAULT_CRAMERI_COLORMAP = "batlow"


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


def get_crameri_colormap(
    name: str = DEFAULT_CRAMERI_COLORMAP,
    reverse: bool = False,
) -> mcolors.Colormap:
    """
    Return a Fabio Crameri scientific colormap.

    Parameters
    ----------
    name : str, default="batlow"
        Name of the Crameri colormap available through ``cmcrameri.cm``.
    reverse : bool, default=False
        Whether to return the reversed colormap.

    Returns
    -------
    matplotlib.colors.Colormap
        Requested scientific colormap.

    Raises
    ------
    ImportError
        If ``cmcrameri`` is not installed.
    AttributeError
        If the requested colormap does not exist.
    """
    try:
        import cmcrameri.cm as cmc
    except ImportError as exc:
        raise ImportError(
            "cmcrameri is required for ITCHI scientific colormap defaults. "
            "Install it with `pip install cmcrameri` or reinstall "
            "itchi-core from pyproject dependencies."
        ) from exc

    colormap_name = f"{name}_r" if reverse else name

    try:
        return getattr(cmc, colormap_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Crameri colormap {colormap_name!r} was not found in cmcrameri.cm."
        ) from exc


def get_default_itchi_colormap() -> mcolors.Colormap:
    """
    Return the default continuous colormap for bounded ITCHI fields.

    The default is the Crameri ``batlow`` scientific colormap. It replaces
    the earlier ``viridis`` fallback for standard ITCHI consumption plots.
    """
    return get_crameri_colormap(DEFAULT_CRAMERI_COLORMAP)


def get_siat_hazard_colormap(
    name: str = "ITCHI_SIAT_Hazard",
) -> mcolors.ListedColormap:
    """
    Return the discrete SIAT-style hazard colormap for ITCHI classes.

    Colors:

    - 0: lightgrey
    - 1: dodgerblue
    - 2: limegreen
    - 3: yellow
    - 4: darkorange
    - 5: red
    """
    return mcolors.ListedColormap(SIAT_DANGER_COLORS, name=name)


def get_siat_hazard_norm(
    bounds: Sequence[int] = SIAT_DANGER_BOUNDS,
) -> mcolors.BoundaryNorm:
    """
    Return the BoundaryNorm associated with SIAT-style hazard classes.
    """
    cmap = get_siat_hazard_colormap()
    return mcolors.BoundaryNorm(bounds, cmap.N)


def get_siat_hazard_cmap_norm() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """
    Return SIAT-style colormap and norm as a pair.
    """
    cmap = get_siat_hazard_colormap()
    norm = mcolors.BoundaryNorm(SIAT_DANGER_BOUNDS, cmap.N)

    return cmap, norm


def _classify_itchi_array_to_siat(
    values: Any,
    thresholds: Sequence[float] = DEFAULT_ITCHI_SIAT_THRESHOLDS,
    clip: bool = True,
) -> np.ndarray:
    """
    Classify a NumPy-like ITCHI array into SIAT-style integer levels.
    """
    threshold_array = np.asarray(thresholds, dtype=float)

    if threshold_array.shape != (6,):
        raise ValueError("thresholds must contain exactly 6 values: 0, ..., 1.")

    if not np.all(np.diff(threshold_array) > 0):
        raise ValueError("thresholds must be strictly increasing.")

    array = np.asarray(values, dtype=float)
    working = np.clip(array, 0.0, 1.0) if clip else array
    levels = np.full(working.shape, np.nan, dtype=float)
    valid = np.isfinite(working)

    levels[valid & (working <= threshold_array[0])] = 0.0

    for level in range(1, 6):
        lower = threshold_array[level - 1]
        upper = threshold_array[level]
        mask = valid & (working > lower) & (working <= upper)
        levels[mask] = float(level)

    return levels


def classify_itchi_to_siat_levels(
    field: Any,
    thresholds: Sequence[float] = DEFAULT_ITCHI_SIAT_THRESHOLDS,
    clip: bool = True,
) -> Any:
    """
    Convert continuous ITCHI values in [0, 1] into SIAT-style classes 0-5.

    The default classification is:

    - ``0``: ITCHI <= 0.0
    - ``1``: 0.0 < ITCHI <= 0.2
    - ``2``: 0.2 < ITCHI <= 0.4
    - ``3``: 0.4 < ITCHI <= 0.6
    - ``4``: 0.6 < ITCHI <= 0.8
    - ``5``: 0.8 < ITCHI <= 1.0

    Parameters
    ----------
    field : array-like or xarray.DataArray
        Continuous ITCHI field.
    thresholds : sequence of float
        Six increasing threshold values. Defaults to equally spaced bounds
        from 0 to 1.
    clip : bool, default=True
        Whether to clip input values into [0, 1] before classification.

    Returns
    -------
    array-like or xarray.DataArray
        Integer-like SIAT hazard classes with NaN preserved.
    """
    import xarray as xr

    if isinstance(field, xr.DataArray):
        classified = xr.apply_ufunc(
            _classify_itchi_array_to_siat,
            field,
            kwargs={"thresholds": thresholds, "clip": clip},
            keep_attrs=True,
        )
        classified.name = f"{field.name or 'itchi'}_siat_level"
        classified.attrs["description"] = "SIAT-style ITCHI hazard level"
        classified.attrs["levels"] = "; ".join(SIAT_DANGER_LABELS)
        return classified

    return _classify_itchi_array_to_siat(
        values=field,
        thresholds=thresholds,
        clip=clip,
    )


def _build_color_kwargs(
    cmap: Any,
    norm: Any | None,
    vmin: float | None,
    vmax: float | None,
) -> dict[str, Any]:
    """
    Build Matplotlib color keyword arguments without mixing norm and vmin/vmax.
    """
    color_kwargs = {"cmap": cmap}

    if norm is not None:
        color_kwargs["norm"] = norm
    else:
        color_kwargs["vmin"] = vmin
        color_kwargs["vmax"] = vmax

    return color_kwargs


def plot_itchi_field(
    field: Any,
    ax: Any,
    lon: Any | None = None,
    lat: Any | None = None,
    crs: Any | None = None,
    cmap: Any | None = None,
    norm: Any | None = None,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    add_colorbar: bool = True,
    colorbar_label: str = "ITCHI",
    colorbar_ticks: Sequence[float] | None = None,
    colorbar_ticklabels: Sequence[str] | None = None,
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
        Colormap. If None, the default Crameri ITCHI colormap is used.
    norm : matplotlib norm or None
        Optional color normalization. When provided, vmin/vmax are not passed.
    vmin, vmax : float or None
        Color normalization bounds for continuous fields.
    add_colorbar : bool
        Whether to add a colorbar.
    colorbar_label : str
        Colorbar label.
    colorbar_ticks : sequence or None
        Optional colorbar tick positions.
    colorbar_ticklabels : sequence or None
        Optional colorbar tick labels.
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
    color_kwargs = _build_color_kwargs(
        cmap=resolved_cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    if lon is not None and lat is not None:
        artist = ax.pcolormesh(
            lon,
            lat,
            field,
            transform=resolved_crs,
            shading=pcolormesh_kwargs.pop("shading", "auto"),
            **color_kwargs,
            **pcolormesh_kwargs,
        )
    elif isinstance(field, xr.DataArray):
        artist = field.plot(
            ax=ax,
            add_colorbar=False,
            **color_kwargs,
            **pcolormesh_kwargs,
        )
    else:
        artist = ax.pcolormesh(
            field,
            shading=pcolormesh_kwargs.pop("shading", "auto"),
            **color_kwargs,
            **pcolormesh_kwargs,
        )

    if add_colorbar:
        colorbar = plt.colorbar(artist, ax=ax, ticks=colorbar_ticks)
        colorbar.set_label(colorbar_label)

        if colorbar_ticklabels is not None:
            colorbar.set_ticklabels(colorbar_ticklabels)

    return artist


def plot_itchi_siat_levels(
    field: Any,
    ax: Any,
    lon: Any | None = None,
    lat: Any | None = None,
    crs: Any | None = None,
    thresholds: Sequence[float] = DEFAULT_ITCHI_SIAT_THRESHOLDS,
    add_colorbar: bool = True,
    colorbar_label: str = "ITCHI hazard level",
    **pcolormesh_kwargs: Any,
) -> Any:
    """
    Classify continuous ITCHI and plot SIAT-style hazard levels.

    This function is intended for communication-oriented hazard maps where
    grey represents level 0 and red represents the maximum hazard class.
    """
    classified = classify_itchi_to_siat_levels(field, thresholds=thresholds)
    cmap, norm = get_siat_hazard_cmap_norm()

    return plot_itchi_field(
        field=classified,
        ax=ax,
        lon=lon,
        lat=lat,
        crs=crs,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        colorbar_label=colorbar_label,
        colorbar_ticks=np.arange(0.5, 6.5, 1.0),
        colorbar_ticklabels=SIAT_DANGER_LABELS,
        **pcolormesh_kwargs,
    )
