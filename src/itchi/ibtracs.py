"""
IBTrACS NetCDF adapter for ITCHI.

This module converts IBTrACS-like NetCDF datasets into the internal ITCHI
track-table contract used by ``snapshot_inputs.py`` and ``tracks.py``.

The adapter is intentionally configurable because IBTrACS exposes several
agency-specific variables. The default configuration prioritizes common USA
fields:

- ``sid`` as storm identifier;
- ``iso_time`` as valid time;
- ``lat`` and ``lon`` as storm-center coordinates;
- ``usa_wind`` as maximum sustained wind;
- ``usa_pres`` as minimum central pressure;
- ``usa_rmw`` as radius of maximum wind;
- ``usa_r34`` as 34-kt wind radii by quadrant.

The output is a pandas DataFrame compatible with
``itchi.tracks.standardize_track_dataframe``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from itchi.tracks import standardize_track_dataframe
from itchi.units import normalize_quadrant_key

DEFAULT_IBTRACS_VARIABLE_MAP: dict[str, str] = {
    "storm_id": "sid",
    "time": "iso_time",
    "lat": "lat",
    "lon": "lon",
    "vmax_kt": "usa_wind",
    "pmin_hpa": "usa_pres",
    "rmw_km": "usa_rmw",
    "r34": "usa_r34",
}

DEFAULT_IBTRACS_R34_QUADRANTS: tuple[str, ...] = (
    "RNE",
    "RSE",
    "RSW",
    "RNW",
)

DEFAULT_IBTRACS_STORM_DIM = "storm"
DEFAULT_IBTRACS_TIME_DIM = "date_time"


def read_ibtracs_dataset(
    path: str | Path,
    engine: str | None = None,
    decode_times: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Read an IBTrACS-like NetCDF dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the IBTrACS NetCDF file.
    engine : str or None, default=None
        Optional xarray backend engine.
    decode_times : bool, default=False
        Whether xarray should decode time variables. IBTrACS commonly stores
        ``iso_time`` as strings, so ``False`` is a conservative default.
    **kwargs : Any
        Additional keyword arguments passed to ``xarray.open_dataset``.

    Returns
    -------
    xarray.Dataset
        Opened IBTrACS dataset.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"IBTrACS file not found: {input_path}")

    return xr.open_dataset(
        input_path,
        engine=engine,
        decode_times=decode_times,
        **kwargs,
    )


def _decode_scalar_text(value: Any) -> str:
    """
    Decode one scalar text value from NetCDF-like content.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()

    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="ignore").strip()

    return str(value).strip()


def _decode_text_value(value: Any) -> str:
    """
    Decode scalar or character-array NetCDF text into a Python string.
    """
    array = np.asarray(value)

    if array.ndim == 0:
        return _decode_scalar_text(array.item())

    flattened = array.ravel()

    if flattened.dtype.kind in {"S", "U"}:
        return "".join(_decode_scalar_text(item) for item in flattened).strip()

    return _decode_scalar_text(flattened[0])


def _decode_text_sequence(values: Any) -> list[str]:
    """
    Decode a one-dimensional string array or two-dimensional char array.
    """
    array = np.asarray(values)

    if array.ndim == 0:
        return [_decode_text_value(array)]

    if array.ndim == 1:
        return [_decode_text_value(value) for value in array]

    return [_decode_text_value(row) for row in array]


def _get_variable_name(
    variable_map: Mapping[str, str],
    canonical_name: str,
    required: bool = True,
) -> str | None:
    """
    Get a source variable name from a canonical-variable map.
    """
    variable_name = variable_map.get(canonical_name)

    if required and variable_name is None:
        raise KeyError(f"Missing variable mapping for {canonical_name!r}.")

    return variable_name


def _require_variable(dataset: xr.Dataset, variable_name: str) -> xr.DataArray:
    """
    Return a required variable from a Dataset.
    """
    if variable_name not in dataset:
        raise KeyError(f"IBTrACS variable {variable_name!r} not found.")

    return dataset[variable_name]


def _infer_storm_dimension(
    dataset: xr.Dataset,
    id_variable: str,
    storm_dim: str | None = None,
) -> str:
    """
    Infer the storm dimension from the identifier variable.
    """
    if storm_dim is not None:
        if storm_dim not in dataset.dims:
            raise KeyError(f"Storm dimension {storm_dim!r} not found.")

        return storm_dim

    if DEFAULT_IBTRACS_STORM_DIM in dataset.dims:
        return DEFAULT_IBTRACS_STORM_DIM

    id_data = _require_variable(dataset, id_variable)

    if not id_data.dims:
        raise ValueError("Storm identifier variable has no dimensions.")

    return id_data.dims[0]


def _infer_time_dimension(
    dataset: xr.Dataset,
    time_variable: str,
    storm_dim: str,
    time_dim: str | None = None,
) -> str:
    """
    Infer the time dimension from the time variable.
    """
    if time_dim is not None:
        if time_dim not in dataset.dims:
            raise KeyError(f"Time dimension {time_dim!r} not found.")

        return time_dim

    if DEFAULT_IBTRACS_TIME_DIM in dataset.dims:
        return DEFAULT_IBTRACS_TIME_DIM

    time_data = _require_variable(dataset, time_variable)

    for dim in time_data.dims:
        if dim != storm_dim:
            return dim

    raise ValueError("Could not infer IBTrACS time dimension.")


def get_ibtracs_storm_ids(
    dataset: xr.Dataset,
    id_variable: str = DEFAULT_IBTRACS_VARIABLE_MAP["storm_id"],
    storm_dim: str | None = None,
) -> list[str]:
    """
    Return decoded storm identifiers from an IBTrACS-like Dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        IBTrACS-like dataset.
    id_variable : str, default="sid"
        Variable containing storm identifiers.
    storm_dim : str or None, default=None
        Storm dimension. If None, it is inferred.

    Returns
    -------
    list[str]
        Decoded storm identifiers.
    """
    resolved_storm_dim = _infer_storm_dimension(
        dataset=dataset,
        id_variable=id_variable,
        storm_dim=storm_dim,
    )

    id_data = _require_variable(dataset, id_variable)
    storm_ids: list[str] = []

    for idx in range(dataset.sizes[resolved_storm_dim]):
        value = id_data.isel({resolved_storm_dim: idx}).values
        storm_ids.append(_decode_text_value(value))

    return storm_ids


def get_ibtracs_storm_index(
    dataset: xr.Dataset,
    storm_id: str,
    id_variable: str = DEFAULT_IBTRACS_VARIABLE_MAP["storm_id"],
    storm_dim: str | None = None,
) -> int:
    """
    Return the integer index of a storm identifier.

    Raises
    ------
    ValueError
        If the storm identifier is absent or duplicated.
    """
    storm_ids = get_ibtracs_storm_ids(
        dataset=dataset,
        id_variable=id_variable,
        storm_dim=storm_dim,
    )

    matches = [idx for idx, value in enumerate(storm_ids) if value == str(storm_id)]

    if not matches:
        raise ValueError(f"Storm {storm_id!r} not found in IBTrACS dataset.")

    if len(matches) > 1:
        raise ValueError(f"Storm {storm_id!r} appears multiple times.")

    return matches[0]


def select_ibtracs_storm(
    dataset: xr.Dataset,
    storm_id: str,
    id_variable: str = DEFAULT_IBTRACS_VARIABLE_MAP["storm_id"],
    storm_dim: str | None = None,
) -> xr.Dataset:
    """
    Select one storm from an IBTrACS-like Dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        IBTrACS-like dataset.
    storm_id : str
        Storm identifier to select.
    id_variable : str, default="sid"
        Identifier variable.
    storm_dim : str or None, default=None
        Storm dimension. If None, it is inferred.

    Returns
    -------
    xarray.Dataset
        Dataset sliced to a single storm.
    """
    resolved_storm_dim = _infer_storm_dimension(
        dataset=dataset,
        id_variable=id_variable,
        storm_dim=storm_dim,
    )

    storm_index = get_ibtracs_storm_index(
        dataset=dataset,
        storm_id=storm_id,
        id_variable=id_variable,
        storm_dim=resolved_storm_dim,
    )

    return dataset.isel({resolved_storm_dim: storm_index})


def _numeric_sequence(
    dataset: xr.Dataset,
    variable_name: str,
    time_dim: str,
) -> np.ndarray:
    """
    Extract a numeric one-dimensional time sequence.
    """
    values = _require_variable(dataset, variable_name)

    if time_dim not in values.dims:
        raise ValueError(
            f"Variable {variable_name!r} does not contain time dimension "
            f"{time_dim!r}."
        )

    return np.asarray(values.transpose(time_dim, ...).values).reshape(
        dataset.sizes[time_dim],
        -1,
    )[:, 0]


def _optional_numeric_sequence(
    dataset: xr.Dataset,
    variable_name: str | None,
    time_dim: str,
) -> np.ndarray | None:
    """
    Extract an optional numeric time sequence.
    """
    if variable_name is None or variable_name not in dataset:
        return None

    return _numeric_sequence(
        dataset=dataset,
        variable_name=variable_name,
        time_dim=time_dim,
    )


def _time_sequence(
    dataset: xr.Dataset,
    variable_name: str,
    time_dim: str,
) -> list[str]:
    """
    Extract and decode time strings from an IBTrACS-like variable.
    """
    time_data = _require_variable(dataset, variable_name)

    if time_dim not in time_data.dims:
        raise ValueError(
            f"Variable {variable_name!r} does not contain time dimension "
            f"{time_dim!r}."
        )

    ordered = time_data.transpose(time_dim, ...).values

    return _decode_text_sequence(ordered)


def _extract_r34_columns(
    dataset: xr.Dataset,
    r34_variable: str | None,
    time_dim: str,
    r34_quadrants: Sequence[str],
) -> dict[str, np.ndarray]:
    """
    Extract R34 quadrant columns from an IBTrACS-like radius variable.
    """
    if r34_variable is None or r34_variable not in dataset:
        return {}

    r34 = dataset[r34_variable]

    if time_dim not in r34.dims:
        raise ValueError(
            f"R34 variable {r34_variable!r} does not contain time dimension "
            f"{time_dim!r}."
        )

    non_time_dims = [dim for dim in r34.dims if dim != time_dim]

    if len(non_time_dims) != 1:
        raise ValueError(
            f"R34 variable {r34_variable!r} must have one non-time quadrant "
            "dimension after storm selection."
        )

    quadrant_dim = non_time_dims[0]
    values = np.asarray(r34.transpose(time_dim, quadrant_dim).values)

    if values.shape[1] < len(r34_quadrants):
        raise ValueError(
            f"R34 variable {r34_variable!r} has {values.shape[1]} quadrants, "
            f"but {len(r34_quadrants)} were requested."
        )

    columns: dict[str, np.ndarray] = {}

    for idx, quadrant_key in enumerate(r34_quadrants):
        quadrant = normalize_quadrant_key(quadrant_key).lower()
        columns[f"r34_{quadrant}"] = values[:, idx]

    return columns


def ibtracs_to_track_dataframe(
    dataset: xr.Dataset,
    storm_id: str,
    variable_map: Mapping[str, str] | None = None,
    r34_quadrants: Sequence[str] = DEFAULT_IBTRACS_R34_QUADRANTS,
    storm_dim: str | None = None,
    time_dim: str | None = None,
    drop_missing_core: bool = True,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Convert one IBTrACS storm to the internal ITCHI track DataFrame.

    Parameters
    ----------
    dataset : xarray.Dataset
        IBTrACS-like dataset containing one or more storms.
    storm_id : str
        Storm identifier to extract.
    variable_map : Mapping[str, str] or None, default=None
        Mapping from ITCHI canonical fields to IBTrACS variables. Missing
        optional mappings are ignored.
    r34_quadrants : sequence of str
        Quadrant order in the IBTrACS R34 variable.
    storm_dim : str or None, default=None
        Storm dimension. If None, it is inferred.
    time_dim : str or None, default=None
        Time dimension. If None, it is inferred.
    drop_missing_core : bool, default=True
        Whether to drop rows with missing time, latitude or longitude.
    standardize : bool, default=True
        Whether to pass the resulting table through
        ``standardize_track_dataframe``.

    Returns
    -------
    pandas.DataFrame
        ITCHI-compatible track table.
    """
    resolved_variable_map = dict(DEFAULT_IBTRACS_VARIABLE_MAP)

    if variable_map is not None:
        resolved_variable_map.update(variable_map)

    id_variable = _get_variable_name(resolved_variable_map, "storm_id")
    time_variable = _get_variable_name(resolved_variable_map, "time")
    lat_variable = _get_variable_name(resolved_variable_map, "lat")
    lon_variable = _get_variable_name(resolved_variable_map, "lon")

    assert id_variable is not None
    assert time_variable is not None
    assert lat_variable is not None
    assert lon_variable is not None

    selected = select_ibtracs_storm(
        dataset=dataset,
        storm_id=storm_id,
        id_variable=id_variable,
        storm_dim=storm_dim,
    )

    resolved_time_dim = _infer_time_dimension(
        dataset=selected,
        time_variable=time_variable,
        storm_dim=_infer_storm_dimension(dataset, id_variable, storm_dim),
        time_dim=time_dim,
    )

    times = _time_sequence(
        dataset=selected,
        variable_name=time_variable,
        time_dim=resolved_time_dim,
    )

    frame_data: dict[str, Any] = {
        "storm_id": [str(storm_id)] * len(times),
        "time": pd.to_datetime(times, errors="coerce"),
        "lat": _numeric_sequence(selected, lat_variable, resolved_time_dim),
        "lon": _numeric_sequence(selected, lon_variable, resolved_time_dim),
    }

    optional_mapping = {
        "vmax_kt": "vmax_kt",
        "pmin_hpa": "pmin_hpa",
        "rmw_km": "rmw_km",
    }

    for canonical_name, output_name in optional_mapping.items():
        variable_name = _get_variable_name(
            resolved_variable_map,
            canonical_name,
            required=False,
        )
        values = _optional_numeric_sequence(selected, variable_name, resolved_time_dim)

        if values is not None:
            frame_data[output_name] = values

    r34_variable = _get_variable_name(resolved_variable_map, "r34", required=False)
    frame_data.update(
        _extract_r34_columns(
            dataset=selected,
            r34_variable=r34_variable,
            time_dim=resolved_time_dim,
            r34_quadrants=r34_quadrants,
        )
    )

    result = pd.DataFrame(frame_data)

    if drop_missing_core:
        result = result.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

    if standardize:
        return standardize_track_dataframe(result)

    return result


def read_ibtracs_track_dataframe(
    path: str | Path,
    storm_id: str,
    variable_map: Mapping[str, str] | None = None,
    r34_quadrants: Sequence[str] = DEFAULT_IBTRACS_R34_QUADRANTS,
    storm_dim: str | None = None,
    time_dim: str | None = None,
    engine: str | None = None,
    decode_times: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read an IBTrACS NetCDF file and return one ITCHI track DataFrame.

    This is a convenience wrapper around ``read_ibtracs_dataset`` and
    ``ibtracs_to_track_dataframe``.
    """
    dataset = read_ibtracs_dataset(
        path=path,
        engine=engine,
        decode_times=decode_times,
        **kwargs,
    )

    try:
        return ibtracs_to_track_dataframe(
            dataset=dataset,
            storm_id=storm_id,
            variable_map=variable_map,
            r34_quadrants=r34_quadrants,
            storm_dim=storm_dim,
            time_dim=time_dim,
        )
    finally:
        dataset.close()
