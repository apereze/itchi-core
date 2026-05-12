"""
Input/output utilities for ITCHI.

This module centralizes reading and writing operations for ITCHI products.

The scientific calculation should remain in modules such as:

- precipitation.py
- geometry.py
- masks.py
- components.py
- index.py
- pipeline.py
- aggregation.py

This module only handles conversion to xarray datasets and file I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

XARRAY_NETCDF_EXTENSIONS: tuple[str, ...] = (".nc", ".nc4", ".netcdf")
XARRAY_ZARR_EXTENSIONS: tuple[str, ...] = (".zarr",)


def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path.

    Returns
    -------
    pathlib.Path
        Created or existing directory path.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def ensure_parent_directory(path: str | Path) -> Path:
    """
    Create the parent directory for a file path if needed.

    Parameters
    ----------
    path : str or pathlib.Path
        File path.

    Returns
    -------
    pathlib.Path
        File path as pathlib.Path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def infer_xarray_format(path: str | Path) -> str:
    """
    Infer xarray storage format from file extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Output or input path.

    Returns
    -------
    str
        Either "netcdf" or "zarr".

    Raises
    ------
    ValueError
        If the format cannot be inferred.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in XARRAY_NETCDF_EXTENSIONS:
        return "netcdf"

    if suffix in XARRAY_ZARR_EXTENSIONS:
        return "zarr"

    raise ValueError(
        "Could not infer xarray format from path extension. "
        "Use .nc, .nc4, .netcdf or .zarr."
    )


def _filter_result_variables(
    result: Mapping[str, Any],
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Filter result variables before conversion to an xarray Dataset.
    """
    selected = dict(result)

    if include is not None:
        include_set = set(include)
        selected = {key: value for key, value in selected.items() if key in include_set}

    if exclude is not None:
        exclude_set = set(exclude)
        selected = {
            key: value for key, value in selected.items() if key not in exclude_set
        }

    return selected


def _numpy_to_dataarray(
    value: Any,
    name: str,
    dims: Sequence[str] | None = None,
    coords: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """
    Convert a NumPy-like value to an xarray DataArray.
    """
    array = np.asarray(value)

    if array.ndim == 0:
        return xr.DataArray(array, name=name)

    if dims is None:
        raise ValueError(
            f"dims must be provided to convert non-scalar variable {name!r} "
            "to an xarray DataArray."
        )

    if len(dims) != array.ndim:
        raise ValueError(
            f"Variable {name!r} has {array.ndim} dimensions, "
            f"but {len(dims)} dimension names were provided."
        )

    usable_coords = None

    if coords is not None:
        usable_coords = {dim: coord for dim, coord in coords.items() if dim in dims}

    return xr.DataArray(
        data=array,
        dims=tuple(dims),
        coords=usable_coords,
        name=name,
    )


def result_to_dataset(
    result: Mapping[str, Any],
    dims: Sequence[str] | None = None,
    coords: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> xr.Dataset:
    """
    Convert an ITCHI result dictionary to an xarray Dataset.

    Parameters
    ----------
    result : Mapping[str, Any]
        Dictionary returned by ITCHI pipeline or aggregation functions.
    dims : sequence of str or None, default=None
        Dimension names used when converting NumPy arrays.
        Required for non-scalar NumPy values.
    coords : Mapping[str, Any] or None, default=None
        Coordinates used when converting NumPy arrays.
    attrs : Mapping[str, Any] or None, default=None
        Global dataset attributes.
    include : sequence of str or None, default=None
        Optional list of variables to keep.
    exclude : sequence of str or None, default=None
        Optional list of variables to exclude.

    Returns
    -------
    xarray.Dataset
        Dataset containing selected ITCHI variables.

    Raises
    ------
    TypeError
        If a value cannot be converted to an xarray variable.
    ValueError
        If dimension metadata are insufficient.
    """
    selected = _filter_result_variables(
        result=result,
        include=include,
        exclude=exclude,
    )

    dataset = xr.Dataset(attrs=dict(attrs or {}))

    for name, value in selected.items():
        if isinstance(value, xr.DataArray):
            dataset[name] = value
            continue

        if isinstance(value, xr.Dataset):
            raise TypeError(
                f"Variable {name!r} is an xarray Dataset. "
                "result_to_dataset expects scalar, NumPy-like or DataArray values."
            )

        dataset[name] = _numpy_to_dataarray(
            value=value,
            name=name,
            dims=dims,
            coords=coords,
        )

    return dataset


def write_dataset(
    dataset: xr.Dataset,
    path: str | Path,
    file_format: str | None = None,
    overwrite: bool = True,
) -> Path:
    """
    Write an xarray Dataset to NetCDF or Zarr.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to write.
    path : str or pathlib.Path
        Output path.
    file_format : {"netcdf", "zarr"} or None, default=None
        Output format. If None, inferred from path extension.
    overwrite : bool, default=True
        Whether to overwrite an existing file or store.

    Returns
    -------
    pathlib.Path
        Written output path.

    Raises
    ------
    FileExistsError
        If output exists and overwrite=False.
    ValueError
        If the file format is unsupported.
    """
    output_path = ensure_parent_directory(path)

    fmt = file_format or infer_xarray_format(output_path)
    fmt = fmt.lower().strip()

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    if fmt == "netcdf":
        dataset.to_netcdf(output_path)
        return output_path

    if fmt == "zarr":
        mode = "w" if overwrite else "w-"
        dataset.to_zarr(output_path, mode=mode)
        return output_path

    raise ValueError(f"Unsupported xarray output format: {file_format}")


def read_dataset(
    path: str | Path,
    file_format: str | None = None,
    chunks: Mapping[str, int] | str | None = None,
) -> xr.Dataset:
    """
    Read an xarray Dataset from NetCDF or Zarr.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path.
    file_format : {"netcdf", "zarr"} or None, default=None
        Input format. If None, inferred from path extension.
    chunks : mapping, str or None, default=None
        Optional chunk configuration for xarray/dask.

    Returns
    -------
    xarray.Dataset
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the format is unsupported.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    fmt = file_format or infer_xarray_format(input_path)
    fmt = fmt.lower().strip()

    if fmt == "netcdf":
        return xr.open_dataset(input_path, chunks=chunks)

    if fmt == "zarr":
        return xr.open_zarr(input_path, chunks=chunks)

    raise ValueError(f"Unsupported xarray input format: {file_format}")


def write_result(
    result: Mapping[str, Any],
    path: str | Path,
    dims: Sequence[str] | None = None,
    coords: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    file_format: str | None = None,
    overwrite: bool = True,
) -> Path:
    """
    Convert an ITCHI result dictionary to a Dataset and write it.

    Parameters
    ----------
    result : Mapping[str, Any]
        Dictionary returned by ITCHI pipeline, aggregation or compiler.
    path : str or pathlib.Path
        Output path.
    dims : sequence of str or None, default=None
        Dimension names for NumPy-like values.
    coords : Mapping[str, Any] or None, default=None
        Coordinates for NumPy-like values.
    attrs : Mapping[str, Any] or None, default=None
        Dataset-level metadata.
    include : sequence of str or None, default=None
        Optional variables to keep.
    exclude : sequence of str or None, default=None
        Optional variables to exclude.
    file_format : {"netcdf", "zarr"} or None, default=None
        Output format. If None, inferred from path extension.
    overwrite : bool, default=True
        Whether to overwrite existing output.

    Returns
    -------
    pathlib.Path
        Written output path.
    """
    dataset = result_to_dataset(
        result=result,
        dims=dims,
        coords=coords,
        attrs=attrs,
        include=include,
        exclude=exclude,
    )

    return write_dataset(
        dataset=dataset,
        path=path,
        file_format=file_format,
        overwrite=overwrite,
    )
