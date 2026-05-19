"""
Event-level ITCHI compiler.

This module orchestrates multiple ITCHI snapshot calculations for one
tropical cyclone event.

The compiler connects:

- pipeline.py      -> snapshot-level ITCHI calculation
- aggregation.py  -> event-level ITCHI products

Important design choice
-----------------------
The snapshot pipeline returns both spatial fields and scalar metadata.
Therefore, the compiler separates outputs into:

1. snapshots:
   Time-stacked spatial variables.

2. metadata:
   Per-snapshot non-spatial metadata, such as radius-resolution sources.

3. event_products:
   Event-level products such as ITCHI_max and ITCHI_acc.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from itchi.aggregation import compute_event_products
from itchi.pipeline import compute_itchi_snapshot_from_grid

SnapshotInput = Mapping[str, Any]
SnapshotResult = Mapping[str, Any]
CompiledEvent = dict[str, Any]


DEFAULT_SNAPSHOT_VARIABLES: tuple[str, ...] = (
    "radius_km",
    "quadrant",
    "R_direct_q",
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
    "ITCHI",
)


DEFAULT_METADATA_FIELDS: tuple[str, ...] = (
    "R_direct_source",
    "R_direct_filled_quadrants",
    "R_direct_used_fallback",
    "ROCLOUD_source",
    "ROCLOUD_filled_quadrants",
    "wind_hazard_source",
)


def _contains_xarray_object(*objects: Any) -> bool:
    """
    Return True if at least one object is an xarray DataArray or Dataset.
    """
    return any(isinstance(obj, xr.DataArray | xr.Dataset) for obj in objects)


def _validate_snapshot_inputs(snapshot_inputs: Sequence[SnapshotInput]) -> None:
    """
    Validate that at least one snapshot input is provided.
    """
    if len(snapshot_inputs) == 0:
        raise ValueError("At least one snapshot input is required.")


def _validate_time_values(
    n_snapshots: int,
    time_values: Sequence[Any] | None,
) -> list[Any] | None:
    """
    Validate optional time values.
    """
    if time_values is None:
        return None

    if len(time_values) != n_snapshots:
        raise ValueError(
            "Length of time_values must match number of snapshot inputs. "
            f"Received {len(time_values)} time values for {n_snapshots} snapshots."
        )

    return list(time_values)


def _stack_variable(
    values: Sequence[Any],
    time_values: Sequence[Any] | None = None,
    time_dim: str = "time",
) -> Any:
    """
    Stack one spatial variable across snapshots.
    """
    if len(values) == 0:
        raise ValueError("Cannot stack an empty sequence of values.")

    if any(_contains_xarray_object(value) for value in values):
        if not all(isinstance(value, xr.DataArray) for value in values):
            raise TypeError(
                "Cannot mix xarray and non-xarray outputs for the same variable."
            )

        if time_values is None:
            return xr.concat(values, dim=time_dim)

        time_coord = xr.DataArray(
            data=list(time_values),
            dims=(time_dim,),
            name=time_dim,
        )

        return xr.concat(values, dim=time_coord)

    return np.stack([np.asarray(value) for value in values], axis=0)


def stack_snapshot_results(
    snapshot_results: Sequence[SnapshotResult],
    variable_names: Sequence[str] = DEFAULT_SNAPSHOT_VARIABLES,
    time_values: Sequence[Any] | None = None,
    time_dim: str = "time",
) -> dict[str, Any]:
    """
    Stack selected spatial snapshot results along a temporal dimension.
    """
    if len(snapshot_results) == 0:
        raise ValueError("At least one snapshot result is required.")

    valid_time_values = _validate_time_values(
        n_snapshots=len(snapshot_results),
        time_values=time_values,
    )

    stacked: dict[str, Any] = {}

    for variable in variable_names:
        missing_indices = [
            idx for idx, result in enumerate(snapshot_results) if variable not in result
        ]

        if missing_indices:
            raise KeyError(
                f"Variable {variable!r} is missing from snapshot result "
                f"indices: {missing_indices}."
            )

        stacked[variable] = _stack_variable(
            values=[result[variable] for result in snapshot_results],
            time_values=valid_time_values,
            time_dim=time_dim,
        )

    return stacked


def extract_snapshot_metadata(
    snapshot_results: Sequence[SnapshotResult],
    metadata_fields: Sequence[str] = DEFAULT_METADATA_FIELDS,
    time_values: Sequence[Any] | None = None,
    time_key: str = "time",
) -> list[dict[str, Any]]:
    """
    Extract per-snapshot metadata from pipeline results.
    """
    valid_time_values = _validate_time_values(
        n_snapshots=len(snapshot_results),
        time_values=time_values,
    )

    metadata_records: list[dict[str, Any]] = []

    for idx, result in enumerate(snapshot_results):
        record: dict[str, Any] = {}

        if valid_time_values is not None:
            record[time_key] = valid_time_values[idx]

        for field in metadata_fields:
            if field in result:
                record[field] = result[field]

        metadata_records.append(record)

    return metadata_records


def _prepare_snapshot_input(
    snapshot_input: SnapshotInput,
    run_quality_control: bool | None,
) -> dict[str, Any]:
    """
    Prepare one snapshot input dictionary before calling the pipeline.
    """
    prepared = dict(snapshot_input)

    if run_quality_control is not None:
        prepared["run_quality_control"] = run_quality_control

    return prepared


def compile_itchi_event(
    snapshot_inputs: Sequence[SnapshotInput],
    time_values: Sequence[Any] | None = None,
    time_dim: str = "time",
    snapshot_variables: Sequence[str] = DEFAULT_SNAPSHOT_VARIABLES,
    metadata_fields: Sequence[str] = DEFAULT_METADATA_FIELDS,
    include_event_products: bool = True,
    run_snapshot_quality_control: bool | None = True,
) -> CompiledEvent:
    """
    Compile ITCHI outputs for one tropical cyclone event.

    Each element in snapshot_inputs must be a dictionary compatible with
    compute_itchi_snapshot_from_grid.

    Parameters
    ----------
    snapshot_inputs : sequence of Mapping
        Sequence of snapshot input dictionaries.
    time_values : sequence or None, default=None
        Optional time coordinate values for stacked snapshot outputs.
    time_dim : str, default="time"
        Name of the temporal dimension.
    snapshot_variables : sequence of str, default=DEFAULT_SNAPSHOT_VARIABLES
        Spatial variables to stack across time.
    metadata_fields : sequence of str, default=DEFAULT_METADATA_FIELDS
        Non-spatial metadata fields to extract per snapshot.
    include_event_products : bool, default=True
        Whether to compute ITCHI_max and ITCHI_acc from stacked ITCHI.
    run_snapshot_quality_control : bool or None, default=True
        If True or False, overrides the run_quality_control argument passed
        to each snapshot pipeline call. If None, each snapshot input controls
        its own quality-control setting.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - snapshots: stacked snapshot-level spatial variables;
        - metadata: per-snapshot metadata records;
        - event_products: event-level products, if requested.
    """
    _validate_snapshot_inputs(snapshot_inputs)

    valid_time_values = _validate_time_values(
        n_snapshots=len(snapshot_inputs),
        time_values=time_values,
    )

    snapshot_results = [
        compute_itchi_snapshot_from_grid(
            **_prepare_snapshot_input(
                snapshot_input=snapshot_input,
                run_quality_control=run_snapshot_quality_control,
            )
        )
        for snapshot_input in snapshot_inputs
    ]

    stacked_snapshots = stack_snapshot_results(
        snapshot_results=snapshot_results,
        variable_names=snapshot_variables,
        time_values=valid_time_values,
        time_dim=time_dim,
    )

    metadata = extract_snapshot_metadata(
        snapshot_results=snapshot_results,
        metadata_fields=metadata_fields,
        time_values=valid_time_values,
        time_key=time_dim,
    )

    compiled: CompiledEvent = {
        "snapshots": stacked_snapshots,
        "metadata": metadata,
    }

    if include_event_products:
        if "ITCHI" not in stacked_snapshots:
            raise KeyError(
                "Cannot compute event products because stacked snapshots "
                "do not contain 'ITCHI'."
            )

        compiled["event_products"] = compute_event_products(
            itchi=stacked_snapshots["ITCHI"],
            dim=time_dim,
            axis=0,
            skipna=True,
        )

    return compiled
